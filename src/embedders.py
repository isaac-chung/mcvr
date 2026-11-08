"""Text embedders behind one interface. The embedder is chosen in the config."""

from __future__ import annotations

import base64
import hashlib
import logging
import threading
import time
from typing import Protocol

import numpy as np

from src.cache import Cache, shared_cache
from src.types import Cost


class Embedder(Protocol):
    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        """L2-normalized float32 vectors; kind is 'query' or 'document'."""
        ...


class SentenceTransformerEmbedder:
    """Local sentence-transformers model (for example Jasper-Token-Compression-600M).

    Extra encode arguments (such as Jasper's compression_ratio) come from the config.
    """

    def __init__(
        self,
        model: str,
        batch_size: int = 16,
        encode_kwargs: dict | None = None,
        query_prompt_name: str | None = "query",
        dtype: str = "float32",
        device: str | None = None,
        query_prefix: str = "",
    ):
        import torch
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(
            model,
            trust_remote_code=True,
            model_kwargs={"torch_dtype": getattr(torch, dtype)},
            tokenizer_kwargs={"padding_side": "left"},
            device=device,
        )
        self.batch_size = batch_size
        self.encode_kwargs = encode_kwargs or {}
        self.query_prompt_name = query_prompt_name
        self.query_prefix = query_prefix

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        prompt = self.query_prompt_name if kind == "query" else None
        if kind == "query" and self.query_prefix:
            texts = [self.query_prefix + t for t in texts]
        vectors = self.model.encode(
            texts,
            batch_size=self.batch_size,
            prompt_name=prompt,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 64,
            **self.encode_kwargs,
        )
        return vectors.astype(np.float32)


class HashEmbedder:
    """Deterministic bag-of-words embedder for tests: no model download."""

    def __init__(self, dim: int = 256):
        self.dim = dim

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            for word in text.lower().split():
                h = int(hashlib.md5(word.encode()).hexdigest(), 16)
                out[i, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norms, 1e-9)


RETRY_STATUS = {429, 500, 502, 503, 504}
OPENAI_DIMS = {"text-embedding-3-small": 1536, "text-embedding-3-large": 3072}


def retryable(error: Exception) -> bool:
    """Rate limits, server errors and dropped connections (openai's exception types)."""
    status = getattr(error, "status_code", None)
    return status in RETRY_STATUS or type(error).__name__ in (
        "APIConnectionError",
        "APITimeoutError",
    )


MAX_BATCH_TOKENS = 200_000  # per embeddings request (API limit 300k)
MAX_INPUT_TOKENS = 8191  # per input text (API limit for the text-embedding-3 models)
log = logging.getLogger(__name__)
_ENCODING = None


def _cl100k():
    global _ENCODING
    if _ENCODING is None:
        import tiktoken

        _ENCODING = tiktoken.get_encoding("cl100k_base")
    return _ENCODING


def truncate_tokens(text: str, max_tokens: int) -> str:
    """text cut to its first max_tokens cl100k_base tokens (the text-embedding-3 tokenizer);
    returned unchanged, without tokenizing, when it cannot be longer (a token is at least one
    UTF-8 byte)."""
    if len(text.encode("utf-8")) <= max_tokens:
        return text
    tokens = _cl100k().encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    return _cl100k().decode(tokens[:max_tokens])


class OpenAIEmbedder:
    """OpenAI embeddings API (key in OPENAI_API_KEY), e.g. text-embedding-3-small.

    Texts are sent in batches, retried with backoff on rate limits, and cached per (model, text)
    so each text is paid for once across runs; queries and documents are embedded alike. `cost`
    counts the API calls and tokens actually paid (cache hits are free) and `versions` the model
    names the API returned.

    A text over max_tokens tokens (the API's per-input limit, 8,191) is embedded as its first
    max_tokens tokens, and cached under that truncated text; texts within the limit are sent and
    cached exactly as before. `truncated` holds the distinct texts cut so far (logged as a
    warning per encode call).
    """

    def __init__(
        self,
        model: str,
        cache: Cache | None = None,
        batch_size: int = 256,
        client=None,
        retries: int = 6,
        max_tokens: int | None = MAX_INPUT_TOKENS,
    ):
        if client is None:
            import openai

            client = openai.OpenAI(max_retries=0)  # we retry ourselves, with longer waits
        self.model, self.cache, self.batch_size = model, cache, batch_size
        self.client, self.retries, self.sleep = client, retries, time.sleep
        self.cost, self.versions = Cost(), set()
        self.max_tokens = max_tokens
        self.truncated: set[str] = set()
        self._lock = threading.Lock()

    @property
    def namespace(self) -> str:
        return f"embed:openai:{self.model}"

    def _batches(self, texts: list[str]) -> list[list[str]]:
        """At most batch_size texts and about MAX_BATCH_TOKENS tokens per request (the API
        rejects requests over 300k tokens; tokens are estimated as characters / 3)."""
        out: list[list[str]] = []
        batch: list[str] = []
        tokens = 0.0
        for text in texts:
            n = len(text) / 3 + 1
            if batch and (len(batch) >= self.batch_size or tokens + n > MAX_BATCH_TOKENS):
                out.append(batch)
                batch, tokens = [], 0.0
            batch.append(text)
            tokens += n
        return out + ([batch] if batch else [])

    def _create(self, batch: list[str]):
        for attempt in range(self.retries + 1):
            try:
                return self.client.embeddings.create(
                    model=self.model, input=batch, encoding_format="float"
                )
            except Exception as error:
                if attempt == self.retries or not retryable(error):
                    raise
                self.sleep(min(60.0, 2.0**attempt))

    def _fetch(self, batch: list[str]) -> dict[str, np.ndarray]:
        start = time.perf_counter()
        resp = self._create(batch)
        rows = sorted(resp.data, key=lambda d: d.index)
        found = {t: np.asarray(d.embedding, np.float32) for t, d in zip(batch, rows, strict=True)}
        cost = Cost(
            llm_calls=1,
            tokens_in=resp.usage.prompt_tokens,
            seconds=time.perf_counter() - start,
            model_version=resp.model,
        )
        with self._lock:
            self.cost, self.versions = self.cost + cost, self.versions | {resp.model}
        if self.cache is not None:
            self.cache.put_many(
                self.namespace,
                [
                    (t, {"v": base64.b64encode(v.tobytes()).decode(), "model": resp.model})
                    for t, v in found.items()
                ],
            )
        return found

    def _input(self, text: str) -> str:
        """What is sent (and cached) for text: text itself, or its first max_tokens tokens."""
        if self.max_tokens is None:
            return text
        sent = truncate_tokens(text, self.max_tokens)
        if sent != text:
            with self._lock:
                self.truncated.add(text)
        return sent

    def encode(self, texts: list[str], kind: str) -> np.ndarray:
        before = len(self.truncated)
        sent = {t: self._input(t) for t in dict.fromkeys(texts) if t.strip()}
        if len(self.truncated) > before:
            log.warning(
                "%d input(s) over %d tokens truncated (%d so far)",
                len(self.truncated) - before,
                self.max_tokens,
                len(self.truncated),
            )
        vectors: dict[str, np.ndarray] = {}
        missing = []
        for text in dict.fromkeys(sent.values()):  # the API rejects empty input
            record = self.cache.get(self.namespace, text) if self.cache is not None else None
            if record is None:
                missing.append(text)
            else:
                vectors[text] = np.frombuffer(base64.b64decode(record["v"]), np.float32)
        for batch in self._batches(missing):
            vectors.update(self._fetch(batch))
        dim = len(next(iter(vectors.values()))) if vectors else OPENAI_DIMS.get(self.model, 1536)
        blank = np.zeros(dim, np.float32)  # an empty text matches nothing
        rows = [vectors.get(sent.get(t, ""), blank) for t in texts]
        out = np.stack(rows) if texts else np.zeros((0, dim))
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.maximum(norms, 1e-9)).astype(np.float32)


def make_embedder(spec: str, **kwargs) -> Embedder:
    """'st:<model name>', 'openai:<model>' or 'hash:<dim>'.

    OpenAI embeddings are cached in results/_cache/embed.sqlite unless a `cache` is passed.
    """
    kind, _, arg = spec.partition(":")
    if kind == "st":
        return SentenceTransformerEmbedder(arg, **kwargs)
    if kind == "openai":
        if "cache" not in kwargs:
            kwargs["cache"] = shared_cache("embed")
        return OpenAIEmbedder(arg, **kwargs)
    if kind == "hash":
        return HashEmbedder(int(arg or 256))
    raise ValueError(f"unknown embedder spec {spec!r}")
