"""OpenAIEmbedder caps each input at max_tokens (cl100k_base) and leaves shorter inputs alone."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np
import tiktoken

from src.cache import Cache
from src.embedders import OpenAIEmbedder, truncate_tokens

ENC = tiktoken.get_encoding("cl100k_base")


class RecordingEmbeddings:
    def __init__(self):
        self.batches: list[list[str]] = []

    def create(self, model, input, encoding_format):
        self.batches.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), float(t.count("a")) + 1.0, 1.0])
            for i, t in enumerate(input)
        ]
        usage = SimpleNamespace(prompt_tokens=sum(len(ENC.encode(t)) for t in input))
        return SimpleNamespace(data=data, usage=usage, model=model + "-v1")


def client():
    return SimpleNamespace(embeddings=RecordingEmbeddings())


def test_truncate_tokens_keeps_short_texts_and_cuts_long_ones():
    short = "a cat on a mat " * 10
    assert truncate_tokens(short, 8191) is short
    long = "the quick brown fox jumps over the lazy dog. " * 2000  # about 20k tokens
    cut = truncate_tokens(long, 100)
    assert len(ENC.encode(cut)) == 100 and long.startswith(cut)
    exact = ENC.decode(ENC.encode(long)[:50])
    assert truncate_tokens(exact, 50) == exact


def test_long_inputs_are_truncated_and_short_ones_unchanged(tmp_path, caplog):
    cache = Cache(tmp_path / "e.sqlite")
    c = client()
    emb = OpenAIEmbedder("m", cache, client=c, max_tokens=64)
    short, long = "a short caption", "word " * 500
    with caplog.at_level(logging.WARNING, logger="src.embedders"):
        out = emb.encode([short, long, long, ""], "document")
    assert out.shape == (4, 3) and np.allclose(out[1], out[2]) and not out[3].any()
    sent = c.embeddings.batches[0]
    assert sent[0] == short  # sent exactly as before
    assert len(ENC.encode(sent[1])) == 64 and long.startswith(sent[1])
    assert emb.truncated == {long} and "1 input(s) over 64 tokens truncated" in caplog.text
    # the short text's cache key is the text itself, so an old cache entry still hits
    assert cache.get(emb.namespace, short) is not None
    plain = OpenAIEmbedder("m", cache, client=client(), max_tokens=None)
    assert np.array_equal(plain.encode([short], "query")[0], out[0])
    assert plain.cost.llm_calls == 0


def test_default_cap_is_the_api_limit():
    emb = OpenAIEmbedder("m", None, client=client())
    assert emb.max_tokens == 8191
    text = "x " * 9000  # 9,000 tokens
    emb.encode([text], "document")
    assert len(ENC.encode(emb.client.embeddings.batches[0][0])) == 8191
