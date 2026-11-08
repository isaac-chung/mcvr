"""LLM calls with caching and token accounting. API keys come from the environment."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from typing import Protocol

import httpx

from src.cache import Cache
from src.types import Cost


class LLM(Protocol):
    def complete(self, prompt: str) -> tuple[str, Cost]: ...


class FakeLLM:
    """Returns canned answers in order (tests)."""

    def __init__(self, answers: list[str]):
        self.answers, self.calls = list(answers), 0

    def complete(self, prompt: str) -> tuple[str, Cost]:
        text = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return text, Cost(llm_calls=1, tokens_in=len(prompt.split()), tokens_out=len(text.split()))


class AnthropicLLM:
    def __init__(self, model: str, max_tokens: int = 200):
        import anthropic

        self.client, self.model, self.max_tokens = anthropic.Anthropic(), model, max_tokens

    def complete(self, prompt: str) -> tuple[str, Cost]:
        start = time.perf_counter()
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        cost = Cost(
            llm_calls=1,
            tokens_in=resp.usage.input_tokens,
            tokens_out=resp.usage.output_tokens,
            seconds=time.perf_counter() - start,
            model_version=resp.model,
        )
        return resp.content[0].text, cost


class OpenAILLM:
    """OpenAI chat model. `effort` sets reasoning_effort for reasoning models (none, low, medium,
    high, xhigh). Reasoning models only accept the default temperature, so it is not sent for
    them; they are therefore not deterministic."""

    def __init__(self, model: str, effort: str | None = None, max_tokens: int = 600):
        import openai

        self.client, self.model, self.effort = openai.OpenAI(), model, effort
        self.send_temperature = effort is None
        # reasoning tokens count against the limit, so leave room for them
        self.max_tokens = 4000 if effort else max_tokens

    def _create(self, prompt: str):
        kwargs = {"reasoning_effort": self.effort} if self.effort else {}
        if self.send_temperature:
            kwargs["temperature"] = 0
        return self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )

    def complete(self, prompt: str) -> tuple[str, Cost]:
        start = time.perf_counter()
        try:
            resp = self._create(prompt)
        except Exception as error:  # some models reject temperature=0; retry once without it
            if self.send_temperature and "temperature" in str(error):
                self.send_temperature = False
                resp = self._create(prompt)
            else:
                raise
        cost = Cost(
            llm_calls=1,
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,  # includes reasoning tokens
            seconds=time.perf_counter() - start,
            model_version=resp.model,
        )
        return resp.choices[0].message.content or "", cost


MEDIA_LEVELS = ("LOW", "MEDIUM", "HIGH", "ULTRA_HIGH")


def media_level(value: str | None) -> str | None:
    """'low' or 'MEDIA_RESOLUTION_LOW' -> 'MEDIA_RESOLUTION_LOW'; None or '' -> None."""
    if not value:
        return None
    name = str(value).strip().upper().removeprefix("MEDIA_RESOLUTION_")
    if name not in MEDIA_LEVELS:
        raise ValueError(f"unknown media resolution {value!r}; use one of {MEDIA_LEVELS}")
    return f"MEDIA_RESOLUTION_{name}"


GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {429, 500, 502, 503, 504}


class GeminiModel:
    """Gemini API over REST (key in GEMINI_API_KEY), shared by the text and vision classes.

    `effort` sets the thinking level (minimal, low, medium, high); left out, the model's default
    applies. temperature and seed go into the cache key (`params`), so repeats with distinct
    seeds are stored separately. Rate limits and server errors are retried with backoff.

    `media_resolution` ("low", "medium", "high", "ultra_high" or the full MEDIA_RESOLUTION_*
    name) sets the tokenization quality of each image, sent per part as
    {"media_resolution": {"level": ...}} on every inline image (the Gemini 3 form; it overrides
    the global generationConfig.mediaResolution). On Gemini 3 an image costs about 280 tokens at
    LOW, 560 at MEDIUM and 1,120 at HIGH or unset. Unset (the default) sends nothing and leaves
    the cache key unchanged; when set, it is part of the cache key.
    """

    def __init__(
        self,
        model: str,
        effort: str | None = None,
        temperature: float = 0.0,
        seed: int | None = None,
        max_tokens: int = 2048,  # thinking tokens count against it
        client=None,
        retries: int = 6,
        media_resolution: str | None = None,
    ):
        self.model, self.effort, self.retries = model, effort, retries
        self.temperature, self.seed, self.max_tokens = temperature, seed, max_tokens
        self.media_resolution = media_level(media_resolution)
        self.client = client or httpx.Client(timeout=180)
        self.sleep = time.sleep
        self.params = {"temperature": temperature, "seed": seed, "max_tokens": max_tokens}
        if self.media_resolution:  # only when set, so existing cache keys are unchanged
            self.params["media_resolution"] = self.media_resolution

    def _generate(self, parts: list[dict], images: int = 0) -> tuple[str, Cost]:
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise RuntimeError("Set the GEMINI_API_KEY environment variable.")
        config: dict = {"temperature": self.temperature, "maxOutputTokens": self.max_tokens}
        if self.seed is not None:
            config["seed"] = self.seed
        if self.effort:
            config["thinkingConfig"] = {"thinkingLevel": self.effort}
        body = {"contents": [{"role": "user", "parts": parts}], "generationConfig": config}
        start = time.perf_counter()
        for attempt in range(self.retries + 1):
            last = attempt == self.retries
            try:
                resp = self.client.post(
                    GEMINI_URL.format(model=self.model), headers={"x-goog-api-key": key}, json=body
                )
            except httpx.TransportError:  # dropped connection or timeout
                if last:
                    raise
            else:
                if resp.status_code not in RETRY_STATUS or last:
                    break
            self.sleep(min(60.0, 2.0**attempt))
        resp.raise_for_status()
        data = resp.json()
        candidates = data.get("candidates") or [{}]  # none when the prompt is blocked
        text = "".join(
            p.get("text", "")
            for p in candidates[0].get("content", {}).get("parts", [])
            if not p.get("thought")
        )
        usage = data.get("usageMetadata", {})
        cost = Cost(
            llm_calls=1,
            tokens_in=usage.get("promptTokenCount", 0),
            # thinking tokens are billed as output, as OpenAI's reasoning tokens are
            tokens_out=usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0),
            seconds=time.perf_counter() - start,
            images=images,
            model_version=data.get("modelVersion", ""),
        )
        return text, cost


class GeminiLLM(GeminiModel):
    def complete(self, prompt: str) -> tuple[str, Cost]:
        return self._generate([{"text": prompt}])


def cache_payload(inner: object, payload: object) -> object:
    """Adds the inner model's sampling params (temperature, seed) to the cache key when it has
    them; models without them keep the pilot's keys, so the pilot caches stay valid."""
    params = getattr(inner, "params", None)
    return {"call": payload, "params": params} if params else payload


class CachedLLM:
    """Stores each (model, prompt, params) answer and its cost, so reruns are free and identical."""

    def __init__(self, inner: LLM, cache: Cache, spec: str):
        self.inner, self.cache, self.spec = inner, cache, spec

    def complete(self, prompt: str) -> tuple[str, Cost]:
        def call() -> dict:
            text, cost = self.inner.complete(prompt)
            return {"text": text, "cost": cost.to_dict()}

        record = self.cache.get_or_compute(
            f"llm:{self.spec}", cache_payload(self.inner, prompt), call
        )
        return record["text"], Cost(**record["cost"])

    def is_cached(self, prompt: str) -> bool:
        """Would complete(prompt) be answered from the cache (no paid call)?"""
        return self.cache.get(f"llm:{self.spec}", cache_payload(self.inner, prompt)) is not None


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """'openai:gpt-6-sol@low' -> ('openai', 'gpt-6-sol', 'low'); the effort part is optional."""
    kind, _, rest = spec.partition(":")
    model, _, effort = rest.partition("@")
    return kind, model, effort or None


def make_llm(spec: str, cache: Cache, **params) -> LLM | None:
    """'anthropic:<model>', 'openai:<model>[@<effort>]' or 'gemini:<model>[@<thinking level>]';
    empty disables LLMs. params (temperature, seed, max_tokens) are for Gemini only."""
    if not spec:
        return None
    kind, model, effort = parse_spec(spec)
    if params and kind != "gemini":
        raise ValueError(f"sampling params are only supported for gemini specs, not {spec!r}")
    if kind == "anthropic":
        inner: LLM = AnthropicLLM(model)
    elif kind == "openai":
        inner = OpenAILLM(model, effort)
    elif kind == "gemini":
        inner = GeminiLLM(model, effort, **params)
    else:
        raise ValueError(f"unknown llm spec {spec!r}")
    return CachedLLM(inner, cache, spec)


class VisionLLM(Protocol):
    def describe(self, prompt: str, images: list[bytes]) -> tuple[str, Cost]: ...


class FakeVision:
    """Returns canned descriptions in order (tests)."""

    def __init__(self, answers: list[str]):
        self.answers, self.calls = list(answers), 0

    def describe(self, prompt: str, images: list[bytes]) -> tuple[str, Cost]:
        text = self.answers[self.calls % len(self.answers)]
        self.calls += 1
        return text, Cost(llm_calls=1, tokens_in=len(prompt.split()), tokens_out=len(text.split()))


class OpenAIVision:
    """OpenAI chat model that looks at JPEG frames (sent at high detail so small text is legible)."""

    def __init__(self, model: str, effort: str | None = None, max_tokens: int = 500):
        import openai

        self.client, self.model, self.effort = openai.OpenAI(), model, effort
        self.send_temperature = effort is None  # reasoning models take only the default
        self.max_tokens = 4000 if effort else max_tokens

    def _create(self, parts: list[dict]):
        kwargs = {"reasoning_effort": self.effort} if self.effort else {}
        if self.send_temperature:
            kwargs["temperature"] = 0
        return self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            messages=[{"role": "user", "content": parts}],
            **kwargs,
        )

    def describe(self, prompt: str, images: list[bytes]) -> tuple[str, Cost]:
        start = time.perf_counter()
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for jpeg in images:
            url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        try:
            resp = self._create(parts)
        except Exception as error:  # some models reject temperature=0; retry once without it
            if self.send_temperature and "temperature" in str(error):
                self.send_temperature = False
                resp = self._create(parts)
            else:
                raise
        cost = Cost(
            llm_calls=1,
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
            seconds=time.perf_counter() - start,
            images=len(images),
            model_version=resp.model,
        )
        return resp.choices[0].message.content or "", cost


class GeminiVision(GeminiModel):
    """Gemini model that looks at JPEG frames, sent inline after the prompt in frame order."""

    def describe(
        self, prompt: str, images: list[bytes], labels: list[str] | None = None
    ) -> tuple[str, Cost]:
        """labels, when given, go as a text part before each image (e.g. "Frame 3 of 16 (t =
        21.4 s):"), so the model can refer to frames by number."""
        if labels is not None and len(labels) != len(images):
            raise ValueError(f"{len(labels)} labels for {len(images)} images")
        parts: list[dict] = [{"text": prompt}]
        for i, jpeg in enumerate(images):
            if labels is not None:
                parts.append({"text": labels[i]})
            data = base64.b64encode(jpeg).decode()
            part: dict = {"inline_data": {"mime_type": "image/jpeg", "data": data}}
            if self.media_resolution:
                part["media_resolution"] = {"level": self.media_resolution}
            parts.append(part)
        return self._generate(parts, images=len(images))


class CachedVision:
    """Stores each (model, prompt, frames, params) answer and its cost; frames are keyed by their
    hash."""

    def __init__(self, inner: VisionLLM, cache: Cache, spec: str):
        self.inner, self.cache, self.spec = inner, cache, spec

    def describe(
        self, prompt: str, images: list[bytes], labels: list[str] | None = None
    ) -> tuple[str, Cost]:
        """labels (Gemini only) are passed on and keyed only when given, so unlabelled calls keep
        their cache keys."""

        def call() -> dict:
            if labels is None:
                text, cost = self.inner.describe(prompt, images)
            else:
                text, cost = self.inner.describe(prompt, images, labels=labels)
            return {"text": text, "cost": cost.to_dict()}

        payload = self._payload(prompt, images, labels)
        record = self.cache.get_or_compute(f"vision:{self.spec}", payload, call)
        return record["text"], Cost(**record["cost"])

    def _payload(self, prompt: str, images: list[bytes], labels: list[str] | None) -> object:
        frames = [hashlib.sha256(img).hexdigest() for img in images]
        key = [prompt, frames] if labels is None else [prompt, frames, list(labels)]
        return cache_payload(self.inner, key)

    def is_cached(self, prompt: str, images: list[bytes], labels: list[str] | None = None) -> bool:
        """Would describe(prompt, images, labels) be answered from the cache (no paid call)?"""
        payload = self._payload(prompt, images, labels)
        return self.cache.get(f"vision:{self.spec}", payload) is not None


def make_vision(spec: str, cache: Cache, **params) -> VisionLLM:
    """'openai:<model>[@<effort>]' or 'gemini:<model>[@<thinking level>]'; params
    (temperature, seed, max_tokens, media_resolution) are for Gemini only. A media_resolution
    of None is dropped, so callers can pass a config value that may be unset."""
    if params.get("media_resolution", "") is None:
        params = {k: v for k, v in params.items() if k != "media_resolution"}
    kind, model, effort = parse_spec(spec)
    if kind == "gemini":
        return CachedVision(GeminiVision(model, effort, **params), cache, spec)
    if kind != "openai":
        raise ValueError(f"unknown vision spec {spec!r}")
    if params:
        raise ValueError(f"sampling params are only supported for gemini specs, not {spec!r}")
    return CachedVision(OpenAIVision(model, effort), cache, spec)
