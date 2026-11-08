"""Gemini adapters, the OpenAI embedder, shared caches and run manifests (all offline)."""

import json
import subprocess
from types import SimpleNamespace

import httpx
import numpy as np
import pandas as pd
import pytest

from src.cache import Cache, shared_cache
from src.config import Config
from src.embedders import OpenAIEmbedder, make_embedder
from src.frames import saved_frames
from src.llm import (
    CachedLLM,
    CachedVision,
    FakeLLM,
    GeminiLLM,
    GeminiVision,
    make_llm,
    make_vision,
)
from src.manifest import runs_cost, write_manifest
from src.types import Cost

# ---- Cost ----------------------------------------------------------------------------------


def test_cost_adds_images_and_merges_versions():
    a = Cost(llm_calls=1, images=8, model_version="gemini-x-001")
    b = Cost(llm_calls=1, images=8, model_version="gemini-x-001,gemini-x-002")
    total = a + b + Cost(searches=1)
    assert total.images == 16 and total.llm_calls == 2 and total.searches == 1
    assert total.model_version == "gemini-x-001,gemini-x-002"


def test_cost_reads_records_cached_before_the_new_fields():
    old = {"searches": 0, "llm_calls": 1, "tokens_in": 5, "tokens_out": 2, "seconds": 0.1}
    assert Cost(**old).images == 0 and Cost(**old).model_version == ""


# ---- caches --------------------------------------------------------------------------------


def test_shared_cache_is_one_file_per_kind(tmp_path):
    llm = shared_cache("llm", tmp_path)
    assert shared_cache("llm", tmp_path) is llm
    assert (tmp_path / "_cache" / "llm.sqlite").exists()
    assert shared_cache("embed", tmp_path) is not llm
    with pytest.raises(ValueError):
        shared_cache("other", tmp_path)


def test_cache_get_and_put_many(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    assert cache.get("ns", "a") is None
    cache.put_many("ns", [("a", {"v": 1}), ("b", {"v": 2})])
    assert cache.get("ns", "b") == {"v": 2}
    assert cache.get_or_compute("ns", "a", lambda: {"v": 9}) == {"v": 1}


def test_config_picks_shared_or_pilot_caches(tmp_path):
    pilot = Config(name="p", results_dir=str(tmp_path))
    shared = Config(name="s", results_dir=str(tmp_path), cache="shared")
    assert pilot.hash() == Config(name="p", results_dir=str(tmp_path)).hash()
    assert pilot.hash() != Config(name="p", results_dir=str(tmp_path), cache="shared").hash()
    pilot.llm_cache(tmp_path / "run" / "llm_cache.sqlite")
    assert (tmp_path / "run" / "llm_cache.sqlite").exists()
    assert shared.vlm_cache(tmp_path / "unused.sqlite") is shared_cache("vlm", tmp_path)
    assert not (tmp_path / "unused.sqlite").exists()


def test_pilot_llm_cache_keys_are_unchanged(tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    CachedLLM(FakeLLM(["x"]), cache, "fake").complete("prompt")
    assert cache.get("llm:fake", "prompt")["text"] == "x"  # same key as before params existed


# ---- Gemini ----------------------------------------------------------------------------------


def gemini_reply(text="a red car", version="gemini-3.5-flash-lite-001"):
    return {
        "candidates": [
            {"content": {"parts": [{"text": "thinking...", "thought": True}, {"text": text}]}}
        ],
        "usageMetadata": {
            "promptTokenCount": 2100,
            "candidatesTokenCount": 120,
            "thoughtsTokenCount": 30,
        },
        "modelVersion": version,
    }


class FakeGeminiServer:
    """httpx transport that records requests and answers from a list of (status, body)."""

    def __init__(self, replies):
        self.replies, self.requests = list(replies), []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        status, body = self.replies.pop(0) if len(self.replies) > 1 else self.replies[0]
        return httpx.Response(status, json=body)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=httpx.MockTransport(self))


@pytest.fixture
def gemini_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def test_gemini_vision_sends_frames_and_reads_usage(gemini_key):
    server = FakeGeminiServer([(200, gemini_reply())])
    vlm = GeminiVision("gemini-3.5-flash-lite", "low", seed=3, client=server.client())
    text, cost = vlm.describe("describe", [b"\xff\xd8one", b"\xff\xd8two"])
    assert text == "a red car"  # thought parts are dropped
    assert cost == Cost(
        llm_calls=1,
        tokens_in=2100,
        tokens_out=150,
        seconds=cost.seconds,
        images=2,
        model_version="gemini-3.5-flash-lite-001",
    )
    request = server.requests[0]
    assert request.headers["x-goog-api-key"] == "test-key"
    assert "key=" not in str(request.url)  # the key never goes into the URL
    assert "models/gemini-3.5-flash-lite:generateContent" in str(request.url)
    body = json.loads(request.content)
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "describe"} and len(parts) == 3
    assert parts[1]["inline_data"]["mime_type"] == "image/jpeg"
    config = body["generationConfig"]
    assert config["seed"] == 3 and config["temperature"] == 0.0
    assert config["thinkingConfig"] == {"thinkingLevel": "low"}


def test_gemini_retries_rate_limits(gemini_key):
    server = FakeGeminiServer([(429, {}), (503, {}), (200, gemini_reply("ok"))])
    llm = GeminiLLM("m", client=server.client())
    waits = []
    llm.sleep = waits.append
    assert llm.complete("hi")[0] == "ok" and len(server.requests) == 3 and waits == [1.0, 2.0]


def test_gemini_gives_up_after_retries(gemini_key):
    llm = GeminiLLM("m", client=FakeGeminiServer([(429, {})]).client(), retries=2)
    llm.sleep = lambda s: None
    with pytest.raises(httpx.HTTPStatusError):
        llm.complete("hi")


def test_gemini_needs_the_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiLLM("m", client=FakeGeminiServer([(200, gemini_reply())]).client()).complete("x")


def test_seeds_are_cached_separately(gemini_key, tmp_path):
    server = FakeGeminiServer([(200, gemini_reply("one")), (200, gemini_reply("two"))])
    cache = Cache(tmp_path / "c.sqlite")

    def vlm(seed):
        inner = GeminiVision("m", seed=seed, temperature=1.0, client=server.client())
        return CachedVision(inner, cache, "gemini:m")

    assert vlm(0).describe("p", [b"a"])[0] == "one"
    assert vlm(1).describe("p", [b"a"])[0] == "two"
    assert vlm(0).describe("p", [b"a"])[0] == "one"  # cached, with its cost and version
    _, cost = vlm(1).describe("p", [b"a"])
    assert len(server.requests) == 2 and cost.images == 1 and cost.model_version


def test_make_llm_and_make_vision_accept_gemini_specs(gemini_key, tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    llm = make_llm("gemini:gemini-3.5-flash-lite@minimal", cache, seed=2)
    assert isinstance(llm.inner, GeminiLLM) and llm.inner.effort == "minimal"
    assert llm.inner.params["seed"] == 2
    vlm = make_vision("gemini:gemini-3.5-flash-lite", cache)
    assert isinstance(vlm.inner, GeminiVision) and vlm.inner.effort is None
    with pytest.raises(ValueError):
        make_vision("openai:gpt-4.1-mini", cache, seed=1)
    with pytest.raises(ValueError):
        make_vision("other:x", cache)


# ---- OpenAI embedder -----------------------------------------------------------------------


class FakeEmbeddings:
    """Stands in for openai.OpenAI().embeddings; vectors are deterministic per text."""

    def __init__(self, fail_first=0):
        self.batches, self.fail_first = [], fail_first

    def create(self, model, input, encoding_format):
        if self.fail_first:
            self.fail_first -= 1
            raise type("RateLimitError", (Exception,), {"status_code": 429})("slow down")
        self.batches.append(list(input))
        data = [
            SimpleNamespace(index=i, embedding=[float(len(t)), float(t.count("a")), 1.0, 0.0])
            for i, t in enumerate(input)
        ]
        data.reverse()  # the API does not promise order; the index field does
        usage = SimpleNamespace(prompt_tokens=sum(len(t.split()) for t in input))
        return SimpleNamespace(data=data, usage=usage, model=model + "-v1")


def fake_client(**kwargs):
    return SimpleNamespace(embeddings=FakeEmbeddings(**kwargs))


def test_openai_embedder_batches_dedupes_and_normalizes(tmp_path):
    client = fake_client()
    emb = OpenAIEmbedder("m", Cache(tmp_path / "e.sqlite"), batch_size=2, client=client)
    texts = ["a cat", "banana", "a cat", "dog", "aardvark"]
    out = emb.encode(texts, "document")
    assert out.shape == (5, 4) and out.dtype == np.float32
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)
    assert np.allclose(out[0], out[2])
    assert [len(b) for b in client.embeddings.batches] == [2, 2]  # 4 unique texts
    assert emb.cost.llm_calls == 2 and emb.cost.tokens_in == 5 and emb.versions == {"m-v1"}
    expected = np.array([6.0, 3.0, 1.0, 0.0]) / np.linalg.norm([6.0, 3.0, 1.0, 0.0])
    assert np.allclose(out[1], expected)


def test_openai_embedder_splits_batches_by_tokens(tmp_path, monkeypatch):
    import src.embedders as emb

    monkeypatch.setattr(emb, "MAX_BATCH_TOKENS", 100)
    client = fake_client()
    embedder = OpenAIEmbedder("m", Cache(tmp_path / "c.sqlite"), batch_size=10, client=client)
    texts = [c * 120 for c in "abcde"]  # about 41 tokens each: two per request
    assert embedder.encode(texts, "doc").shape == (5, 4)
    assert [len(b) for b in client.embeddings.batches] == [2, 2, 1]


def test_openai_embedder_reuses_the_cache_across_instances(tmp_path):
    cache = Cache(tmp_path / "e.sqlite")
    first = OpenAIEmbedder("m", cache, client=fake_client()).encode(["x y", "z"], "document")
    client = fake_client()
    again = OpenAIEmbedder("m", cache, client=client)
    second = again.encode(["z", "x y"], "query")
    assert client.embeddings.batches == [] and again.cost.llm_calls == 0
    assert np.array_equal(first[::-1], second)
    other = OpenAIEmbedder("other-model", cache, client=client)
    other.encode(["z"], "query")
    assert client.embeddings.batches == [["z"]]  # the model is part of the key


def test_openai_embedder_retries_rate_limits_and_handles_blank_text(tmp_path):
    client = fake_client(fail_first=2)
    emb = OpenAIEmbedder("text-embedding-3-small", None, client=client)
    waits = []
    emb.sleep = waits.append
    out = emb.encode(["", "words"], "document")
    assert waits == [1.0, 2.0] and client.embeddings.batches == [["words"]]
    assert not out[0].any() and out.shape == (2, 4)
    assert OpenAIEmbedder("text-embedding-3-small", None, client=client).encode(
        ["  "], "query"
    ).shape == (1, 1536)


def test_make_embedder_openai_spec(tmp_path):
    cache = Cache(tmp_path / "e.sqlite")
    emb = make_embedder("openai:text-embedding-3-small", cache=cache, client=fake_client())
    assert isinstance(emb, OpenAIEmbedder) and emb.cache is cache
    assert emb.namespace == "embed:openai:text-embedding-3-small"


# ---- manifest ------------------------------------------------------------------------------


def test_write_manifest_records_versions_and_keeps_created(tmp_path):
    cfg = Config(name="mc", seed=4)
    cost = Cost(llm_calls=3, images=24, model_version="gemini-3.5-flash-lite-001")
    path = write_manifest(tmp_path, cfg, models={"describe": "gemini:x"}, cost=cost)
    first = json.loads(path.read_text())
    assert first["config_hash"] == cfg.hash() and first["seeds"] == {"seed": 4}
    assert first["model_versions"] == ["gemini-3.5-flash-lite-001"]
    assert first["cost"]["images"] == 24 and first["models"] == {"describe": "gemini:x"}
    assert len(first["git"]["commit"] or "x" * 40) == 40
    second = json.loads(write_manifest(tmp_path, cfg, seeds={"runs": [0, 1, 2]}).read_text())
    assert second["created"] == first["created"] and second["seeds"] == {"runs": [0, 1, 2]}


def test_runs_cost_sums_runs_parquet(tmp_path):
    assert runs_cost(tmp_path) == Cost()
    pd.DataFrame(
        [
            {**Cost(searches=1, llm_calls=1, tokens_in=10).to_dict(), "query_id": "q1"},
            {**Cost(searches=2, model_version="v2").to_dict(), "query_id": "q2"},
        ]
    ).to_parquet(tmp_path / "runs.parquet")
    total = runs_cost(tmp_path)
    assert (total.searches, total.llm_calls, total.tokens_in) == (3, 1, 10)
    assert total.model_version == "v2"


# ---- frames saved to disk ------------------------------------------------------------------


@pytest.fixture
def wide_clip(tmp_path):
    path = tmp_path / "wide.mp4"
    cmd = ["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc=duration=12:size=1280x720:rate=5"]
    try:
        subprocess.run([*cmd, str(path)], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        pytest.skip("ffmpeg not available")
    return path


def test_saved_frames_cover_the_video_and_are_reused(wide_clip, tmp_path, monkeypatch):
    out = tmp_path / "frames" / "v1"
    files = saved_frames(wide_clip, 8, out)
    assert [f.name for f in files] == [f"{i}.jpg" for i in range(8)]
    record = json.loads((out / "frames.json").read_text())
    assert record["times"][0] < 1.0 and record["times"][-1] > 11.0  # spread over the whole video
    width = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=width,height", "-of", "csv=p=0"]
        + [str(files[0])],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert width == "768,432"
    monkeypatch.setattr("src.frames._grab", lambda *a: pytest.fail("frames were resampled"))
    assert saved_frames(wide_clip, 8, out) == files
    with pytest.raises(ValueError):
        saved_frames(wide_clip, 16, out)


# ---- Gemini media resolution ---------------------------------------------------------------


def test_gemini_media_resolution_goes_on_each_image_part(gemini_key):
    server = FakeGeminiServer([(200, gemini_reply())])
    vlm = GeminiVision("m", media_resolution="low", client=server.client())
    vlm.describe("p", [b"a", b"b"], labels=["Frame 1", "Frame 2"])
    parts = json.loads(server.requests[0].content)["contents"][0]["parts"]
    images = [p for p in parts if "inline_data" in p]
    assert len(images) == 2
    assert all(p["media_resolution"] == {"level": "MEDIA_RESOLUTION_LOW"} for p in images)
    assert all("media_resolution" not in p for p in parts if "text" in p)
    assert vlm.params["media_resolution"] == "MEDIA_RESOLUTION_LOW"


def test_gemini_without_media_resolution_sends_and_keys_as_before(gemini_key, tmp_path):
    server = FakeGeminiServer([(200, gemini_reply("one")), (200, gemini_reply("two"))])
    vlm = GeminiVision("m", client=server.client())
    assert "media_resolution" not in vlm.params
    vlm.describe("p", [b"a"])
    parts = json.loads(server.requests[0].content)["contents"][0]["parts"]
    assert all("media_resolution" not in p for p in parts)
    body = json.loads(server.requests[0].content)
    assert "mediaResolution" not in body["generationConfig"]
    # the key without a resolution equals the key the code wrote before the option existed
    cache = Cache(tmp_path / "c.sqlite")
    plain = CachedVision(GeminiVision("m", client=server.client()), cache, "gemini:m")
    old_key = {
        "call": ["p", [__import__("hashlib").sha256(b"a").hexdigest()]],
        "params": {"temperature": 0.0, "seed": None, "max_tokens": 2048},
    }
    cache.put_many("vision:gemini:m", [(old_key, {"text": "cached", "cost": Cost().to_dict()})])
    assert plain.describe("p", [b"a"])[0] == "cached"
    low = CachedVision(
        GeminiVision("m", media_resolution="MEDIA_RESOLUTION_LOW", client=server.client()),
        cache,
        "gemini:m",
    )
    assert not low.is_cached("p", [b"a"])  # a set resolution is its own cache entry
    assert low.describe("p", [b"a"])[0] == "two" and len(server.requests) == 2


def test_media_resolution_names_and_make_vision(gemini_key, tmp_path, monkeypatch):
    from src.llm import media_level

    monkeypatch.setenv("OPENAI_API_KEY", "test")
    assert media_level("medium") == "MEDIA_RESOLUTION_MEDIUM"
    assert media_level("MEDIA_RESOLUTION_HIGH") == "MEDIA_RESOLUTION_HIGH"
    assert media_level(None) is None and media_level("") is None
    with pytest.raises(ValueError):
        media_level("tiny")
    cache = Cache(tmp_path / "c.sqlite")
    vlm = make_vision("gemini:m", cache, media_resolution="low")
    assert vlm.inner.media_resolution == "MEDIA_RESOLUTION_LOW"
    unset = make_vision("gemini:m", cache, media_resolution=None)  # a config value left unset
    assert unset.inner.media_resolution is None and "media_resolution" not in unset.inner.params
    make_vision("openai:gpt-4.1-mini", cache, media_resolution=None)  # None is not a param
    with pytest.raises(ValueError):
        make_vision("openai:gpt-4.1-mini", cache, media_resolution="low")
