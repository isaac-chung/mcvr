"""data/mc/<dataset>/ loading, and the guarantee that captions never reach the index."""

import json
import re
from pathlib import Path

import pytest

from src.datasets import mc
from src.embedders import HashEmbedder
from src.llm import FakeVision

CAPTIONS = {
    "v1": "A woman in a zebra-striped coat juggles seven oranges beside a blue food truck.",
    "v2": "Two boys paint a mural that reads HELLO SUMMER on a brick wall at dusk.",
    "v3": "A chef flips pancakes while a parrot watches from the counter of a tiny cafe.",
}
DESCRIPTIONS = {
    "v1": "A person in a patterned coat tosses fruit in a street; a vehicle is parked nearby.",
    "v2": "Children with brushes stand at a wall painted with large letters.",
    "v3": "A cook works at a stove in a small kitchen while a bird sits nearby.",
}


@pytest.fixture
def mc_dir(tmp_path):
    root = tmp_path / "data" / "mc" / "toy"
    root.mkdir(parents=True)
    durations = {"v1": 12.0, "v2": 45.5, "v3": 200.0}
    with open(root / "videos.jsonl", "w") as f:
        for vid, dur in durations.items():
            row = {
                "video_id": vid,
                "path": f"data/mc/toy/videos/{vid}.mp4",
                "duration_s": dur,
                "width": 1280,
                "height": 720,
                "has_audio": True,
                "source": "youtube",
                "split": "val",
                "length_bin": mc.length_bin(dur),
            }
            f.write(json.dumps(row) + "\n")
    with open(root / "captions.jsonl", "w") as f:
        for vid, cap in CAPTIONS.items():
            row = {"video_id": vid, "caption": cap, "caption_origin": "human", "parts": []}
            f.write(json.dumps(row) + "\n")
    return root


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (5, "lt30s"),
        (29.9, "lt30s"),
        (30, "30to60s"),
        (60, "30to60s"),
        (90, "1to2min"),
        (120, "1to2min"),
        (121, "2to4min"),
        (240, "2to4min"),
    ],
)
def test_length_bins(seconds, expected):
    assert mc.length_bin(seconds) == expected


def test_over_four_minutes_has_no_bin():
    with pytest.raises(ValueError, match="240"):
        mc.length_bin(240.5)


def test_corpus_loads_videos_jsonl(mc_dir):
    corpus = mc.McCorpus.load(mc_dir)
    assert corpus.doc_ids == ["v1", "v2", "v3"] and corpus.name == "toy"
    assert corpus.videos["v3"].length_bin == "2to4min"
    assert corpus.by_bin() == {"lt30s": ["v1"], "30to60s": ["v2"], "1to2min": [], "2to4min": ["v3"]}
    assert corpus.video_path("v1") == mc_dir / "videos" / "v1.mp4"
    assert corpus.frames_dir("v1") == mc_dir / "frames" / "v1"
    assert corpus.frames_dir("v1", 16) == mc_dir / "frames16" / "v1"


def test_stored_length_bin_is_recomputed(mc_dir):
    rows = [json.loads(line) for line in (mc_dir / "videos.jsonl").read_text().splitlines()]
    rows[2]["length_bin"] = "gt2min"  # the old label
    (mc_dir / "videos.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert mc.McCorpus.load(mc_dir).videos["v3"].length_bin == "2to4min"


def make_second_dataset(mc_dir):
    other = mc_dir.parent / "other"
    other.mkdir()
    row = {"video_id": "v1", "path": "data/mc/other/videos/v1.mp4", "duration_s": 70.0}
    (other / "videos.jsonl").write_text(json.dumps(row) + "\n")
    caption = {"video_id": "v1", "caption": "A red kite with a long tail loops above a beach."}
    (other / "captions.jsonl").write_text(json.dumps(caption) + "\n")
    return other


def test_pooled_corpus_uses_global_ids(mc_dir):
    other = make_second_dataset(mc_dir)
    corpus = mc.McCorpus.pool(mc_dir.parent, ["toy", "other"])
    assert corpus.doc_ids == ["toy/v1", "toy/v2", "toy/v3", "other/v1"]
    assert corpus.locate("other/v1") == (other, "v1")
    assert corpus.frames_dir("other/v1", 16) == other / "frames16" / "v1"
    assert corpus.videos["other/v1"].dataset == "other"
    assert corpus.by_bin()["1to2min"] == ["other/v1"]
    picked = mc.McCorpus.pool(mc_dir.parent, ["toy", "other"], ids=["other/v1", "toy/v2"])
    assert picked.doc_ids == ["other/v1", "toy/v2"]
    with pytest.raises(KeyError):
        mc.McCorpus.pool(mc_dir.parent, ["toy"], ids=["toy/nope"])
    assert mc.split_global_id("vript_long/a__sc001-002") == ("vript_long", "a__sc001-002")


def test_pooled_describe_writes_each_dataset_and_leaks_no_caption(mc_dir, tmp_path, monkeypatch):
    other = make_second_dataset(mc_dir)
    corpus = mc.McCorpus.pool(mc_dir.parent, ["toy", "other"])
    captions = {**mc.load_captions(mc_dir), **mc.load_captions(other)}
    assert not mc.caption_leaks([repr(vars(corpus))], captions, min_chars=10)
    monkeypatch.setattr(mc.McCorpus, "frames", lambda self, d, n=8: [d.encode()] * n)
    prompts = []

    class SpyVision(FakeVision):
        def describe(self, prompt, images):
            prompts.append(prompt)
            return super().describe(prompt, images)

    texts = [DESCRIPTIONS["v1"], DESCRIPTIONS["v2"], DESCRIPTIONS["v3"], "A kite over sand."]
    mc.describe_corpus(corpus, SpyVision(texts), "prompts/describe_v4.txt", "fake", 16, 1)
    assert mc.load_descriptions(other) == {"v1": "A kite over sand."}
    assert mc.load_descriptions(mc_dir) == DESCRIPTIONS
    descriptions = mc.corpus_descriptions(corpus, "describe_v4.txt")
    assert descriptions["other/v1"] == "A kite over sand." and len(descriptions) == 4
    assert mc.corpus_descriptions(corpus, "describe_v3.txt") == {}
    embedder = SpyEmbedder()
    index = mc.description_index(corpus, descriptions, embedder, tmp_path / "index")
    assert index.doc_ids == corpus.doc_ids
    assert mc.caption_leaks(embedder.seen + prompts, captions, min_chars=10) == []


def test_index_store_hash_depends_on_the_doc_ids():
    a = mc.index_settings(["x/1", "x/2"], "openai:e", "gemini:g", "prompts/describe_v4.txt", 16)
    b = mc.index_settings(["x/1"], "openai:e", "gemini:g", "prompts/describe_v4.txt", 16)
    assert mc.index_store("results", a) != mc.index_store("results", b)
    assert mc.index_store("results", a) == mc.index_store("results", dict(a))


def test_videos_jsonl_with_caption_text_is_rejected(mc_dir):
    rows = (mc_dir / "videos.jsonl").read_text().splitlines()
    leaked = {**json.loads(rows[0]), "caption": CAPTIONS["v1"]}
    (mc_dir / "videos.jsonl").write_text("\n".join([json.dumps(leaked), *rows[1:]]) + "\n")
    with pytest.raises(ValueError, match="caption"):
        mc.McCorpus.load(mc_dir)


def test_captions_load_separately(mc_dir):
    captions = mc.load_captions(mc_dir)
    assert captions["v2"].caption == CAPTIONS["v2"] and captions["v2"].caption_origin == "human"


class SpyEmbedder(HashEmbedder):
    """Records every text that reaches the embedder."""

    def __init__(self):
        super().__init__(64)
        self.seen: list[str] = []

    def encode(self, texts, kind):
        self.seen += texts
        return super().encode(texts, kind)


def test_no_caption_text_reaches_the_index(mc_dir, tmp_path, monkeypatch):
    corpus = mc.McCorpus.load(mc_dir)
    # the corpus object holds no caption text at all
    assert not mc.caption_leaks([repr(vars(corpus))], mc.load_captions(mc_dir), min_chars=10)
    # describe from (fake) frames; the describer sees only the prompt and the frames
    monkeypatch.setattr(mc.McCorpus, "frames", lambda self, vid, n=8: [vid.encode()] * n)
    prompts = []

    class SpyVision(FakeVision):
        def describe(self, prompt, images):
            prompts.append(prompt)
            return super().describe(prompt, images)

    vlm = SpyVision([DESCRIPTIONS[v] for v in corpus.doc_ids])
    records = mc.describe_corpus(corpus, vlm, "prompts/describe_v4.txt", "fake", workers=1)
    descriptions = mc.load_descriptions(mc_dir)
    assert descriptions == DESCRIPTIONS and records["v1"]["n_frames"] == 8
    embedder = SpyEmbedder()
    index = mc.description_index(corpus, descriptions, embedder, tmp_path / "index")
    index.search("someone juggling oranges", 3)
    captions = mc.load_captions(mc_dir)
    index_inputs = embedder.seen[:3]
    assert index_inputs == [DESCRIPTIONS[v] for v in corpus.doc_ids]
    assert mc.caption_leaks(index_inputs + prompts, captions, min_chars=10) == []
    for cap in captions.values():
        assert all(cap.caption not in text for text in embedder.seen + prompts)


def test_caption_leaks_finds_a_copied_caption(mc_dir):
    captions = mc.load_captions(mc_dir)
    texts = ["intro. " + CAPTIONS["v2"].upper() + " outro.", DESCRIPTIONS["v1"]]
    assert mc.caption_leaks(texts, captions) == ["v2"]


def test_indexing_and_system_code_never_reads_captions():
    """Structural guarantee: only mining and label code may load captions."""
    root = Path(mc.__file__).resolve().parents[1]
    pattern = re.compile(r"load_captions|captions\.jsonl")
    offenders = [
        str(p.relative_to(root))
        for p in root.rglob("*.py")
        if p.parts[-2] in {"indexes", "systems", "src"}  # pilot datasets have their own
        and p.name not in {"constraints.py", "negatives.py", "labels.py"}
        and pattern.search(p.read_text())
    ]
    assert offenders == []


def test_describe_corpus_merges_into_descriptions_jsonl(mc_dir, monkeypatch):
    corpus = mc.McCorpus.load(mc_dir)
    monkeypatch.setattr(mc.McCorpus, "frames", lambda self, vid, n=8: [b"x"] * n)
    mc.describe_corpus(corpus, FakeVision(["first"]), "prompts/describe_v4.txt", "f", 8, 1, ["v1"])
    mc.describe_corpus(corpus, FakeVision(["second"]), "prompts/describe_v4.txt", "f", 8, 1, ["v2"])
    assert mc.load_descriptions(mc_dir) == {"v1": "first", "v2": "second"}
    assert mc.description_cost(mc_dir).llm_calls == 2
    assert mc.McCorpus.load(mc_dir).to_corpus({"v1": "first"}).texts["DESC"]["v3"] == ""


def test_rate_limit_spaces_paid_calls_and_counts_their_cost():
    limit = mc.RateLimit(per_minute=60)  # one call per second
    now, slept = [100.0], []
    limit.clock = lambda: now[0]
    limit.sleep = slept.append
    vision = mc.RateLimitedVision(FakeVision(["a"]), limit)
    for _ in range(3):
        vision.describe("p", [b"x"])
    assert slept == [1.0, 2.0] and vision.cost.llm_calls == 3
