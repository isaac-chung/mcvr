"""Describe-prompt variants (mc.descriptions_file) and dev/description_length_dev.py, offline."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.datasets import mc
from src.llm import FakeLLM, FakeVision

ROOT = Path(__file__).resolve().parents[1]


def load_script():
    path = ROOT / "dev" / "description_length_dev.py"
    spec = importlib.util.spec_from_file_location("variant_index", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def toy(tmp_path):
    root = tmp_path / "data" / "mc" / "toy"
    root.mkdir(parents=True)
    with open(root / "videos.jsonl", "w") as f:
        f.writelines(
            json.dumps({"video_id": vid, "path": "", "duration_s": dur}) + "\n"
            for vid, dur in {"v1": 12.0, "v2": 45.0}.items()
        )
    return root


def test_descriptions_file_keeps_the_main_prompt_in_descriptions_jsonl():
    assert mc.descriptions_file("prompts/describe_v4.txt") == "descriptions.jsonl"
    assert mc.descriptions_file("prompts/describe_v4b.txt") == "descriptions_describe_v4b.jsonl"


def test_variant_descriptions_do_not_overwrite_the_main_ones(toy, monkeypatch):
    corpus = mc.McCorpus.pool(toy.parent, ["toy"])
    monkeypatch.setattr(mc.McCorpus, "frames", lambda self, d, n=8: [b"x"] * n)
    mc.describe_corpus(corpus, FakeVision(["short"]), "prompts/describe_v4.txt", "f", 16, 1)
    variant = mc.descriptions_file("prompts/describe_v4b.txt")
    mc.describe_corpus(
        corpus, FakeVision(["a longer one"]), "prompts/describe_v4b.txt", "f", 16, 1, None, variant
    )
    assert mc.load_descriptions(toy) == {"v1": "short", "v2": "short"}
    main = mc.corpus_descriptions(corpus, "describe_v4.txt")
    longer = mc.corpus_descriptions(corpus, "describe_v4b.txt", variant)
    assert main == {"toy/v1": "short", "toy/v2": "short"}
    assert longer == {"toy/v1": "a longer one", "toy/v2": "a longer one"}
    assert mc.corpus_descriptions(corpus, "describe_v4.txt", variant) == {}
    a = mc.index_settings(corpus.doc_ids, "e", "d", "prompts/describe_v4.txt", 16)
    b = mc.index_settings(corpus.doc_ids, "e", "d", "prompts/describe_v4b.txt", 16)
    assert mc.index_store("r", a) != mc.index_store("r", b)


def test_word_stats_by_dataset_and_length_bin(toy):
    vi = load_script()
    corpus = mc.McCorpus.pool(toy.parent, ["toy"])
    stats = vi.description_words(corpus, {"toy/v1": "one two three", "toy/v2": ""})
    assert stats["overall"]["all"] == {"n": 1, "median": 3.0, "p10": 3.0, "p90": 3.0}
    assert set(stats["length_bin"]) == {"lt30s"} and set(stats["dataset"]) == {"toy"}
    assert vi.word_stats([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])["median"] == 5.5


ROWS = [
    {
        "query_id": "toy/v1#k1",
        "source": "toy/v1",
        "k": 1,
        "dataset": "toy",
        "length_bin": "lt30s",
        "facts": [{"text": "A red cap."}],
    },
    {
        "query_id": "toy/v1#k2",
        "source": "toy/v1",
        "k": 2,
        "dataset": "toy",
        "length_bin": "lt30s",
        "facts": [{"text": "A red cap."}, {"text": "A sign reads OPEN."}],
    },
]


def test_fact_coverage_from_one_judgment_per_source():
    vi = load_script()
    facts = vi.source_facts(list(reversed(ROWS)))
    assert facts == {"toy/v1": ["A red cap.", "A sign reads OPEN."]}
    answer = json.dumps(
        {
            "facts": [
                {"fact": 1, "answer": "yes", "quote": "red cap"},
                {"fact": 2, "answer": "yes", "quote": ""},  # no quote: unclear
            ]
        }
    )
    llm = FakeLLM([answer])
    judged = {
        "toy/v1": vi.judge_description(
            llm, "{caption}\n{facts}", "A man in a red cap.", facts["toy/v1"]
        )
    }
    assert llm.calls == 1 and judged["toy/v1"]["answers"] == ["yes", "unclear"]
    rows = vi.coverage_rows(ROWS, facts, judged)
    assert [r["yes"] for r in rows] == [1.0, 0.5]
    other = [{**r, "yes": 0.0} for r in rows]
    diff = vi.paired_rows(rows, other, "yes")
    means = vi.grouped_means(diff, "yes", "k")
    assert means[1]["mean"] == 1.0 and means[2]["mean"] == 0.5


def test_describe_projection_scales_to_the_sample():
    vi = load_script()
    cost = mc.Cost(llm_calls=5, tokens_in=1_000_000, tokens_out=0, images=80)
    proj = vi.describe_projection(cost, 5, 500)
    assert proj["usd_measured"] == pytest.approx(vi.PRICE_IN)
    assert proj["usd_projected"] == pytest.approx(100 * vi.PRICE_IN)
