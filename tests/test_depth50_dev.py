"""Deeper rerank pool (dev/depth50_dev.py): pure helpers; offline."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev/depth50_dev.py"


@pytest.fixture(scope="module")
def dp():
    spec = importlib.util.spec_from_file_location("deep_pool", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FIRST = [(f"d{i}", 1.0 - i / 100) for i in range(1, 7)]  # d1 .. d6


def test_rescore_reranks_only_the_head_ties_in_first_order(dp):
    scores = {"d1": 2, "d2": 9, "d3": 9, "d4": 10, "d5": 0, "d6": 10}
    out = dp.rescore(FIRST, scores, 3)
    assert [d for d, _ in out] == ["d2", "d3", "d1", "d4", "d5", "d6"]
    assert max(s for _, s in out[3:]) < min(s for _, s in out[:3])
    deep = dp.rescore(FIRST, scores, 6)
    assert [d for d, _ in deep][:3] == ["d4", "d6", "d2"]


def test_rank_of_and_share_in_top(dp):
    assert dp.rank_of(FIRST, "d3") == 3 and dp.rank_of(FIRST, "x") is None
    rows = [
        {"query_id": "a", "k": 1, "source": "d2"},
        {"query_id": "b", "k": 2, "source": "d5"},
    ]
    qrels = {"a": {"d2": 1}, "b": {"d5": 1, "d1": 0}}
    rk = {"a": FIRST, "b": FIRST}
    top3 = dp.share_in_top(rk, rows, qrels, 3)
    assert top3["all"] == {"n": 2, "share": 0.5} and top3["2"]["share"] == 0.0
    assert dp.share_in_top(rk, rows, qrels, 5)["all"]["share"] == 1.0


def test_rescued_counts_sources_from_the_deep_band(dp):
    rows = [
        {"query_id": "a", "k": 1, "source": "d5"},  # S0 rank 5, deep rank 1
        {"query_id": "b", "k": 1, "source": "d4"},  # S0 rank 4, deep rank 4
        {"query_id": "c", "k": 2, "source": "d1"},  # S0 rank 1: outside the band
    ]
    s0 = {q: FIRST for q in "abc"}
    deep = {"a": [("d5", 9), *FIRST], "b": FIRST, "c": FIRST}
    out = dp.rescued(s0, deep, rows, 4, 6, 2)
    assert out["all"] == {"n": 2, "into_top": 1, "to_rank1": 1}
    assert "2" not in out


def test_qrels_from_lists(dp):
    assert dp.qrels_from_lists({"q": ["a", "b"]}) == {"q": {"a": 1, "b": 1}}
