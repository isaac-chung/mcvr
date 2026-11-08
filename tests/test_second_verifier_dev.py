"""Second-verifier check (dev/second_verifier_dev.py): helpers, offline."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev/second_verifier_dev.py"


@pytest.fixture(scope="module")
def sv():
    spec = importlib.util.spec_from_file_location("second_verifier", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_fixed_splitter_returns_saved_constraints(sv):
    rows = [{"query_id": "a", "query": "red car"}, {"query_id": "b", "query": "dog"}]
    by_text = sv.constraints_by_text(rows, {"a": ["red", "car"], "b": ["dog"]})
    cons, cost = sv.FixedSplitter(by_text)("red car")
    assert cons == ["red", "car"] and cost.tokens_in == 0 and cost.llm_calls == 0


def test_constraints_by_text_rejects_conflicting_splits(sv):
    rows = [{"query_id": "a", "query": "x"}, {"query_id": "b", "query": "x"}]
    with pytest.raises(ValueError):
        sv.constraints_by_text(rows, {"a": ["1"], "b": ["2"]})


def test_prompts_match_the_checkers(sv):
    from src.llm import FakeLLM
    from src.systems.rerank import DescriptionScore
    from src.systems.verify import DescriptionCheck

    seen = []

    class Spy(FakeLLM):
        def complete(self, prompt):
            seen.append(prompt)
            return super().complete(prompt)

    descs = {"d": "a red car"}
    DescriptionCheck(Spy(['{"constraints": []}']), "{query}|{constraints}|{description}", descs)(
        "q", ["c1"], "d"
    )
    DescriptionScore(Spy(['{"score": 1}']), "{query}|{evidence}", descs)("q", "d")
    assert seen[0] == sv.s2_prompt("{query}|{constraints}|{description}", "q", ["c1"], "a red car")
    assert seen[1] == sv.s4_prompt("{query}|{evidence}", "q", "a red car")


def test_project_counts_only_uncached(sv):
    class Half:
        def is_cached(self, p):
            return p.startswith("S2")

    rows = [{"query_id": "a", "query": "q"}]
    s0 = {"a": [("d1", 1.0), ("d2", 0.5), ("d3", 0.1)]}
    tokens, calls = sv.project(
        Half(), rows, s0, {"a": ["c"]}, {}, "S2 {query}", "S4 {query}", n=2, chars_per_token=1
    )
    assert calls == 2  # only the S4 prompts of the top 2
    assert tokens == 2 * len("S4 q")


def test_verdict_agreement(sv):
    a = sv.verdict_items({"q": {"verdicts": {"d1": ["yes", "no"], "d2": ["unclear", "yes"]}}})
    b = sv.verdict_items({"q": {"verdicts": {"d1": ["yes", "no"], "d2": ["unclear", "no"]}}})
    ag = sv.verdict_agreement(a, b)
    assert ag["n"] == 4 and ag["agree"] == 0.75
    assert ag["share_a"]["yes"] == 0.5 and ag["share_b"]["yes"] == 0.25
    assert sv.verdict_agreement(a, a)["kappa"] == pytest.approx(1.0)


def test_score_spearman_skips_constant(sv):
    a = {"q1": {"x": 1, "y": 2, "z": 3}, "q2": {"x": 5, "y": 5, "z": 5}}
    b = {"q1": {"x": 3, "y": 2, "z": 1}, "q2": {"x": 1, "y": 2, "z": 3}}
    out = sv.score_spearman(a, b)
    assert out["n"] == 1 and out["constant"] == 1 and out["mean"] == pytest.approx(-1.0)


def test_gap_difference_and_top1(sv):
    rows = [{"query_id": q, "source": s, "k": 1} for q, s in (("a", "s1"), ("b", "s2"))]
    m = "ndcg@10"
    sc = {
        "v": {"S2": {"a": {m: 0.2}, "b": {m: 0.4}}, "S4": {"a": {m: 0.6}, "b": {m: 0.8}}},
        "r": {"S2": {"a": {m: 0.2}, "b": {m: 0.4}}, "S4": {"a": {m: 0.4}, "b": {m: 0.6}}},
    }
    d = sv.gap_difference(sc, "v", "r", rows, m)["all"]
    assert d["n"] == 2 and d["mean"] == pytest.approx(0.2)
    ra = {"a": [("x", 1.0)], "b": [("y", 1.0)]}
    rb = {"a": [("x", 1.0)], "b": [("z", 1.0)]}
    assert sv.top1_agreement(ra, rb, rows) == 0.5
