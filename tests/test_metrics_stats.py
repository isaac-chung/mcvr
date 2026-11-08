import numpy as np
import pytest

from src.metrics import judged_at_k, ndcg_at_k, plan_scores, recall_at_k
from src.stats import bootstrap_ci, paired_summary, permutation_p

GRADES = {"a": 3, "b": 1, "c": 0}


def test_ndcg_perfect_and_worse():
    perfect = [("a", 1), ("b", 1), ("c", 1)]
    worse = [("b", 1), ("a", 1), ("c", 1)]
    assert ndcg_at_k(perfect, GRADES) == pytest.approx(1.0)
    assert ndcg_at_k(worse, GRADES) < 1.0
    assert ndcg_at_k([("x", 1)], GRADES) == 0.0
    assert ndcg_at_k([("a", 1)], {}) == 0.0


def test_recall_and_judged():
    ranking = [("a", 1), ("x", 1), ("b", 1)]
    assert recall_at_k(ranking, GRADES, 2) == 0.5  # relevant = a, b
    assert judged_at_k(ranking, GRADES, 3) == pytest.approx(2 / 3)


def test_plan_scores():
    scores = plan_scores(("ASR", "VIS", "DESC"), ("DESC",))
    assert scores == {"plan_precision": 1 / 3, "plan_recall": 1.0, "plan_overhead": 2.0}


def test_ndcg_matches_ir_measures_when_available():
    ir_measures = pytest.importorskip("ir_measures")
    qrels = [ir_measures.Qrel("q", d, g) for d, g in GRADES.items()]
    run = [ir_measures.ScoredDoc("q", d, s) for d, s in [("b", 3.0), ("a", 2.0), ("x", 1.0)]]
    ref = ir_measures.calc_aggregate([ir_measures.nDCG @ 10], qrels, run)[ir_measures.nDCG @ 10]
    ours = ndcg_at_k([("b", 3.0), ("a", 2.0), ("x", 1.0)], GRADES, 10)
    assert ours == pytest.approx(ref, abs=1e-6)


def test_bootstrap_ci_contains_mean_and_permutation_detects_effect():
    rng = np.random.default_rng(1)
    values = rng.normal(0.3, 0.1, size=60)
    clusters = [str(i // 2) for i in range(60)]
    lo, hi = bootstrap_ci(values, clusters, n=2000)
    assert lo < values.mean() < hi
    assert permutation_p(values, clusters, n=2000) < 0.01
    assert permutation_p(rng.normal(0, 0.1, size=60), clusters, n=2000) > 0.05


def test_paired_summary_uses_shared_queries_only():
    a = {"q1": 0.5, "q2": 0.7, "q3": 0.9}
    b = {"q1": 0.4, "q2": 0.6}
    s = paired_summary(a, b, {"q1": "x", "q2": "y", "q3": "z"})
    assert s["n"] == 2 and s["mean"] == pytest.approx(0.1)
