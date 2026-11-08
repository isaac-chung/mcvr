"""Smoke pilot follow-up (gold-constraint rows, balanced frame subset, splitter error), offline."""

import importlib.util
import json
from pathlib import Path

import pytest

from src.cache import Cache
from src.llm import CachedLLM, FakeLLM
from src.systems.decompose import GoldConstraints
from src.systems.verify import ConstraintVerifier, DescriptionCheck
from src.types import Cost, Query

SCRIPT = Path(__file__).resolve().parents[1] / "dev/verifiers_dev.py"


@pytest.fixture(scope="module")
def pv():
    spec = importlib.util.spec_from_file_location("pilot_verify", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows(per_k=(10, 10, 9, 8)):
    out = []
    for k, n in enumerate(per_k, 1):
        for i in range(n):
            out.append(
                {
                    "query_id": f"d/v{i}#k{k}",
                    "source": f"d/v{i}",
                    "k": k,
                    "dataset": "d",
                    "length_bin": "lt30s",
                    "query": f"q{i} k{k}",
                    "facts": [{"text": f"fact {j} of v{i}"} for j in range(k)],
                }
            )
    return out


def test_gold_constraints_returns_facts_without_cost():
    gold = GoldConstraints.from_rows(rows())
    facts, cost = gold("q3  k2")
    assert facts == ["fact 0 of v3", "fact 1 of v3"] and cost == Cost()
    with pytest.raises(KeyError):
        gold("unknown")


def test_gold_verifier_checks_the_facts():
    llm = FakeLLM(['{"constraints": [{"constraint": 1, "answer": "yes"}]}'] * 2)
    gold = GoldConstraints({"a red cap": ["The man wears a red cap."]})
    check = DescriptionCheck(llm, "{constraints} | {description}", {"x": "X", "y": "Y"})
    out = ConstraintVerifier("S2-gold", gold, check).rerank(
        Query("q", "a red cap", "x"), [("x", 1.0), ("y", 0.5)]
    )
    assert out.trace["constraints"] == ["The man wears a red cap."]
    assert out.cost.llm_calls == 2  # no split call


def test_subset_balanced_is_nested_and_balanced(pv):
    rs = rows((60, 60, 50, 40))
    small, big = pv.subset_ids(rs, 5), pv.subset_balanced(rs, 150)
    assert set(small) <= set(big) and len(big) == 150
    counts = [sum(q.endswith(f"#k{k}") for q in big) for k in (1, 2, 3, 4)]
    assert counts == [38, 38, 37, 37]
    assert set(pv.subset_balanced(rs, 40)) <= set(pv.subset_balanced(rs, 100))
    assert pv.per_k_counts(rows((3, 3, 3, 1)), 12) == {1: 3, 2: 3, 3: 3, 4: 1}


def test_projection_and_fit(pv):
    rs = rows((20, 20, 20, 20))
    plan = {"S3": (["a", "b", "c"], 10), "S2-gold": (["a"], 20)}
    proj = pv.project_new(plan, {"S3": {"a"}}, {"S3": 6000, "S2-gold": 300})
    assert proj["systems"]["S3"]["calls"] == 20 and proj["total_in"] == 20 * 6000 + 20 * 300
    fitted = pv.fit_subset(rs, 40, 0, 25 * 60000, lambda ids: len(ids) * 60000)
    assert fitted == 25


def test_token_overlap_and_parse_match(pv):
    assert pv.token_overlap("The jersey reads Sun 22.", "a jersey reads Sun 22") == 1.0
    assert pv.token_overlap("red cap", "blue car") == 0.0
    m = pv.parse_match(
        '{"facts": [{"fact": 1, "match": "full"}, {"fact": 2, "match": "weird"}],'
        ' "constraints": [{"constraint": 1, "fact": 1}, {"constraint": 2, "fact": null},'
        ' {"constraint": 3, "fact": 9}]}',
        2,
        3,
    )
    assert m == {"facts": ["full", "none"], "constraints": [1, None, None]}
    assert pv.parse_match("garbled", 1, 1) == {"facts": ["none"], "constraints": [None]}


def test_split_stats(pv):
    rs = rows((1, 1, 0, 0))
    splits = {"d/v0#k1": ["fact 0 of v0", "a man"], "d/v0#k2": ["fact 0 of v0", "fact 1 of v0"]}
    matches = {
        "d/v0#k1": {"facts": ["full"], "constraints": [1, None]},
        "d/v0#k2": {"facts": ["full", "full"], "constraints": [1, 2]},
    }
    st = pv.split_stats(rs, splits, matches)
    assert st["all"]["mean_constraints"] == 2.0 and st[1]["share_gt_k"] == 1.0
    assert st[2]["share_eq_k"] == 1.0 and st["all"]["overlap_fact_recall@0.5"] == 1.0
    assert st["all"]["llm_constraint_precision"] == 0.75 and st["all"]["llm_exact_split"] == 0.5


def test_metered_keeps_cache_keys(tmp_path, pv):
    cache = Cache(tmp_path / "c.sqlite")
    CachedLLM(FakeLLM(["a"]), cache, "x:y").complete("p")
    log = []
    again = CachedLLM(pv.Metered(FakeLLM(["b"]), log), cache, "x:y")
    assert again.complete("p")[0] == "a" and log == []
    assert again.complete("other")[0] == "b" and len(log) == 1


def test_read_rankings_derives_s2_gold_at_10(tmp_path, pv):
    first = [f"v{i}" for i in range(12)]
    s0 = {"query_id": "q", "ranking": [[d, 1.0 - i / 100] for i, d in enumerate(first)]}
    verdicts = {d: ["no"] for d in first}
    verdicts["v11"] = ["yes"]  # outside the top 10: must not move up at N = 10
    verdicts["v5"] = ["yes"]
    gold = {
        "query_id": "q",
        "ranking": s0["ranking"],
        "trace": {"first_stage": first, "verdicts": verdicts},
    }
    (tmp_path / "S0.jsonl").write_text(json.dumps(s0) + "\n")
    (tmp_path / "S2-gold.jsonl").write_text(json.dumps(gold) + "\n")
    systems = pv.read_rankings(tmp_path)
    assert systems["S2-gold@10"]["q"][0][0] == "v5"
    assert [d for d, _ in systems["S2-gold@10"]["q"]].index("v11") == 11


def test_evaluate_followup_pairs(pv):
    rs = rows((2, 2, 0, 0))
    hit = {r["query_id"]: [(r["source"], 1.0), ("z", 0.0)] for r in rs}
    miss = {r["query_id"]: [("z", 1.0), (r["source"], 0.0)] for r in rs}
    systems = {"S2-gold": hit, "S2": miss, "S3-gold": hit, "S3": hit}
    qrels = pv.load_qrels(None, rs)
    ev = pv.evaluate_followup(rs, systems, qrels, [rs[0]["query_id"]])
    p = ev["pairs"]["S2-gold - S2"]
    assert p["recall@1"]["overall"]["all"]["mean"] == 1.0 and set(p["mrr"]["k"]) == {1, 2}
    assert ev["pairs"]["S3-gold - S3"]["ndcg@10"]["overall"]["all"]["n"] == 1
    assert "length_bin" in p["ndcg@10"]
    lines = pv.followup_lines("t", ev, ["S2", "S2-gold"], ["S3", "S3-gold"])
    assert any(line.startswith("| S2-gold - S2 | all | all | 4 |") for line in lines)
