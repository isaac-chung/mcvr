"""Tier 1 (pipeline/run_graded_systems.py, src.systems.verify graded variant), offline."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.systems.verify import (
    GRADE_FAILED,
    GRADE_MISSING,
    DescriptionCheck,
    GradedVerifier,
    aggregate_grades,
    parse_grades,
)
from src.types import Cost, Query

ROOT = Path(__file__).resolve().parents[1]


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def t1():
    return load("pipeline/run_graded_systems.py", "_test_tier1")


def test_parse_grades_scale_missing_and_failed():
    text = json.dumps(
        {"constraints": [{"constraint": 1, "score": 10}, {"constraint": 3, "score": "2"}]}
    )
    assert parse_grades(text, 3) == [10.0, GRADE_MISSING, 2.0]
    assert parse_grades('{"constraints": {"1": 14, "2": -3}}', 2) == [10.0, 0.0]
    assert parse_grades("no json here", 2) == [GRADE_FAILED] * 2
    assert parse_grades('{"constraints": [{"constraint": 1, "score": "high"}]}', 1) == [
        GRADE_FAILED
    ]
    assert aggregate_grades([10, 5, 0]) == 5.0 and aggregate_grades([10, 5, 0], "min") == 0.0
    assert aggregate_grades([]) == GRADE_FAILED


class PromptLLM:
    def __init__(self, fn):
        self.fn, self.prompts = fn, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        return self.fn(prompt), Cost(llm_calls=1, tokens_in=10, tokens_out=2)


def test_graded_verifier_mean_min_and_trace():
    grades = {"D-A": [10, 10], "D-B": [10, 0], "D-C": [6, 6]}

    def answer(prompt):
        for key, g in grades.items():
            if key in prompt:
                return json.dumps(
                    {"constraints": [{"constraint": i + 1, "score": s} for i, s in enumerate(g)]}
                )
        return "garbage"

    llm = PromptLLM(answer)
    check = DescriptionCheck(
        llm,
        "Q {query}\n{constraints}\nD {description}",
        {"a": "D-A", "b": "D-B", "c": "D-C"},
        parse_grades,
    )
    splitter = lambda text: (["c1", "c2"], Cost())
    cands = [("c", 1.0), ("b", 0.9), ("a", 0.8), ("d", 0.7)]
    mean = GradedVerifier("S2g", splitter, check, workers=2).rerank(Query("q", "x", "a"), cands)
    assert [d for d, _ in mean.ranking] == ["a", "c", "b", "d"]  # 10, 6, 5, failed -1
    assert mean.trace["grades"]["d"] == [GRADE_FAILED] * 2 and mean.trace["aggregate"] == "mean"
    assert mean.cost.llm_calls == 4
    low = GradedVerifier("S2g", splitter, check, aggregate="min").rerank(
        Query("q", "x", "a"), cands
    )
    assert [d for d, _ in low.ranking] == ["a", "c", "b", "d"] and low.ranking[2][1] == 0.0


def test_rescore_graded_uses_saved_grades(t1):
    line = {
        "trace": {
            "first_stage": ["a", "b", "c"],
            "grades": {"a": [0, 10], "b": [6, 6], "c": [10, 10]},
        }
    }
    first = [("a", 3.0), ("b", 2.0), ("c", 1.0), ("d", 0.5)]
    assert [d for d, _ in t1.rescore_graded(line, first, 3, "mean")] == ["c", "b", "a", "d"]
    assert [d for d, _ in t1.rescore_graded(line, first, 3, "min")] == ["c", "b", "a", "d"]
    assert [d for d, _ in t1.rescore_graded(line, first, 2, "mean")] == ["b", "a", "c", "d"]


def test_apply_rule_shares_gain_and_top1_counts(t1):
    rows = []
    for i in range(20):
        fires = i % 2 == 0
        rows.append(
            {
                "query_id": f"q{i}",
                "source": f"s{i // 2}",
                "q4": 0.5,
                "q5": 0.5 + (0.2 if fires else 0.0),
                "r4": 0.0,
                "r5": 0.0,
                "s4_margin": 1.0 if fires else 5.0,
                "c_base": 100.0,
                "c5": 1000.0,
                "top1_changed": fires,
            }
        )
    r = t1.apply_rule(rows, lambda r: r["s4_margin"] <= 2, n_boot=200)
    assert r["share"] == 0.5 and r["kept"] == pytest.approx(1.0)
    assert r["top1_changes_caught"] == 10 and r["top1_changes_missed"] == 0
    assert r["tokens_per_query"] == pytest.approx(600.0)
    never = t1.apply_rule(rows, lambda r: False, n_boot=200)
    assert (
        never["share"] == 0.0
        and never["kept"] == pytest.approx(0.0)
        and never["top1_changes_missed"] == 10
    )


def test_fill_plan_counts_unjudged_pairs_once(t1):
    rows = [
        {"query_id": "q1", "source": "s", "twin": "t"},
        {"query_id": "q2", "source": "s", "twin": "t"},
        {"query_id": "q3", "source": "u", "twin": None},
    ]
    rank = {
        "S4@50": {"q1": ["s", "a", "t"], "q2": ["a", "b"], "q3": ["u"]},
        "S2g@50": {"q1": ["c", "a"], "q3": ["v"]},
    }
    judged = {"s": {"s", "a"}, "u": {"u"}}
    plan = t1.fill_plan(rows, rank, judged, ["S2g@50"], top=10)
    assert plan["pairs"] == 3  # (s,b), (s,c), (u,v); twin and judged skipped, (s,a) once
    assert plan["by_system"] == {"S4@50": 1, "S2g@50": 2} and plan["only_new_systems"] == 2
    assert plan["query_ids"] == ["q1", "q2", "q3"]


def test_holm(t1):
    assert t1.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]  # step-down keeps the running max
