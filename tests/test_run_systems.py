"""Full-run systems and judge scripts (pipeline/run_systems.py, judge.py): new pieces, offline."""

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.cache import Cache
from src.llm import CachedLLM, FakeLLM

ROOT = Path(__file__).resolve().parents[1]


def load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fs():
    return load("pipeline/run_systems.py", "_test_full_systems")


@pytest.fixture(scope="module")
def fj():
    return load("pipeline/judge_labels.py", "_test_full_judge")


def row(q, k, src, ds="vript_short", b="lt30s", twin=None):
    return {"query_id": q, "k": k, "source": src, "dataset": ds, "length_bin": b, "twin": twin,
            "query": q, "facts": [{"text": "x", "type": "object"}]}  # fmt: skip


def test_nested_sample_balanced_seeded_vript_only(fs):
    rows = [
        row(f"v{b}{k}{i}", k, f"s{b}{k}{i}", b=b)
        for b in fs.BINS
        for k in (1, 2, 3, 4)
        for i in range(12)
    ]
    rows += [row(f"x{i}", 1, f"x{i}", ds="shot2story") for i in range(30)]
    a = fs.nested_sample(rows, 150, seed=1)
    assert a == fs.nested_sample(list(reversed(rows)), 150, seed=1)
    assert len(a) == 150 and not any(q.startswith("x") for q in a)
    cell = {r["query_id"]: (r["length_bin"], r["k"]) for r in rows}
    per = {}
    for q in a:
        per[cell[q]] = per.get(cell[q], 0) + 1
    assert len(per) == 16 and set(per.values()) <= {9, 10}
    # a cell short of queries hands its share to the others
    short = [
        r
        for r in rows
        if not (cell.get(r["query_id"]) == ("lt30s", 4) and r["query_id"] != "vlt30s40")
    ]
    b = fs.nested_sample(short, 150, seed=1)
    assert len(b) == 150 and "vlt30s40" in b and sum(cell[q] == ("lt30s", 4) for q in b) == 1


def test_both_in_top_and_rescore(fs):
    rows = [row("q1", 1, "a", twin="b"), row("q2", 1, "a", twin="c")]
    ranks = {
        "q1": [("x", 1), ("b", 0.9), ("a", 0.8)],
        "q2": [("a", 1)] + [(f"d{i}", 0) for i in range(10)] + [("c", 0)],
    }
    assert fs.both_in_top(ranks, rows, 3) == ["q1"]
    s0 = [("a", 3.0), ("b", 2.0), ("c", 1.0), ("d", 0.5)]
    line = {
        "trace": {
            "first_stage": ["a", "b", "c"],
            "verdicts": {"a": ["no"], "b": ["yes"], "c": ["yes"]},
        }
    }
    assert [d for d, _ in fs.rescore(line, s0, 2)] == ["b", "a", "c", "d"]
    line4 = {"trace": {"first_stage": ["a", "b", "c"], "scores": {"a": 1, "b": 2, "c": 9}}}
    assert [d for d, _ in fs.rescore(line4, s0, 3)] == ["c", "b", "a", "d"]


def test_reseeded_gives_fresh_cache_keys(fs, tmp_path):
    cache = Cache(tmp_path / "c.sqlite")
    base = CachedLLM(FakeLLM(["one"]), cache, "fake")
    base.complete("p")
    assert base.is_cached("p")
    again = CachedLLM(fs.Reseeded(FakeLLM(["two"]), 1), cache, "fake")
    assert not again.is_cached("p")
    assert again.complete("p")[0] == "two" and base.complete("p")[0] == "one"
    assert not CachedLLM(fs.Reseeded(FakeLLM(["x"]), 2), cache, "fake").is_cached("p")


def test_ols_boot_recovers_slope(fs):
    rng = np.random.default_rng(0)
    n = 400
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 0.5 + 0.3 * x + rng.normal(scale=0.1, size=n)
    X = np.column_stack([np.ones(n), x, z])
    r = fs.ols_boot(y, X, [f"c{i // 2}" for i in range(n)], 500)
    assert abs(r["beta"][1] - 0.3) < 0.03 and r["ci_low"][1] < 0.3 < r["ci_high"][1]
    assert abs(r["beta"][2]) < 0.03 and r["vif"][0] is None and r["vif"][1] < 1.2


def test_ledger_stops_at_shared_and_total_caps(fs, tmp_path):
    S = SimpleNamespace(out=tmp_path, caps={"desc_systems": 100, "frames": 1000}, total_cap=150)
    (tmp_path / "retrieve").mkdir()
    (tmp_path / "retrieve" / "spend.json").write_text(json.dumps({"entries": [{"tokens_in": 60}]}))
    led = fs.Ledger(S, "desc")
    led.check("desc", 40)
    with pytest.raises(SystemExit):
        led.check("desc", 41)  # shared desc_systems cap: 60 + 41 > 100
    with pytest.raises(SystemExit):
        fs.Ledger(S, "frames").check("frames", 91)  # total cap: 60 + 91 > 150


def select_setup(fs, tmp_path, twin_both_min):
    twin = [row(f"t{i}", 1 + i % 4, f"s{i}", twin=f"w{i}") for i in range(8)]
    nat = [row(f"n{i}", 1 + i % 4, f"s{i}") for i in range(8)]
    sets = {"natural": nat, "twin": twin}
    S = SimpleNamespace(out=tmp_path, runs=tmp_path / "runs", top=2, twin_both_min=twin_both_min,
                        cfg={"nested_frame_sample": 4, "second_verifier_subset": 4}, nested_ids=None,
                        S4="S4@50")  # fmt: skip
    for name, rs in sets.items():
        (S.runs / name).mkdir(parents=True)
        with open(S.runs / name / "S0.jsonl", "w") as f:  # twin-both only for t0, t1
            for r in rs:
                ranking = [
                    [r["source"], 1.0],
                    [r["twin"] if r["query_id"] in ("t0", "t1") else "z", 0.5],
                ]
                f.write(json.dumps({"query_id": r["query_id"], "ranking": ranking}) + "\n")
    return S, sets


def test_select_keeps_s0_pool_when_twin_both_is_large(fs, tmp_path):
    S, sets = select_setup(fs, tmp_path, 2)
    fs.stage_select(S, {}, sets, argparse.Namespace(force=False))
    sub = json.loads((tmp_path / "select/subsets.json").read_text())
    assert sub["pool"] == "S0" and sub["twin_both"]["twin"] == ["t0", "t1"]
    assert len(sub["nested_sample"]["natural"]) == 4 and len(sub["second"]["twin"]) == 4
    assert sub["frames"] == {"natural": sub["nested_sample"]["natural"], "twin": ["t0", "t1"]}


def test_select_decision_3_waits_for_s4_then_uses_its_top(fs, tmp_path):
    S, sets = select_setup(fs, tmp_path, 3)
    args = argparse.Namespace(force=False)
    fs.stage_select(S, {}, sets, args)
    assert not (tmp_path / "select/subsets.json").exists()
    assert json.loads((tmp_path / "select/pending.json").read_text())["pool"] == "S4@50"
    for name, rs in sets.items():  # S4@50 puts source and twin on top for t0, t2, t3
        with open(S.runs / name / "S4@50.jsonl", "w") as f:
            for r in rs:
                hit = r["query_id"] in ("t0", "t2", "t3")
                ranking = [[r["source"], 9], [r["twin"] if hit else "z", 8]]
                f.write(json.dumps({"query_id": r["query_id"], "ranking": ranking}) + "\n")
    fs.stage_select(S, {}, sets, args)
    sub = json.loads((tmp_path / "select/subsets.json").read_text())
    assert sub["pool"] == "S4@50" and sub["twin_both_s0"] == 2
    assert sub["twin_both"]["twin"] == ["t0", "t2", "t3"]
    assert not (tmp_path / "select/pending.json").exists()


def test_s6_rows_margin_and_costs(fs):
    ranks = {
        "twin": {"S4@50": {"q": [("a", 9.0), ("b", 6.0)]}, "S5": {"q": [("b", 8.0), ("a", 7.0)]}}
    }
    lines = {"twin": {"S4@50": {"q": {"source": "a", "cost": {"tokens_in": 100}}},
                      "S5": {"q": {"source": "a", "cost": {"tokens_in": 5000}}}}}  # fmt: skip
    rows = fs.s6_rows(ranks, lines, {"twin": {"q": {"a": 1}}}, {"twin": ["q", "missing"]}, "S4@50")
    assert len(rows) == 1 and rows[0]["s4_margin"] == 3.0
    assert rows[0]["q4"] == 1.0 and rows[0]["r5"] == 0.0 and rows[0]["c5"] == 5000.0


def test_prompt_test_acceptance(fj):
    rows = [
        {"query_id": "s#k1", "source": "s", "k": 1},
        {"query_id": "s#twin-k1", "source": "s", "k": 1, "set": "twin"},
    ]
    specs = ("g", "o")
    yes = {"g": {"answer": "yes"}, "o": {"answer": "yes"}}
    judged = {"s": {"s": {"f0": yes, "f1": {"g": {"answer": "no"}, "o": {"answer": "yes"}}}}}
    acc = fj.acceptance(rows, {"s#k1": ["f0"], "s#twin-k1": ["f0", "f1"]}, judged, specs)
    assert (
        acc["both"]["rate"] == 0.5 and acc["o"]["rate"] == 1.0 and acc["both twin"]["rate"] == 0.0
    )


def test_description_drop_counts_unclear_and_skips_unchecked_sources(fs):
    S = SimpleNamespace(S2="S2@50")
    trace = {"verdicts": {"a": ["yes", "unclear", "no", "yes"], "x": ["yes"]}}
    lines = {"natural": {"S2@50": {
        "q1": {"source": "a", "trace": trace},
        "q2": {"source": "b", "trace": trace},  # source outside the pool: no verdict
    }}}  # fmt: skip
    sets = {"natural": [row("q1", 2, "a"), row("q2", 2, "b")]}
    d = fs.description_drop(S, sets, lines)
    assert set(d) == {("natural", "q1")} and d[("natural", "q1")]["yes"] == 0.5
    v = fs.drop_verdicts(S, sets, lines)["natural"]["k=2"]
    assert (v["queries"], v["checked"], v["constraints"]) == (2, 1, 4)
    assert (v["yes"], v["no"], v["unclear"]) == (0.5, 0.25, 0.25)
    assert fs.COVARIATES[-1] == "scenes_per_min" and "caption_words_per_min" not in fs.COVARIATES


def test_frame_verifier_slug_and_cache_key(fs, tmp_path):
    assert fs.frame_slug("gemini:gemini-3.8-flash@low", None) == "gemini-3.8-flash-default"
    assert fs.frame_slug("gemini:gemini-3.8-flash@low", "medium") == "gemini-3.8-flash-medium"
    S = SimpleNamespace(cfg={"results_dir": str(tmp_path)})
    low = fs.gemini_vision_models(S, "gemini:gemini-3.8-flash@low", "low")
    unset = fs.gemini_vision_models(S, "gemini:gemini-3.8-flash@low", None)
    assert low.media_resolution == "MEDIA_RESOLUTION_LOW" and unset.media_resolution is None
    # the media resolution is part of the cache key, so the variants never share answers
    assert low.vlm_cached._payload("p", [b"x"], ["Frame 1 of 1:"]) != unset.vlm_cached._payload(
        "p", [b"x"], ["Frame 1 of 1:"]
    )
    with pytest.raises(ValueError):
        fs.gemini_vision_models(S, "openai:gpt-6-sol@low", None)
