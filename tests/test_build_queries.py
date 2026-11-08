"""Full-run query build (pipeline/build_queries.py): offline helpers."""

import importlib.util
import json
import sys
from pathlib import Path

import pytest

from src.cache import Cache
from src.datasets.mc import Video
from src.llm import CachedVision, FakeVision
from src.types import Cost

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    folder = ROOT / "pipeline"
    sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cap_budget_stops_on_actual_spend_and_ledger_counts_paid_calls(tmp_path):
    b = load("build_queries")

    class LabelFake(FakeVision):
        def describe(self, prompt, images, labels=None):
            text, _ = super().describe(prompt, images)
            return text, Cost(llm_calls=1, tokens_in=1000, tokens_out=5)

    ledger = b.Ledger()
    budget = b.CapBudget(cap=5000, prior=2500, ledger=ledger, specs=("gemini:fake",))
    inner = CachedVision(LabelFake(["x"]), Cache(tmp_path / "c.sqlite"), "gemini:fake")
    vision = b.PaidVision(inner, budget, "gemini:fake", ledger)
    vision.describe("p", [b"1"])  # projected 1150 + words, actual 1000
    vision.describe("p", [b"1"])  # cached: not paid
    assert ledger.by_spec["gemini:fake"].tokens_in == 1000 and budget.calls == 1
    with pytest.raises(b.B2.BudgetExceeded):  # 2500 + 1000 + 1150 + ... > 5000 after one more
        vision.describe("q", [b"2", b"3"])


def test_record_spend_totals_and_prior(tmp_path):
    b = load("build_queries")
    path = tmp_path / "spend.json"
    ledger = b.Ledger()
    ledger.add(
        "gemini:gemini-3.5-flash-lite", Cost(llm_calls=2, tokens_in=1_000_000, tokens_out=100_000)
    )
    ledger.add("openai:gpt-6-astra", Cost(llm_calls=1, tokens_in=500))
    b.record_spend(path, "build", "canary", ledger)
    spend = b.record_spend(path, "build", "main", ledger)
    assert spend["totals"]["build"]["tokens_in"] == 2 * 1_000_500
    assert spend["gemini_usd"]["usd"] == pytest.approx(2 * (0.30 + 0.25))
    assert b.prior_tokens(spend) == 2 * 1_000_500
    assert b.prior_tokens(spend, "index_chunks") == 0


def test_youtube_ids_across_datasets():
    b = load("build_queries")
    s2s = Video("abcdefghijk.3", "x", 5.0, dataset="shot2story")
    long_ = Video("span1", "x", 70.0, dataset="vript_long", source_video_id="abcdefghijk")
    short_yt = Video("abcdefghijk", "x", 20.0, dataset="vript_short")
    tiktok = Video("7212345678901234567", "x", 20.0, dataset="vript_short")
    care = Video("v_12_0", "x", 9.0, dataset="carebench")
    assert {b.youtube_id(v) for v in (s2s, long_, short_yt)} == {"abcdefghijk"}
    assert b.youtube_id(tiktok) is None and b.youtube_id(care) is None


def test_run_sources_retries_failures_and_stops_on_budget(monkeypatch):
    b = load("build_queries")
    monkeypatch.setattr(b.time, "sleep", lambda s: None)
    seen = {}

    def flaky(g):
        seen[g] = seen.get(g, 0) + 1
        if g == "a" and seen[g] < 3:
            raise RuntimeError("429")
        return g.upper()

    results, errors = b.run_sources(["a", "b"], flaky, 2, rounds=4, wait_s=1, log=lambda m: None)
    assert results == {"a": "A", "b": "B"} and not errors and seen["a"] == 3
    results, errors = b.run_sources(
        ["a"], lambda g: 1 / 0, 1, rounds=1, wait_s=1, log=lambda m: None
    )
    assert not results and "ZeroDivisionError" in errors["a"]

    def broke(g):
        raise b.B2.BudgetExceeded("cap")

    with pytest.raises(b.B2.BudgetExceeded):
        b.run_sources(["a"], broke, 1, rounds=2, wait_s=1, log=lambda m: None)


def test_yield_check():
    b = load("build_queries")
    nat = [{"k": k} for k in (1, 2, 3, 4)] * 300
    twin = [{"k": k} for k in (1, 2, 3)] * 300 + [{"k": 4}] * 149
    yc = b.yield_check(nat, twin)
    assert yc["pass_queries"] and not yc["pass_twin_k4"] and yc["per_k"]["twin"][4] == 149




def test_build_report_runs_on_a_tiny_build(tmp_path):
    r = load("build_queries_report")
    out = tmp_path / "full"
    out.mkdir()
    g = {
        "key": "n1",
        "text": "A dog",
        "type": "object",
        "role": "natural",
        "window": {"kind": "whole"},
        "agreement": "all_yes",
        "answers": {"gemini:x": {"answer": "yes"}, "openai:y": {"answer": "yes"}},
        "confirmed": True,
        "index": {"visible": False, "n_frames": 40},
        "index_dropped": True,
    }
    rec = {
        "source": "vript_long/s",
        "dataset": "vript_long",
        "length_bin": "1to2min",
        "grounding": [g],
        "cost": {"ground_local": {"gemini:x": Cost(llm_calls=1).to_dict()}},
    }
    (out / "constraints.jsonl").write_text(json.dumps(rec) + "\n")
    (out / "queries.jsonl").write_text(json.dumps({"source": "vript_long/s", "k": 1}) + "\n")
    (out / "queries_twin.jsonl").write_text("")
    (out / "twins.jsonl").write_text(json.dumps({"source": "vript_long/s", "log": []}) + "\n")
    cfg = tmp_path / "cfg.yaml"
    cfg.write_text(f"out_dir: {out}\ngrounding_media_resolution: low\ncaps: {{build: 46e6}}\n")
    r.main(["--config", str(cfg)])
    text = (out / "build_report.md").read_text()
    assert "| vript bin=1to2min | 1 | 1 | 1.00 | 40.0 |" in text
    assert "| vript_long/1to2min | 1 | 1 | 0 | 0 | 0 | 1 |" in text
