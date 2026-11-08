"""Shared loader for the 2026-11-05 review reanalyses (the paper/ scripts that import it). No model calls.

Loads the saved test-set runs, labels and helpers exactly as paper/fig2_trigger_sweep.py and
pipeline/run_graded_systems.py --stage eval do (same 100-source draw, same qrels, same rescoring from saved
grades, same cluster bootstrap T1.boot with seed 0 and 10,000 resamples), so numbers match the
paper. Run from the repo root:

    uv run --offline --with matplotlib python paper/tab_reagg.py

Data-key names vs paper names: S2g@50 (runs) = paper S2 / S2@50 (graded, descriptions);
S3g = paper S3 (graded, frames); S2@50 (runs) = paper S2b (binary); S3 (runs) = paper S3b.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/full/review_20261105"

_spec = importlib.util.spec_from_file_location("_t1", ROOT / "pipeline/run_graded_systems.py")
T1 = importlib.util.module_from_spec(_spec)
sys.modules["_t1"] = T1
_spec.loader.exec_module(T1)
FS, TS, PV = T1.FS, T1.TS, T1.PV
N_BOOT = 10_000
SET_LABEL = {"natural": "nested", "twin": "near-miss"}


def load():
    """(S, sets, sub, ranks, lines, qrels{kind: {set: qrels}}) on the 722 test queries."""
    args = argparse.Namespace(config=None, out=None, n_boot=N_BOOT, tier1_config=None, all_sources=False)
    S = T1.settings(args)
    ctx = FS.load_context(S, need_index=False)
    S.all_sets = FS.load_sets(S, ctx)
    sets = T1.draw_sets(S, S.all_sets)
    sub = FS.read_subsets(S)
    ranks, lines = T1.load_runs(S, sets, sub)
    for name, rs in sets.items():
        keep = {r["query_id"] for r in rs}
        ranks[name] = {s: {q: v for q, v in r.items() if q in keep} for s, r in ranks[name].items()}
        lines[name] = {s: {q: v for q, v in r.items() if q in keep} for s, r in lines[name].items()}
    qp = S.judge_dir / "qrels_expanded.jsonl"
    qrels = {
        "expanded": {n: PV.load_qrels(qp, rs) for n, rs in sets.items()},
        "strict": {n: PV.load_qrels(None, rs) for n, rs in sets.items()},
    }
    return S, sets, sub, ranks, lines, qrels


def score(rank_by_q: dict, qrels_set: dict) -> dict:
    return PV.score_system(rank_by_q, qrels_set)


def boot(diffs, clusters, n_boot: int = N_BOOT, seed: int = 0) -> dict:
    return T1.boot(diffs, clusters, n_boot, seed)


def paired(sa: dict, sb: dict, rows: list[dict], metric: str = "ndcg@10") -> dict | None:
    return T1.paired(sa, sb, rows, N_BOOT, metric)


def mean_ci(values, clusters, n_boot: int = N_BOOT) -> dict:
    """Mean with 95% cluster-bootstrap CI (values need not be differences)."""
    return boot(list(values), list(clusters), n_boot)


def fmt(d: dict | None, p: bool = False) -> str:
    if not d:
        return "-"
    s = f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"
    return s + (f" (n={d['n']})" if p else "")


def dump(name: str, obj) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    path.write_text(json.dumps(obj, indent=1, default=T1._json))
    return path


def tokens(lines_set: dict, system: str, qids) -> tuple[np.ndarray, np.ndarray]:
    """Per-query logged input and output tokens of a system (as the cost tables)."""
    ln = lines_set[system]
    tin = np.array([float(ln[q]["cost"]["tokens_in"]) for q in qids])
    tout = np.array([float(ln[q]["cost"]["tokens_out"]) for q in qids])
    return tin, tout
