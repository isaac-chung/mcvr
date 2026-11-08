"""Rebuild the S6 margin sweep (and the S4 top-score sweep) from saved per-query outputs.

No model calls: reuses pipeline/run_graded_systems.py's loaders (saved runs + qrels) exactly as
run_graded_systems.py --stage eval does, then TS.sweep / TS.crossfit / TS.point_cis.
Run from the repo root:  env -u OPENAI_API_KEY -u GEMINI_API_KEY .venv/bin/python -I <this> <out.csv>
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_t1", ROOT / "pipeline/run_graded_systems.py")
T1 = importlib.util.module_from_spec(spec)
sys.modules["_t1"] = T1
spec.loader.exec_module(T1)
FS, TS = T1.FS, T1.TS

out_csv = Path(sys.argv[1])
N_BOOT = int(sys.argv[2]) if len(sys.argv) > 2 else 2000

args = argparse.Namespace(config=None, out=None, n_boot=FS.N_BOOT, tier1_config=None, all_sources=False)
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
qrels_path = S.judge_dir / "qrels_expanded.jsonl"
qrels = {n: FS.PV.load_qrels(qrels_path, rs) for n, rs in sets.items()}
rows = T1.trigger_rows(S, sets, ranks, lines, qrels, sub)
dev = FS.dev_s6_rows()
for r in dev:  # dev rows lack s4_top / k; add s4_top from the saved S4@50 ranking
    pass

groups = {
    "test pooled": rows,
    "test nested": [r for r in rows if r["set"] == "natural"],
    "test near-miss": [r for r in rows if r["set"] == "twin"],
    "test contested": [r for r in rows if r["set"] == "twin" and r["contested"]],
}

T = S.tier["trigger"]
t_m, t_k, t_s = float(T["margin_threshold"]), int(T["k_threshold"]), float(T["top_score_threshold"])


def point(rs, fire):
    n = len(rs)
    q4 = sum(r["q4"] for r in rs) / n
    q5 = sum(r["q5"] for r in rs) / n
    trig = [bool(fire(r)) for r in rs]
    q = sum(r["q5"] if f else r["q4"] for r, f in zip(rs, trig)) / n
    cost = sum(r["c_base"] + (r["c5"] if f else 0.0) for r, f in zip(rs, trig)) / n
    caught = sum(1 for r, f in zip(rs, trig) if f and r.get("top1_changed"))
    changes = sum(1 for r in rs if r.get("top1_changed"))
    return {"share": sum(trig) / n, "cost": cost, "ndcg10": q, "kept_ndcg": TS.kept(q, q4, q5),
            "caught": caught, "changes": changes}


out = []
summary = {}
for gname, rs in groups.items():
    n = len(rs)
    for feat, direction, label in (("s4_margin", "le", "margin <= t"), ("s4_top", "le", "S4 top score <= t")):
        pts = TS.sweep(rs, feat, direction)
        summary[f"{gname} | {label} | AUC"] = TS.auc_summary(pts)
        for p in pts:
            t = p["t"]
            cis = TS.point_cis(rs, feat, direction, t) if (math.isfinite(t) and gname == "test pooled") else {}
            caught = sum(1 for r in rs if TS.fires(r[feat], t, direction) and r["top1_changed"])
            out.append({
                "group": gname, "n": n, "kind": "sweep", "rule": label, "threshold": t,
                "share_sent": p["share"], "share_sent_ci": cis.get("share_ci"),
                "gain_kept": p["kept_ndcg"], "gain_kept_ci": cis.get("kept_ci"),
                "ndcg10": p["ndcg10"], "tokens_per_query": p["cost"],
                "top1_changes_caught": caught, "top1_changes": sum(r["top1_changed"] for r in rs),
                "marked": "dev threshold (margin <= 2)" if (feat == "s4_margin" and t == t_m) else
                          ("top score < 8 (= <= 7)" if (feat == "s4_top" and t == t_s - 1) else ""),
            })
    # named operating points
    share_m = sum(r["s4_margin"] <= t_m for r in rows) / len(rows)  # pooled share, as run_graded_systems.py
    named = {
        "margin <= 2 (dev threshold)": lambda r: r["s4_margin"] <= t_m,
        "k >= 3": lambda r: r["k"] >= t_k,
        "S4 top score < 8": lambda r: r["s4_top"] < t_s,
        f"random, share {share_m:.3f} (pooled margin share)": lambda r: r["u"] < share_m,
        "always (S5)": lambda r: True,
        "never (S4@50)": lambda r: False,
    }
    for name, fire in named.items():
        p = point(rs, fire)
        out.append({"group": gname, "n": n, "kind": "operating point", "rule": name, "threshold": "",
                    "share_sent": p["share"], "share_sent_ci": None, "gain_kept": p["kept_ndcg"],
                    "gain_kept_ci": None, "ndcg10": p["ndcg10"], "tokens_per_query": p["cost"],
                    "top1_changes_caught": p["caught"], "top1_changes": p["changes"], "marked": name})
    # cross-fit curve (5 folds over sources, 20 reps), margin and top score
    for feat, label in (("s4_margin", "margin"), ("s4_top", "S4 top score")):
        for frac in (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0):
            cf = TS.crossfit(rs, feat, "le", frac)
            out.append({"group": gname, "n": n, "kind": "cross-fit", "rule": f"{label}, keep >= {frac:g}",
                        "threshold": "per fold", "share_sent": cf["share"][0], "share_sent_ci": cf["share"][1:],
                        "gain_kept": cf["kept"][0], "gain_kept_ci": cf["kept"][1:], "ndcg10": "",
                        "tokens_per_query": cf["cost"][0], "top1_changes_caught": "", "top1_changes": "",
                        "marked": "cross-fit (paper: keep >= 0.8 / 0.9)" if (feat == "s4_margin" and frac in (0.8, 0.9)) else ""})

# dev sweep (the 120 dev twin-both queries the threshold was tuned on)
dev_pts = TS.sweep(dev, "s4_margin", "le")
summary["dev | margin <= t | AUC"] = TS.auc_summary(dev_pts)
for p in dev_pts:
    out.append({"group": "dev (120 twin-both queries; smoke_v2)", "n": len(dev), "kind": "sweep", "rule": "margin <= t",
                "threshold": p["t"], "share_sent": p["share"], "share_sent_ci": None, "gain_kept": p["kept_ndcg"],
                "gain_kept_ci": None, "ndcg10": p["ndcg10"], "tokens_per_query": p["cost"],
                "top1_changes_caught": "", "top1_changes": "",
                "marked": "dev threshold (margin <= 2)" if p["t"] == t_m else ""})
_dev_s4 = {ln["query_id"]: ln for ln in T1.read_jsonl(ROOT / FS.DEV / "deep_pool/twin/S4@50.jsonl")}
for r in dev:  # dev S4 top score from the saved dev S4@50 ranking
    r["s4_top"] = float(_dev_s4[r["query_id"]]["ranking"][0][1])
dev_tpts = TS.sweep(dev, "s4_top", "le")
summary["dev | S4 top score <= t | AUC"] = TS.auc_summary(dev_tpts)
for p in dev_tpts:
    out.append({"group": "dev (120 twin-both queries; smoke_v2)", "n": len(dev), "kind": "sweep",
                "rule": "S4 top score <= t", "threshold": p["t"], "share_sent": p["share"], "share_sent_ci": None,
                "gain_kept": p["kept_ndcg"], "gain_kept_ci": None, "ndcg10": p["ndcg10"],
                "tokens_per_query": p["cost"], "top1_changes_caught": "", "top1_changes": "",
                "marked": "top score < 8 (= <= 7)" if p["t"] == t_s - 1 else ""})


def fmt(v):
    if v is None or v == "":
        return ""
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, float):
        if math.isinf(v):
            return "-inf (never)" if v < 0 else "inf"
        if math.isnan(v):
            return "nan"
        return f"{v:.4f}"
    return str(v)


cols = ["group", "n", "kind", "rule", "threshold", "share_sent", "share_sent_ci", "gain_kept", "gain_kept_ci",
        "ndcg10", "tokens_per_query", "top1_changes_caught", "top1_changes", "marked"]
with out_csv.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(cols)
    for row in out:
        w.writerow([fmt(row[c]) for c in cols])
print(json.dumps({k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in summary.items()}, indent=1))
print("rows", len(out), "test rows", len(rows), "dev rows", len(dev))
