"""Re-review pass (2026-11-05): offline only, no model calls.

    uv run --offline --with matplotlib python paper/length_and_k_trends.py

R1. Length slope of S5 - S4@50 per doubling of duration within Vript (tier1.eval_length),
    expanded and strict labels (reproduces the paper's -0.011 / -0.013 first).
R2. Trend over k of S5 - S4@50: OLS of the per-query gain on k (+ set indicator when pooled),
    cluster bootstrap over sources; pooled and per set; expanded and strict labels; with the
    minimal slope detectable at 80% power (2.8 x bootstrap SE), as for length.
R3. Dev set (the 120 near-miss dev queries the trigger threshold was set on;
    results/smoke_v2): S5 - S4@50 and S5 - S4@10 nDCG@10, expanded and strict labels.
    Reproduces the pilot report's strict S5 - S4@10 (+0.043 [+0.003, +0.093]) first.
    Note: dev S5 reranks S0's top 10 (pilot), not S4@50's top 10.
R4. Contested subgroup check: S5 - S4@50 on the 96 contested near-miss queries (paper +0.134).
Writes results/full/review_20261105/rereview.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

T1, FS, PV = C.T1, C.FS, C.PV

S, sets, sub, ranks, lines, qrels = C.load()
pool = sub["pool"]
print("pool", pool)
out: dict = {}

for kind in ("expanded", "strict"):
    scores = {n: {s: PV.score_system(r, qrels[kind][n]) for s, r in ranks[n].items()} for n in sets}
    # R1
    L = T1.eval_length(S, sets, scores, sub, C.N_BOOT)
    s = L["slope_log2_duration"]
    out[f"length_{kind}"] = {"n": L["n"], "clusters": L["clusters"], **s}
    print(f"[R1 {kind}] n={L['n']} ({L['clusters']} src) slope {s['beta']:+.4f} "
          f"[{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}] se {s['se']:.4f} mde {s['mde_80_power']:.4f}")
    # R2
    for grp in ("pooled", "natural", "twin", "pooled_nmk2", "twin_k2"):
        Y, X, cl = [], [], []
        for name, rs in sets.items():
            if not grp.startswith("pooled") and not grp.startswith(name):
                continue
            sc = scores[name]
            for r in rs:
                q = r["query_id"]
                if grp in ("pooled_nmk2", "twin_k2") and name == "twin" and r["k"] == 1:
                    continue
                if q not in sc.get("S5", {}) or q not in sc[pool]:
                    continue
                Y.append(sc["S5"][q]["ndcg@10"] - sc[pool][q]["ndcg@10"])
                X.append([1.0, float(r["k"])] + ([float(name == "twin")] if grp.startswith("pooled") else []))
                cl.append(r["source"])
        fit = FS.ols_boot(Y, X, cl, C.N_BOOT)
        cell = {"n": len(Y), "clusters": fit["clusters"], "beta": fit["beta"][1],
                "ci": [fit["ci_low"][1], fit["ci_high"][1]], "se": fit["se"][1],
                "mde_80_power": 2.8 * fit["se"][1], "mean_gain": float(np.mean(Y))}
        out[f"k_trend_{kind}_{grp}"] = cell
        print(f"[R2 {kind} {grp}] n={cell['n']} slope/k {cell['beta']:+.4f} [{cell['ci'][0]:+.4f}, "
              f"{cell['ci'][1]:+.4f}] se {cell['se']:.4f} mde {cell['mde_80_power']:.4f} mean {cell['mean_gain']:+.3f}")
    # R4
    tw = scores["twin"]
    cont = set(sub["twin_both"]["twin"])
    rows = [r for r in sets["twin"] if r["query_id"] in cont]
    d = C.paired(tw["S5"], tw[pool], rows)
    out[f"contested_{kind}"] = d
    print(f"[R4 {kind}] contested S5 - S4@50 {C.fmt(d, True)}")

# R3 dev
DEV, DEV_DATA = Path("results/smoke_v2"), Path("data/mc/smoke_v2")
ids = json.loads((DEV / "pilot/frame_subset_twin_both.json").read_text())["twin"]
qrows = {r["query_id"]: r for r in T1.read_jsonl(DEV_DATA / "queries_twin.jsonl")}
keep = [qrows[q] for q in ids]
runs = {
    "S4@50": DEV / "deep_pool/twin/S4@50.jsonl",
    "S4@10": DEV / "pilot/twin/S4.jsonl",
    "S0": DEV / "pilot/twin/S0.jsonl",
    "S5": DEV / "pilot/twin/S5.jsonl",
}
rk = {}
for s, p in runs.items():
    rk[s] = {ln["query_id"]: [(d, x) for d, x in ln["ranking"]] for ln in T1.read_jsonl(p)}
src = {r["query_id"]: r["source"] for r in keep}
for kind, qp in (("expanded", DEV_DATA / "judge/qrels_expanded.jsonl"), ("strict", None)):
    qr = PV.load_qrels(qp, keep)
    sc = {s: PV.score_system({q: v for q, v in r.items() if q in src}, qr) for s, r in rk.items()}
    rows = [{"query_id": q, "source": src[q]} for q in ids]
    for a, b in (("S5", "S4@10"), ("S5", "S4@50"), ("S4@50", "S4@10"), ("S5", "S0")):
        d = C.paired(sc[a], sc[b], rows)
        out[f"dev_{kind}_{a}-{b}"] = d
        print(f"[R3 dev {kind}] {a} - {b}: {C.fmt(d, True)}")
    for s in sc:
        v = [sc[s][q]["ndcg@10"] for q in ids if q in sc[s]]
        print(f"   {s}: n={len(v)} mean {np.mean(v):.3f}")
# how many dev S4@50 sources is the source in top 10?
in10 = sum(1 for q in ids if src[q] in [d for d, _ in rk["S4@50"][q][:10]])
print("dev: source in S4@50 top 10:", in10, "of", len(ids), "; sources:", len(set(src.values())))
out["dev_source_in_s4at50_top10"] = in10
print("wrote", C.dump("rereview.json", out))
