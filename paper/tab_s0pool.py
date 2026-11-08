"""Review item W4 (2026-11-06): does the frame gain survive the pre-planned S0 pool? Offline, no model calls.

    uv run --offline --with matplotlib python paper/tab_s0pool.py

Analysis fixed in run_plan_w4_s0pool_20261106.md (section 4), nothing added to the primary set. Run files:
S0, S4@50 (stored S4 grades for all 50 S0 candidates), S5 (whole-query frame check on S4@50's top 10),
S5_s0top10 (the same check on S0's top 10, the new run). S4@10 = S0's top 10 reordered by the stored
S4 grades (systems.rescore, as in tab_w1_pools.py mirror pool b). All 722 test queries (nested =
natural 350, near-miss = twin 372); nDCG@10, expanded labels primary, strict secondary; paired cluster
bootstrap over source videos (review_common.boot: 10,000 resamples, seed 0).

Primary (pooled, expanded): S5@S0top10 - S4@10 (same ten candidates). Secondary, labelled as such:
S5@S0top10 - S0; S5 - S4@50 (existing, S4@50's pool); share of the oracle headroom recovered,
(S5 - S4)/(oracle - S4) on each pool (ratio of means, cluster bootstrap of the ratio, same draws);
the contested subgroup under S0's pool (near-miss queries with source and HN video both in S0's top 10);
HN-above-source rate on that subgroup (rank of the HN video 'twin' above the source, ranks only);
source in / out of S0's top 10 split of the primary gain.
Dev comparison: appdx_06_full_results.tex 'Frame gain on the development queries' is S5 (on S0's top 10)
- S4@50 over 120 dev near-miss queries with source and HN in S0's top 10 (S4@50 differs from S4@10 by
-0.002 there): the matching test quantity is the contested-subgroup gain under S0's pool, not pooled.
Decision table (pooled, expanded, primary): CI above 0 -> gain does not depend on the post-hoc pool;
CI spans 0 -> needs a strong first stage; CI below 0 -> frames hurt on S0's pool.
Sanity: 722 queries, 10 candidates each, ids equal S0's top 10, scores finite, S4@50 / S5 reproduced.

Writes results/full/review_20261106/w4_s0pool.json and w4_s0pool.md.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

OUTDIR = C.ROOT / "results/full/review_20261106"
SETLAB = {"natural": "nested", "twin": "near-miss"}
RUNS = C.ROOT / "results/full/runs"


def ndcg_of(order, qrel):
    return C.PV.query_metrics([(d, 0.0) for d in order], qrel)["ndcg@10"]


def read(path):
    return [json.loads(x) for x in open(path)]


def ratio_boot(num, den, clusters, n_boot=C.N_BOOT, seed=0):
    """Ratio of means (sum num / sum den) with a source-cluster bootstrap."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(uniq))
    sd = np.bincount(inv, weights=den, minlength=len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    dd = sd[idx].sum(1)
    r = np.where(dd > 0, sn[idx].sum(1) / np.where(dd > 0, dd, 1), np.nan)
    return {"mean": float(num.sum() / den.sum()), "ci_low": float(np.nanpercentile(r, 2.5)),
            "ci_high": float(np.nanpercentile(r, 97.5)), "n": int(len(num))}


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    groups = {"nested": sets["natural"], "near-miss": sets["twin"],
              "pooled": [r for rs in sets.values() for r in rs]}
    out: dict = {"n_queries": {k: len(v) for k, v in groups.items()}}

    # ---- load the new run, build pools ----
    new = {n: {x["query_id"]: x for x in read(RUNS / n / "S5_s0top10.jsonl")} for n in sets}
    sanity = {"queries": {}, "errors_in_cost_s0top10": {}, "unparsed_scores": 0, "not_10_candidates": 0,
              "ids_not_equal_S0_top10": 0, "non_finite_scores": 0, "query_ids_equal_test_set": True,
              "S4_10_pool_not_equal_S0_top10": 0}
    rk = {n: {"S5s0": {}, "S4_10": {}} for n in sets}
    for n, rs in sets.items():
        ids = {r["query_id"] for r in rs}
        sanity["queries"][SETLAB[n]] = len(new[n])
        sanity["query_ids_equal_test_set"] &= set(new[n]) == ids
        sanity["errors_in_cost_s0top10"][SETLAB[n]] = json.loads((RUNS / n / "cost_s0top10.json").read_text()).get("n_errors")
        for q in ids:
            s0pool = [d for d, _ in ranks[n]["S0"][q][:10]]
            top = new[n][q]["ranking"][:10]
            sanity["not_10_candidates"] += len(top) != 10
            sanity["ids_not_equal_S0_top10"] += set(d for d, _ in top) != set(s0pool)
            for _, s in top:
                if not isinstance(s, (int, float)):
                    sanity["unparsed_scores"] += 1
                elif not math.isfinite(s):
                    sanity["non_finite_scores"] += 1
            rk[n]["S5s0"][q] = [(d, float(s)) for d, s in new[n][q]["ranking"]]
            rk[n]["S4_10"][q] = ranks[n]["S4@10"][q][:10]
            sanity["S4_10_pool_not_equal_S0_top10"] += set(d for d, _ in rk[n]["S4_10"][q]) != set(s0pool)
    out["sanity"] = sanity

    # ---- scores ----
    SC = {}
    for kind in ("expanded", "strict"):
        SC[kind] = {}
        for n in sets:
            qr = qrels[kind][n]
            orc0, orc4 = {}, {}
            for q in rk[n]["S5s0"]:
                p0 = [d for d, _ in ranks[n]["S0"][q][:10]]
                p4 = [d for d, _ in ranks[n]["S4@50"][q][:10]]
                orc0[q] = {"ndcg@10": ndcg_of(sorted(p0, key=lambda d: -qr[q].get(d, 0)), qr[q])}
                orc4[q] = {"ndcg@10": ndcg_of(sorted(p4, key=lambda d: -qr[q].get(d, 0)), qr[q])}
            SC[kind][n] = {
                "S0": C.score(ranks[n]["S0"], qr), "S4@10": C.score(rk[n]["S4_10"], qr),
                "S5@S0top10": C.score(rk[n]["S5s0"], qr), "S4@50": C.score(ranks[n]["S4@50"], qr),
                "S5@S4top10": C.score(ranks[n]["S5"], qr), "oracle S0 top10": orc0, "oracle S4@50 top10": orc4}
        SC[kind]["pooled"] = {s: {q: v for n in sets for q, v in SC[kind][n][s].items()} for s in SC[kind]["natural"]}

    def sc(kind, g):
        return SC[kind]["pooled"] if g == "pooled" else SC[kind]["natural" if g == "nested" else "twin"]

    def mean(s, rows, name):
        return C.mean_ci([s[name][r["query_id"]]["ndcg@10"] for r in rows], [r["source"] for r in rows])

    def headroom(s, rows, sys_, base, orc):
        q = [r["query_id"] for r in rows]
        return ratio_boot([s[sys_][i]["ndcg@10"] - s[base][i]["ndcg@10"] for i in q],
                          [s[orc][i]["ndcg@10"] - s[base][i]["ndcg@10"] for i in q], [r["source"] for r in rows])

    names = ["S0", "S4@10", "S5@S0top10", "S4@50", "S5@S4top10", "oracle S0 top10", "oracle S4@50 top10"]
    out["results"] = {}
    for kind in ("expanded", "strict"):
        res = {}
        for g, rows in groups.items():
            s = sc(kind, g)
            res[g] = {
                "ndcg@10": {k: mean(s, rows, k) for k in names},
                "PRIMARY S5@S0top10 - S4@10": C.paired(s["S5@S0top10"], s["S4@10"], rows),
                "secondary S5@S0top10 - S0": C.paired(s["S5@S0top10"], s["S0"], rows),
                "secondary S4@10 - S0": C.paired(s["S4@10"], s["S0"], rows),
                "existing S5@S4top10 - S4@50": C.paired(s["S5@S4top10"], s["S4@50"], rows),
                "secondary headroom share, S0 pool": headroom(s, rows, "S5@S0top10", "S4@10", "oracle S0 top10"),
                "secondary headroom share, S4@50 pool": headroom(s, rows, "S5@S4top10", "S4@50", "oracle S4@50 top10"),
            }
        out["results"][kind] = res

    # ---- source in / out of S0's top 10 (secondary) ----
    s0top = {r["query_id"]: {d for d, _ in ranks["natural" if r["query_id"] in new["natural"] else "twin"]["S0"][r["query_id"]][:10]}
             for r in groups["pooled"]}
    s = SC["expanded"]["pooled"]
    gain = {r["query_id"]: s["S5@S0top10"][r["query_id"]]["ndcg@10"] - s["S4@10"][r["query_id"]]["ndcg@10"] for r in groups["pooled"]}
    tot = sum(gain.values())
    out["secondary_source_in_S0_top10"] = {}
    for lab, flag in (("source in S0 top 10", True), ("source not in S0 top 10", False)):
        rows = [r for r in groups["pooled"] if (r["source"] in s0top[r["query_id"]]) == flag]
        out["secondary_source_in_S0_top10"][lab] = {
            "n": len(rows), "S5@S0top10 - S4@10": C.paired(s["S5@S0top10"], s["S4@10"], rows),
            "share of total gain (sum)": float(sum(gain[r["query_id"]] for r in rows) / tot)}

    # ---- contested subgroup under S0's pool (secondary) ----
    tw = sets["twin"]
    cont = [r for r in tw if {r["source"], r["twin"]} <= s0top[r["query_id"]]]
    cont50 = [r for r in tw if {r["source"], r["twin"]} <= {d for d, _ in ranks["twin"]["S4@50"][r["query_id"]][:10]}]
    out["contested"] = {"n_S0_top10": len(cont), "n_S4@50_top10": len(cont50),
                        "frozen_subset_twin_both_equals_S4@50_def": set(sub["twin_both"]["twin"]) == {r["query_id"] for r in cont50}}

    def hn_above(rank_by_q, rows):
        v = []
        for r in rows:
            order = [d for d, _ in rank_by_q[r["query_id"]][:10]]
            v.append(float(order.index(r["twin"]) < order.index(r["source"])))
        return v

    for kind in ("expanded", "strict"):
        st = SC[kind]["twin"]
        out["contested"][kind] = {
            "secondary S5@S0top10 - S4@10": C.paired(st["S5@S0top10"], st["S4@10"], cont),
            "secondary S5@S0top10 - S0": C.paired(st["S5@S0top10"], st["S0"], cont),
            "secondary S5@S0top10 - S4@50 (dev-style definition)": C.paired(st["S5@S0top10"], st["S4@50"], cont),
            "ndcg@10": {k: mean(st, cont, k) for k in names},
            "secondary headroom share, S0 pool": headroom(st, cont, "S5@S0top10", "S4@10", "oracle S0 top10"),
        }
    cl = [r["source"] for r in cont]
    v0, v4, v5 = hn_above(ranks["twin"]["S0"], cont), hn_above(rk["twin"]["S4_10"], cont), hn_above(rk["twin"]["S5s0"], cont)
    out["contested"]["HN above source, S0 pool"] = {
        "n": len(cont), "S0": C.mean_ci(v0, cl), "S4@10": C.mean_ci(v4, cl), "S5@S0top10": C.mean_ci(v5, cl),
        "S5@S0top10 - S4@10 (paired)": C.boot(list(np.array(v5) - np.array(v4)), cl)}
    cl50 = [r["source"] for r in cont50]
    out["contested"]["HN above source, S4@50 pool (reference)"] = {
        "n": len(cont50), "S4@50": C.mean_ci(hn_above(ranks["twin"]["S4@50"], cont50), cl50),
        "S5@S4top10": C.mean_ci(hn_above(ranks["twin"]["S5"], cont50), cl50)}

    # ---- sanity: reproduce existing numbers ----
    pe = SC["expanded"]["pooled"]
    allq = [r["query_id"] for r in groups["pooled"]]
    out["sanity"]["reproduce"] = {
        "S4@50 pooled expanded (expect .634)": float(np.mean([pe["S4@50"][q]["ndcg@10"] for q in allq])),
        "S5@S4top10 pooled expanded (expect .711)": float(np.mean([pe["S5@S4top10"][q]["ndcg@10"] for q in allq])),
        "oracle S4@50 pool (expect .720)": float(np.mean([pe["oracle S4@50 top10"][q]["ndcg@10"] for q in allq])),
        "S4@10 nested / near-miss (expect .530 / .530)": [float(np.mean([SC["expanded"][n]["S4@10"][r["query_id"]]["ndcg@10"] for r in sets[n]])) for n in ("natural", "twin")],
        "S5 - S4@50 pooled (expect +0.077 [+0.059, +0.097])": out["results"]["expanded"]["pooled"]["existing S5@S4top10 - S4@50"]}
    d = out["results"]["expanded"]["pooled"]["PRIMARY S5@S0top10 - S4@10"]
    out["decision"] = ("gain > 0, CI above 0: the frame gain does not depend on the post-hoc pool" if d["ci_low"] > 0 else
                       "CI below 0: frames hurt on S0's pool" if d["ci_high"] < 0 else
                       "CI spans 0: the gain needs a strong first stage")
    out["decision_inputs"] = d
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "w4_s0pool.json").write_text(json.dumps(out, indent=1, default=C.T1._json))

    # ---- markdown ----
    f = C.fmt
    rr = lambda v: f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    md = ["# W4: frames on S0's top 10 (2026-11-06)", "", "nDCG@10, 722 test queries, 95% cluster-bootstrap CI over sources (10,000, seed 0). Plan: run_plan_w4_s0pool_20261106.md section 4.", "",
          f"**Decision (pooled, expanded, primary S5@S0top10 - S4@10 = {f(d)}): {out['decision']}.**", ""]
    for kind in ("expanded", "strict"):
        md += [f"## {kind}", "", "| group | S0 | S4@10 | S5@S0top10 | S4@50 | S5@S4top10 | oracle S0 pool | oracle S4@50 pool |", "|---|---|---|---|---|---|---|---|"]
        for g, r in out["results"][kind].items():
            md.append(f"| {g} | " + " | ".join(f"{r['ndcg@10'][k]['mean']:.3f}" for k in names) + " |")
        md += ["", "| group | PRIMARY S5@S0top10 - S4@10 | S5@S0top10 - S0 (sec.) | S5 - S4@50 (existing) | headroom share S0 pool (sec.) | headroom share S4@50 pool (sec.) |", "|---|---|---|---|---|---|"]
        for g, r in out["results"][kind].items():
            md.append(f"| {g} | {f(r['PRIMARY S5@S0top10 - S4@10'])} | {f(r['secondary S5@S0top10 - S0'])} | {f(r['existing S5@S4top10 - S4@50'])} | "
                      f"{rr(r['secondary headroom share, S0 pool'])} | {rr(r['secondary headroom share, S4@50 pool'])} |")
        md.append("")
    md += ["## Source in / out of S0's top 10 (secondary, pooled expanded)", ""]
    for k, v in out["secondary_source_in_S0_top10"].items():
        md.append(f"- {k}: n={v['n']}, gain {f(v['S5@S0top10 - S4@10'])}, share of total gain {v['share of total gain (sum)']:.3f}")
    c = out["contested"]
    md += ["", f"## Contested subgroup under S0's pool (secondary): n={c['n_S0_top10']} (S4@50 definition: n={c['n_S4@50_top10']})", ""]
    for kind in ("expanded", "strict"):
        for k, v in c[kind].items():
            if k == "ndcg@10":
                md.append(f"- {kind} nDCG@10: " + ", ".join(f"{a} {b['mean']:.3f}" for a, b in v.items()))
            elif "headroom" in k:
                md.append(f"- {kind} {k}: {rr(v)}")
            else:
                md.append(f"- {kind} {k}: {f(v)}")
    for k in ("HN above source, S0 pool", "HN above source, S4@50 pool (reference)"):
        v = c[k]
        md.append(f"- {k} (n={v['n']}): " + ", ".join(f"{a} {b['mean']*100:+.1f}% [{b['ci_low']*100:+.1f}, {b['ci_high']*100:+.1f}]" for a, b in v.items() if a != "n"))
    md += ["", "## Sanity", "", "```", json.dumps(out["sanity"], indent=1, default=C.T1._json), "```", ""]
    (OUTDIR / "w4_s0pool.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
