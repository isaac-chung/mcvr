"""Review item W5 (2026-11-06): long descriptions (S4L) vs frames (S5) vs short descriptions (S4) on the same ten. Offline.

    uv run --offline --with matplotlib python paper/tab_w5_long_descriptions.py

Analysis fixed in run_plan_w5_longdesc_20261106.md (section 5). Modelled on tab_s0pool.py (same loaders,
metrics, review_common.boot: paired cluster bootstrap over source videos, 10,000 resamples, seed 0).
Pools: S4@50's top 10 (S4 short = S4@50, S5 = S5, S4L = S4L_s4pool) and S0's top 10 (S4 short = S4@10, i.e. S0's
top 10 reordered by stored S4 grades, S5 = S5_s0top10, S4L = S4L_s0pool). nDCG@10, expanded labels primary,
strict secondary; nested (natural 350), near-miss (twin 372), pooled (722).
PRIMARY (per plan, per pool, pooled expanded): S5 - S4L and S4L - S4; TOST margin +-0.02 on S5 - S4L (90% interval).
SECONDARY: share of frame gain recovered (S4L - S4)/(S5 - S4); headroom shares; source in/out of pool split;
contested subgroups (HN-above-source); cost per query; describer tokens per video; fallback-pair count.
Decision table (plan, pooled, expanded) applied per pool at the end.
Writes results/full/review_20261106/w5_longdesc.json and .md.
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
MARGIN = 0.02
FALLBACK = "vript_short/2oAeEoKRKGQ"


def read(path):
    return [json.loads(x) for x in open(path)]


def ndcg_of(order, qrel):
    return C.PV.query_metrics([(d, 0.0) for d in order], qrel)["ndcg@10"]


def ratio_boot(num, den, clusters, n_boot=C.N_BOOT, seed=0):
    num, den = np.asarray(num, float), np.asarray(den, float)
    uniq, inv = np.unique(np.asarray(clusters), return_inverse=True)
    sn = np.bincount(inv, weights=num, minlength=len(uniq))
    sd = np.bincount(inv, weights=den, minlength=len(uniq))
    idx = np.random.default_rng(seed).integers(0, len(uniq), size=(n_boot, len(uniq)))
    dd = sd[idx].sum(1)
    r = np.where(dd > 0, sn[idx].sum(1) / np.where(dd > 0, dd, 1), np.nan)
    return {"mean": float(num.sum() / den.sum()), "ci_low": float(np.nanpercentile(r, 2.5)),
            "ci_high": float(np.nanpercentile(r, 97.5)), "n": int(len(num))}


def tost(d):
    inside = d["ci90_low"] > -MARGIN and d["ci90_high"] < MARGIN
    return {"margin": MARGIN, "ci90": [d["ci90_low"], d["ci90_high"]],
            "equivalent_within_margin": bool(inside)}


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    groups = {"nested": sets["natural"], "near-miss": sets["twin"], "pooled": [r for rs in sets.values() for r in rs]}
    out: dict = {"n_queries": {k: len(v) for k, v in groups.items()}}

    new = {p: {n: {x["query_id"]: x for x in read(RUNS / n / f"S4L_{p}.jsonl")} for n in sets} for p in ("s4pool", "s0pool")}
    s0top = {n: {q: [d for d, _ in ranks[n]["S0"][q][:10]] for q in ranks[n]["S0"]} for n in sets}
    s4top = {n: {q: [d for d, _ in ranks[n]["S4@50"][q][:10]] for q in ranks[n]["S4@50"]} for n in sets}
    POOL = {"S4@50 pool": ("s4pool", s4top), "S0 pool": ("s0pool", s0top)}
    SHORT = {"S4@50 pool": "S4@50", "S0 pool": "S4@10"}
    FRAMES = {"S4@50 pool": "S5", "S0 pool": "S5_s0top10"}
    sanity = {"queries": {}, "not_10_candidates": 0, "ids_not_equal_pool": 0, "unparsed_scores": 0,
              "non_finite_scores": 0, "query_ids_equal_test_set": True, "errors_in_cost_files": {}, "S5_s0top10_ids_not_equal_S0_top10": 0}
    rk = {pl: {n: {"S4L": {}, "S4": {}, "S5": {}} for n in sets} for pl in POOL}
    for pl, (pf, top) in POOL.items():
        for n, rs in sets.items():
            ids = {r["query_id"] for r in rs}
            sanity["queries"][f"{pl} {SETLAB[n]}"] = len(new[pf][n])
            sanity["query_ids_equal_test_set"] &= set(new[pf][n]) == ids
            sanity["errors_in_cost_files"][f"S4L_{pf} {SETLAB[n]}"] = json.loads((RUNS / n / f"cost_S4L_{pf}.json").read_text()).get("n_errors")
            frames_runs = ranks[n]["S5"] if pl == "S4@50 pool" else {x["query_id"]: x["ranking"] for x in read(RUNS / n / "S5_s0top10.jsonl")}
            for q in ids:
                tn = new[pf][n][q]["ranking"][:10]
                sanity["not_10_candidates"] += len(tn) != 10
                sanity["ids_not_equal_pool"] += set(d for d, _ in tn) != set(top[n][q])
                for _, s in tn:
                    if not isinstance(s, (int, float)):
                        sanity["unparsed_scores"] += 1
                    elif not math.isfinite(s):
                        sanity["non_finite_scores"] += 1
                rk[pl][n]["S4L"][q] = [(d, float(s)) for d, s in new[pf][n][q]["ranking"]]
                rk[pl][n]["S4"][q] = ranks[n][SHORT[pl]][q][:10]
                fr = frames_runs[q]
                rk[pl][n]["S5"][q] = [(d, float(s)) for d, s in fr][:10]
                sanity["ids_not_equal_pool"] += set(d for d, _ in rk[pl][n]["S5"][q]) != set(top[n][q])
                sanity["ids_not_equal_pool"] += set(d for d, _ in rk[pl][n]["S4"][q]) != set(top[n][q])
    out["sanity"] = sanity

    # ---- scores ----
    SC = {}
    for kind in ("expanded", "strict"):
        SC[kind] = {}
        for pl, (pf, top) in POOL.items():
            SC[kind][pl] = {}
            for n in sets:
                qr = qrels[kind][n]
                SC[kind][pl][n] = {
                    "S4 short": C.score(rk[pl][n]["S4"], qr), "S4L": C.score(rk[pl][n]["S4L"], qr), "S5": C.score(rk[pl][n]["S5"], qr),
                    "oracle": {q: {"ndcg@10": ndcg_of(sorted(top[n][q], key=lambda d: -qr[q].get(d, 0)), qr[q])} for q in top[n]}}
            SC[kind][pl]["pooled"] = {s: {q: v for n in sets for q, v in SC[kind][pl][n][s].items()} for s in SC[kind][pl]["natural"]}
        SC[kind]["_ref"] = {n: {"S4@50": C.score(ranks[n]["S4@50"], qrels[kind][n]), "S5": C.score(ranks[n]["S5"], qrels[kind][n]),
                                "S4@10": C.score({q: ranks[n]["S4@10"][q][:10] for q in ranks[n]["S4@10"]}, qrels[kind][n])} for n in sets}

    def sc(kind, pl, g):
        return SC[kind][pl]["pooled"] if g == "pooled" else SC[kind][pl]["natural" if g == "nested" else "twin"]

    def mean(s, rows, name):
        return C.mean_ci([s[name][r["query_id"]]["ndcg@10"] for r in rows], [r["source"] for r in rows])

    def hshare(s, rows, a, b):
        q = [r["query_id"] for r in rows]
        return ratio_boot([s[a][i]["ndcg@10"] - s[b][i]["ndcg@10"] for i in q], [s["oracle"][i]["ndcg@10"] - s[b][i]["ndcg@10"] for i in q], [r["source"] for r in rows])

    def recovered(s, rows):
        q = [r["query_id"] for r in rows]
        return ratio_boot([s["S4L"][i]["ndcg@10"] - s["S4 short"][i]["ndcg@10"] for i in q],
                          [s["S5"][i]["ndcg@10"] - s["S4 short"][i]["ndcg@10"] for i in q], [r["source"] for r in rows])

    names = ["S4 short", "S4L", "S5", "oracle"]
    out["results"] = {}
    for kind in ("expanded", "strict"):
        out["results"][kind] = {}
        for pl in POOL:
            res = {}
            for g, rows in groups.items():
                s = sc(kind, pl, g)
                d = C.paired(s["S5"], s["S4L"], rows)
                res[g] = {
                    "ndcg@10": {k: mean(s, rows, k) for k in names},
                    "PRIMARY S5 - S4L": d, "TOST on S5 - S4L (+-0.02)": tost(d),
                    "PRIMARY S4L - S4 short": C.paired(s["S4L"], s["S4 short"], rows),
                    "secondary S5 - S4 short": C.paired(s["S5"], s["S4 short"], rows),
                    "secondary share of frame gain recovered by S4L": recovered(s, rows),
                    "secondary headroom share S4L": hshare(s, rows, "S4L", "S4 short"),
                    "secondary headroom share S5": hshare(s, rows, "S5", "S4 short")}
            out["results"][kind][pl] = res

    # ---- source in / out of pool (secondary, pooled expanded) ----
    out["secondary_source_in_pool"] = {}
    for pl, (pf, top) in POOL.items():
        s = SC["expanded"][pl]["pooled"]
        pooled = groups["pooled"]
        tn = {r["query_id"]: ("natural" if r["query_id"] in new[pf]["natural"] else "twin") for r in pooled}
        o = {}
        for lab, flag in (("source in pool", True), ("source not in pool", False)):
            rows = [r for r in pooled if (r["source"] in top[tn[r["query_id"]]][r["query_id"]]) == flag]
            o[lab] = {"n": len(rows), "S5 - S4L": C.paired(s["S5"], s["S4L"], rows), "S4L - S4 short": C.paired(s["S4L"], s["S4 short"], rows),
                      "S5 - S4 short": C.paired(s["S5"], s["S4 short"], rows)}
        out["secondary_source_in_pool"][pl] = o

    # ---- contested subgroups ----
    tw = sets["twin"]
    out["contested"] = {}

    def hn_above(rank_by_q, rows):
        v = []
        for r in rows:
            order = [d for d, _ in rank_by_q[r["query_id"]][:10]]
            v.append(float(order.index(r["twin"]) < order.index(r["source"])))
        return v

    for pl, (pf, top) in POOL.items():
        cont = [r for r in tw if {r["source"], r["twin"]} <= set(top["twin"][r["query_id"]])]
        cl = [r["source"] for r in cont]
        v = {k: hn_above(rk[pl]["twin"][k], cont) for k in ("S4", "S4L", "S5")}
        o = {"n": len(cont), "HN above source": {("S4 short" if k == "S4" else k): C.mean_ci(x, cl) for k, x in v.items()},
             "HN above source S5 - S4L (paired)": C.boot(list(np.array(v["S5"]) - np.array(v["S4L"])), cl),
             "HN above source S4L - S4 short (paired)": C.boot(list(np.array(v["S4L"]) - np.array(v["S4"])), cl)}
        for kind in ("expanded", "strict"):
            st = SC[kind][pl]["twin"]
            o[kind] = {"ndcg@10": {k: mean(st, cont, k) for k in names}, "S5 - S4L": C.paired(st["S5"], st["S4L"], cont),
                       "S4L - S4 short": C.paired(st["S4L"], st["S4 short"], cont)}
        out["contested"][pl] = o

    # ---- fallback video pairs ----
    fb = {}
    for pl, (pf, top) in POOL.items():
        qs = [(n, q) for n in sets for q, t in top[n].items() if FALLBACK in t]
        fb[pl] = {"pairs": len(qs), "queries": len(qs)}
    uni = {(n, q) for pl, (pf, top) in POOL.items() for n in sets for q, t in top[n].items() if FALLBACK in t}
    fb["union query-pools"] = len(uni)
    fd = {"fallback_description_words": None}
    for ln in open(C.ROOT / "data/mc/vript_short/descriptions_fps05_v4d.jsonl"):
        d = json.loads(ln)
        if d["video_id"] == "2oAeEoKRKGQ":
            fd = {"fallback_description_words_in_file": len(d["description"].split()), "note": d.get("fallback")}
    fb["fallback file entry"] = fd
    out["fallback"] = fb

    # ---- cost ----
    def cj(n, f):
        return json.loads((RUNS / n / f).read_text())

    def per_call(files, key):
        tin = sum(cj(n, f)[key]["tokens_in"] for n, f in files)
        cal = sum(cj(n, f)[key]["calls"] for n, f in files)
        tout = sum(cj(n, f)[key]["tokens_out"] for n, f in files)
        return {"in_per_call": tin / cal, "out_per_call": tout / cal, "in_per_query_10": 10 * tin / cal, "out_per_query_10": 10 * tout / cal, "calls": cal}

    cost = {"S4 short (S4@50 run, per call)": per_call([(n, "cost.json") for n in sets], "S4@50"),
            "S4L S4@50 pool": per_call([(n, "cost_S4L_s4pool.json") for n in sets], "S4L_s4pool (s4long)"),
            "S4L S0 pool": per_call([(n, "cost_S4L_s0pool.json") for n in sets], "S4L_s0pool (s4long)"),
            "S5 S0 pool": per_call([(n, "cost_s0top10.json") for n in sets], "S5_s0top10 (s0pool_frames)")}
    # S5 on S4@50's pool: logged per-query cost lines (as the paper's cost table)
    allq = {n: [r["query_id"] for r in sets[n]] for n in sets}
    tin5 = np.concatenate([C.tokens(lines[n], "S5", allq[n])[0] for n in sets])
    cost["S5 S4@50 pool (logged per query, mean)"] = {"in_per_query": float(tin5.mean())}
    out["cost_per_query_input_tokens"] = cost
    ld = json.loads((C.ROOT / "results/full/long_desc/spend.json").read_text())["entries"]
    le = [e for e in ld if e["stage"] == "long_desc"][0]
    ix = json.loads((C.ROOT / "results/full/index/build.json").read_text())["describe"]["recorded"]
    ixv = json.loads((C.ROOT / "results/full/index/build.json").read_text())["settings"]["n_docs"]
    out["describer_tokens_per_video"] = {
        "long (v4d, 3,470 pool videos)": {"in": le["tokens_in"] / le["videos"], "out": le["tokens_out"] / le["videos"], "usd_total": le["usd"], "videos": le["videos"]},
        "index (v4, all 8,603 videos)": {"in": ix["tokens_in"] / ixv, "out": ix["tokens_out"] / ixv, "videos": ixv},
        "note": "one-off indexing cost; long set covers pool videos only"}
    wl = []
    for ds in ("carebench", "shot2story", "vript_long", "vript_short"):
        wl += [len(json.loads(x)["description"].split()) for x in open(C.ROOT / f"data/mc/{ds}/descriptions_fps05_v4d.jsonl")]
    out["long_description_words"] = {"n_in_files": len(wl), "mean": float(np.mean(wl))}

    # ---- sanity: reproduce ----
    pe = SC["expanded"]
    ref = lambda name: float(np.mean([pe["_ref"][n][name][r["query_id"]]["ndcg@10"] for n in sets for r in sets[n]]))
    out["sanity"]["reproduce"] = {
        "S4@50 pooled (expect .634)": ref("S4@50"), "S5 pooled (expect .711)": ref("S5"), "S4@10 pooled (expect .530)": ref("S4@10"),
        "S5@S0top10 pooled (expect .570)": float(np.mean(list(pe["S0 pool"]["pooled"]["S5"][q]["ndcg@10"] for q in pe["S0 pool"]["pooled"]["S5"]))),
        "S4 short on S4@50 pool equals S4@50 (max abs diff)": float(max(abs(pe["S4@50 pool"]["pooled"]["S4 short"][q]["ndcg@10"] - pe["_ref"]["natural" if q in pe["_ref"]["natural"]["S4@50"] else "twin"]["S4@50"][q]["ndcg@10"]) for q in pe["S4@50 pool"]["pooled"]["S4 short"]))}

    # ---- decision table ----
    dec = {}
    for pl in POOL:
        r = out["results"]["expanded"][pl]["pooled"]
        a, b, t = r["PRIMARY S5 - S4L"], r["PRIMARY S4L - S4 short"], r["TOST on S5 - S4L (+-0.02)"]
        rows = []
        if a["ci_low"] > 0:
            rows.append("S5 - S4L > 0, CI above 0: frames beat longer text on the same candidates; Finding 3 stands against a stronger text control")
        if t["equivalent_within_margin"]:
            rows.append("S5 ~ S4L (90% interval inside +-0.02): long descriptions recover the frame gain at ~1/20 of query cost")
        if b["ci_high"] <= 0 or b["mean"] <= 0:
            rows.append("S4L - S4 <= 0: length does not help")
        if a["ci_high"] < 0:
            rows.append("S4L - S5 > 0 (CI): drop the 'frames beat text' wording")
        if not rows:
            rows.append("no table row fires cleanly (see intervals)")
        dec[pl] = {"S5 - S4L": a, "S4L - S4": b, "TOST": t, "rows": rows}
    out["decision"] = dec
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "w5_longdesc.json").write_text(json.dumps(out, indent=1, default=C.T1._json))

    # ---- markdown ----
    f = C.fmt
    rr = lambda v: f"{v['mean']:.3f} [{v['ci_low']:.3f}, {v['ci_high']:.3f}]"
    md = ["# W5: long descriptions (S4L) vs frames (S5) vs short descriptions (S4), same ten (2026-11-06)", "",
          "nDCG@10, 722 test queries, 95% cluster-bootstrap CI over sources (10,000, seed 0). Plan: run_plan_w5_longdesc_20261106.md section 5. Primary = S5 - S4L and S4L - S4 short, pooled expanded; all else secondary.", ""]
    for pl in POOL:
        md += [f"## Decision, {pl} (pooled, expanded)", ""] + [f"- {x}" for x in dec[pl]["rows"]] + [""]
    for kind in ("expanded", "strict"):
        for pl in POOL:
            md += [f"## {kind}, {pl}", "", "| group | S4 short | S4L | S5 | oracle | PRIMARY S5 - S4L | 90% (TOST +-0.02) | PRIMARY S4L - S4 short | S5 - S4 (sec.) | S4L share of frame gain (sec.) | headroom S4L / S5 (sec.) |", "|---|---|---|---|---|---|---|---|---|---|---|"]
            for g, r in out["results"][kind][pl].items():
                t = r["TOST on S5 - S4L (+-0.02)"]
                md.append(f"| {g} | " + " | ".join(f"{r['ndcg@10'][k]['mean']:.3f}" for k in names) + f" | {f(r['PRIMARY S5 - S4L'])} | [{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}] {'equiv' if t['equivalent_within_margin'] else 'not equiv'} | {f(r['PRIMARY S4L - S4 short'])} | {f(r['secondary S5 - S4 short'])} | {rr(r['secondary share of frame gain recovered by S4L'])} | {rr(r['secondary headroom share S4L'])} / {rr(r['secondary headroom share S5'])} |")
            md.append("")
    md += ["## Source in / out of pool (secondary, pooled expanded)", ""]
    for pl, o in out["secondary_source_in_pool"].items():
        for k, v in o.items():
            md.append(f"- {pl}, {k}: n={v['n']}, S5 - S4L {f(v['S5 - S4L'])}, S4L - S4 {f(v['S4L - S4 short'])}, S5 - S4 {f(v['S5 - S4 short'])}")
    md += ["", "## Contested subgroups (secondary)", ""]
    for pl, o in out["contested"].items():
        md.append(f"- {pl}: n={o['n']}; HN above source: " + ", ".join(f"{k} {v['mean']*100:.1f}% [{v['ci_low']*100:.1f}, {v['ci_high']*100:.1f}]" for k, v in o["HN above source"].items())
                  + f"; S5 - S4L {o['HN above source S5 - S4L (paired)']['mean']*100:+.1f} pts [{o['HN above source S5 - S4L (paired)']['ci_low']*100:+.1f}, {o['HN above source S5 - S4L (paired)']['ci_high']*100:+.1f}]; S4L - S4 {o['HN above source S4L - S4 short (paired)']['mean']*100:+.1f} pts [{o['HN above source S4L - S4 short (paired)']['ci_low']*100:+.1f}, {o['HN above source S4L - S4 short (paired)']['ci_high']*100:+.1f}]")
        for kind in ("expanded", "strict"):
            md.append(f"  - {kind} nDCG@10: " + ", ".join(f"{k} {v['mean']:.3f}" for k, v in o[kind]["ndcg@10"].items()) + f"; S5 - S4L {f(o[kind]['S5 - S4L'])}; S4L - S4 {f(o[kind]['S4L - S4 short'])}")
    md += ["", "## Fallback description", "", "```", json.dumps(out["fallback"], indent=1), "```", "", "## Cost", "", "```",
           json.dumps({"cost_per_query_input_tokens": cost, "describer_tokens_per_video": out["describer_tokens_per_video"], "long_description_words": out["long_description_words"]}, indent=1, default=C.T1._json), "```", "",
           "## Sanity", "", "```", json.dumps(out["sanity"], indent=1, default=C.T1._json), "```", ""]
    (OUTDIR / "w5_longdesc.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
