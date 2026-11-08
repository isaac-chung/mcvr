"""100-source reframing: stratified draw, overlap with the 380, power on the draw, Tier 1 cost."""

import argparse
import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, "pipeline")
import importlib.util

spec = importlib.util.spec_from_file_location("_t1", "pipeline/run_graded_systems.py")
T1 = importlib.util.module_from_spec(spec)
sys.modules["_t1"] = T1
spec.loader.exec_module(T1)
FS, PV, TS = T1.FS, T1.PV, T1.TS

ap = argparse.ArgumentParser()
ap.add_argument("--n-sources", type=int, default=100)
ap.add_argument("--seed", type=int, default=1)
ap.add_argument("--n-boot", type=int, default=4000)
ap.add_argument("--overlap-only", action="store_true")
ap.add_argument("--out", help="write the draw (sources, query ids, strata, seed) to this JSON file")
args = ap.parse_args()
OUT = Path("results/full/select")
S = T1.settings(argparse.Namespace(config=None, tier1_config=None, out=None, n_boot=args.n_boot, all_sources=True))
ctx = FS.load_context(S, need_index=False)
sets = FS.load_sets(S, ctx)
sub = FS.read_subsets(S)
rows_all = [r for rs in sets.values() for r in rs]

# ---- 1. stratified draw of sources (strata = configs/mc_full.yaml test_strata)
strata = {}
for r in rows_all:
    key = (r["dataset"], r["length_bin"] if r["dataset"] in FS.VRIPT else "any")
    strata.setdefault(key, set()).add(r["source"])
keys = sorted(strata)
total = sum(len(v) for v in strata.values())
alloc = {k: args.n_sources * len(strata[k]) // total for k in keys}
rem = args.n_sources - sum(alloc.values())
for k in keys[:rem]:
    alloc[k] += 1
drawn = set()
for k in keys:
    ids = sorted(strata[k])
    random.Random(f"{args.seed}|{k}").shuffle(ids)
    drawn |= set(ids[: alloc[k]])
print(f"strata (sources): { {f'{k[0]}/{k[1]}': len(v) for k, v in strata.items()} } total {total}")
print(
    f"draw: { {f'{k[0]}/{k[1]}': alloc[k] for k in keys} } = {len(drawn)} sources, seed {args.seed}"
)
D = {n: [r for r in rs if r["source"] in drawn] for n, rs in sets.items()}
print(
    f"queries in draw: { {n: len(v) for n, v in D.items()} } of { {n: len(v) for n, v in sets.items()} }; by k nested { {k: sum(r['k'] == k for r in D['natural']) for k in (1, 2, 3, 4)} }, near-miss { {k: sum(r['k'] == k for r in D['twin']) for k in (1, 2, 3, 4)} }"
)
draw_json = {
    "n_sources": args.n_sources,
    "seed": args.seed,
    "rule": "stratified by (dataset, Vript length bin), proportional allocation, seeded shuffle of sorted source ids per stratum",
    "strata": {f"{k[0]}/{k[1]}": {"sources": len(strata[k]), "drawn": alloc[k]} for k in keys},
    "sources": sorted(drawn),
    "queries": {n: [r["query_id"] for r in rs] for n, rs in D.items()},
}
Path(args.out or OUT / f"draw{args.n_sources}_seed{args.seed}.json").write_text(
    json.dumps(draw_json, indent=1)
)

# ---- 2. overlap with the 380 frame-subset queries, and the second-verifier subsets
frames = {n: set(v) for n, v in sub["frames"].items()}
ov = {n: [r for r in D[n] if r["query_id"] in frames[n]] for n in sets}
print(
    f"overlap with the 380: contested {len(ov['twin'])} of 230, nested sample {len(ov['natural'])} of 150, total {sum(map(len, ov.values()))}; by k contested { {k: sum(r['k'] == k for r in ov['twin']) for k in (1, 2, 3, 4)} }"
)
second = {n: [r for r in D[n] if r["query_id"] in set(sub["second"][n])] for n in sets}
print(
    f"overlap with the second-verifier subsets: { {n: len(v) for n, v in second.items()} } of 150 each"
)
if args.overlap_only:
    print(
        json.dumps(
            {
                "seed": args.seed,
                "contested": len(ov["twin"]),
                "nested_sample": len(ov["natural"]),
                "total": sum(map(len, ov.values())),
                "contested_k1": sum(r["k"] == 1 for r in ov["twin"]),
                "contested_k4": sum(r["k"] == 4 for r in ov["twin"]),
                "second": {n: len(v) for n, v in second.items()},
                "queries": {n: len(v) for n, v in D.items()},
            }
        )
    )
    sys.exit(0)

# ---- 3. power: the paper's claims on the draw (existing runs)
ranks, lines = T1.load_runs(S, sets, sub)
qrels = {n: PV.load_qrels(S.judge_dir / "qrels_expanded.jsonl", rs) for n, rs in sets.items()}
qstrict = {n: PV.load_qrels(None, rs) for n, rs in sets.items()}
sc = {n: {s: PV.score_system(r, qrels[n]) for s, r in ranks[n].items()} for n in sets}
ss = {n: {s: PV.score_system(r, qstrict[n]) for s, r in ranks[n].items()} for n in sets}


def cell(a, b, rows, scores, label):
    d = T1.by_group(scores[a], scores[b], rows, args.n_boot)
    print(f"  {label:38s} " + " | ".join(f"{g}: {T1.ci(d.get(g))}" for g in ("all", "k=1", "k=4")))


print("\nclaims on the draw (expanded labels; mean [95% CI] (n)):")
for n in sets:
    L = T1.SET_LABEL[n]
    cell("S1", "S0", D[n], sc[n], f"{L} S1 - S0 (splitting hurts)")
    cell("S4@50", "S0", D[n], sc[n], f"{L} S4 - S0")
    cell("S4@50", "S2@50", D[n], sc[n], f"{L} S4 - S2@50 (main verifier)")
    cell("S4@10", "S2@10", D[n], sc[n], f"{L} S4@10 - S2@10")
    cell(
        "S4@50_gpt-5.5", "S2@50_gpt-5.5", second[n], sc[n], f"{L} S4 - S2 (gpt-5.5, second subset)"
    )
    cell("S4@50", "S2@50", D[n], ss[n], f"{L} S4 - S2@50 (strict)")
cell("S5", "S4@50", ov["twin"], sc["twin"], "contested S5 - S4@50")
cell("S5", "S4@50", ov["natural"], sc["natural"], "nested sample S5 - S4@50")
cell("S3", "S5", ov["twin"], sc["twin"], "contested S3 - S5 (TOST +-0.02)")
d = T1.paired(sc["twin"]["S3"], sc["twin"]["S5"], ov["twin"], args.n_boot)
if d:
    print(
        f"    TOST 90% CI [{d['ci90_low']:+.3f}, {d['ci90_high']:+.3f}] -> {'equivalent' if -0.02 < d['ci90_low'] and d['ci90_high'] < 0.02 else 'not shown'}"
    )
# trigger on the overlap
rows_t = T1.trigger_rows(S, sets, ranks, lines, qrels, sub)
keep = {r["query_id"] for rs in ov.values() for r in rs}
tr = [r for r in rows_t if r["query_id"] in keep and r["set"] == "twin"]
if len(tr) >= 5:
    a = T1.apply_rule(tr, lambda r: r["s4_margin"] <= 2, args.n_boot)
    print(
        f"  trigger margin<=2 on contested overlap (n {a['n']}): sent {a['share']:.1%} [{a['share_ci'][0]:.1%}, {a['share_ci'][1]:.1%}], kept {a['kept']:.1%} [{a['kept_ci'][0]:.1%}, {a['kept_ci'][1]:.1%}], top-1 changes caught/missed {a['top1_changes_caught']}/{a['top1_changes_missed']}"
    )
# length regression on the overlap (S5 available) and the projected MDE after B on the draw
meta = FS.source_meta(S)
Y, X, cl = [], [], []
for n in sets:
    for r in ov[n]:
        q = r["query_id"]
        if r["dataset"] in FS.VRIPT and q in sc[n]["S5"]:
            x = FS.covariates({**r, "set": n}, meta)
            if x:
                Y.append(sc[n]["S5"][q]["ndcg@10"] - sc[n]["S4@50"][q]["ndcg@10"])
                X.append(x)
                cl.append(r["source"])
j = FS.COVARIATES.index("log2_duration")
vript_draw = sum(
    1
    for n in sets
    for r in D[n]
    if r["dataset"] in FS.VRIPT and FS.covariates({**r, "set": n}, meta)
)
if len(Y) > 10:
    fit = FS.ols_boot(Y, X, cl, args.n_boot)
    se = fit["se"][j]
    print(
        f"  length slope on overlap Vript (n {len(Y)}, {fit['clusters']} sources): {fit['beta'][j]:+.4f} [{fit['ci_low'][j]:+.4f}, {fit['ci_high'][j]:+.4f}], MDE(80%) {2.8 * se:.4f}"
    )
    print(
        f"  after B on the draw: n ~{vript_draw} Vript queries -> projected SE {se * math.sqrt(len(Y) / vript_draw):.4f}, MDE(80%) {2.8 * se * math.sqrt(len(Y) / vript_draw):.4f} (full set: n 297 SE 0.0138 -> all ~1,200: MDE {2.8 * 0.0138 * math.sqrt(297 / 1200):.4f})"
    )
# S4 vs S2 gap at the current full-set width, scaled
n_full = sum(map(len, sets.values()))
n_draw = sum(map(len, D.values()))
print(
    f"  interval widths scale by sqrt({n_full}/{n_draw}) = {math.sqrt(n_full / n_draw):.2f}x on the draw"
)

# ---- 4. Tier 1 cost on the draw (from the cache; prices USD per 1M: sol 2/10, gpt-5.5 5/30)
ids = {n: [r["query_id"] for r in rs] for n, rs in D.items()}
first = {n: FS.read_ranking(S.runs / n / "S0.jsonl") for n in sets}
m = FS.text_models(FS.VERIFIER_SPEC, S)
tmpl = Path(S.tier["graded_desc_prompt"]).read_text()
P = T1.desc_prompts(S, sets, ids, first, ctx["descriptions"], tmpl, S.depth_g)
new = [p for p in dict.fromkeys(P.values()) if not m.verifier_cached.is_cached(p)]
tin = sum(P2 := FS.P2.text_tokens(p) for p in new)
a1 = tin / 1e6 * 2 + len(new) * 97 / 1e6 * 10
print(
    f"\nTier 1 cost on the draw:\n  A1 S2g@50: {len(P)} calls, {len(new)} uncached -> {tin / 1e6:.1f}M in, ${a1:.0f} (already cached {len(P) - len(new)})"
)
n2 = sum(min(150, len(D[n])) for n in sets)
a3 = n2 * 50 * (303 * 5 + 97 * 30) / 1e6
print(f"  A3 gpt-5.5 S2g@50 on 150 per set within the draw: {n2 * 50} calls, ${a3:.0f} (no cache)")
n_fr = sum(map(len, ov.values()))
a2 = n_fr * 10 * (8800 * 2 + 100 * 10) / 1e6
print(
    f"  A2 S3g on the frame-subset queries in the draw: {n_fr} queries, {n_fr * 10} calls, ${a2:.0f}"
)
n_b = n_draw - n_fr
b = n_b * 10 * (8650 * 2 + 35 * 10) / 1e6
print(f"  B  S5 on the other draw queries: {n_b} queries, {n_b * 10} calls, ${b:.0f}")
plan = T1.fill_plan(
    [r for rs in D.values() for r in rs],
    T1.rankings(S),
    T1.judged_pairs(S),
    S.tier["fill"]["systems"],
    S.top,
)
d_new = 0.12 * (
    n_draw * 10 * 2 + n_fr * 10
)  # plan's ~12% of new top-10 slots (S2g, S5, S3g) before dedup
dcost = (plan["pairs"] + d_new) * 0.039
print(
    f"  D  fill: {plan['pairs']} existing unjudged pairs in the draw's top 10s + ~{d_new:.0f} from the new systems -> ~${dcost:.0f} at $0.039 per pair"
)
print(
    f"  total ~${a1 + a2 + a3 + b + dcost:.0f} standard (text stages A1/A3 halve with the Batch API); full-set plan was ~$730"
)
