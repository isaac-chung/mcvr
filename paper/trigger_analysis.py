"""Review item O3 (2026-11-05): uncertainty on the margin trigger's curve, the cross-fit point's
sampling CI, a fixed-budget policy, and money per query. No model calls.

    uv run --offline --with matplotlib python paper/trigger_analysis.py

Rows are pipeline/run_graded_systems.py trigger_rows (as paper/fig2_trigger_sweep.py): all 722 test queries,
expanded labels, q4 / q5 = nDCG@10 of S4@50 / S5, s4_margin = S4@50 top-two score margin,
c_base = S4@50 input tokens, c5 = the frame step's input tokens (S5 line).

1. Threshold rule "send iff margin <= t" for every t: share sent and gain kept
   (sum(d_t) / sum(d_5), d = gain over S4@50) with pointwise 95% cluster-bootstrap bands (10,000
   resamples of source videos; the same resamples for every t). Pooled and per set.
2. Cross-fit point (target keep >= 0.8 and >= 0.9; five folds over sources): its sampling CI by
   resampling sources (2,000 resamples), running the cross-fit (one random fold split per
   resample) inside each; duplicated sources count as distinct clusters.
3. Fixed-budget policy: send the x% of queries with the smallest S4@50 margin (ties: lower S4
   top score first, then the seeded uniform draw of trigger_rows), x = 0, 5, ..., 100; gain kept
   with 95% bootstrap CI; against the threshold rule at the same share (linear interpolation
   between the rule's points) and against random at x (expected kept = x).
   Per-batch quantile: in deployment it needs the margin distribution of the batch (or a
   threshold set from a past batch).
4. Money: gpt-6-sol at US$2 / US$10 per 1M input / output tokens, the only verifier price recorded
   in the repo (docs/full_run_results.md, 2026-10-26, "OpenAI list prices ... standard tier"; also
   docs/description_length_ablation_plan.md, "not re-checked"). No price is recorded for
   gpt-6-luna (splitter) or text-embedding-3-small (query embedding), so these are listed as
   tokens only and left out of the dollars. Output tokens include reasoning tokens.

Writes results/full/review_20261105/o3_trigger.json and o3_trigger_curve.csv.
"""

from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

PRICE_IN, PRICE_OUT = 2.0, 10.0  # US$ per 1M tokens, gpt-6-sol (docs/full_run_results.md)
N_BOOT_CF = 2000
GRID = [x / 100 for x in range(0, 101, 5)]


def cluster_index(rows):
    src = sorted({r["source"] for r in rows})
    pos = {s: i for i, s in enumerate(src)}
    return np.array([pos[r["source"]] for r in rows]), len(src)


def boot_picks(n_clusters, n_boot=C.N_BOOT, seed=0):
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(n_boot, n_clusters))
    # weight of each cluster in each resample
    w = np.zeros((n_boot, n_clusters))
    for j in range(n_clusters):
        w[:, j] = (picks == j).sum(axis=1)
    return w


def per_cluster(vals, cidx, n_c):
    out = np.zeros(n_c)
    np.add.at(out, cidx, vals)
    return out


def policy_stats(rows, fire, W, cidx, n_c):
    """share, kept (point) and 95% bootstrap CIs for a boolean fire vector."""
    fire = np.asarray(fire, dtype=float)
    d5 = np.array([r["q5"] - r["q4"] for r in rows])
    d = d5 * fire
    cnt = per_cluster(np.ones(len(rows)), cidx, n_c)
    sf = per_cluster(fire, cidx, n_c)
    sd = per_cluster(d, cidx, n_c)
    sd5 = per_cluster(d5, cidx, n_c)
    share_b = (W @ sf) / (W @ cnt)
    den = W @ sd5
    kept_b = np.where(den > 0, (W @ sd) / np.where(den > 0, den, 1), np.nan)
    kept_b = kept_b[np.isfinite(kept_b)]
    return {
        "share": float(fire.mean()),
        "share_ci": [float(x) for x in np.percentile(share_b, [2.5, 97.5])],
        "kept": float(d.sum() / d5.sum()) if d5.sum() > 0 else math.nan,
        "kept_ci": [float(x) for x in np.percentile(kept_b, [2.5, 97.5])],
        "tokens_in": float(np.mean([r["c_base"] + (r["c5"] if f else 0.0) for r, f in zip(rows, fire)])),
        "ndcg10": float(np.mean([r["q5"] if f else r["q4"] for r, f in zip(rows, fire)])),
    }


def budget_fire(rows, x):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["s4_margin"], rows[i]["s4_top"], rows[i]["u"]))
    m = int(round(x * len(rows)))
    fire = np.zeros(len(rows), dtype=bool)
    fire[order[:m]] = True
    return fire


def interp(curve, x):
    """Threshold rule's kept at share x, linear between its points (share sorted)."""
    pts = sorted((p["share"], p["kept"]) for p in curve)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return math.nan


def crossfit_boot(rows, frac, n_boot=N_BOOT_CF, seed=1):
    """Sampling CI of the cross-fit operating point: resample sources, cross-fit inside."""
    by_src: dict = {}
    for r in rows:
        by_src.setdefault(r["source"], []).append(r)
    srcs = sorted(by_src)
    rng = np.random.default_rng(seed)
    kept, share, cost = [], [], []
    for b in range(n_boot):
        pick = rng.integers(0, len(srcs), size=len(srcs))
        rs = [{**r, "source": f"{srcs[j]}#{i}", "query_id": f"{r['query_id']}#{i}"}
              for i, j in enumerate(pick) for r in by_src[srcs[j]]]
        if sum(r["q5"] - r["q4"] for r in rs) <= 0:
            continue
        cf = C.TS.crossfit(rs, "s4_margin", "le", frac, folds=5, reps=1)
        kept.append(cf["kept"][0])
        share.append(cf["share"][0])
        cost.append(cf["cost"][0])
    f = lambda v: [float(x) for x in np.nanpercentile(v, [2.5, 97.5])]  # noqa: E731
    return {"n_boot": len(kept), "kept_ci": f(kept), "share_ci": f(share), "tokens_ci": f(cost)}


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    q_exp = qrels["expanded"]
    rows = C.T1.trigger_rows(S, sets, ranks, lines, q_exp, sub)
    # output tokens per query for money
    for r in rows:
        r["o4"] = float(lines[r["set"]]["S4@50"][r["query_id"]]["cost"]["tokens_out"])
        r["o5"] = float(lines[r["set"]]["S5"][r["query_id"]]["cost"]["tokens_out"])
    T = S.tier["trigger"]
    t_m = float(T["margin_threshold"])
    groups = {
        "pooled": rows,
        "nested": [r for r in rows if r["set"] == "natural"],
        "near-miss": [r for r in rows if r["set"] == "twin"],
    }
    out: dict = {"n": len(rows), "price_usd_per_1M": {"gpt-6-sol": [PRICE_IN, PRICE_OUT]}, "groups": {}}
    csv_rows = []
    for g, rs in groups.items():
        cidx, n_c = cluster_index(rs)
        W = boot_picks(n_c)
        thr = sorted({r["s4_margin"] for r in rs})
        curve = []
        never = policy_stats(rs, [False] * len(rs), W, cidx, n_c)
        never["t"] = None
        curve.append(never)
        for t in thr:
            p = policy_stats(rs, [r["s4_margin"] <= t for r in rs], W, cidx, n_c)
            p["t"] = t
            curve.append(p)
        dev = next(p for p in curve if p["t"] == t_m)
        budget = []
        for x in GRID:
            p = policy_stats(rs, budget_fire(rs, x), W, cidx, n_c)
            p["x"] = x
            p["threshold_rule_kept_at_share"] = interp(curve, p["share"])
            budget.append(p)
        # top-score fixed budget as a second ordering, for reference
        cell = {
            "n": len(rs),
            "threshold_curve": curve,
            "dev_threshold": {"t": t_m, **dev},
            "budget_at_dev_share": policy_stats(rs, budget_fire(rs, dev["share"]), W, cidx, n_c),
            "fixed_budget": budget,
        }
        if g == "pooled":
            for frac in (0.8, 0.9):
                pt = C.TS.crossfit(rs, "s4_margin", "le", frac)
                cell[f"crossfit_keep{frac}"] = {
                    "point(mean over 20 fold splits)": {"kept": pt["kept"][0], "share": pt["share"][0],
                                                        "tokens": pt["cost"][0]},
                    "fold_split_spread": {"kept": pt["kept"][1:], "share": pt["share"][1:]},
                    "sampling_ci(bootstrap over sources)": crossfit_boot(rs, frac),
                }
        out["groups"][g] = cell
        for p in curve:
            csv_rows.append([g, "threshold", "" if p["t"] is None else p["t"], p["share"], *p["share_ci"],
                             p["kept"], *p["kept_ci"], p["tokens_in"], p["ndcg10"]])
        for p in budget:
            csv_rows.append([g, "budget", p["x"], p["share"], *p["share_ci"], p["kept"], *p["kept_ci"],
                             p["tokens_in"], p["ndcg10"]])

    # money per query (pooled, all 722 queries); tokens as logged (table 1 convention)
    allq = [(r["set"], r["query_id"]) for r in rows]

    def mean_of(system, field):
        return float(np.mean([float(lines[s][system][q]["cost"][field]) for s, q in allq]))

    fire = [r["s4_margin"] <= t_m for r in rows]
    money = {}
    tok = {
        "S0": (0.0, 0.0),
        "S2@50": (mean_of("S2g@50", "tokens_in"), mean_of("S2g@50", "tokens_out")),
        "S4@50": (mean_of("S4@50", "tokens_in"), mean_of("S4@50", "tokens_out")),
        "S5": (mean_of("S4@50", "tokens_in") + mean_of("S5", "tokens_in"),
               mean_of("S4@50", "tokens_out") + mean_of("S5", "tokens_out")),
        "S3": (mean_of("S4@50", "tokens_in") + mean_of("S3g", "tokens_in"),
               mean_of("S4@50", "tokens_out") + mean_of("S3g", "tokens_out")),
        "S6 (margin <= 2)": (float(np.mean([r["c_base"] + (r["c5"] if f else 0) for r, f in zip(rows, fire)])),
                             float(np.mean([r["o4"] + (r["o5"] if f else 0) for r, f in zip(rows, fire)]))),
    }
    split_in, split_out = mean_of("S1", "tokens_in"), mean_of("S1", "tokens_out")
    for s, (ti, to) in tok.items():
        money[s] = {
            "tokens_in": ti, "tokens_out": to,
            "usd_in": ti * PRICE_IN / 1e6, "usd_out": to * PRICE_OUT / 1e6,
            "usd_total": ti * PRICE_IN / 1e6 + to * PRICE_OUT / 1e6,
            "usd_per_1000_queries": 1000 * (ti * PRICE_IN / 1e6 + to * PRICE_OUT / 1e6),
            "unpriced": {"query embedding (text-embedding-3-small)": "1 call",
                         **({"splitter (gpt-6-luna) tokens_in / tokens_out": [split_in, split_out]}
                            if s in ("S2@50", "S3") else {})},
        }
    out["money_pooled"] = money
    path = C.dump("o3_trigger.json", out)
    with (C.OUT / "o3_trigger_curve.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["group", "kind", "threshold_or_budget", "share", "share_lo", "share_hi", "kept", "kept_lo",
                    "kept_hi", "tokens_in_per_query", "ndcg10"])
        w.writerows(csv_rows)

    for g, c in out["groups"].items():
        d = c["dev_threshold"]
        b = c["budget_at_dev_share"]
        print(f"\n== {g} (n={c['n']}) dev threshold: share {d['share']:.3f} {d['share_ci']} kept {d['kept']:.3f} "
              f"[{d['kept_ci'][0]:.3f}, {d['kept_ci'][1]:.3f}] tokens {d['tokens_in']:.0f}")
        print(f"   budget at same share: kept {b['kept']:.3f} [{b['kept_ci'][0]:.3f}, {b['kept_ci'][1]:.3f}]")
        print("   threshold curve:", "; ".join(
            f"t={p['t']} s={p['share']:.3f} k={p['kept']:.3f}[{p['kept_ci'][0]:.2f},{p['kept_ci'][1]:.2f}]"
            for p in c["threshold_curve"]))
        print("   budget:", "; ".join(
            f"x={p['x']:.2f} k={p['kept']:.3f}[{p['kept_ci'][0]:.2f},{p['kept_ci'][1]:.2f}] thr={p['threshold_rule_kept_at_share']:.3f} tok={p['tokens_in'] / 1000:.1f}k"
            for p in c["fixed_budget"]))
        for frac in (0.8, 0.9):
            if f"crossfit_keep{frac}" in c:
                print(f"   cross-fit keep>={frac}:", c[f"crossfit_keep{frac}"])
    print("\nmoney (pooled, gpt-6-sol at $2/$10 per 1M):")
    for s, m in money.items():
        print(f"   {s:18s} in {m['tokens_in']:9.0f} out {m['tokens_out']:7.0f}  ${m['usd_total']:.4f}/query  "
              f"(${m['usd_per_1000_queries']:.2f} per 1k)  in-only ${m['usd_in']:.4f}")
    print(f"   splitter (unpriced): in {split_in:.0f} out {split_out:.0f}")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
