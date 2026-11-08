"""Review item O2 (2026-11-05): where the frame gain S5 - S4@50 comes from, and how close it is
to what any reordering of S4@50's top 10 could reach. No model calls.

    uv run --offline --with matplotlib python paper/frame_gain_decomposition.py

1. Split S5 - S4@50 (nDCG@10) by whether the source is in S4@50's top 10 (the pool S5 reorders):
   counts, mean gain with 95% cluster-bootstrap CI, and each part's contribution to the overall
   mean (share of queries x mean gain).
2. Oracle ceilings within the same pool (the candidates S5 sees; ranks below 10 untouched):
     src@1   the source moved to rank 1, the other nine in S4@50 order (no change if absent)
     ideal   the top 10 sorted by label (relevant first, S4@50 order within), i.e. the best nDCG@10
             any reranker of this pool can reach under the given labels
   Reported as nDCG@10 and as gains over S4@50, with S5's share of each ceiling's headroom
   (sum of S5 gains / sum of ceiling gains, cluster-bootstrap CI over sources).
Per set (nested, near-miss) and pooled; expanded labels (primary) and strict labels (where src@1
and ideal coincide, since only the source is relevant).

Writes results/full/review_20261105/o2_frame_decomp.json.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

TOP = 10


def src_first(rank: list, source: str) -> list:
    head = rank[:TOP]
    ids = [d for d, _ in head]
    if source not in ids:
        return rank
    i = ids.index(source)
    return [head[i]] + head[:i] + head[i + 1:] + rank[TOP:]


def ideal(rank: list, grades: dict) -> list:
    head = rank[:TOP]
    order = sorted(range(len(head)), key=lambda i: (-(grades.get(head[i][0], 0) > 0), i))
    return [head[i] for i in order] + rank[TOP:]


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    out: dict = {"pool": "S4@50 top 10", "labels": {}}
    for kind in ("expanded", "strict"):
        per_q = []  # dicts with q4, q5, src1, ideal, in_pool, set, source
        for name, rs in sets.items():
            g = qrels[kind][name]
            for r in rs:
                q = r["query_id"]
                r4 = ranks[name]["S4@50"][q]
                m = {
                    "q4": C.PV.query_metrics(r4, g[q])["ndcg@10"],
                    "q5": C.PV.query_metrics(ranks[name]["S5"][q], g[q])["ndcg@10"],
                    "src1": C.PV.query_metrics(src_first(r4, r["source"]), g[q])["ndcg@10"],
                    "ideal": C.PV.query_metrics(ideal(r4, g[q]), g[q])["ndcg@10"],
                }
                m.update(set=name, source=r["source"], query_id=q,
                         in_pool=r["source"] in {d for d, _ in r4[:TOP]},
                         any_rel_in_pool=any(g[q].get(d, 0) > 0 for d, _ in r4[:TOP]))
                per_q.append(m)
        groups = {
            "nested": [m for m in per_q if m["set"] == "natural"],
            "near-miss": [m for m in per_q if m["set"] == "twin"],
            "pooled": per_q,
        }
        L = {}
        for gname, ms in groups.items():
            cell = {"n": len(ms)}
            cl = [m["source"] for m in ms]
            for key in ("q4", "q5", "src1", "ideal"):
                cell[f"ndcg10[{key}]"] = C.mean_ci([m[key] for m in ms], cl)
            for key in ("q5", "src1", "ideal"):
                cell[f"gain[{key} - q4]"] = C.mean_ci([m[key] - m["q4"] for m in ms], cl)
            cell["S5 share of headroom: src1"] = {
                "ratio": sum(m["q5"] - m["q4"] for m in ms) / max(1e-12, sum(m["src1"] - m["q4"] for m in ms)),
                "ci": C.TS.ratio_ci([m["q5"] - m["q4"] for m in ms], [m["src1"] - m["q4"] for m in ms], cl, C.N_BOOT),
            }
            cell["S5 share of headroom: ideal"] = {
                "ratio": sum(m["q5"] - m["q4"] for m in ms) / max(1e-12, sum(m["ideal"] - m["q4"] for m in ms)),
                "ci": C.TS.ratio_ci([m["q5"] - m["q4"] for m in ms], [m["ideal"] - m["q4"] for m in ms], cl, C.N_BOOT),
            }
            cell["S5 gap to ideal (ideal - S5)"] = C.mean_ci([m["ideal"] - m["q5"] for m in ms], cl)
            split = {}
            for part, sel in (("source in pool", True), ("source not in pool", False)):
                sm = [m for m in ms if m["in_pool"] == sel]
                if not sm:
                    continue
                scl = [m["source"] for m in sm]
                d = [m["q5"] - m["q4"] for m in sm]
                split[part] = {
                    "n": len(sm),
                    "share_of_queries": len(sm) / len(ms),
                    "S4@50": sum(m["q4"] for m in sm) / len(sm),
                    "S5": sum(m["q5"] for m in sm) / len(sm),
                    "ideal": sum(m["ideal"] - 0 for m in sm) / len(sm),
                    "S5 - S4@50": C.mean_ci(d, scl),
                    "contribution_to_overall_gain": sum(d) / len(ms),
                    "ideal - S4@50": C.mean_ci([m["ideal"] - m["q4"] for m in sm], scl),
                    "n_with_any_relevant_in_pool": sum(m["any_rel_in_pool"] for m in sm),
                }
            # share of the overall gain from source-in-pool queries (cluster bootstrap of a ratio)
            d_all = [m["q5"] - m["q4"] for m in ms]
            d_in = [x if m["in_pool"] else 0.0 for x, m in zip(d_all, ms)]
            split["share of overall gain from source-in-pool queries"] = {
                "ratio": sum(d_in) / sum(d_all) if sum(d_all) else None,
                "ci": C.TS.ratio_ci(d_in, d_all, cl, C.N_BOOT),
            }
            cell["split"] = split
            L[gname] = cell
        out["labels"][kind] = L
    path = C.dump("o2_frame_decomp.json", out)
    for kind in ("expanded", "strict"):
        print(f"\n== {kind}")
        for gname, c in out["labels"][kind].items():
            print(f" {gname} (n={c['n']}): S4 {c['ndcg10[q4]']['mean']:.3f} S5 {c['ndcg10[q5]']['mean']:.3f} "
                  f"src@1 {c['ndcg10[src1]']['mean']:.3f} ideal {c['ndcg10[ideal]']['mean']:.3f} | "
                  f"S5-S4 {C.fmt(c['gain[q5 - q4]'])} src1-S4 {C.fmt(c['gain[src1 - q4]'])} ideal-S4 {C.fmt(c['gain[ideal - q4]'])} "
                  f"ideal-S5 {C.fmt(c['S5 gap to ideal (ideal - S5)'])}")
            h = c["S5 share of headroom: ideal"]
            h1 = c["S5 share of headroom: src1"]
            print(f"   S5 share of headroom: src@1 {h1['ratio']:.3f} [{h1['ci'][0]:.3f}, {h1['ci'][1]:.3f}]  "
                  f"ideal {h['ratio']:.3f} [{h['ci'][0]:.3f}, {h['ci'][1]:.3f}]")
            for part, s in c["split"].items():
                if "n" in s:
                    print(f"   {part}: n={s['n']} ({100 * s['share_of_queries']:.1f}%) S4 {s['S4@50']:.3f} S5 {s['S5']:.3f} "
                          f"ideal {s['ideal']:.3f} gain {C.fmt(s['S5 - S4@50'])} contrib {s['contribution_to_overall_gain']:+.4f} "
                          f"ideal-S4 {C.fmt(s['ideal - S4@50'])} anyrel {s['n_with_any_relevant_in_pool']}")
                else:
                    print(f"   {part}: {s['ratio']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]")
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
