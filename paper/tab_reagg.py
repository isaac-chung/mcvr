"""Review item O1 (2026-11-05): re-aggregate the cached per-constraint grades of the graded
per-constraint checks, S2@50 on descriptions (data key S2g@50) and S3 on frames (data key S3g),
and compare with the whole-query reranks S4@50 / S5 on the same candidates. No model calls.

    uv run --offline --with matplotlib python paper/tab_reagg.py

Grades come from each saved line's trace (trace.grades[doc] = one 0-10 integer per split
constraint, trace.first_stage = the candidates in pool order). Every aggregation reranks the same
candidates with the same grades; ties keep the pool order (S0 rank for S2; S4@50 rank for S3, as
the saved S3g run), candidates outside the pool keep their first-stage order (merge_reranked).

Aggregations (score of a candidate from its grades g_1..g_k):
  mean      mean grade (the paper's pre-set aggregation; reproduces the saved runs)
  min       minimum grade (the paper's secondary aggregation)
  u0        unmentioned -> 0: each grade of exactly 5 set to 0, then the mean
  nn        mean over grades != 5 only; a candidate whose grades are all 5 scores 5 (neutral, i.e.
            between a contradicted and a stated candidate), ties by pool order
  n8        number of grades >= 8 ("stated")
  net       #grades >= 8 minus #grades <= 2 ("stated" minus "contradicted")
Within a query k is fixed, so counts rank exactly as shares would.

Writes results/full/review_20261105/o1_reaggregate.json and prints a summary.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402
from src.systems.base import merge_reranked  # noqa: E402
from src.systems.verify import rank_by_score  # noqa: E402

MARGIN = 0.02


def agg(g: list[float], how: str) -> float:
    if not g:
        return -1.0  # GRADE_FAILED (none occur in the cached test runs)
    if how == "mean":
        return sum(g) / len(g)
    if how == "min":
        return min(g)
    if how == "u0":
        return sum(0.0 if x == 5 else x for x in g) / len(g)
    if how == "nn":
        v = [x for x in g if x != 5]
        return sum(v) / len(v) if v else 5.0
    if how == "n8":
        return float(sum(x >= 8 for x in g))
    if how == "net":
        return float(sum(x >= 8 for x in g) - sum(x <= 2 for x in g))
    raise ValueError(how)


AGGS = ["mean", "min", "u0", "nn", "n8", "net"]
NAMES = {
    "mean": "mean (paper)",
    "min": "min (paper, secondary)",
    "u0": "unmentioned 5 -> 0, mean",
    "nn": "mean over grades != 5",
    "n8": "count of grades >= 8",
    "net": "#>=8 minus #<=2",
}


def rerank(line: dict, first: list, how: str) -> list:
    head = line["trace"]["first_stage"]
    scores = [agg(line["trace"]["grades"][d], how) for d in head]
    return merge_reranked(first, rank_by_score([(d, 0.0) for d in head], scores))


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    rows_all = [r for rs in sets.values() for r in rs]
    out: dict = {"aggregations": NAMES, "tost_margin": MARGIN, "labels": {}}
    # share of candidates whose grades are all 5 (no evidence either way), per system
    diag = {}
    for sysn in ("S2g@50", "S3g"):
        n_c = n_all5 = n_g = n_5 = 0
        for name in sets:
            for ln in lines[name][sysn].values():
                for d in ln["trace"]["first_stage"]:
                    g = ln["trace"]["grades"][d]
                    n_c += 1
                    n_all5 += all(x == 5 for x in g)
                    n_g += len(g)
                    n_5 += sum(x == 5 for x in g)
        diag[sysn] = {"candidates": n_c, "share_all5": n_all5 / n_c, "grades": n_g, "share_grade5": n_5 / n_g}
    out["grade_diagnostics"] = diag
    # how often the source ends up tied at the top under each aggregation (descriptions)
    new_ranks = {name: {} for name in sets}
    for name in sets:
        s0 = ranks[name]["S0"]
        pool = ranks[name]["S4@50"]
        for how in AGGS:
            new_ranks[name][f"S2:{how}"] = {
                q: rerank(ln, s0[q], how) for q, ln in lines[name]["S2g@50"].items()
            }
            new_ranks[name][f"S3:{how}"] = {
                q: rerank(ln, pool[q], how) for q, ln in lines[name]["S3g"].items()
            }
        # sanity: mean / min reproduce the saved / tier1-rescored runs
        for q in ranks[name]["S2g@50"]:
            assert [d for d, _ in new_ranks[name]["S2:mean"][q][:10]] == [d for d, _ in ranks[name]["S2g@50"][q][:10]]
            assert [d for d, _ in new_ranks[name]["S2:min"][q][:10]] == [d for d, _ in ranks[name]["S2g@50-min"][q][:10]]
        for q in ranks[name]["S3g"]:
            assert [d for d, _ in new_ranks[name]["S3:mean"][q][:10]] == [d for d, _ in ranks[name]["S3g"][q][:10]]
            assert [d for d, _ in new_ranks[name]["S3:min"][q][:10]] == [d for d, _ in ranks[name]["S3g-min"][q][:10]]

    for kind in ("expanded", "strict"):
        L: dict = {"desc": {}, "frames": {}}
        sc = {
            name: {
                **{s: C.score(new_ranks[name][s], qrels[kind][name]) for s in new_ranks[name]},
                **{s: C.score(ranks[name][s], qrels[kind][name]) for s in ("S0", "S4@50", "S5")},
            }
            for name in sets
        }
        pooled = {s: {q: v for name in sets for q, v in sc[name][s].items()} for s in sc["natural"]}
        groups = {"nested": ("natural", sets["natural"]), "near-miss": ("twin", sets["twin"]),
                  "pooled": (None, rows_all)}
        for how in AGGS:
            for side, mine, ref in (("desc", f"S2:{how}", "S4@50"), ("frames", f"S3:{how}", "S5")):
                cell = {}
                for g, (name, rs) in groups.items():
                    s = pooled if name is None else sc[name]
                    m_mine = C.mean_ci([s[mine][r["query_id"]]["ndcg@10"] for r in rs], [r["source"] for r in rs])
                    d_ref = C.paired(s[ref], s[mine], rs)  # ref - mine
                    d_mean = C.paired(s[mine], s[f"{mine.split(':')[0]}:mean"], rs)  # mine - mean
                    c = {"ndcg10": m_mine, f"{ref} - this": d_ref, "this - mean": d_mean}
                    if side == "desc":
                        c["this - S0"] = C.paired(s[mine], s["S0"], rs)
                        c["k>=2: S4@50 - this"] = C.paired(s[ref], s[mine], [r for r in rs if int(r["k"]) >= 2])
                    else:
                        c["this - S4@50"] = C.paired(s[mine], s["S4@50"], rs)
                        c["tost"] = {
                            "ci90": [d_ref["ci90_low"], d_ref["ci90_high"]],
                            "equivalent": -MARGIN < d_ref["ci90_low"] and d_ref["ci90_high"] < MARGIN,
                        }
                    cell[g] = c
                L[side][how] = cell
        out["labels"][kind] = L

    path = C.dump("o1_reaggregate.json", out)
    print("grade diagnostics", diag)
    for kind in ("expanded", "strict"):
        L = out["labels"][kind]
        print(f"\n== {kind} labels: descriptions (S2 aggregations; S4@50 - this)")
        for how in AGGS:
            c = L["desc"][how]
            print(f"  {NAMES[how]:28s} " + "  ".join(
                f"{g}: nDCG {c[g]['ndcg10']['mean']:.3f}  gap {C.fmt(c[g]['S4@50 - this'])}" for g in ("nested", "near-miss")))
        print(f"== {kind} labels: frames (S3 aggregations; S5 - this; 90% CI; TOST +-{MARGIN})")
        for how in AGGS:
            c = L["frames"][how]
            print(f"  {NAMES[how]:28s} " + "  ".join(
                f"{g}: nDCG {c[g]['ndcg10']['mean']:.3f} d {C.fmt(c[g]['S5 - this'])} "
                f"90%[{c[g]['tost']['ci90'][0]:+.3f},{c[g]['tost']['ci90'][1]:+.3f}] eq={c[g]['tost']['equivalent']}"
                for g in ("nested", "near-miss", "pooled")))
    print(f"\nwrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
