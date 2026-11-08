"""Review item W1 (2026-11-06): is the S4 vs S2 contrast confounded? Offline, no model calls.

    uv run --offline --with matplotlib python paper/tab_w1_pools.py

Paper names: S2 = run key S2g@50 (graded per-constraint on descriptions, S0's top 50), S4@50 =
whole-query graded rerank of S0's top 50, S3 = S3g (graded per-constraint on frames of S4@50's top
10), S5 = whole-query on the frames of S4@50's top 10. All 722 test queries (nested = natural 350,
near-miss = twin 372); nDCG@10, expanded labels primary, strict secondary; paired cluster bootstrap
over source videos (review_common.boot: 10,000 resamples, seed 0).

1. Same-pool 2x2 on S4@50's top 10 (the pool S5 and S3 rerank). S2'g reranks that pool by the stored
   S2g@50 grades (mean over constraints; ties by S0 rank, as S2's own tie-break; variant: ties by
   S4@50 rank). Cells: description whole = S4@50, description per-constraint = S2'g, frame whole = S5,
   frame per-constraint = S3. Gaps: S4@50 - S2'g (descriptions), S5 - S3 (frames), interaction =
   (S4@50 - S2'g) - (S5 - S3) per query. Oracle = best reordering of the pool under the labels.
   Coverage: every pool candidate must have a stored S2g@50 grade.
2. Split quality. No finished split_match_v1 labels exist for the test set (the paid judge stopped at
   the daily limit, results/full/rebuttal/split_check/stopped.json), so the class is the free one the
   paper uses (splitter_errors.py): number of splitter constraints (runs/<set>/splits.json) vs k:
   '=k' exact, '>k' over-split, '<k' under-split. A stricter 'exact+match' class additionally needs
   a one-to-one token-F1 >= 0.5 match between the k gold facts and the k constraints (greedy,
   best pair first). S4@50 - S2@50 and S4@50 - S2'g are reported per class.
3. Task E (human labels, 100 test sources, 715 facts: stated / implied / absent in the description,
   data/mc/full/human100/taskE.csv; fact numbers per query from key.json taskBD, which lists each
   query's fact numbers). (a) Per query: share of facts implied (any implied vs none; also
   'no fact absent'), with the S4@50 - S2@50 gap per group. (b) Per fact: the grade S2 gives the
   source for the constraint that matches the fact (token F1 >= 0.5, best constraint, queries with
   the splitter's count = k only and all queries), by Task E label: mean grade, share exactly 5, share
   >= 8; the same for S3 (frames) where the source is in S3's pool. (c) Paper's model-judge
   figure vs human shares.

Writes results/full/review_20261106/w1_offline.json and w1_offline.md.
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402
from src.systems.verify import aggregate_grades  # noqa: E402

OUTDIR = C.ROOT / "results/full/review_20261106"
HUMAN = C.ROOT / "data/mc/full/human100"
DPC = "desc per-constraint (S2'g)"
GD = "gap_desc S4@50 - S2'g"
SP = "S4@50 - S2'g (same pool)"
SETLAB = {"natural": "nested", "twin": "near-miss"}


# ---- helpers -----------------------------------------------------------------------------------


def tok(t: str) -> list[str]:
    stop = {"a", "an", "the", "is", "are", "of", "in", "on", "with", "and", "to", "at", "it"}
    return [w for w in "".join(c if c.isalnum() else " " for c in t.lower()).split() if w not in stop]


def f1(a: str, b: str) -> float:
    x, y = tok(a), tok(b)
    c = sum(min(x.count(w), y.count(w)) for w in set(x))
    if not c:
        return 0.0
    p, r = c / len(y), c / len(x)
    return 2 * p * r / (p + r)


def match_one_to_one(facts: list[str], cons: list[str], thr=0.5) -> dict[int, int]:
    """Greedy one-to-one fact -> constraint matching by token F1 >= thr (best pair first)."""
    pairs = sorted(((f1(f, c), i, j) for i, f in enumerate(facts) for j, c in enumerate(cons)), reverse=True)
    out, used = {}, set()
    for s, i, j in pairs:
        if s < thr:
            break
        if i not in out and j not in used:
            out[i] = j
            used.add(j)
    return out


def ndcg_of(order, qrel):
    return C.PV.query_metrics([(d, 0.0) for d in order], qrel)["ndcg@10"]


def ci(d):
    return C.fmt(d)


def groups_of(sets, extra=None):
    rows_all = [r for rs in sets.values() for r in rs]
    return {"nested": sets["natural"], "near-miss": sets["twin"], "pooled": rows_all}


def pair_cell(sa, sb, rows):
    d = C.paired(sa, sb, rows)
    return d


def diff_of_diffs(a, b, c, d, rows):
    """Mean of (a-b) - (c-d) per query with a cluster-bootstrap CI."""
    v = [(a[r["query_id"]]["ndcg@10"] - b[r["query_id"]]["ndcg@10"])
         - (c[r["query_id"]]["ndcg@10"] - d[r["query_id"]]["ndcg@10"]) for r in rows]
    return C.boot(v, [r["source"] for r in rows])


# ---- main --------------------------------------------------------------------------------------


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    groups = groups_of(sets)
    allrows = groups["pooled"]
    out: dict = {"n_queries": {k: len(v) for k, v in groups.items()}}

    # ---------------- 1. same-pool 2x2 ----------------
    cover = {"queries": 0, "pool_candidates": 0, "with_S2_grade": 0, "queries_fully_covered": 0,
             "S5_pool_equals_S4_top10": 0, "S3_pool_equals_S4_top10": 0}
    rk = {n: {"S2p": {}, "S2p_s4tie": {}, "oracle": {}} for n in sets}
    for n in sets:
        for q, line in lines[n]["S2g@50"].items():
            pool = [d for d, _ in ranks[n]["S4@50"][q][:10]]
            gr = line["trace"]["grades"]
            pos = {d: i for i, d in enumerate(line["trace"]["first_stage"])}  # = S0 order
            have = [d for d in pool if d in gr]
            cover["queries"] += 1
            cover["pool_candidates"] += len(pool)
            cover["with_S2_grade"] += len(have)
            cover["queries_fully_covered"] += len(have) == len(pool)
            cover["S5_pool_equals_S4_top10"] += {d for d, _ in ranks[n]["S5"][q][:10]} == set(pool)
            cover["S3_pool_equals_S4_top10"] += {d for d, _ in ranks[n]["S3g"][q][:10]} == set(pool)
            sc = {d: aggregate_grades(gr[d], "mean") if d in gr else -1.0 for d in pool}
            rk[n]["S2p"][q] = [(d, sc[d]) for d in sorted(pool, key=lambda d: (-sc[d], pos.get(d, 99)))]
            s4pos = {d: i for i, d in enumerate(pool)}
            rk[n]["S2p_s4tie"][q] = [(d, sc[d]) for d in sorted(pool, key=lambda d: (-sc[d], s4pos[d]))]
    out["coverage"] = cover
    out["S2g_pool_equals_S0_top50"] = sum(
        [d for d, _ in ranks[n]["S0"][q][:50]] == lines[n]["S2g@50"][q]["trace"]["first_stage"]
        for n in sets for q in lines[n]["S2g@50"]) / cover["queries"]

    out["same_pool"] = {}
    SC = {}  # kind -> name -> system -> qid -> metrics
    for kind in ("expanded", "strict"):
        SC[kind] = {}
        for n in sets:
            qr = qrels[kind][n]
            SC[kind][n] = {
                "S0": C.score(ranks[n]["S0"], qr),
                "S2": C.score(ranks[n]["S2g@50"], qr),
                "S4": C.score(ranks[n]["S4@50"], qr),
                "S5": C.score(ranks[n]["S5"], qr),
                "S3": C.score(ranks[n]["S3g"], qr),
                "S2p": C.score(rk[n]["S2p"], qr),
                "S2p_s4tie": C.score(rk[n]["S2p_s4tie"], qr),
            }
            orc = {}
            for q in rk[n]["S2p"]:
                pool = [d for d, _ in ranks[n]["S4@50"][q][:10]]
                orc[q] = {"ndcg@10": ndcg_of(sorted(pool, key=lambda d: -qr[q].get(d, 0)), qr[q])}
            SC[kind][n]["oracle"] = orc
        pooled = {s: {q: v for n in sets for q, v in SC[kind][n][s].items()} for s in SC[kind]["natural"]}
        SC[kind]["pooled"] = pooled
        res = {}
        for g, rows in groups.items():
            s = SC[kind]["pooled"] if g == "pooled" else SC[kind]["natural" if g == "nested" else "twin"]
            cells = {name: C.mean_ci([s[k][r["query_id"]]["ndcg@10"] for r in rows], [r["source"] for r in rows])
                     for name, k in (("desc whole (S4@50)", "S4"), ("desc per-constraint (S2'g)", "S2p"),
                                     ("frame whole (S5)", "S5"), ("frame per-constraint (S3)", "S3"),
                                     ("S2@50 on S0 top 50 (paper S2)", "S2"), ("oracle on pool", "oracle"))}
            res[g] = {
                "cells": cells,
                "gap_desc S4@50 - S2'g": C.paired(s["S4"], s["S2p"], rows),
                "gap_frames S5 - S3": C.paired(s["S5"], s["S3"], rows),
                "interaction (desc gap - frame gap)": diff_of_diffs(s["S4"], s["S2p"], s["S5"], s["S3"], rows),
                "whole framing gap, frames - desc: S5 - S4@50": C.paired(s["S5"], s["S4"], rows),
                "per-constraint framing, frames - desc: S3 - S2'g": C.paired(s["S3"], s["S2p"], rows),
                "S4@50 - S2@50 (full 50 pool, paper)": C.paired(s["S4"], s["S2"], rows),
                "S2@50 - S2'g (pool restriction effect)": C.paired(s["S2"], s["S2p"], rows),
                "gap_desc with S4-order tie-break": C.paired(s["S4"], s["S2p_s4tie"], rows),
            }
        out["same_pool"][kind] = res

    # ---------------- 1b. mirror pools (S4 grades vs S2 grades on three pools) ----------------
    # (a) pool = S2's own top 10 (S2g@50): S2 grades = S2's order as is; S4' = same pool by stored S4@50
    # scores (ties: S0 order, as S4). (b) pool = S0's top 10: S4 grades (= S4@10) vs S2 grades (S2g@10).
    # (c) pool = S4@50's top 10: S4@50 vs S2'g (the section 1 gap).
    def by_scores(pool, sc, pos):
        return [(d, sc[d]) for d in sorted(pool, key=lambda d: (-sc[d], pos.get(d, 99)))]

    mrk = {n: {p: {"S4g": {}, "S2g": {}} for p in "abc"} for n in sets}
    mpool = {n: {p: {} for p in "abc"} for n in sets}
    chk = {"S2_order_reproduced": 0, "S4_order_reproduced": 0, "queries": 0}
    for n in sets:
        for q, l2 in lines[n]["S2g@50"].items():
            l4 = lines[n]["S4@50"][q]
            fs = l2["trace"]["first_stage"]
            pos = {d: i for i, d in enumerate(fs)}
            g2 = {d: aggregate_grades(l2["trace"]["grades"][d], "mean") for d in fs if d in l2["trace"]["grades"]}
            s4 = {d: float(v) for d, v in l4["trace"]["scores"].items()}
            assert list(l4["trace"]["first_stage"]) == list(fs)
            full2 = by_scores(fs, {d: g2.get(d, -1.0) for d in fs}, pos)
            full4 = by_scores(fs, s4, pos)
            chk["queries"] += 1
            chk["S2_order_reproduced"] += [d for d, _ in full2[:10]] == [d for d, _ in ranks[n]["S2g@50"][q][:10]]
            chk["S4_order_reproduced"] += [d for d, _ in full4[:10]] == [d for d, _ in ranks[n]["S4@50"][q][:10]]
            pools = {"a": [d for d, _ in ranks[n]["S2g@50"][q][:10]], "b": list(fs[:10]),
                     "c": [d for d, _ in ranks[n]["S4@50"][q][:10]]}
            for p, pool in pools.items():
                mpool[n][p][q] = pool
                mrk[n][p]["S4g"][q] = by_scores(pool, {d: s4[d] for d in pool}, pos)
                mrk[n][p]["S2g"][q] = by_scores(pool, {d: g2.get(d, -1.0) for d in pool}, pos)
    out["mirror_pools"] = {"check": chk, "definition": (
        "pools: a = S2's own top 10 (S2g@50), b = S0's top 10, c = S4@50's top 10; each pool ranked by stored "
        "S4@50 grades (ties: S0 order) vs stored S2g@50 mean grades (ties: S0 order); gap = S4 grades - S2 grades "
        "on the same ten candidates; oracle = best reordering of the pool under the labels. nDCG@10, paired "
        "cluster bootstrap over sources (10,000, seed 0). For pool a the S2 side is S2's own top 10.")}
    PN = {"a": "S2 top 10 (S2's own pool)", "b": "S0 top 10", "c": "S4@50 top 10"}
    for kind in ("expanded", "strict"):
        res = {}
        sc = {}
        for n in sets:
            qr = qrels[kind][n]
            sc[n] = {p: {"S4g": C.score(mrk[n][p]["S4g"], qr), "S2g": C.score(mrk[n][p]["S2g"], qr),
                         "oracle": {q: {"ndcg@10": ndcg_of(sorted(pl, key=lambda d: -qr[q].get(d, 0)), qr[q])}
                                    for q, pl in mpool[n][p].items()}} for p in "abc"}
        for p in "abc":
            res[PN[p]] = {}
            for g, rows in groups.items():
                get = (lambda sy: {q: v for n in sets for q, v in sc[n][p][sy].items()}) if g == "pooled" else \
                      (lambda sy, nn=("natural" if g == "nested" else "twin"): sc[nn][p][sy])
                a_, b_, o_ = get("S4g"), get("S2g"), get("oracle")
                res[PN[p]][g] = {
                    "gap S4 grades - S2 grades": C.paired(a_, b_, rows),
                    "S4 grades": C.mean_ci([a_[r["query_id"]]["ndcg@10"] for r in rows], [r["source"] for r in rows]),
                    "S2 grades": C.mean_ci([b_[r["query_id"]]["ndcg@10"] for r in rows], [r["source"] for r in rows]),
                    "oracle": C.mean_ci([o_[r["query_id"]]["ndcg@10"] for r in rows], [r["source"] for r in rows]),
                }
        out["mirror_pools"][kind] = res

    # ---------------- 2. split quality ----------------
    splits = {n: json.loads((C.ROOT / f"results/full/runs/{n}/splits.json").read_text()) for n in sets}
    cls = {}
    for n, rs in sets.items():
        for r in rs:
            cons = splits[n][r["query_id"]]
            k = int(r["k"])
            c = "=k" if len(cons) == k else (">k" if len(cons) > k else "<k")
            facts = [f["text"] if isinstance(f, dict) else f for f in r["facts"]]
            m = match_one_to_one(facts, cons)
            cls[r["query_id"]] = {"class": c, "exact_match": c == "=k" and len(m) == k}
    out["split_class_counts"] = {
        SETLAB[n]: {c: sum(cls[r["query_id"]]["class"] == c for r in rs) for c in ("=k", ">k", "<k")}
        | {"exact+match": sum(cls[r["query_id"]]["exact_match"] for r in rs), "n": len(rs)}
        for n, rs in sets.items()
    }
    out["split_class_definition"] = (
        "'=k'/'>k'/'<k': count of splitter constraints (runs/<set>/splits.json) vs k gold facts; "
        "'exact+match': =k and a one-to-one token-F1>=0.5 match of all k facts to the k constraints. "
        "No finished split_match_v1 labels exist for the test set (judge stopped at its daily limit).")
    out["by_split"] = {}
    for kind in ("expanded", "strict"):
        res = {}
        for g, rows in groups.items():
            s = SC[kind]["pooled"] if g == "pooled" else SC[kind]["natural" if g == "nested" else "twin"]
            cell = {}
            for cname, pred in (("all", lambda c: True), ("=k", lambda c: c["class"] == "=k"),
                                ("exact+match", lambda c: c["exact_match"]),
                                (">k", lambda c: c["class"] == ">k"), ("<k", lambda c: c["class"] == "<k")):
                rs = [r for r in rows if pred(cls[r["query_id"]])]
                if len(rs) < 5:
                    cell[cname] = {"n": len(rs)}
                    continue
                cell[cname] = {
                    "n": len(rs),
                    "S4@50": float(np.mean([s["S4"][r["query_id"]]["ndcg@10"] for r in rs])),
                    "S2@50": float(np.mean([s["S2"][r["query_id"]]["ndcg@10"] for r in rs])),
                    "S4@50 - S2@50": C.paired(s["S4"], s["S2"], rs),
                    "S4@50 - S2'g (same pool)": C.paired(s["S4"], s["S2p"], rs),
                }
            res[g] = cell
        out["by_split"][kind] = res

    # ---------------- 3. Task E ----------------
    key = json.loads((HUMAN / "key.json").read_text())
    E_items = {it["item"]: it for it in key["tasks"]["taskE"]}
    lab = {}
    with open(HUMAN / "taskE.csv") as fh:
        for r in csv.DictReader(fh):
            if r["annotator"]:
                lab[(E_items[r["item"]]["source"], int(r["constraint"]))] = r["answer"]
    nfacts = {it["source"]: len(it["facts"]) for it in E_items.values()}
    assert len(lab) == sum(nfacts.values()), (len(lab), sum(nfacts.values()))
    qfacts = {}
    for it in key["tasks"]["taskBD"]:
        for q in it["queries"]:
            qfacts[q["query_id"]] = (it["source"], q["facts"], [it["facts"][c - 1]["text"] for c in q["facts"]])
    test_ids = {r["query_id"] for r in allrows}
    out["taskE_coverage"] = {
        "sources_labelled": len(E_items), "facts_labelled": len(lab),
        "label_counts": dict(Counter(lab.values())),
        "test_queries_with_all_fact_labels": sum(q in qfacts for q in test_ids),
        "test_queries": len(test_ids)}
    qinfo = {}
    for q in test_ids:
        if q not in qfacts:
            continue
        src, idx, _ = qfacts[q]
        got = [lab[(src, c)] for c in idx]
        qinfo[q] = {"n_implied": got.count("implied"), "n_stated": got.count("stated"),
                    "n_absent": got.count("absent"), "k": len(got)}

    def qgroup(name):
        return {
            "any implied": lambda i: i["n_implied"] > 0,
            "no implied": lambda i: i["n_implied"] == 0,
            "any implied, no absent": lambda i: i["n_implied"] > 0 and i["n_absent"] == 0,
            "all stated": lambda i: i["n_stated"] == i["k"],
            "no fact absent": lambda i: i["n_absent"] == 0,
            "some absent": lambda i: i["n_absent"] > 0,
        }[name]

    out["taskE_by_query"] = {}
    for kind in ("expanded", "strict"):
        res = {}
        for g, rows in groups.items():
            s = SC[kind]["pooled"] if g == "pooled" else SC[kind]["natural" if g == "nested" else "twin"]
            cell = {}
            for name in ("any implied", "no implied", "any implied, no absent", "all stated", "no fact absent", "some absent"):
                rs = [r for r in rows if r["query_id"] in qinfo and qgroup(name)(qinfo[r["query_id"]])]
                if len(rs) < 5:
                    cell[name] = {"n": len(rs)}
                    continue
                cell[name] = {
                    "n": len(rs),
                    "S4@50 - S2@50": C.paired(s["S4"], s["S2"], rs),
                    "S4@50 - S2'g (same pool)": C.paired(s["S4"], s["S2p"], rs),
                    "S5 - S3 (frames)": C.paired(s["S5"], s["S3"], rs),
                }
            res[g] = cell
        out["taskE_by_query"][kind] = res
    # implied share per query as a continuous association (pooled, expanded)
    s = SC["expanded"]["pooled"]
    rs = [r for r in allrows if r["query_id"] in qinfo]
    gap = np.array([s["S4"][r["query_id"]]["ndcg@10"] - s["S2"][r["query_id"]]["ndcg@10"] for r in rs])
    for nm, f in (("share_implied", lambda i: i["n_implied"] / i["k"]), ("share_stated", lambda i: i["n_stated"] / i["k"]),
                  ("share_absent", lambda i: i["n_absent"] / i["k"])):
        x = np.array([f(qinfo[r["query_id"]]) for r in rs])
        out.setdefault("taskE_gap_correlation_pearson", {})[nm] = float(np.corrcoef(x, gap)[0, 1]) if x.std() else None
    out["taskE_query_label_mix"] = {
        SETLAB[n]: {"queries": sum(r["query_id"] in qinfo for r in rs_),
                    "any_implied": sum(qinfo[r["query_id"]]["n_implied"] > 0 for r in rs_ if r["query_id"] in qinfo),
                    "all_stated": sum(qinfo[r["query_id"]]["n_stated"] == qinfo[r["query_id"]]["k"] for r in rs_ if r["query_id"] in qinfo),
                    "any_implied_no_absent": sum(qinfo[r["query_id"]]["n_implied"] > 0 and qinfo[r["query_id"]]["n_absent"] == 0
                                                 for r in rs_ if r["query_id"] in qinfo)}
        for n, rs_ in sets.items()}

    # (b) per-fact grade on the source, by Task E label
    def fact_grades(system, query_filter):
        rows_ = []
        match_tot = fact_tot = 0
        for n, rs_ in sets.items():
            for r in rs_:
                q = r["query_id"]
                if q not in qfacts or not query_filter(cls[q]):
                    continue
                ln = lines[n][system][q]
                gr = ln["trace"]["grades"]
                src, idx, ftexts = qfacts[q]
                cons = ln["trace"]["constraints"] if "constraints" in ln["trace"] else splits[n][q]
                for c, ft in zip(idx, ftexts):
                    fact_tot += 1
                    sc = [f1(ft, cn) for cn in cons]
                    j = int(np.argmax(sc))
                    if sc[j] < 0.5 or src not in gr:
                        continue
                    match_tot += 1
                    rows_.append({"label": lab[(src, c)], "grade": gr[src][j], "source": src, "set": SETLAB[n]})
        cells = {}
        for lbl in ("stated", "implied", "absent"):
            rr = [x for x in rows_ if x["label"] == lbl]
            cl = [x["source"] for x in rr]
            if not rr:
                cells[lbl] = {"n": 0}
                continue
            cells[lbl] = {
                "n": len(rr),
                "mean_grade": C.mean_ci([x["grade"] for x in rr], cl),
                "share_5": C.mean_ci([float(x["grade"] == 5) for x in rr], cl),
                "share_ge8": C.mean_ci([float(x["grade"] >= 8) for x in rr], cl),
                "share_le2": C.mean_ci([float(x["grade"] <= 2) for x in rr], cl),
            }
        return {"fact_slots": fact_tot, "matched_with_source_graded": match_tot, "by_label": cells}

    out["taskE_fact_grades"] = {
        "S2 (descriptions), queries with count=k": fact_grades("S2g@50", lambda c: c["class"] == "=k"),
        "S2 (descriptions), all queries": fact_grades("S2g@50", lambda c: True),
        "S3 (frames), queries with count=k": fact_grades("S3g", lambda c: c["class"] == "=k"),
    }
    # (c) model judge vs human shares
    judge = {}
    jd = json.loads((C.ROOT / "results/full/fact_filter_judge/judged.json").read_text())["judged"]
    jk = [json.loads(x) for x in open(C.ROOT / "results/full/fact_filter_judge/key.jsonl")]
    for l in jk:
        if l["source"] in jd:
            judge[(l["source"], l["text"])] = jd[l["source"]]["answers"][l["pos"] - 1]
    by_text = {}
    for it in E_items.values():
        for c, f_ in enumerate(it["facts"], 1):
            by_text[(it["source"], c)] = f_["text"]
    both = [(lab[k], judge[(k[0], t)]) for k, t in by_text.items() if (k[0], t) in judge]
    jc = Counter(j for _, j in both)
    hc = Counter(h for h, _ in both)
    out["judge_vs_human"] = {
        "human_all_715": {a: out["taskE_coverage"]["label_counts"].get(a, 0) / len(lab) for a in ("stated", "implied", "absent")},
        "matched_facts": len(both), "human_on_matched": {a: hc[a] / len(both) for a in hc},
        "judge_on_matched": {a: jc[a] / len(both) for a in jc}, "crosstab_human_to_judge": {
            h: dict(Counter(j for hh, j in both if hh == h)) for h in ("stated", "implied", "absent")},
        "source_of_paper_figure": (
            "'<=12.2% stated' = tables/fact_filter_types.tex row Scene (max over the 9 fact types) of the text-only "
            "model judge (results/full/fact_filter_judge_optA, 3,812 mined facts of the 100 test sources, labels "
            "stated / contradicted / unclear, no 'implied'); human Task E = data/mc/full/human100/taskE.csv, the 715 "
            "facts that appear in test queries, labels stated / implied / absent."),
    }

    p = C.dump  # same dumper, other folder
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / "w1_offline.json").write_text(json.dumps(out, indent=1, default=C.T1._json))

    # ---------------- markdown ----------------
    md = ["# W1 offline checks (2026-11-06)", "", "nDCG@10, 722 test queries, 95% cluster-bootstrap CI over sources (10,000, seed 0). Expanded labels unless stated.", ""]
    cv = out["coverage"]
    md += [f"Coverage: {cv['with_S2_grade']}/{cv['pool_candidates']} pool candidates have a stored S2 grade; "
           f"{cv['queries_fully_covered']}/{cv['queries']} queries fully covered; S5 / S3 rerank exactly S4@50's top 10 in "
           f"{cv['S5_pool_equals_S4_top10']} / {cv['S3_pool_equals_S4_top10']} queries.", "", "## 1. Same-pool 2x2", ""]
    for kind in ("expanded", "strict"):
        md += [f"**{kind}**", "", "| group | desc whole S4@50 | desc per-c S2'g | frame whole S5 | frame per-c S3 | oracle | gap desc | gap frames | interaction |", "|---|---|---|---|---|---|---|---|---|"]
        for g, r in out["same_pool"][kind].items():
            c = r["cells"]
            md.append(f"| {g} | {c['desc whole (S4@50)']['mean']:.3f} | {c[DPC]['mean']:.3f} | "
                      f"{c['frame whole (S5)']['mean']:.3f} | {c['frame per-constraint (S3)']['mean']:.3f} | {c['oracle on pool']['mean']:.3f} | "
                      f"{ci(r[GD])} | {ci(r['gap_frames S5 - S3'])} | {ci(r['interaction (desc gap - frame gap)'])} |")
        md.append("")
    mp = out["mirror_pools"]
    md += ["## 1b. Mirror pools (S4 grades vs S2 grades on the same ten candidates)", "", f"Check: {json.dumps(mp['check'])}", ""]
    for kind in ("expanded", "strict"):
        md += [f"**{kind}**", "", "| pool | group | S4 grades | S2 grades | gap S4 - S2 | oracle |", "|---|---|---|---|---|---|"]
        for pn, r in mp[kind].items():
            for g, d in r.items():
                md.append(f"| {pn} | {g} | {d['S4 grades']['mean']:.3f} | {d['S2 grades']['mean']:.3f} | "
                          f"{ci(d['gap S4 grades - S2 grades'])} | {d['oracle']['mean']:.3f} |")
        md.append("")
    md += ["## 2. By split class (count of splitter constraints vs k)", ""]
    for kind in ("expanded", "strict"):
        md += [f"**{kind}**", "", "| group | class | n | S4@50 - S2@50 | S4@50 - S2'g |", "|---|---|---|---|---|"]
        for g, cell in out["by_split"][kind].items():
            for cn, d in cell.items():
                if "S4@50 - S2@50" in d:
                    md.append(f"| {g} | {cn} | {d['n']} | {ci(d['S4@50 - S2@50'])} | {ci(d[SP])} |")
        md.append("")
    md += ["## 3. Task E", "", f"Coverage: {json.dumps(out['taskE_coverage'])}", "", "**By query (expanded)**", "",
           "| group | query set | n | S4@50 - S2@50 | S4@50 - S2'g | S5 - S3 |", "|---|---|---|---|---|---|"]
    for g, cell in out["taskE_by_query"]["expanded"].items():
        for cn, d in cell.items():
            if "S4@50 - S2@50" in d:
                md.append(f"| {cn} | {g} | {d['n']} | {ci(d['S4@50 - S2@50'])} | {ci(d[SP])} | {ci(d['S5 - S3 (frames)'])} |")
    md += ["", "**Per-fact grade on the source by Task E label**", "", "| system / queries | label | n | mean grade | share = 5 | share >= 8 |", "|---|---|---|---|---|---|"]
    for sysn, r in out["taskE_fact_grades"].items():
        for lbl, d in r["by_label"].items():
            if d["n"]:
                md.append(f"| {sysn} | {lbl} | {d['n']} | {d['mean_grade']['mean']:.2f} | {d['share_5']['mean']:.2f} | {d['share_ge8']['mean']:.2f} |")
    md += ["", f"Judge vs human: {json.dumps(out['judge_vs_human'], default=str)}"]
    (OUTDIR / "w1_offline.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
