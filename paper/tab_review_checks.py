"""Reviewer-requested checks on the saved full-run outputs. FREE: no model calls; reads
results/full/runs, results/full/select/subsets.json and data/mc/full/judge only.

    env -u OPENAI_API_KEY -u GEMINI_API_KEY uv run python paper/tab_review_checks.py \
        [--tables-dir results/paper/tables]

1. Like-for-like S2 on the frame candidates (review Q8): S2' = S2's per-constraint yes-count
   ranking (S2's scoring and tie-break: ties in candidate order) applied to S4@50's top 10, the
   candidates S3 / S5 rerank, from the description verdicts S2@50 already saved (every one of
   S4@50's top 10 is in S0's top 50). Means and paired CIs on both frame subsets, overall and
   per k.
2. Paired TOST for S3 vs S5 (Q3): equivalence at margin m when the 90% cluster-bootstrap CI of
   S3 - S5 lies inside (-m, +m); m in 0.01, 0.02, 0.03.
3. Accepted-source subset (Q4): queries whose own source both judges accept; the main paired
   comparisons recomputed there under expanded and strict labels.
4. Per-k absolute nDCG@10 / R@1 (Q2, Q7) on the frame subsets and on all queries.
5. Judge-pool counts (Q5): extras reached only at k >= 3, sampled vs unjudged; where the
   expanded positives come from; unjudged top-10 candidates per system.

Reuses the eval code unchanged (pipeline/run_systems.py load_runs / paired_cells / means,
pilot_verify load_qrels / score_system / paired, judge_pool + src.judge_frames pools and
labels, src.stats.bootstrap_ci for the 90% interval with the same resampling as the 95%
one). Writes results/full/review_checks/{report.md, summary.json, tables/review_*.tex} and, with
--tables-dir, copies the .tex tables there.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JD = _load(ROOT / "pipeline/judge_labels.py", "_rc_judge")
FS = JD.FS
PV = FS.PV
jf = JD.jf

from src.stats import bootstrap_ci

OUT = Path("results/full/review_checks")
S2F = "S2'"  # S2's description verdicts on S4@50's top 10
FRAMES = {"twin-both": ("twin", "twin_both"), "nested sample": ("natural", "nested_sample")}
SET_LABEL = {"natural": "nested", "twin": "twin"}
ALL_PAIRS = [("S1", "S0"), ("S2@50", "S0"), ("S4@50", "S0"), ("S4@50", "S2@50")]
FRAME_PAIRS = [(S2F, "S2@10"), ("S3", S2F), ("S5", "S4@50"), ("S4@50", S2F), ("S3", "S5")]
ACC_FRAME_PAIRS = [("S5", "S4@50"), ("S3", "S5"), ("S3", S2F)]
MARGINS = (0.01, 0.02, 0.03)
KS = ("all", "1", "2", "3", "4")
ALL_QUERY_SYS = ("S0", "S1", "S2@50", "S4@50")  # run on every query (judge pool systems)
FRAME_SYS = ("S3", "S5")
METRICS = ("ndcg@10", "recall@1")


# ---- inputs ------------------------------------------------------------------------------------


def s2_on(s4_rank, s2_line, top: int):
    """S2's yes-count ranking of S4@50's top `top` (ties in S4@50 order), then S4@50's tail, as
    S3 is built; plus the candidates without a saved S2 verdict."""
    head = [d for d, _ in s4_rank[:top]]
    verdicts = s2_line["trace"]["verdicts"]
    missing = [d for d in head if d not in verdicts]
    scores = [sum(a == "yes" for a in verdicts.get(d, [])) for d in head]
    ranked = FS.rank_by_score([(d, 0.0) for d in head], scores)
    return FS.merge_reranked(s4_rank, ranked), missing


def load(n_boot: int, sources_file: str | None = None) -> dict:
    """sources_file: the test set (systems.py --sources-file): the query sets, the frozen
    subsets, the runs and the judged pairs restricted to the drawn sources."""
    S = FS.settings(argparse.Namespace(smoke=False, config=None, out=None, n_boot=n_boot))
    sets = FS.load_sets(S, FS.load_context(S, need_index=False))
    if sources_file:
        sets = FS.restrict_sets(S, sets, sources_file)
    sub = FS.read_subsets(S)
    ranks, lines = {}, {}
    for name in sets:
        ranks[name], lines[name] = FS.load_runs(S.runs / name, S.rescore)
        if S.keep:
            ranks[name] = {
                s: {q: v for q, v in r.items() if q in S.keep[name]} for s, r in ranks[name].items()
            }
            lines[name] = {
                s: {q: v for q, v in r.items() if q in S.keep[name]} for s, r in lines[name].items()
            }
    frames, coverage = {}, {}
    for label, (name, key) in FRAMES.items():
        ids = sub[key][name]
        frames[label] = (name, [r for r in sets[name] if r["query_id"] in set(ids)])
        gaps, s3_same, same_split, short = [], 0, 0, 0
        ranks[name].setdefault(S2F, {})
        for q in ids:
            s4, s2 = ranks[name]["S4@50"][q], lines[name]["S2@50"][q]
            ranks[name][S2F][q], missing = s2_on(s4, s2, S.top)
            gaps += [(q, d) for d in missing]
            # the construction reproduces S3 exactly from S3's own verdicts
            s3 = lines[name]["S3"][q]
            assert s3["trace"]["first_stage"] == [d for d, _ in s4[: S.top]], q
            rebuilt, _ = s2_on(s4, s3, S.top)
            s3_same += [d for d, _ in rebuilt] == [d for d, _ in ranks[name]["S3"][q]]
            cons = s2["trace"]["constraints"]
            same_split += cons == s3["trace"]["constraints"]
            short += sum(
                len(s2["trace"]["verdicts"][d]) != len(cons)
                for d, _ in s4[: S.top]
                if d in s2["trace"]["verdicts"]
            )
        coverage[label] = {
            "queries": len(ids),
            "candidates": len(ids) * S.top,
            "missing_verdicts": len(gaps),
            "missing": gaps[:20],
            "verdict_lists_wrong_length": short,
            "same_constraints_as_S3": same_split,
            "S3_rebuilt_from_its_verdicts": s3_same,
        }
    qrels = {
        "expanded": {
            n: PV.load_qrels(S.judge_dir / "qrels_expanded.jsonl", rs) for n, rs in sets.items()
        },
        "strict": {n: PV.load_qrels(None, rs) for n, rs in sets.items()},
    }
    scores = {
        kind: {n: {s: PV.score_system(r, qr[n]) for s, r in ranks[n].items()} for n in sets}
        for kind, qr in qrels.items()
    }
    return {
        "S": S,
        "sets": sets,
        "sub": sub,
        "ranks": ranks,
        "frames": frames,
        "coverage": coverage,
        "qrels": qrels,
        "scores": scores,
    }


def by_k(cells: dict) -> dict:
    """paired_cells / means output -> {"all"|k: cell} for one metric."""
    return {"all": cells["None"]["all"], **{str(k): c for k, c in cells.get("k", {}).items()}}


def pair(sa, sb, rows, n_boot) -> dict:
    return {m: by_k(c) for m, c in FS.paired_cells(sa, sb, rows, n_boot).items()}


def abs_means(sc, rows, n_boot) -> dict:
    """{metric: {"all"|k: {n, mean, ci_low, ci_high}}}, CI by the same cluster bootstrap."""
    out = {}
    for m in METRICS:
        out[m] = {}
        for g in KS:
            rs = [r for r in rows if r["query_id"] in sc and (g == "all" or str(r["k"]) == g)]
            if rs:
                c = FS.P2.share_ci(
                    [sc[r["query_id"]][m] for r in rs], [r["source"] for r in rs], n_boot
                )
                out[m][g] = {
                    "n": c["n"],
                    "mean": c["rate"],
                    "ci_low": c["ci_low"],
                    "ci_high": c["ci_high"],
                }
    return out


def status(c: dict) -> str:
    return "+" if c["ci_low"] > 0 else "-" if c["ci_high"] < 0 else "0"


# ---- checks ------------------------------------------------------------------------------------


def check_frames(D: dict) -> dict:
    out = {}
    for kind in ("expanded", "strict"):
        out[kind] = {}
        for label, (name, rows) in D["frames"].items():
            sc = D["scores"][kind][name]
            out[kind][label] = {
                "means": {
                    s: abs_means(sc[s], rows, D["S"].n_boot)
                    for s in ("S0", "S2@10", "S4@10", S2F, "S4@50", "S3", "S5")
                },
                "paired": {
                    f"{a} - {b}": pair(sc[a], sc[b], rows, D["S"].n_boot) for a, b in FRAME_PAIRS
                },
            }
    return out


def tost(sa, sb, rows, metric, n_boot) -> dict:
    rs = [r for r in rows if r["query_id"] in sa and r["query_id"] in sb]
    diffs = np.array([sa[r["query_id"]][metric] - sb[r["query_id"]][metric] for r in rs])
    clusters = [r["source"] for r in rs]
    lo, hi = bootstrap_ci(diffs, clusters, n=n_boot, alpha=0.10, seed=0)
    lo95, hi95 = bootstrap_ci(diffs, clusters, n=n_boot, alpha=0.05, seed=0)
    ref = PV.paired(sa, sb, rs, metric, None, n_boot)["all"]  # same resampling as the eval
    assert abs(ref["ci_low"] - lo95) < 1e-12 and abs(ref["ci_high"] - hi95) < 1e-12
    shown = {str(m): bool(lo > -m and hi < m) for m in MARGINS}
    return {
        "n": len(rs),
        "mean": float(diffs.mean()),
        "ci90": [lo, hi],
        "ci95": [lo95, hi95],
        "equivalent": shown,
        "smallest_margin": next((m for m in MARGINS if shown[str(m)]), None),
        "min_margin_exact": float(max(0.0, -lo, hi)),
    }


def check_tost(D: dict) -> dict:
    out = {}
    for kind in ("expanded", "strict"):
        out[kind] = {}
        for label, (name, rows) in D["frames"].items():
            sc = D["scores"][kind][name]
            out[kind][label] = {}
            for m in METRICS:
                cell = {"all": tost(sc["S3"], sc["S5"], rows, m, D["S"].n_boot)}
                for k in ("1", "2", "3", "4"):
                    cell[k] = tost(
                        sc["S3"], sc["S5"], [r for r in rows if str(r["k"]) == k], m, D["S"].n_boot
                    )
                out[kind][label][m] = cell
    return out


def judge_rows(S) -> tuple:
    """JD.setup's (jcfg, rows, facts, of_query) and the judged pairs, restricted to the kept
    sources under --sources-file."""
    jcfg, jrows, facts, of_query, _ = JD.setup(S)
    judged = JD.JP.read_judged(S.judge_dir / "judged.jsonl")
    if S.keep_sources is not None:
        jrows = [r for r in jrows if r["source"] in S.keep_sources]
        judged = {s: v for s, v in judged.items() if s in S.keep_sources}
    return jcfg, jrows, facts, of_query, judged


def accepted_ids(D: dict) -> tuple[dict, dict]:
    jcfg, jrows, _facts, of_query, judged = judge_rows(D["S"])
    specs = tuple(jcfg["models"])
    acc = {
        r["query_id"]: jf.relevant(judged[r["source"]][r["source"]], of_query[r["query_id"]], specs)
        for r in jrows
        if r["source"] in judged.get(r["source"], {})
    }
    rates = {}
    for name, rows in D["sets"].items():
        v = [acc[r["query_id"]] for r in rows if r["query_id"] in acc]
        rates[name] = {"n": len(rows), "judged": len(v), "accepted": int(sum(v))}
    for label, (_name, rows) in D["frames"].items():
        v = [acc[r["query_id"]] for r in rows if r["query_id"] in acc]
        rates[label] = {"n": len(rows), "judged": len(v), "accepted": int(sum(v))}
    allv = [acc[r["query_id"]] for rs in D["sets"].values() for r in rs if r["query_id"] in acc]
    rates["all"] = {"n": len(allv), "accepted": int(sum(allv)), "rate": sum(allv) / len(allv)}
    return acc, rates


def check_accepted(D: dict, acc: dict) -> dict:
    nb = D["S"].n_boot
    groups = [(SET_LABEL[n], n, rs, ALL_PAIRS) for n, rs in D["sets"].items()]
    groups += [(label, n, rs, ACC_FRAME_PAIRS) for label, (n, rs) in D["frames"].items()]
    out = {}
    for kind in ("expanded", "strict"):
        out[kind] = {}
        for label, name, rows, pairs in groups:
            sc = D["scores"][kind][name]
            keep = [r for r in rows if acc.get(r["query_id"])]
            cell = {}
            for a, b in pairs:
                full, sub = pair(sc[a], sc[b], rows, nb), pair(sc[a], sc[b], keep, nb)
                cell[f"{a} - {b}"] = {
                    "all": full,
                    "accepted": sub,
                    "changed": {
                        m: [
                            g
                            for g in KS
                            if g in full[m]
                            and g in sub[m]
                            and status(full[m][g]) != status(sub[m][g])
                        ]
                        for m in METRICS
                    },
                }
            out[kind][label] = {"n_all": len(rows), "n_accepted": len(keep), "pairs": cell}
    return out


def check_per_k(D: dict) -> dict:
    nb = D["S"].n_boot
    out = {}
    for kind in ("expanded", "strict"):
        out[kind] = {}
        for name, rows in D["sets"].items():
            sc = D["scores"][kind][name]
            out[kind][SET_LABEL[name]] = {
                s: abs_means(sc[s], rows, nb)
                for s in ("S0", "S1", "S2@10", "S2@50", "S4@10", "S4@50")
            }
    return out


def check_pool(D: dict) -> dict:
    S = D["S"]
    jcfg, jrows, _facts, of_query, judged = judge_rows(S)
    full_k, top = jcfg.get("full_k", 2), jcfg.get("top", jf.TOP)
    rank = JD.rankings(S)
    pools, plan, planned = JD.JP.judge_tasks(jrows, rank, jcfg)

    def core(e) -> bool:
        return bool({"source", "twin"} & set(e["why"]))

    def extra(e) -> bool:  # reached only by queries with k > full_k
        return not core(e) and (e["min_k"] is None or e["min_k"] > full_k)

    pairs = [(s, d, e) for s, p in pools.items() for d, e in p.items()]
    ex = [(s, d, e) for s, d, e in pairs if extra(e)]
    src_level = {
        "pool_pairs": len(pairs),
        "planned": len(planned),
        "judged_pairs": sum(len(v) for v in judged.values()),
        "planned_not_judged": sum(d not in judged.get(s, {}) for s, d in planned),
        "extras_k_ge_3": len(ex),
        "extras_sampled": sum(plan[s][d] for s, d, _ in ex),
        "extras_unsampled": sum(not plan[s][d] for s, d, _ in ex),
        "extras_judged": sum(d in judged.get(s, {}) for s, d, _ in ex),
        "by_min_k": {
            str(k): {
                "extras": sum(e["min_k"] == k for _, _, e in ex),
                "sampled": sum(plan[s][d] for s, d, e in ex if e["min_k"] == k),
            }
            for k in (3, 4)
        },
    }
    # (query, candidate): each query's pool candidates (union of the systems' top 10), without
    # its source and twin
    rows = [r for rs in D["sets"].values() for r in rs]
    q_level: dict = {}
    for r in rows:
        src = r["source"]
        for d in jf.judged_in_top(r, rank, top):
            if d in (src, r.get("twin")):
                continue
            e = pools[src][d]
            for g in ("all", str(r["k"])):
                c = q_level.setdefault(
                    g, {"pairs": 0, "only_k_ge_3": 0, "sampled": 0, "unjudged": 0}
                )
                c["pairs"] += 1
                if extra(e):
                    c["only_k_ge_3"] += 1
                    c["sampled"] += d in judged.get(src, {})
                c["unjudged"] += d not in judged.get(src, {})
    # where the expanded positives (other than the source) come from
    sys_pool = (*ALL_QUERY_SYS, *FRAME_SYS)
    qrels = D["qrels"]["expanded"]
    pos = {
        "positives": 0,
        "in_top10_of": {s: 0 for s in sys_pool},
        "only_in": {s: 0 for s in sys_pool},
    }
    pos |= {"in_some_all_query_system": 0, "in_no_top10_of_query": 0, "via_k_ge_3_sample": 0}
    pos["outside_pool"] = 0  # judged only in a fill for runs outside runs/ (local rerankers)
    frame_pos = {"positives": 0, "in_S3_or_S5_not_S4@50": 0, "in_all_query_sys_not_frame_sys": 0}
    frame_ids = {r["query_id"] for _, rs in D["frames"].values() for r in rs}
    for name, rs in D["sets"].items():
        rk = D["ranks"][name]
        for r in rs:
            q, src = r["query_id"], r["source"]
            heads = {s: {d for d, _ in rk[s][q][:top]} for s in sys_pool if s in rk and q in rk[s]}
            for d, g in qrels[name][q].items():
                if g <= 0 or d == src:
                    continue
                pos["positives"] += 1
                hit = [s for s, h in heads.items() if d in h]
                for s in hit:
                    pos["in_top10_of"][s] += 1
                if len(hit) == 1:
                    pos["only_in"][hit[0]] += 1
                pos["in_some_all_query_system"] += any(s in ALL_QUERY_SYS for s in hit)
                pos["in_no_top10_of_query"] += not hit
                if d in pools.get(src, {}):
                    pos["via_k_ge_3_sample"] += extra(pools[src][d])
                else:
                    pos["outside_pool"] += 1
                if q in frame_ids:
                    frame_pos["positives"] += 1
                    fr = {s for s in hit if s in FRAME_SYS}
                    frame_pos["in_S3_or_S5_not_S4@50"] += bool(fr) and "S4@50" not in hit
                    frame_pos["in_all_query_sys_not_frame_sys"] += not fr and bool(
                        set(hit) & set(ALL_QUERY_SYS)
                    )
    # unjudged (counted as non-relevant) candidates in each system's top 10
    unj = {}
    for name, rs in D["sets"].items():
        rk = D["ranks"][name]
        for s in (
            "S0",
            "S1",
            "S2@10",
            "S2@50",
            "S2g@10",
            "S2g@50",
            "S4@10",
            "S4@50",
            S2F,
            "S3",
            "S3g",
            "S5",
        ):
            for r in rs:
                q, src = r["query_id"], r["source"]
                if q not in rk.get(s, {}):
                    continue
                head = [d for d, _ in rk[s][q][:top]]
                miss = [d for d in head if d not in judged.get(src, {})]
                for g in ("all", str(r["k"])):
                    c = (
                        unj.setdefault(SET_LABEL[name], {})
                        .setdefault(s, {})
                        .setdefault(
                            g,
                            {
                                "queries": 0,
                                "slots": 0,
                                "unjudged": 0,
                                "queries_with_unjudged": 0,
                                "unjudged_at_1": 0,
                            },
                        )
                    )
                    c["queries"] += 1
                    c["slots"] += len(head)
                    c["unjudged"] += len(miss)
                    c["queries_with_unjudged"] += bool(miss)
                    c["unjudged_at_1"] += bool(head) and head[0] in miss
    fn = jf.false_negative_estimate(
        [r for r in jrows if r["source"] in judged],
        of_query,
        rank,
        pools,
        plan,
        judged,
        tuple(jcfg["models"]),
        full_k,
        top,
    )
    return {
        "source_level": src_level,
        "query_level": q_level,
        "positives": pos,
        "frame_subset_positives": frame_pos,
        "unjudged_top10": unj,
        "false_negative_estimate": fn,
    }


def regression_checks(D: dict, res: dict) -> dict:
    """Recomputed cells against results/full/eval/summary.json."""
    ref = json.loads((D["S"].out / "eval" / "summary.json").read_text())["labels"]
    out = {}
    for kind in ("expanded", "strict"):
        a = ref[kind]["paired"]["natural"]["S4@50 - S0"]["ndcg@10"]["None"]["all"]["mean"]
        b = res["accepted"][kind]["nested"]["pairs"]["S4@50 - S0"]["all"]["ndcg@10"]["all"]["mean"]
        c = ref[kind]["subset_paired"]["twin-both"]["S5 - S4@50"]["ndcg@10"]["None"]["all"]
        d = res["frames"][kind]["twin-both"]["paired"]["S5 - S4@50"]["ndcg@10"]["all"]
        out[kind] = {
            "S4@50 - S0 nested": abs(a - b) < 1e-12,
            "S5 - S4@50 twin-both": abs(c["mean"] - d["mean"]) < 1e-12
            and abs(c["ci_low"] - d["ci_low"]) < 1e-12,
        }
        assert all(out[kind].values()), out
    return out


# ---- report ------------------------------------------------------------------------------------


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def ci(c) -> str:
    return "-" if not c else f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def report(res: dict) -> str:
    L = ["# Reviewer checks (paper/tab_review_checks.py; no model calls)", ""]
    L += [
        (
            "All from saved outputs: results/full/runs, results/full/select/subsets.json, "
            "data/mc/full/judge. Expanded labels primary. CIs: paired cluster bootstrap over source "
            f"videos, {res['n_boot']:,} resamples (the eval code's), 95% unless stated."
        ),
        "",
        "## 1. S2' = S2's description verdicts on S4@50's top 10 (review Q8)",
        "",
        (
            "S2' ranks S4@50's top 10 by S2@50's per-constraint yes-count (ties in S4@50 order), "
            "then S4@50's tail, exactly as S3 is built from S3's frame verdicts."
        ),
        "",
        (
            "| subset | queries | candidates | missing verdicts | wrong-length verdict lists | "
            "same constraints as S3 | S3 rebuilt from its verdicts |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for label, c in res["coverage"].items():
        L.append(
            f"| {label} | {c['queries']} | {c['candidates']} | {c['missing_verdicts']} | "
            f"{c['verdict_lists_wrong_length']} | {c['same_constraints_as_S3']} | "
            f"{c['S3_rebuilt_from_its_verdicts']} |"
        )
    for kind in ("expanded", "strict"):
        for label, cell in res["frames"][kind].items():
            for m in METRICS:
                L += ["", f"### {kind}, {label}: {m} means by k", ""]
                L += ["| system | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
                for s, mm in cell["means"].items():
                    L.append(
                        f"| {s} | "
                        + " | ".join(f3(mm[m].get(g, {}).get("mean")) for g in KS)
                        + " |"
                    )
                L += ["", f"### {kind}, {label}: {m} paired differences", ""]
                L += ["| pair | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
                for p, pc in cell["paired"].items():
                    L.append(f"| {p} | " + " | ".join(ci(pc[m].get(g)) for g in KS) + " |")
    L += ["", "## 2. Equivalence of S3 and S5 (paired TOST, 90% CI inside +-margin; Q3)", ""]
    L += [
        "| labels | subset | metric | n | S3 - S5 | 90% CI | eq. at .01 / .02 / .03 | smallest | exact |"
    ]
    L += ["|---|---|---|---|---|---|---|---|---|"]
    for kind, by in res["tost"].items():
        for label, bm in by.items():
            for m, cell in bm.items():
                t = cell["all"]
                eq = " / ".join("yes" if t["equivalent"][str(x)] else "no" for x in MARGINS)
                L.append(
                    f"| {kind} | {label} | {m} | {t['n']} | {t['mean']:+.3f} | "
                    f"[{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}] | {eq} | "
                    f"{t['smallest_margin'] if t['smallest_margin'] else 'none'} | {t['min_margin_exact']:.4f} |"
                )
    L += [
        "",
        "Per k (expanded, nDCG@10): smallest margin of .01/.02/.03 at which equivalence is shown.",
        "",
    ]
    L += ["| subset | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|"]
    for label, bm in res["tost"]["expanded"].items():
        c = bm["ndcg@10"]
        L.append(
            f"| {label} | "
            + " | ".join(
                f"{c[k]['smallest_margin'] or 'none'} (n {c[k]['n']}, exact {c[k]['min_margin_exact']:.4f})"
                for k in ("1", "2", "3", "4")
            )
            + " |"
        )
    a = res["acceptance"]
    L += ["", "## 3. Accepted-source subset (Q4)", ""]
    L += [
        f"Both judges accept the source on its own query for {a['all']['accepted']} of "
        f"{a['all']['n']} evaluated queries ({a['all']['rate']:.1%}); "
        + "; ".join(f"{k} {v['accepted']}/{v['n']}" for k, v in a.items() if k != "all")
        + ". Changed = groups where the CI's sign status (above 0 / includes 0 / below 0) differs "
        "from all queries.",
        "",
    ]
    for kind, by in res["accepted"].items():
        L += [f"### {kind} labels", ""]
        L += [
            "| group | pair | n all / accepted | nDCG@10 all | nDCG@10 accepted | R@1 all | R@1 accepted | changed (nDCG; R@1) |"
        ]
        L += ["|---|---|---|---|---|---|---|---|"]
        for label, cell in by.items():
            for p, pc in cell["pairs"].items():
                ch = pc["changed"]
                L.append(
                    f"| {label} | {p} | {cell['n_all']} / {cell['n_accepted']} | "
                    f"{ci(pc['all']['ndcg@10']['all'])} | {ci(pc['accepted']['ndcg@10']['all'])} | "
                    f"{ci(pc['all']['recall@1']['all'])} | {ci(pc['accepted']['recall@1']['all'])} | "
                    f"{','.join(ch['ndcg@10']) or '-'}; {','.join(ch['recall@1']) or '-'} |"
                )
        L += [""]
        L += ["Per k on the accepted subset (nDCG@10):", ""]
        L += ["| group | pair | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|---|"]
        for label, cell in by.items():
            for p, pc in cell["pairs"].items():
                L.append(
                    f"| {label} | {p} | "
                    + " | ".join(ci(pc["accepted"]["ndcg@10"].get(g)) for g in KS)
                    + " |"
                )
        L += [""]
    L += ["## 4. Per-k absolute scores (Q2, Q7)", ""]
    for kind in ("expanded", "strict"):
        for label, cell in res["frames"][kind].items():
            L += [f"### {kind}, {label}: nDCG@10 [95% CI]", ""]
            L += ["| system | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
            for s, mm in cell["means"].items():
                L.append(f"| {s} | " + " | ".join(ci(mm["ndcg@10"].get(g)) for g in KS) + " |")
            L += [""]
        for set_, cell in res["per_k"][kind].items():
            for m in METRICS:
                L += [f"### {kind}, all {set_} queries: {m}", ""]
                L += ["| system | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
                for s, mm in cell.items():
                    L.append(
                        f"| {s} | "
                        + " | ".join(f3(mm[m].get(g, {}).get("mean")) for g in KS)
                        + " |"
                    )
                L += [""]
    p = res["pool"]
    sl, ql, po, fp = (
        p["source_level"],
        p["query_level"],
        p["positives"],
        p["frame_subset_positives"],
    )
    L += ["## 5. Judge pool (Q5)", ""]
    L += [
        (
            f"- (source, candidate) pairs in the pools: {sl['pool_pairs']}; planned for judging "
            f"{sl['planned']}, judged {sl['judged_pairs']} (planned but not judged: {sl['planned_not_judged']})."
        ),
        (
            f"- Extras reached only through k >= 3 queries (not source or twin): {sl['extras_k_ge_3']}; "
            f"sampled (25%) {sl['extras_sampled']}, left unjudged {sl['extras_unsampled']} "
            f"(judged in all: {sl['extras_judged']}); by min k: {sl['by_min_k']}."
        ),
        f"- (query, candidate) pairs (each query's pool candidates, without source and twin): "
        f"{ql['all']['pairs']}; entered only through k >= 3 queries {ql['all']['only_k_ge_3']} "
        f"(sampled and judged {ql['all']['sampled']}, unjudged {ql['all']['unjudged']}). By k: "
        + "; ".join(f"k={k}: {ql[k]}" for k in ("1", "2", "3", "4") if k in ql)
        + ".",
        (
            f"- Expanded positives other than the source: {po['positives']} (query, video) pairs; in the "
            f"top 10 of a system run on all queries (S0, S1, S2@50, S4@50) for that query: "
            f"{po['in_some_all_query_system']}; in no system's top 10 for that query (pooled through "
            f"another query of the same source): {po['in_no_top10_of_query']}; labelled through the "
            f"25% k >= 3 sample: {po['via_k_ge_3_sample']}."
        ),
        (
            f"- Per system, positives in its top 10: {po['in_top10_of']}; found by that system alone: "
            f"{po['only_in']}."
        ),
        (
            f"- Frame-subset queries: {fp['positives']} positives; in S3 or S5's top 10 but not S4@50's: "
            f"{fp['in_S3_or_S5_not_S4@50']} (frame systems add no candidates); found by all-query "
            f"systems but in neither S3 nor S5's top 10: {fp['in_all_query_sys_not_frame_sys']}."
        ),
        (
            f"- Estimated missed positives among unjudged extras (judge_frames.false_negative_estimate): "
            f"{json.dumps(p['false_negative_estimate'].get('all'))}."
        ),
        "",
        "Unjudged candidates (counted as non-relevant) in each system's top 10:",
        "",
        "| set | system | queries | top-10 slots | unjudged | share | queries with any | unjudged at rank 1 | unjudged k=3 / k=4 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for set_, by in p["unjudged_top10"].items():
        for s, c in by.items():
            a_ = c["all"]
            L.append(
                f"| {set_} | {s} | {a_['queries']} | {a_['slots']} | {a_['unjudged']} | "
                f"{a_['unjudged'] / max(1, a_['slots']):.1%} | {a_['queries_with_unjudged']} | "
                f"{a_['unjudged_at_1']} | {c.get('3', {}).get('unjudged', 0)} / {c.get('4', {}).get('unjudged', 0)} |"
            )
    return "\n".join(L) + "\n"


# ---- LaTeX -------------------------------------------------------------------------------------


def tnum(v: float) -> str:
    return f"${v:.3f}$"


def tci(c: dict) -> str:
    return f"${c['mean']:+.3f}$\\,{{\\scriptsize[${c['ci_low']:+.3f}$, ${c['ci_high']:+.3f}$]}}"


def tsys(s: str) -> str:
    return {S2F: "S2$'$"}.get(s, s)


def tpair(p: str) -> str:
    a, b = p.split(" - ")
    return f"{tsys(a)} $-$ {tsys(b)}"


HEADER = "% Generated by paper/tab_review_checks.py from saved full-run outputs; do not edit by hand.\n"


def table_frames(res: dict) -> str:
    fr, to = res["frames"]["expanded"], res["tost"]["expanded"]
    labels = list(FRAMES)
    n = {lab: fr[lab]["means"]["S5"]["ndcg@10"]["all"]["n"] for lab in labels}
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{2pt}",
    ]
    L += ["\\begin{tabular}{lcc}", "\\toprule"]
    L += [
        " & Twin-both & Nested sample \\\\",
        f" & ($n={n['twin-both']}$) & ($n={n['nested sample']}$) \\\\",
    ]
    L += ["\\midrule", "\\multicolumn{3}{l}{\\emph{nDCG@10}} \\\\"]
    for s in (S2F, "S4@50", "S3", "S5"):
        L.append(
            f"{tsys(s)} & "
            + " & ".join(tnum(fr[lab]["means"][s]["ndcg@10"]["all"]["mean"]) for lab in labels)
            + " \\\\"
        )
    L += ["\\midrule", "\\multicolumn{3}{l}{\\emph{Paired difference, 95\\% CI}} \\\\"]
    for p in ("S3 - S2'", "S4@50 - S2'", "S5 - S4@50", "S3 - S5"):
        L.append(
            f"{tpair(p)} & "
            + " & ".join(tci(fr[lab]["paired"][p]["ndcg@10"]["all"]) for lab in labels)
            + " \\\\"
        )
    L += ["\\midrule", "\\multicolumn{3}{l}{\\emph{Equivalence of S3 and S5 (paired TOST)}} \\\\"]
    L.append(
        "90\\% CI & "
        + " & ".join(
            f"{{\\scriptsize[${to[lab]['ndcg@10']['all']['ci90'][0]:+.3f}$, ${to[lab]['ndcg@10']['all']['ci90'][1]:+.3f}$]}}"
            for lab in labels
        )
        + " \\\\"
    )
    exact = [to[lab]["ndcg@10"]["all"]["min_margin_exact"] for lab in labels]
    L.append("Smallest margin & " + " & ".join(f"$\\pm{m:.3f}$" for m in exact) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    L.append(
        "\\caption{Like-for-like comparison on the frame candidates (expanded labels). "
        "S2$'$ applies S2's per-constraint description verdicts (from the S2@50 run, no new calls) "
        "to S4@50's top 10, with S2's yes-count scoring and "
        "tie-break. Intervals are paired cluster-bootstrap intervals over source videos. "
        "Equivalence (two one-sided tests at $\\alpha=.05$): the 90\\% interval of S3 $-$ S5 "
        "lies inside $\\pm$ the margin; we give the smallest such margin, since no margin was set before the main run.}"
    )
    L += ["\\label{tab:rc-frames}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_accepted(res: dict) -> str:
    acc, a = res["accepted"], res["acceptance"]
    any_change = any(
        "all" in pc["changed"]["ndcg@10"]
        for kind in acc.values()
        for cell in kind.values()
        for pc in cell["pairs"].values()
    )
    L = [HEADER + "\\begin{table*}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}"]
    L += ["\\begin{tabular}{lrcccc}", "\\toprule"]
    L += [
        "& & \\multicolumn{2}{c}{All queries} & \\multicolumn{2}{c}{Source accepted by both judges} \\\\",
        "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
        "Comparison & $n$ & Expanded & Strict & Expanded & Strict \\\\",
        "\\midrule",
    ]
    heads = {
        "nested": "Nested, all queries",
        "twin": "Twin, all queries",
        "twin-both": "Twin-both (frame subset)",
        "nested sample": "Nested sample (frame subset)",
    }
    for i, (label, title) in enumerate(heads.items()):
        e, s = acc["expanded"][label], acc["strict"][label]
        if i:
            L.append("\\addlinespace[2pt]")
        L.append(f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}}}} \\\\")
        for p in e["pairs"]:
            changed = (
                "all" in e["pairs"][p]["changed"]["ndcg@10"] + s["pairs"][p]["changed"]["ndcg@10"]
            )
            mark = "$^\\dagger$" if changed else ""
            L.append(
                f"{tpair(p)}{mark} & {e['n_accepted']}/{e['n_all']} & "
                f"{tci(e['pairs'][p]['all']['ndcg@10']['all'])} & "
                f"{tci(s['pairs'][p]['all']['ndcg@10']['all'])} & "
                f"{tci(e['pairs'][p]['accepted']['ndcg@10']['all'])} & "
                f"{tci(s['pairs'][p]['accepted']['ndcg@10']['all'])} \\\\"
            )
    L += ["\\bottomrule", "\\end{tabular}"]
    L.append(
        "\\caption{Main paired comparisons (nDCG@10, 95\\% cluster-bootstrap intervals) restricted "
        f"to the queries whose own source both frame judges accept ({a['all']['accepted']} of "
        f"{a['all']['n']}, {100 * a['all']['rate']:.0f}\\%); $n$ is accepted/all. "
        "S2$'$: S2's description verdicts applied to S4@50's top 10. "
        + (
            "$^\\dagger$: the interval's position relative to zero differs from all queries.}"
            if any_change
            else "No interval changes its position relative to zero (above, including, below) "
            "between all queries and the accepted subset.}"
        )
    )
    L += ["\\label{tab:rc-accepted}", "\\end{table*}"]
    return "\n".join(L) + "\n"


def table_per_k(res: dict) -> str:
    fr, pk = res["frames"]["expanded"], res["per_k"]["expanded"]
    L = [HEADER + "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{4pt}"]
    L += [
        "\\begin{tabular}{lccccc}",
        "\\toprule",
        "System & All & $k=1$ & $k=2$ & $k=3$ & $k=4$ \\\\",
    ]
    panels = [
        ("Twin-both", fr["twin-both"]["means"], ("S0", "S4@10", S2F, "S4@50", "S3", "S5")),
        ("Nested sample", fr["nested sample"]["means"], ("S0", "S4@10", S2F, "S4@50", "S3", "S5")),
        ("All twin queries", pk["twin"], ("S0", "S1", "S2@50", "S4@10", "S4@50")),
        ("All nested queries", pk["nested"], ("S0", "S1", "S2@50", "S4@10", "S4@50")),
    ]
    for title, means, systems in panels:
        ns = "/".join(str(means["S0"]["ndcg@10"][g]["n"]) for g in KS)
        L += ["\\midrule", f"\\multicolumn{{6}}{{l}}{{\\emph{{{title}}} ($n$ = {ns})}} \\\\"]
        for s in systems:
            L.append(
                f"{tsys(s)} & "
                + " & ".join(tnum(means[s]["ndcg@10"][g]["mean"]) for g in KS)
                + " \\\\"
            )
    L += ["\\bottomrule", "\\end{tabular}"]
    L.append(
        "\\caption{Absolute nDCG@10 by number of constraints $k$ (expanded labels); $n$ is "
        "all/$k{=}1$/$k{=}2$/$k{=}3$/$k{=}4$. "
        "S2$'$ applies S2's description verdicts to S4@50's top 10.}"
    )
    L += ["\\label{tab:rc-perk}", "\\end{table}"]
    return "\n".join(L) + "\n"


# ---- main --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--tables-dir", type=Path, help="also copy the .tex tables here")
    ap.add_argument("--sources-file", help="the test set (as systems.py --sources-file)")
    args = ap.parse_args(argv)
    D = load(args.n_boot, args.sources_file)
    acc, rates = accepted_ids(D)
    res = {
        "n_boot": args.n_boot,
        "coverage": D["coverage"],
        "frames": check_frames(D),
        "tost": check_tost(D),
        "acceptance": rates,
        "accepted": check_accepted(D, acc),
        "per_k": check_per_k(D),
        "pool": check_pool(D),
    }
    res["regression"] = regression_checks(D, res)
    (OUT / "tables").mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "report.md").write_text(report(res))
    tables = {
        "review_frames.tex": table_frames(res),
        "review_accepted.tex": table_accepted(res),
        "review_per_k.tex": table_per_k(res),
    }
    for name, text in tables.items():
        (OUT / "tables" / name).write_text(text)
        if args.tables_dir:
            args.tables_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(OUT / "tables" / name, args.tables_dir / name)
    print(f"wrote {OUT}/report.md, {OUT}/summary.json, {OUT}/tables/ ({', '.join(tables)})")


if __name__ == "__main__":
    main()
