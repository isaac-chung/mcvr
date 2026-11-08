"""Human labels vs model labels (human_label_plan.md, "What we report"). FREE: no model calls;
reads data/mc/full/human/{key.json, taskA.csv, taskA_overlap.csv, taskB.csv}, results/full/runs/twin/S5.jsonl
and data/mc/full/judge/qrels_expanded.jsonl.

    uv run python paper/tab_human.py [--human data/mc/full/human]

Task A (one item per source; human/make_label_sheets.py) gives a met / not met / can't
tell label for every (video, fact) on the item. From it:

1. Per constraint, people vs each frame judge and vs both judges together (yes = every judge
   says yes, as the expanded labels): precision and recall of the judge's yes against the
   human "met", Cohen's kappa on met vs not met. Judge "unsure" counts as not yes (as
   src.judge_frames.relevant); human "can't tell" is left out.
2. Per (query, video): a video meets the query when every one of the query's facts is met
   (undecided when one is can't tell and none is not met). Against the judges' expanded label
   and the S5 frame verifier (score >= t of 10 for t in 10 / 9 / 8, on the candidates it scored).
3. S5 vs S4@50 at rank 1 under human labels next to two model labels: the eval's expanded
   qrels (data/mc/full/judge/qrels_expanded.jsonl: the source always relevant, the twin never,
   any other judged video when both judges say yes to every fact) and the judges alone (both
   say yes to every fact, on any video, the source and twin included): the share of queries
   whose top 1 meets the query per system, the paired difference S5 - S4@50 with a 95% cluster
   bootstrap CI over sources (src.stats.bootstrap_ci, seed 0), and the win rate (S5 right,
   S4@50 wrong) / (exactly one right) with the same resampling. Overall and per k.
4. Twin validity per twin query: d met on the source, d not met on the twin, every shared fact
   of the query met on the twin; per k, under human and model labels.
5. Agreement: the second annotator's overlap items vs the first annotator, per constraint
   (kappa on the three labels and on met vs the rest, raw agreement, the disagreements).

Task B (one source video per query the two frame judges reject; present / absent / can't tell
for every fact of the query):

6. The share of rejected facts (any judge not yes) that people find present (judges too
   strict) vs absent (the mined fact is wrong), overall, by set, k, rejection pattern (which
   judges said no) and fact type; the same share for the facts both judges accepted. Can't
   tell is left out of the shares. Per query: the share of rejected sources that meet every
   fact for people, and source acceptance per set x k corrected with it: (judge-accepted +
   rejected x that share) / judged, from the stratum counts in key.json (each stratum's own
   share, and the share pooled over k within the set). Only eval queries count (those in
   results/full/runs/*/S0.jsonl): the draw also reached the queries of a source the eval drops
   (missing from the index); those items are left out and the stratum counts recomputed from
   the eval queries (make_label_sheet.load: judged.jsonl and the same fact keys).

The first annotator's taskA.csv is the human label everywhere except 5. Writes
results/full/human/{report.md, summary.json, tables/human_{top1,agree,k}.tex}; with --tables-dir,
copies the .tex tables there.

On the 100-source sheets (--human data/mc/full/human100, key.json "plan" =
human_labeling_100src.md; docs/human_labeling_100src.md) it reports which CSVs exist and, once
taskA, taskBD and taskC are exported, writes results/full/human100/{report.md, summary.json}:
S5 vs S4@50 at rank 1 on all 235 disagreements by set and k; judges vs people per fact off the
source video (the Task A source column is the Task B+D answer) and per (query, video); HN
validity over Task A's near-miss items and Task C (all 98 sources); every fact on its source
(Task B+D) with judge vs human query acceptance on all 722 queries, also with the facts noted
as borderline counted absent (BORDERLINE); Task E (does the index description state each fact:
stated / implied / absent, by fact type, corpus part and description length, per query by set
and k, and against the text-only fact judge of results/full/fact_filter_judge); kappa on the
Task A, B+D and E overlaps; the Task C overlap (taskC_overlap.csv, every Task C item: kappa,
met vs rest with a source bootstrap CI, and HN validity under each annotator's labels); and tables/human_{top1,agree,k}.tex (--tables-dir copies them).
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

from src.datasets.mc import read_jsonl
from src.grounding_v2 import cohen_kappa
from src.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MS = _load(ROOT / "human/make_label_sheets.py", "_an_make_label_sheet")

OUT = ROOT / "results/full/human"
S5_RUN = ROOT / "results/full/runs/twin/S5.jsonl"
QRELS = ROOT / "data/mc/full/judge/qrels_expanded.jsonl"
RUNS = ROOT / "results/full/runs"
H2H = ("human", "expanded", "judges")
N_BOOT = 10000
THRESHOLDS = (10, 9, 8)
KS = ("all", "1", "2", "3", "4")
SET_LABEL = {"natural": "nested", "twin": "twin"}


# ---- inputs ------------------------------------------------------------------------------------


def read_labels(path: Path, items: dict) -> tuple[str, dict, list]:
    """(annotator, {(item, video, constraint): answer}, notes) of one exported sheet."""
    rows = list(csv.DictReader(path.open(newline="")))
    who = {r["annotator"] for r in rows}
    assert len(who) == 1, f"{path}: annotators {who}"
    labels, notes = {}, []
    for r in rows:
        if r["question"] == "note":
            notes.append((r["item"], r["answer"]))
            continue
        it = items[r["item"]]
        c = int(r["constraint"])
        assert r["video"] in it["videos"] and 1 <= c <= len(it["facts"]), r
        labels[(r["item"], r["video"], c)] = r["answer"]
    return who.pop(), labels, notes


def judge_yes(item: dict, video: str, c: int, spec: str | None) -> bool | None:
    """One judge's (or with spec None, every judge's) yes on a fact; None when unjudged."""
    answers = item["videos"][video]["judges"]
    if answers is None:
        return None
    a = answers[c - 1]
    return a[spec] == "yes" if spec else all(v == "yes" for v in a.values())


def meets(labels: dict, item: str, video: str, facts: list[int]) -> bool | None:
    got = [labels[(item, video, c)] for c in facts]
    if "not_met" in got:
        return False
    return True if all(g == "met" for g in got) else None


# ---- statistics --------------------------------------------------------------------------------


def binary(truth: list[bool], pred: list[bool]) -> dict:
    """Precision / recall of pred against truth, kappa, raw agreement."""
    tp = sum(t and p for t, p in zip(truth, pred, strict=True))
    fp = sum(p and not t for t, p in zip(truth, pred, strict=True))
    fn = sum(t and not p for t, p in zip(truth, pred, strict=True))
    n = len(truth)
    return {
        "n": n,
        "human_yes": sum(truth),
        "model_yes": sum(pred),
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "agreement": sum(t == p for t, p in zip(truth, pred, strict=True)) / n if n else None,
        "kappa": cohen_kappa([str(t) for t in truth], [str(p) for p in pred]),
    }


def mean_ci(values: list[float], clusters: list[str]) -> dict:
    if not values:
        return {"n": 0}
    lo, hi = bootstrap_ci(np.array(values, float), clusters, n=N_BOOT, seed=0)
    return {
        "n": len(values),
        "sources": len(set(clusters)),
        "mean": float(np.mean(values)),
        "ci_low": lo,
        "ci_high": hi,
    }


def ratio_ci(num: list[float], den: list[float], clusters: list[str]) -> dict:
    """sum(num) / sum(den) with a 95% percentile CI, resampling clusters as bootstrap_ci."""
    groups: dict[str, list[float]] = {}
    for a, b, c in zip(num, den, clusters, strict=True):
        g = groups.setdefault(c, [0.0, 0.0])
        g[0] += a
        g[1] += b
    sums = np.array(list(groups.values()))
    if not sums[:, 1].sum():
        return {"n": 0}
    idx = np.random.default_rng(0).integers(0, len(sums), size=(N_BOOT, len(sums)))
    n_, d_ = sums[idx, 0].sum(axis=1), sums[idx, 1].sum(axis=1)
    rates = n_[d_ > 0] / d_[d_ > 0]
    return {
        "n": int(sums[:, 1].sum()),
        "wins": int(sums[:, 0].sum()),
        "rate": float(sums[:, 0].sum() / sums[:, 1].sum()),
        "ci_low": float(np.quantile(rates, 0.025)),
        "ci_high": float(np.quantile(rates, 0.975)),
    }


# ---- checks ------------------------------------------------------------------------------------


def per_constraint(items: dict, labels: dict, specs: list[str]) -> dict:
    out = {}
    for name, spec in [(s, s) for s in specs] + [("both judges", None)]:
        truth, pred, unjudged = [], [], 0
        for (i, v, c), a in sorted(labels.items()):
            if a == "cant_tell":
                continue
            y = judge_yes(items[i], v, c, spec)
            if y is None:
                unjudged += 1
                continue
            truth.append(a == "met")
            pred.append(y)
        out[name] = binary(truth, pred) | {"unjudged": unjudged}
    out["cant_tell"] = sum(a == "cant_tell" for a in labels.values())
    return out


def query_rows(items: dict, labels: dict, s5: dict, qrels: dict) -> list[dict]:
    """One row per (query, video) of each item: human and model 'meets the query'."""
    rows = []
    for i, it in items.items():
        for q in it["queries"]:
            for v, info in it["videos"].items():
                judged = info["judges"] is not None
                rows.append(
                    {
                        "item": i,
                        "query_id": q["query_id"],
                        "source": it["source"],
                        "k": q["k"],
                        "video": v,
                        "role": info["role"],
                        "top1_of": [s for s, lab in q["top1"].items() if lab == v],
                        "human": meets(labels, i, v, q["facts"]),
                        "judges": judged and all(judge_yes(it, v, c, None) for c in q["facts"]),
                        "expanded": info["video"] in qrels[q["query_id"]],
                        "judged": judged,
                        "s5": s5[q["query_id"]].get(info["video"]),
                    }
                )
    return rows


def per_query_video(rows: list[dict]) -> dict:
    dec = [r for r in rows if r["human"] is not None]
    jud = [r for r in dec if r["judged"]]
    out = {
        "pairs": len(rows),
        "undecided": len(rows) - len(dec),
        "both judges (expanded)": binary([r["human"] for r in jud], [r["judges"] for r in jud]),
    }
    scored = [r for r in dec if r["s5"] is not None]
    for t in THRESHOLDS:
        out[f"S5 score >= {t}"] = binary(
            [r["human"] for r in scored], [r["s5"] >= t for r in scored]
        )
    out["S5 unscored"] = len(dec) - len(scored)
    return out


def head_to_head(rows: list[dict]) -> dict:
    by_q: dict[str, dict] = {}
    for r in rows:
        for s in r["top1_of"]:
            by_q.setdefault(r["query_id"], {"k": r["k"], "source": r["source"]})[s] = r
    out = {}
    for kind in H2H:
        out[kind] = {}
        for g in KS:
            qs = [q for q in by_q.values() if g == "all" or str(q["k"]) == g]
            ok = [q for q in qs if q["S5"][kind] is not None and q["S4@50"][kind] is not None]
            src = [q["source"] for q in ok]
            a5 = [float(q["S5"][kind]) for q in ok]
            a4 = [float(q["S4@50"][kind]) for q in ok]
            out[kind][g] = {
                "queries": len(qs),
                "undecided": len(qs) - len(ok),
                "S5": mean_ci(a5, src),
                "S4@50": mean_ci(a4, src),
                "S5 - S4@50": mean_ci([x - y for x, y in zip(a5, a4, strict=True)], src),
                "win_rate": ratio_ci(
                    [float(x > y) for x, y in zip(a5, a4, strict=True)],
                    [float(x != y) for x, y in zip(a5, a4, strict=True)],
                    src,
                ),
            }
    unjudged = sum(not q[s]["judged"] for q in by_q.values() for s in ("S5", "S4@50"))
    out["unjudged_top1"] = unjudged
    return out


def twin_validity(items: dict, labels: dict) -> dict:
    def lab(kind, it, i, v, c):
        if kind == "human":
            return labels[(i, v, c)]
        y = judge_yes(it, v, c, None)
        return "met" if y else "not_met"

    out = {}
    for kind in ("human", "model"):
        rows = []
        for i, it in items.items():
            src = next(v for v, x in it["videos"].items() if x["role"] == "source")
            twn = next(v for v, x in it["videos"].items() if x["role"] == "twin")
            for q in it["queries"]:
                d = q["facts"][q["d_index"] - 1]
                shared = [c for c in q["facts"] if c != d]
                assert it["facts"][d - 1]["role"] == "distinct", q["query_id"]
                r = {
                    "k": q["k"],
                    "source": it["source"],
                    "d_in_source": lab(kind, it, i, src, d) == "met",
                    "d_absent_twin": lab(kind, it, i, twn, d) == "not_met",
                    "shared_in_twin": all(lab(kind, it, i, twn, c) == "met" for c in shared),
                }
                r["valid"] = r["d_in_source"] and r["d_absent_twin"] and r["shared_in_twin"]
                rows.append(r)
        out[kind] = {}
        for g in KS:
            rs = [r for r in rows if g == "all" or str(r["k"]) == g]
            src = [r["source"] for r in rs]
            out[kind][g] = {
                c: mean_ci([float(r[c]) for r in rs], src)
                for c in ("d_in_source", "d_absent_twin", "shared_in_twin", "valid")
            }
    return out


def overlap(items: dict, first: dict, second: dict) -> dict:
    keys = sorted(second)
    missing = [k for k in keys if k not in first]
    keys = [k for k in keys if k in first]
    a, b = [first[k] for k in keys], [second[k] for k in keys]
    return {
        "n": len(keys),
        "items": len({k[0] for k in keys}),
        "missing_in_first": len(missing),
        "agreement": sum(x == y for x, y in zip(a, b, strict=True)) / len(keys),
        "kappa_3way": cohen_kappa(a, b),
        "kappa_met": cohen_kappa([str(x == "met") for x in a], [str(y == "met") for y in b]),
        "first": dict(Counter(a)),
        "second": dict(Counter(b)),
        "disagreements": [
            {
                "item": i,
                "video": v,
                "constraint": c,
                "fact": items[i]["facts"][c - 1]["text"],
                "first": first[(i, v, c)],
                "second": second[(i, v, c)],
            }
            for i, v, c in keys
            if first[(i, v, c)] != second[(i, v, c)]
        ],
    }


def eval_strata(eval_ids: set[str]) -> dict:
    """{set: {k: {"judged", "rejected"}}} over the eval queries, as make_label_sheet counts."""
    D = MS.load(yaml.safe_load(MS.CONFIG.read_text()))
    out: dict[str, dict[str, Counter]] = {}
    for q, r in D["rows"].items():
        own = D["judged"].get(r["source"], {}).get(r["source"])
        if q not in eval_ids or own is None:
            continue
        c = out.setdefault(SET_LABEL[r["set"]], {}).setdefault(str(r["k"]), Counter())
        c["judged"] += 1
        c["rejected"] += not MS.relevant(own, D["of_query"][q], D["specs"])
    return {v: {k: dict(by[k]) for k in sorted(by)} for v, by in sorted(out.items())}


def task_b(items: dict, labels: dict, strata: dict) -> dict:
    facts = []
    for i, it in items.items():
        judges = it["videos"]["S"]["judges"]
        for c, f in enumerate(it["facts"], 1):
            no = sorted(s.split(":")[-1] for s, a in judges[c - 1].items() if a != "yes")
            facts.append(
                {
                    "item": i,
                    "set": SET_LABEL[it["set"]],
                    "k": str(it["k"]),
                    "source": it["source"],
                    "type": f["type"],
                    "text": f["text"],
                    "rejected": c in it["rejected"],
                    "pattern": " + ".join(no) or "accepted",
                    "human": labels[(i, "S", c)],
                }
            )

    def present(rs):
        rs = [r for r in rs if r["human"] != "cant_tell"]
        return mean_ci([float(r["human"] == "present") for r in rs], [r["source"] for r in rs])

    rej = [r for r in facts if r["rejected"]]
    groups = {"all": rej}
    groups |= {f"set {v}": [r for r in rej if r["set"] == v] for v in ("nested", "twin")}
    groups |= {f"k {k}": [r for r in rej if r["k"] == k] for k in KS[1:]}
    groups |= {
        f"rejected by {p}": [r for r in rej if r["pattern"] == p]
        for p in sorted({r["pattern"] for r in rej})
    }
    groups |= {
        f"type {t}": [r for r in rej if r["type"] == t] for t in sorted({r["type"] for r in rej})
    }
    out = {
        "facts": len(facts),
        "rejected_facts": len(rej),
        "labels": dict(Counter(r["human"] for r in facts)),
        "rejected_labels": dict(Counter(r["human"] for r in rej)),
        "rejected_present": {g: present(rs) | {"facts": len(rs)} for g, rs in groups.items()},
        "accepted_present": present([r for r in facts if not r["rejected"]]),
        "absent": [
            {"item": r["item"], "fact": r["text"], "by": r["pattern"]}
            for r in rej
            if r["human"] == "absent"
        ],
    }
    # per query: does the rejected source meet every fact for people?
    q = {}
    for i, it in items.items():
        got = [labels[(i, "S", c)] for c in range(1, len(it["facts"]) + 1)]
        meets_ = False if "absent" in got else (True if all(g == "present" for g in got) else None)
        q[i] = {
            "set": SET_LABEL[it["set"]],
            "k": str(it["k"]),
            "source": it["source"],
            "meets": meets_,
        }
    corr = {}
    for v, by_k in strata.items():
        dec = [r for r in q.values() if r["set"] == v and r["meets"] is not None]
        pooled = sum(r["meets"] for r in dec) / len(dec) if dec else None
        corr[v] = {}
        for k, st in by_k.items():
            ds = [r for r in dec if r["k"] == k]
            own = sum(r["meets"] for r in ds) / len(ds) if ds else None
            judged, rejected = st["judged"], st["rejected"]
            acc = (judged - rejected) / judged
            corr[v][k] = {
                "judged": judged,
                "rejected": rejected,
                "labelled": len(ds),
                "rejected_meet": own,
                "judge_acceptance": acc,
                "corrected": acc + rejected / judged * own if own is not None else None,
                "corrected_pooled": acc + rejected / judged * pooled
                if pooled is not None
                else None,
            }
        st = corr[v].values()
        judged = sum(c["judged"] for c in st)
        corr[v]["all"] = {
            "judged": judged,
            "rejected": sum(c["rejected"] for c in st),
            "labelled": sum(c["labelled"] for c in st),
            "rejected_meet": pooled,
            "judge_acceptance": sum(c["judge_acceptance"] * c["judged"] for c in st) / judged,
            "corrected": sum(c["corrected"] * c["judged"] for c in st) / judged
            if all(c["corrected"] is not None for c in st)
            else None,
            "corrected_pooled": sum(c["corrected_pooled"] * c["judged"] for c in st) / judged
            if pooled is not None
            else None,
        }
    rs = [r for r in q.values() if r["meets"] is not None]
    out["queries"] = len(q)
    out["queries_undecided"] = len(q) - len(rs)
    out["rejected_sources_meet"] = mean_ci(
        [float(r["meets"]) for r in rs], [r["source"] for r in rs]
    )
    out["acceptance"] = corr
    return out


# ---- report ------------------------------------------------------------------------------------


def f(x, p=3) -> str:
    return "-" if x is None else f"{x:.{p}f}"


def ci(c: dict) -> str:
    if not c.get("n"):
        return "-"
    m = c.get("mean", c.get("rate"))
    return f"{m:.3f} [{c['ci_low']:.3f}, {c['ci_high']:.3f}]"


def report(res: dict) -> str:
    A = res["taskA"]
    L = ["# Human labels (Task A)", ""]
    L += [
        (
            f"Annotator {A['annotator']}: {A['items']} items (sources), {A['queries']} queries, "
            f"{A['labels']} (video, fact) labels, {A['per_constraint']['cant_tell']} can't tell."
        ),
        "",
        "## 1. Per constraint: judges vs people (human 'met' as truth)",
        "",
        "| judge | n | human met | judge yes | precision | recall | agreement | kappa | unjudged |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, c in A["per_constraint"].items():
        if isinstance(c, dict):
            L.append(
                f"| {name} | {c['n']} | {c['human_yes']} | {c['model_yes']} | {f(c['precision'])} "
                f"| {f(c['recall'])} | {f(c['agreement'])} | {f(c['kappa'])} | {c['unjudged']} |"
            )
    P = A["per_query_video"]
    L += [
        "",
        "## 2. Per (query, video): meets every fact of the query",
        "",
        (
            f"{P['pairs']} pairs, {P['undecided']} undecided (a can't tell and no not met); S5 has no "
            f"score for {P['S5 unscored']} (outside that query's S4@50 top 10)."
        ),
        "",
        "| model label | n | human yes | model yes | precision | recall | agreement | kappa |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in P.items():
        if isinstance(c, dict):
            L.append(
                f"| {name} | {c['n']} | {c['human_yes']} | {c['model_yes']} | {f(c['precision'])} "
                f"| {f(c['recall'])} | {f(c['agreement'])} | {f(c['kappa'])} |"
            )
    H = A["head_to_head"]
    L += [
        "",
        "## 3. S5 vs S4@50 at rank 1 (share of queries whose top 1 meets the query)",
        "",
        (
            "expanded = the eval's qrels (source always relevant, twin never, other videos when "
            "both judges say yes to every fact); judges = both judges yes to every fact on any "
            f"video. Unjudged top-1 videos count as not relevant ({H['unjudged_top1']} of them). "
            "95% cluster bootstrap CIs over sources."
        ),
        "",
        "| labels | k | queries | S5 | S4@50 | S5 - S4@50 | S5 win rate (wins / decisive) |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in H2H:
        for g in KS:
            c = H[kind][g]
            w = c["win_rate"]
            L.append(
                f"| {kind} | {g} | {c['queries']} | {ci(c['S5'])} | {ci(c['S4@50'])} | "
                f"{ci(c['S5 - S4@50'])} | {ci(w)} ({w.get('wins', 0)} / {w.get('n', 0)}) |"
            )
    T = A["twin_validity"]
    L += [
        "",
        "## 4. Twin validity (per twin query)",
        "",
        "| labels | k | d in source | d absent from twin | shared facts in twin | all three |",
        "|---|---|---|---|---|---|",
    ]
    for kind in ("human", "model"):
        for g in KS:
            c = T[kind][g]
            L.append(
                f"| {kind} | {g} (n={c['valid']['n']}) | {ci(c['d_in_source'])} | "
                f"{ci(c['d_absent_twin'])} | {ci(c['shared_in_twin'])} | {ci(c['valid'])} |"
            )
    L += ["", "## 5. Agreement between annotators (overlap)", ""]
    O = res.get("overlap")
    if O is None:
        L.append("taskA_overlap.csv not found.")
    else:
        L += [
            (
                f"{O['annotator']} vs {A['annotator']}: {O['n']} labels on {O['items']} items; raw "
                f"agreement {f(O['agreement'])}, kappa {f(O['kappa_3way'])} (met / not met / can't "
                f"tell), {f(O['kappa_met'])} (met vs the rest). Labels: first {O['first']}, second "
                f"{O['second']}."
            ),
            "",
            "| item | video | fact | first | second |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {d['item']} | {d['video']} | {d['fact']} | {d['first']} | {d['second']} |"
            for d in O["disagreements"]
        ]
    B = res.get("taskB")
    L += ["", "## 6. Task B: sources the judges rejected", ""]
    if B is None:
        L.append("taskB.csv not found.")
    else:
        L += [
            (
                f"{B['drawn']} drawn; left out (not eval queries: their source is missing from the "
                f"index): {', '.join(f'{i} {q}' for i, q in B['dropped_not_in_eval']) or 'none'}."
            ),
            "",
            (
                f"{B['queries']} queries, {B['facts']} facts, {B['rejected_facts']} rejected by at "
                f"least one judge. Labels on the rejected facts: {B['rejected_labels']}; on all "
                f"facts: {B['labels']}."
            ),
            "",
            "Share of rejected facts that people find present (can't tell left out):",
            "",
            "| group | rejected facts | present |",
            "|---|---|---|",
        ]
        L += [f"| {g} | {c['facts']} | {ci(c)} |" for g, c in B["rejected_present"].items()]
        L += [
            "",
            f"Facts both judges accepted, found present: {ci(B['accepted_present'])}.",
            "",
            "Rejected facts people find absent: "
            + (
                "; ".join(f"{a['item']} '{a['fact']}' (no: {a['by']})" for a in B["absent"])
                or "none"
            )
            + ".",
            "",
            (
                f"Rejected sources that meet every fact of their query for people: "
                f"{ci(B['rejected_sources_meet'])} ({B['queries_undecided']} undecided)."
            ),
            "",
            "Source acceptance (both judges yes to every fact), corrected with the human share:",
            "",
            "| set | k | judged | rejected | labelled | rejected that meet | judges | corrected | corrected (pooled share) |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for v, by_k in B["acceptance"].items():
            for k, c in by_k.items():
                L.append(
                    f"| {v} | {k} | {c['judged']} | {c['rejected']} | {c['labelled']} | "
                    f"{f(c['rejected_meet'], 2)} | {f(c['judge_acceptance'])} | {f(c['corrected'])} "
                    f"| {f(c['corrected_pooled'])} |"
                )
    L += ["", "## Notes", ""]
    for who, notes in res["notes"].items():
        L += [f"- {who} {i}: {t.strip()}" for i, t in notes]
    return "\n".join(L) + "\n"


# ---- LaTeX -------------------------------------------------------------------------------------

HEADER = "% Generated by paper/tab_human.py from the human labels; do not edit by hand.\n"
JUDGE_NAME = {
    "gemini:gemini-3.5-flash-lite": "Gemini",
    "openai:gpt-6-astra": "\\texttt{gpt-6-astra}",
}


def tnum(v: float | None) -> str:
    return "--" if v is None else f"${v:.3f}$"


def tci(c: dict, sign: bool = False) -> str:
    m = c.get("mean", c.get("rate"))
    fmt = "+.3f" if sign else ".3f"
    return f"${m:{fmt}}$\\,{{\\scriptsize[${c['ci_low']:{fmt}}$, ${c['ci_high']:{fmt}}$]}}"


def table_top1(res: dict) -> str:
    H = res["taskA"]["head_to_head"]
    n = H["human"]["all"]["queries"]
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    L += ["\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lcccc}", "\\toprule"]
    L.append("Labels & S5 & S4@50 & S5 $-$ S4@50 & S5 wins \\\\")
    L.append("\\midrule")
    names = {"human": "People", "expanded": "Expanded (eval)", "judges": "Judges, every video"}
    for kind, name in names.items():
        c = H[kind]["all"]
        w = c["win_rate"]
        L.append(
            f"{name} & {tnum(c['S5']['mean'])} & {tnum(c['S4@50']['mean'])} & "
            f"{tci(c['S5 - S4@50'], sign=True)} & {w['wins']}/{w['n']} \\\\"
        )
    L += ["\\bottomrule", "\\end{tabular}}"]
    L.append(
        "\\caption{S5 against S4@50 on the "
        f"{n} twin-both queries where they rank a different video first: the share of queries "
        "whose top video meets every constraint, the paired difference with a 95\\% "
        "cluster-bootstrap interval over source videos, and S5's wins among the queries where "
        "exactly one top video meets the query. Expanded labels count the source as relevant "
        "and the twin as not; judges, every video applies the two judges' fact answers to every "
        "video, the source and twin included.}"
    )
    L += ["\\label{tab:human-top1}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_agree(res: dict) -> str:
    A = res["taskA"]
    pc, pq, O = A["per_constraint"], A["per_query_video"], res.get("overlap")
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
    ]
    L += ["\\begin{tabular}{lcccc}", "\\toprule"]
    L.append("Against people & $n$ & Precision & Recall & $\\kappa$ \\\\")
    L += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Per fact on a video}} \\\\"]

    def row(name, c):
        return (
            f"{name} & {c['n']} & {tnum(c['precision'])} & {tnum(c['recall'])} & "
            f"{tnum(c['kappa'])} \\\\"
        )

    for spec, name in JUDGE_NAME.items():
        L.append(row(name, pc[spec]))
    L.append(row("Both judges", pc["both judges"]))
    if O:
        L.append(f"Second annotator & {O['n']} & -- & -- & {tnum(O['kappa_met'])} \\\\")
    L += ["\\midrule", "\\multicolumn{5}{l}{\\emph{Per video meeting a whole query}} \\\\"]
    L.append(row("Both judges", pq["both judges (expanded)"]))
    for t in THRESHOLDS:
        op = "=" if t == 10 else "\\ge"
        L.append(row(f"S5 score ${op} {t}$", pq[f"S5 score >= {t}"]))
    L += ["\\bottomrule", "\\end{tabular}"]
    L.append(
        "\\caption{Model labels against the first annotator on the Task A videos (people's "
        "\\emph{met} as truth; \\emph{can't tell} left out). Precision and recall are those of "
        "the model's yes. Judge \\emph{unsure} counts as no. A video meets a query when every "
        "constraint of the query is met; S5 scores each video 0--10 for the whole query. Second "
        "annotator: agreement with the first on the overlap items, $\\kappa$ on met against "
        "not met or can't tell.}"
    )
    L += ["\\label{tab:human-agree}", "\\end{table}"]
    return "\n".join(L) + "\n"


def table_k(res: dict) -> str:
    T, B = res["taskA"]["twin_validity"]["human"], res.get("taskB")
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
    ]
    L += ["\\resizebox{\\columnwidth}{!}{%", "\\begin{tabular}{lccccc}", "\\toprule"]
    L.append(
        " & Twin & Rejected facts & \\multicolumn{2}{c}{Nested acceptance} & Twin acceptance \\\\"
    )
    L.append("\\cmidrule(lr){4-5}\\cmidrule(lr){6-6}")
    L.append("$k$ & pairs valid & present & Judges & Corrected & Judges / corr. \\\\")
    L.append("\\midrule")
    for g in KS[1:] + KS[:1]:
        if g == "all":
            L.append("\\midrule")
        tw = T[g]["valid"]
        cells = [f"{tw['mean']:.2f} ({tw['n']})"]
        if B:
            fp = B["rejected_present"][g if g == "all" else f"k {g}"]
            ne, tn = B["acceptance"]["nested"][g], B["acceptance"]["twin"][g]
            cells += [
                f"{fp['mean']:.2f} ({fp['n']})",
                f"{ne['judge_acceptance']:.2f}",
                f"{ne['corrected']:.2f}",
                f"{tn['judge_acceptance']:.2f} / {tn['corrected']:.2f}",
            ]
        else:
            cells += ["--"] * 4
        L.append(f"{'All' if g == 'all' else g} & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}}"]
    L.append(
        "\\caption{Per number of constraints $k$, from people's labels ($n$ in brackets). "
        "Twin pairs valid: $d$ met on the source, $d$ not met on the twin and every shared fact "
        "met on the twin (Task A, 66 twin queries). Rejected facts present: facts at least one "
        "judge rejected on the source that people find present (Task B, \\emph{can't tell} left "
        "out). Source acceptance: the share of queries whose source both judges accept on every "
        "fact; corrected adds the rejected sources that meet every fact for people, at the rate "
        "of their own set and $k$ (4--8 labelled queries each).}"
    )
    L += ["\\label{tab:human-k}", "\\end{table}"]
    return "\n".join(L) + "\n"


TABLES = {"human_top1.tex": table_top1, "human_agree.tex": table_agree, "human_k.tex": table_k}


def table_top1_100(res: dict) -> str:
    H = res["taskA"]["head_to_head"]
    n = H["all"]["human"]["all"]["queries"]
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Labels & Set & S5 & S4@50 & S5 $-$ S4@50 & S5 wins \\\\",
        "\\midrule",
    ]
    rows = [("People", g, "human") for g in SETS] + [
        ("Expanded (eval)", "all", "expanded"),
        ("Judges, every video", "all", "judges"),
    ]
    N = res["taskA"]["head_to_head_no_source"]
    rows += [
        ("People, neither top video the source", "all", "human", N),
        ("Expanded, same queries", "all", "expanded", N),
    ]
    for name, g, kind, *h in rows:
        c = (h[0] if h else H[g])[kind]["all"]
        w = c["win_rate"]
        if name in ("Expanded (eval)", "People, neither top video the source"):
            L.append("\\midrule")
        L.append(
            f"{name if g == 'all' else ''} & {g} ({c['queries']}) & {tnum(c['S5']['mean'])} & "
            f"{tnum(c['S4@50']['mean'])} & {tci(c['S5 - S4@50'], sign=True)} & "
            f"{w['wins']}/{w['n']} \\\\"
        )
    L += [
        "\\bottomrule",
        "\\end{tabular}}",
        (
            f"\\caption{{S5 against S4@50 on all {n} test queries where they rank a different "
            "video first (Task A): the share of queries whose top video meets every constraint, the "
            "paired difference with a 95\\% cluster-bootstrap interval over source videos, and "
            "S5's wins among the queries where exactly one top video meets the query. Expanded "
            "labels count the source as relevant and the HN video as not; judges, every video "
            "applies the two judges' fact answers to every video, the source and HN video "
            "included. Queries with a \\emph{can't tell} on a deciding fact are left out (2). "
            f"S5's top video is the source on {res['taskA']['top1_source']['S5']} queries "
            f"(S4@50's on {res['taskA']['top1_source']['S4@50']}), and "
            f"{res['taskA']['s5_wins_source_first']} of S5's wins are the source; the last two "
            "rows leave out every query where either top video is the source. A top-1 contrast "
            "on these queries, not nDCG@10.}"
        ),
        "\\label{tab:human-top1}",
        "\\end{table}",
    ]
    return "\n".join(L) + "\n"


def table_agree_100(res: dict) -> str:
    A = res["taskA"]
    P = A["per_query_video_off_source"]["all"]
    O = res.get("overlap_taskA", {}).get("all")
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{4pt}",
        "\\begin{tabular}{lcccc}",
        "\\toprule",
        "Against people & $n$ & Precision & Recall & $\\kappa$ \\\\",
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Per fact on a video (Task A, off the source)}} \\\\",
    ]

    def row(name, c):
        return (
            f"{name} & {c['n']} & {tnum(c['precision'])} & {tnum(c['recall'])} & "
            f"{tnum(c['kappa'])} \\\\"
        )

    for spec, name in JUDGE_NAME.items():
        L.append(row(name, A["per_constraint"][spec]))
    L.append(row("Both judges", A["per_constraint"]["both judges"]))
    if O:
        L.append(f"Second annotator & {O['n']} & -- & -- & {tnum(O['kappa_met'])} \\\\")
    L += [
        "\\midrule",
        "\\multicolumn{5}{l}{\\emph{Per video meeting a whole query (off the source)}} \\\\",
        row("Expanded labels", P["expanded qrels"]),
        row("S5 score $= 10$", P["S5 score >= 10"]),
        row("S5 score $\\ge 9$", P["S5 score >= 9"]),
        row("S5 score $\\ge 8$", P["S5 score >= 8"]),
        row("S5 score $\\ge 7$", P["S5 score >= 7"]),
        "\\bottomrule",
        "\\end{tabular}",
        (
            "\\caption{Model labels against the first annotator on the Task A videos other than "
            "the source, that is each disagreement query's two top videos and the HN video, not "
            "the judged pool (people's \\emph{met} as truth; \\emph{can't tell} left out). "
            "Precision and recall are those of the model's yes. Judge \\emph{unsure} counts as "
            "no. A video meets a query when every constraint of the query is met; off the source "
            "the expanded labels are the two judges' answers on every fact. S5 scores each video "
            "0--10 for the whole query, on the videos in that query's S4@50 top 10. Second "
            "annotator: agreement with the first on the 25 overlap items, $\\kappa$ on met against "
            "not met or can't tell.}"
        ),
        "\\label{tab:human-agree}",
        "\\end{table}",
    ]
    return "\n".join(L) + "\n"


def table_k_100(res: dict) -> str:
    T, B, S = res["hn_validity"]["human"], res["taskBD"], res["taskBD_strict"]
    L = [
        HEADER + "\\begin{table}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{lccc}",
        "\\toprule",
        " & HN & Nested acceptance & Near-miss acceptance \\\\",
        "$k$ & valid & Judges / people (strict) & Judges / people (strict) \\\\",
        "\\midrule",
    ]

    def p(x):
        return f"{x:.2f}".lstrip("0") if x < 1 else "1.00"

    for k in KS[1:] + ("all",):
        if k == "all":
            L.append("\\midrule")
        cells = [f"{p(T[k]['valid']['mean'])} ({T[k]['valid']['n']})"]
        for g in ("nested", "near-miss"):
            b, s = B["acceptance"][g][k], S["acceptance"][g][k]
            cells.append(
                f"{p(b['judges']['mean'])} / {p(b['human']['mean'])} ({p(s['human']['mean'])})"
            )
        L.append(f"{'All' if k == 'all' else k} & " + " & ".join(cells) + " \\\\")
    rp, sp = B["rejected_present"]["all"], S["rejected_present"]["all"]
    L += [
        "\\bottomrule",
        "\\end{tabular}}",
        (
            "\\caption{Per number of constraints $k$, from people's labels ($n$ in brackets). HN "
            "valid: $d$ met on the source, $d$ not met on the HN video and every shared fact met on "
            f"the HN video (Task A and C: the HN pair of all {res['hn_validity']['sources']} near-miss "
            f"sources, on {res['hn_validity']['queries']} of their queries; Task A carries only the "
            "disagreement queries). "
            "Source acceptance: the share of queries whose source both judges accept on every fact, "
            "against the share whose every fact people find present on the source (Task B+D, all "
            f"{B['queries']} test queries); strict counts the {len(S['flagged'])} facts the "
            "annotator noted as borderline as absent. Of the facts at least one judge rejected, "
            f"people find {rp['mean'] * 100:.1f}\\% present ({sp['mean'] * 100:.1f}\\% strict; "
            "\\emph{can't tell} left out).}"
        ),
        "\\label{tab:human-k}",
        "\\end{table}",
    ]
    return "\n".join(L) + "\n"


TABLES_100 = {
    "human_top1.tex": table_top1_100,
    "human_agree.tex": table_agree_100,
    "human_k.tex": table_k_100,
}


def status_100src(human: Path, key: dict) -> str:
    """The 100-source sheets (docs/human_labeling_100src.md): what each page needs and what its
    exported CSV holds. The analysis of these sheets (merged Task A, Task B+D, C and E, corrected
    acceptance over all 722 queries, HN validity on all 98 sources, kappa on the three overlaps)
    is a later step; until then this only reports what is missing."""
    L = [
        (
            f"{human}: 100-source sheets (plan {key['plan']}, seed {key['seed']}); analysis of these"
            " sheets is not implemented yet, so nothing is written. Status of the exported CSVs:"
        )
    ]
    pages = list(key["tasks"]) + [f"{t}_overlap" for t in key.get("overlap", {})]
    for page in pages:
        items = key["tasks"][page.removesuffix("_overlap")]
        if page.endswith("_overlap"):
            ids = set(key["overlap"][page.removesuffix("_overlap")])
            items = [it for it in items if it["item"] in ids]
        need = sum(len(it["videos"]) * len(it["facts"]) for it in items)
        path = human / f"{page}.csv"
        if not path.exists():
            L.append(f"  {page}.csv: missing ({len(items)} items, {need} answers needed)")
            continue
        rows = [
            r
            for r in csv.DictReader(path.open(newline=""))
            if r["question"] not in ("note", "askable")
        ]
        who = sorted({r["annotator"] for r in rows})
        L.append(
            f"  {page}.csv: {len(rows)} of {need} answers ({len(items)} items)"
            + (f", annotator {', '.join(who)}" if who else ", empty")
        )
    return "\n".join(L)


# ---- 100-source sheets (docs/human_labeling_100src.md) -----------------------------------------

OUT_100 = ROOT / "results/full/human100"
S5_RUNS = {v: RUNS / v / "S5.jsonl" for v in ("natural", "twin")}
SETS = ("all", "nested", "near-miss")
BD_MEETS = {"present": "met", "absent": "not_met", "cant_tell": "cant_tell"}
# Task B+D facts the annotator flagged as borderline in a note, counted absent in the strict
# variant (item: facts). B55's note confirms its fact (checked = plaid), so it is not here.
BORDERLINE = {
    "B01": [2],  # blue walls are blue-black bands
    "B10": [4],  # partial hand, mostly fingers
    "B21": [4, 7],  # unclear whether the woman wears a dress
    "B25": [2],  # pink very faint
    "B33": [4],  # "two static frames" is a guess
    "B37": [3, 5],  # the green thing may be a table or a box (f3 box, f5 table)
    "B59": [1, 4, 5],  # shirt borderline purple / dark blue
}


def read_page(human: Path, page: str, items: dict) -> tuple[str, dict, list]:
    """read_labels plus the check that the page is complete."""
    who, labels, notes = read_labels(human / f"{page}.csv", items)
    need = {
        (i, v, c)
        for i, it in items.items()
        for v in it["videos"]
        for c in range(1, len(it["facts"]) + 1)
    }
    assert set(labels) == need, f"{page}.csv: {len(need - set(labels))} labels missing"
    return who, labels, notes


def by_set(rows: list[dict], fn) -> dict:
    """fn on the rows of each set (nested / near-miss) and on all of them."""
    return {g: fn([r for r in rows if g == "all" or r["set"] == g]) for g in SETS}


def head_to_head_by_set(rows: list[dict]) -> dict:
    return by_set(rows, head_to_head)


def task_bd(items: dict, labels: dict) -> dict:
    """Task B+D: every fact of every query on all 100 sources, present / absent / can't tell on
    the source. Per fact: judge yes (both judges) against human present. Per query: accepted by
    the judges (no rejected fact) against accepted by people (every fact present)."""
    facts, queries = [], []
    for i, it in items.items():
        judges = it["videos"]["S"]["judges"]
        sets = {}
        for q in it["queries"]:
            for c in q["facts"]:
                sets.setdefault(c, set()).add(SET_LABEL[q["set"]])
        for c, f_ in enumerate(it["facts"], 1):
            no = sorted(s.split(":")[-1] for s, a in judges[c - 1].items() if a != "yes")
            facts.append(
                {
                    "item": i,
                    "c": c,
                    "source": it["source"],
                    "sets": sorted(sets.get(c, ())),
                    "type": f_["type"],
                    "text": f_["text"],
                    "rejected": c in it["rejected"],
                    "pattern": " + ".join(no) or "accepted",
                    "human": labels[(i, "S", c)],
                }
            )
        for q in it["queries"]:
            got = [labels[(i, "S", c)] for c in q["facts"]]
            human = (
                False if "absent" in got else (True if all(g == "present" for g in got) else None)
            )
            queries.append(
                {
                    "query_id": q["query_id"],
                    "set": "near-miss" if q["set"] == "twin" else "nested",
                    "k": str(q["k"]),
                    "source": it["source"],
                    "judges": not q["rejected"],
                    "human": human,
                }
            )

    def present(rs):
        rs = [r for r in rs if r["human"] != "cant_tell"]
        return mean_ci([float(r["human"] == "present") for r in rs], [r["source"] for r in rs])

    rej = [r for r in facts if r["rejected"]]
    acc = [r for r in facts if not r["rejected"]]
    groups = {"all": rej}
    groups |= {
        f"rejected by {p}": [r for r in rej if r["pattern"] == p]
        for p in sorted({r["pattern"] for r in rej})
    }
    groups |= {
        f"type {t}": [r for r in rej if r["type"] == t] for t in sorted({r["type"] for r in rej})
    }
    dec = [r for r in facts if r["human"] != "cant_tell"]
    out = {
        "items": len(items),
        "facts": len(facts),
        "rejected_facts": len(rej),
        "labels": dict(Counter(r["human"] for r in facts)),
        "rejected_labels": dict(Counter(r["human"] for r in rej)),
        "rejected_present": {g: present(rs) | {"facts": len(rs)} for g, rs in groups.items()},
        "accepted_present": present(acc) | {"facts": len(acc)},
        "per_fact": binary(
            [r["human"] == "present" for r in dec], [not r["rejected"] for r in dec]
        ),
        "not_present": [
            {
                "item": r["item"],
                "c": r["c"],
                "fact": r["text"],
                "human": r["human"],
                "by": r["pattern"],
            }
            for r in facts
            if r["human"] != "present"
        ],
    }
    # per query, set x k: judge acceptance, human acceptance, and how they cross
    acc_q = {}
    for g in SETS:
        acc_q[g] = {}
        for k in KS:
            rs = [
                r for r in queries if (g == "all" or r["set"] == g) and (k == "all" or r["k"] == k)
            ]
            d = [r for r in rs if r["human"] is not None]
            src = [r["source"] for r in d]
            acc_q[g][k] = {
                "queries": len(rs),
                "undecided": len(rs) - len(d),
                "judges": mean_ci([float(r["judges"]) for r in rs], [r["source"] for r in rs]),
                "judges_decided": mean_ci([float(r["judges"]) for r in d], src),
                "human": mean_ci([float(r["human"]) for r in d], src),
                "judge_rejected_human_accepts": sum(r["human"] and not r["judges"] for r in d),
                "judge_accepted_human_rejects": sum(r["judges"] and not r["human"] for r in d),
                "judge_rejected": sum(not r["judges"] for r in rs),
            }
    out["queries"] = len(queries)
    out["acceptance"] = acc_q
    out["per_query"] = binary(
        [r["human"] for r in queries if r["human"] is not None],
        [r["judges"] for r in queries if r["human"] is not None],
    )
    out["human_rejected_queries"] = [
        {"query_id": r["query_id"], "judges": r["judges"], "human": r["human"]}
        for r in queries
        if r["human"] is not True
    ]
    return out


def overlap_split(items: dict, first: dict, second: dict, prefilled: dict) -> dict:
    """overlap() on every cell, on the cells off the source video (the first annotator's Task A
    source cells are their Task B+D answers), and on the cells the overlap page had not
    pre-filled (the first annotator's own pre-fill is not excluded)."""

    def sub(keep):
        s = {k: v for k, v in second.items() if keep(k)}
        return overlap(items, first, s) if s else {"n": 0}

    src = {
        i: next(v for v, x in it["videos"].items() if x["role"] == "source")
        for i, it in items.items()
    }
    return {
        "all": overlap(items, first, second),
        "off source video": sub(lambda k: k[1] != src[k[0]]),
        "not pre-filled": sub(lambda k: f"c|{k[1]}|{k[2]}" not in prefilled.get(k[0], {})),
    }


FACT_JUDGE = ROOT / "results/full/fact_filter_judge"
E_LABELS = ("stated", "implied", "absent")


def task_e(items: dict, labels: dict, bd: dict) -> dict:
    """Task E: does the index description state each fact (stated / implied / absent)? Per fact
    by type and corpus part, per query by set and k (every fact stated, or stated or implied),
    and against the text-only fact judge (results/full/fact_filter_judge, natural facts only;
    matched on source and fact text)."""
    part = {}
    for p in RUNS.glob("*/S0.jsonl"):
        for ln in read_jsonl(p):
            part[ln["source"]] = (
                f"{ln['dataset']} {ln['length_bin']}"
                if ln["dataset"].startswith("vript")
                else ln["dataset"]
            )
    judge = {}
    if (FACT_JUDGE / "judged.json").exists():
        answers = json.loads((FACT_JUDGE / "judged.json").read_text())["judged"]
        for ln in read_jsonl(FACT_JUDGE / "key.jsonl"):
            if ln["source"] in answers:
                judge[(ln["source"], ln["text"])] = answers[ln["source"]]["answers"][ln["pos"] - 1]
    bd_of = {it["source"]: it for it in bd.values()}
    facts, queries = [], []
    for i, it in items.items():
        words = len(it["videos"]["D"]["text"].split())
        in_set: dict[int, set] = {}
        for q in bd_of[it["source"]]["queries"]:
            for c in q["facts"]:
                in_set.setdefault(c, set()).add("near-miss" if q["set"] == "twin" else "nested")
        for c, f_ in enumerate(it["facts"], 1):
            facts.append(
                {
                    "item": i,
                    "source": it["source"],
                    "type": f_["type"],
                    "part": part.get(it["source"], it["dataset"]),
                    "words": words,
                    "sets": in_set.get(c, set()),
                    "human": labels[(i, "D", c)],
                    "judge": judge.get((it["source"], f_["text"])),
                }
            )
        for q in bd_of[it["source"]]["queries"]:
            got = [labels[(i, "D", c)] for c in q["facts"]]
            queries.append(
                {
                    "set": "near-miss" if q["set"] == "twin" else "nested",
                    "k": str(q["k"]),
                    "source": it["source"],
                    "stated": all(g == "stated" for g in got),
                    "stated_or_implied": "absent" not in got,
                    "share_stated": sum(g == "stated" for g in got) / len(got),
                }
            )

    def shares(rs):
        src = [r["source"] for r in rs]
        return {"facts": len(rs)} | {
            a: mean_ci([float(r["human"] == a) for r in rs], src) for a in E_LABELS
        }

    groups = {"all": facts}
    for s in ("nested", "near-miss"):
        groups[f"in {s} queries"] = [r for r in facts if s in r["sets"]]
        groups |= {
            f"in {s} queries, type {t}": [r for r in facts if s in r["sets"] and r["type"] == t]
            for t in sorted({r["type"] for r in facts})
        }
    groups |= {
        f"type {t}": [r for r in facts if r["type"] == t]
        for t in sorted({r["type"] for r in facts})
    }
    groups |= {
        f"part {p}": [r for r in facts if r["part"] == p]
        for p in sorted({r["part"] for r in facts})
    }
    for lo, hi in ((0, 30), (30, 50), (50, 1000)):
        groups[f"description {lo}-{hi} words"] = [r for r in facts if lo <= r["words"] < hi]
    per_query = {}
    for g in SETS:
        per_query[g] = {}
        for k in KS:
            rs = [
                r for r in queries if (g == "all" or r["set"] == g) and (k == "all" or r["k"] == k)
            ]
            src = [r["source"] for r in rs]
            per_query[g][k] = {
                c: mean_ci([float(r[c]) for r in rs], src)
                for c in ("stated", "stated_or_implied", "share_stated")
            }
    jr = [r for r in facts if r["judge"] is not None]
    return {
        "items": len(items),
        "facts": len(facts),
        "labels": dict(Counter(r["human"] for r in facts)),
        "words": {
            "median": float(
                np.median([len(it["videos"]["D"]["text"].split()) for it in items.values()])
            )
        },
        "by_fact": {g: shares(rs) for g, rs in groups.items() if rs},
        "per_query": per_query,
        "vs_fact_judge": {
            "matched": len(jr),
            "crosstab": {
                h: dict(Counter(r["judge"] for r in jr if r["human"] == h)) for h in E_LABELS
            },
            "stated": binary(
                [r["human"] == "stated" for r in jr], [r["judge"] == "yes" for r in jr]
            ),
            "stated_or_implied": binary(
                [r["human"] != "absent" for r in jr], [r["judge"] == "yes" for r in jr]
            ),
        },
    }


def per_query_video_100(rows: list[dict]) -> dict:
    """Per (query, video) against people: the both-judges rule on every video (the source
    included), the eval's expanded qrels (source always relevant, HN never) and S5's score."""
    dec = [r for r in rows if r["human"] is not None]
    out = {"pairs": len(rows), "undecided": len(rows) - len(dec)}
    for name, key in (("judges rule", "judges"), ("expanded qrels", "expanded")):
        b = binary([r["human"] for r in dec], [r[key] for r in dec])
        b["missed"] = sum(r["human"] and not r[key] for r in dec)
        b["added"] = sum(r[key] and not r["human"] for r in dec)
        out[name] = b
    scored = [r for r in dec if r["s5"] is not None]
    for t_ in THRESHOLDS + (7,):
        out[f"S5 score >= {t_}"] = binary(
            [r["human"] for r in scored], [r["s5"] >= t_ for r in scored]
        )
    out["S5 unscored"] = len(dec) - len(scored)
    return out


def off_source_misses(rows: list[dict]) -> dict:
    """Off the source: per set and k, videos people find relevant and how many expanded misses."""
    out = {}
    for r in rows:
        if r["role"] == "source" or r["human"] is not True:
            continue
        c = out.setdefault(f"{r['set']} k{r['k']}", {"relevant": 0, "missed": 0})
        c["relevant"] += 1
        c["missed"] += not r["expanded"]
    return dict(sorted(out.items()))


def kappa_ci(triples: list[tuple]) -> dict:
    """Cohen's kappa of (a, b) with a 95% percentile CI resampling clusters (third element)."""
    by: dict[str, list] = {}
    for a, b, c in triples:
        by.setdefault(c, []).append((a, b))
    groups = list(by.values())
    rng = np.random.default_rng(0)
    ks = []
    for _ in range(N_BOOT):
        pick = [x for g in rng.integers(0, len(groups), len(groups)) for x in groups[g]]
        k_ = cohen_kappa([str(a) for a, _ in pick], [str(b) for _, b in pick])
        if k_ is not None and not np.isnan(k_):
            ks.append(k_)
    return {"ci_low": float(np.quantile(ks, 0.025)), "ci_high": float(np.quantile(ks, 0.975))}


def analyze_100src(human: Path, key: dict) -> dict:
    A = {it["item"]: it for it in key["tasks"]["taskA"]}
    C = {it["item"]: it for it in key["tasks"]["taskC"]}
    BD = {it["item"]: it for it in key["tasks"]["taskBD"]}
    who, a_lab, notes = read_page(human, "taskA", A)
    _, c_lab, c_notes = read_page(human, "taskC", C)
    _, bd_lab, bd_notes = read_page(human, "taskBD", BD)
    res: dict = {"notes": {who: notes + c_notes + bd_notes}}

    # Task A source cells are the Task B answer (pre-filled); check they still agree
    src_of = {it["source"]: i for i, it in BD.items()}
    fact_of = {
        (it["source"], f_["key"]): c for it in BD.values() for c, f_ in enumerate(it["facts"], 1)
    }
    mismatch = []
    for i, it in A.items():
        v = next(v for v, x in it["videos"].items() if x["role"] == "source")
        for c, f_ in enumerate(it["facts"], 1):
            b = bd_lab[(src_of[it["source"]], "S", fact_of[(it["source"], f_["key"])])]
            if BD_MEETS[b] != a_lab[(i, v, c)]:
                mismatch.append((i, c, a_lab[(i, v, c)], b))

    s5 = {ln["query_id"]: ln["trace"]["scores"] for p in S5_RUNS.values() for ln in read_jsonl(p)}
    qrels = {r["query_id"]: set(r["relevant"]) for r in read_jsonl(QRELS)}
    rows = query_rows(A, a_lab, s5, qrels)
    for r in rows:
        r["set"] = A[r["item"]]["set_label"]
    top1: dict[str, dict] = {}
    for r in rows:
        for s in r["top1_of"]:
            top1.setdefault(r["query_id"], {})[s] = r
    src_top1 = {q for q, by in top1.items() if any(r["role"] == "source" for r in by.values())}
    off_src = {k: v for k, v in a_lab.items() if A[k[0]]["videos"][k[1]]["role"] != "source"}
    res["taskA"] = {
        "annotator": who,
        "items": len(A),
        "items_by_set": dict(Counter(it["set_label"] for it in A.values())),
        "queries": sum(len(it["queries"]) for it in A.values()),
        "labels": len(a_lab),
        "labels_off_source": len(off_src),
        "source_vs_taskBD_mismatch": mismatch,
        "per_constraint": per_constraint(A, off_src, key["judges"]),
        "per_query_video": by_set(rows, per_query_video_100),
        "per_query_video_off_source": by_set(
            [r for r in rows if r["role"] != "source"], per_query_video_100
        ),
        "off_source_misses_by_k": off_source_misses(rows),
        "head_to_head": head_to_head_by_set(rows),
        "head_to_head_no_source": head_to_head([r for r in rows if r["query_id"] not in src_top1]),
        "s5_wins_source_first": sum(
            q["S5"]["human"] is True
            and q["S4@50"]["human"] is False
            and q["S5"]["role"] == "source"
            for q in top1.values()
        ),
        "top1_source": {
            s: sum(q[s]["role"] == "source" for q in top1.values()) for s in ("S5", "S4@50")
        },
    }

    # HN validity on all near-miss sources: Task A near-miss items + Task C
    twin_items = {i: it for i, it in A.items() if it["set"] == "twin"} | C
    twin_labels = {k: v for k, v in a_lab.items() if k[0] in twin_items} | c_lab
    res["hn_validity"] = {
        "sources": len({it["source"] for it in twin_items.values()}),
        "queries": sum(len(it["queries"]) for it in twin_items.values()),
        "from_taskA": sum(1 for i in twin_items if i in A),
        "from_taskC": len(C),
    } | twin_validity(twin_items, twin_labels)
    res["hn_invalid"] = [
        {"item": i, "query_id": q["query_id"], "k": q["k"]}
        for i, it in twin_items.items()
        for q in it["queries"]
        if not _hn_valid(it, i, q, twin_labels)
    ]
    res["hn_full_construction"] = {
        "valid": sum(
            _hn_valid(it, i, q, twin_labels, shared_on_source=True)
            for i, it in twin_items.items()
            for q in it["queries"]
        ),
        "queries": res["hn_validity"]["queries"],
    }
    conflicts = []
    for i, it in C.items():
        v = next(v for v, x in it["videos"].items() if x["role"] == "source")
        for c, f_ in enumerate(it["facts"], 1):
            b = bd_lab[(src_of[it["source"]], "S", fact_of[(it["source"], f_["key"])])]
            if BD_MEETS[b] != c_lab[(i, v, c)]:
                conflicts.append(
                    {"item": i, "c": c, "fact": f_["text"], "taskC": c_lab[(i, v, c)], "taskBD": b}
                )
    res["taskC_vs_taskBD_source"] = conflicts
    res["taskBD"] = {"annotator": who} | task_bd(BD, bd_lab)
    flagged = [(i, "S", c) for i, cs in BORDERLINE.items() for c in cs]
    strict = bd_lab | {k: "absent" for k in flagged}
    res["taskBD_strict"] = {
        "flagged": [
            {"item": i, "c": c, "fact": BD[i]["facts"][c - 1]["text"], "label": bd_lab[(i, v, c)]}
            for i, v, c in flagged
        ]
    } | task_bd(BD, strict)

    # overlaps
    if (human / "taskA_overlap.csv").exists():
        ids = set(key["overlap"]["taskA"])
        items = {i: A[i] for i in ids}
        who2, second, n2 = read_page(human, "taskA_overlap", items)
        res["overlap_taskA"] = {"annotator": who2} | overlap_split(
            A, a_lab, second, key["prefill"].get("taskA_overlap", {})
        )
        res["notes"][who2] = n2
    if (human / "taskBD_overlap.csv").exists():
        ids = set(key["overlap"]["taskBD"])
        items = {i: BD[i] for i in ids}
        who3, second, n3 = read_page(human, "taskBD_overlap", items)
        res["overlap_taskBD"] = {"annotator": who3} | overlap(BD, bd_lab, second)
        res["notes"].setdefault(who3, []).extend(n3)
    if (human / "taskC_overlap.csv").exists():
        # every Task C item (key.json "overlap" has no taskC entry)
        who5, second, n5 = read_page(human, "taskC_overlap", C)
        O = res["overlap_taskC"] = {"annotator": who5} | overlap(C, c_lab, second)
        keys = sorted(second)
        O["kappa_met_ci"] = kappa_ci([(c_lab[k] == "met", second[k] == "met", k[0]) for k in keys])
        for d in O["disagreements"]:
            d["role"] = C[d["item"]]["videos"][d["video"]]["role"]
        valid = [
            (i, q["query_id"], _hn_valid(it, i, q, c_lab), _hn_valid(it, i, q, second))
            for i, it in C.items()
            for q in it["queries"]
        ]
        O["hn_validity"] = {
            "queries": len(valid),
            "valid_first": sum(a for *_, a, _ in valid),
            "valid_second": sum(b for *_, b in valid),
            "agree": sum(a == b for *_, a, b in valid),
            "differ": [
                {"item": i, "query_id": q, "first": a, "second": b}
                for i, q, a, b in valid
                if a != b
            ],
        }
        res["notes"].setdefault(who5, []).extend(n5)
    if (human / "taskE.csv").exists():
        E = {it["item"]: it for it in key["tasks"]["taskE"]}
        _, e_lab, e_notes = read_page(human, "taskE", E)
        res["taskE"] = {"annotator": who} | task_e(E, e_lab, BD)
        res["notes"][who].extend(e_notes)
        if (human / "taskE_overlap.csv").exists():
            items = {i: E[i] for i in key["overlap"]["taskE"]}
            who4, second, n4 = read_page(human, "taskE_overlap", items)
            res["overlap_taskE"] = {"annotator": who4} | overlap(E, e_lab, second)
            keys = sorted(second)
            res["overlap_taskE"]["kappa_stated"] = cohen_kappa(
                [str(e_lab[k] == "stated") for k in keys],
                [str(second[k] == "stated") for k in keys],
            )
            res["overlap_taskE"]["kappa_stated_ci"] = kappa_ci(
                [(e_lab[k] == "stated", second[k] == "stated", k[0]) for k in keys]
            )
            res["notes"].setdefault(who4, []).extend(n4)
    res["missing_pages"] = [
        p for p in ("taskE", "taskE_overlap") if not (human / f"{p}.csv").exists()
    ]
    return res


def _hn_valid(it: dict, i: str, q: dict, labels: dict, shared_on_source: bool = False) -> bool:
    src = next(v for v, x in it["videos"].items() if x["role"] == "source")
    twn = next(v for v, x in it["videos"].items() if x["role"] == "twin")
    d = q["facts"][q["d_index"] - 1]
    shared = [c for c in q["facts"] if c != d]
    return (
        labels[(i, src, d)] == "met"
        and labels[(i, twn, d)] == "not_met"
        and all(labels[(i, twn, c)] == "met" for c in shared)
        and (not shared_on_source or all(labels[(i, src, c)] == "met" for c in shared))
    )


def report_100src(res: dict) -> str:
    A = res["taskA"]
    L = ["# Human labels on the 100-source test set (Tasks A-D)", ""]
    L += [
        (
            f"Annotator {A['annotator']}. Task A: {A['items']} items {A['items_by_set']}, "
            f"{A['queries']} disagreement queries, {A['labels']} (video, fact) labels, of which "
            f"{A['labels_off_source']} off the source video (the source column is the Task B+D "
            f"answer; {len(A['source_vs_taskBD_mismatch'])} cells differ from it). Pages not yet "
            f"exported: {', '.join(res['missing_pages']) or 'none'}."
        ),
        "",
        "## 1. S5 vs S4@50 at rank 1 on every top-1 disagreement",
        "",
        (
            "Share of queries whose top 1 meets every fact. expanded = the eval's qrels; judges = "
            "both frame judges say yes to every fact. 95% cluster bootstrap over sources; win "
            "rate = S5 right and S4@50 wrong over the queries where exactly one is right."
        ),
        "",
        "| set | labels | k | queries | S5 | S4@50 | S5 - S4@50 | S5 win rate (wins / decisive) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in SETS:
        H = A["head_to_head"][g]
        for kind in H2H:
            for k in KS:
                c = H[kind][k]
                w = c["win_rate"]
                L.append(
                    f"| {g} | {kind} | {k} | {c['queries']} | {ci(c['S5'])} | {ci(c['S4@50'])} | "
                    f"{ci(c['S5 - S4@50'])} | {ci(w)} ({w.get('wins', 0)} / {w.get('n', 0)}) |"
                )
    L += [
        "",
        "## 2. Task A per constraint, off the source video: judges vs people",
        "",
        "| judge | n | human met | judge yes | precision | recall | agreement | kappa |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, c in A["per_constraint"].items():
        if isinstance(c, dict):
            L.append(
                f"| {name} | {c['n']} | {c['human_yes']} | {c['model_yes']} | {f(c['precision'])} "
                f"| {f(c['recall'])} | {f(c['agreement'])} | {f(c['kappa'])} |"
            )
    L += [
        f"\nCan't tell: {A['per_constraint']['cant_tell']}.",
        "",
        "## 3. Per (query, video): meets the query, people vs model labels",
        "",
        "| set | model label | n | human yes | model yes | precision | recall | agreement | kappa |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for g in SETS:
        for name, c in A["per_query_video"][g].items():
            if isinstance(c, dict):
                L.append(
                    f"| {g} | {name} | {c['n']} | {c['human_yes']} | {c['model_yes']} | "
                    f"{f(c['precision'])} | {f(c['recall'])} | {f(c['agreement'])} | {f(c['kappa'])} |"
                )
    T = res["hn_validity"]
    L += [
        "",
        "",
        (
            "Off the source video (the expanded qrels count the source relevant, so these are "
            "the pairs where they can miss):"
        ),
        "",
        "| set | model label | n | human yes | model yes | precision | recall | kappa | missed | added |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for g in SETS:
        for name, c in A["per_query_video_off_source"][g].items():
            if isinstance(c, dict):
                L.append(
                    f"| {g} | {name} | {c['n']} | {c['human_yes']} | {c['model_yes']} | "
                    f"{f(c['precision'])} | {f(c['recall'])} | {f(c['kappa'])} | "
                    f"{c.get('missed', '')} | {c.get('added', '')} |"
                )
    N = A["head_to_head_no_source"]
    L += [
        "",
        f"Off-source expanded misses by set and k: {A['off_source_misses_by_k']}.",
        "",
        (
            f"Top-1 source: S5 {A['top1_source']['S5']}, S4@50 {A['top1_source']['S4@50']}; S5 "
            f"wins that are the source: {A['s5_wins_source_first']}. Without queries where either "
            f"top video is the source ({N['human']['all']['queries']} queries): people S5 "
            f"{ci(N['human']['all']['S5'])}, S4@50 {ci(N['human']['all']['S4@50'])}, difference "
            f"{ci(N['human']['all']['S5 - S4@50'])}, wins {N['human']['all']['win_rate'].get('wins')}"
            f"/{N['human']['all']['win_rate'].get('n')}; expanded difference "
            f"{ci(N['expanded']['all']['S5 - S4@50'])}."
        ),
        "",
        "## 4. Near-miss (HN) validity on all near-miss sources",
        "",
        (
            f"{T['sources']} sources ({T['from_taskA']} Task A items, {T['from_taskC']} Task C), "
            f"{T['queries']} queries. Valid = d met on the source, d not met on the HN, every "
            "shared fact met on the HN. model = both judges."
        ),
        "",
        "| labels | k | d in source | d absent from HN | shared facts in HN | valid |",
        "|---|---|---|---|---|---|",
    ]
    for kind in ("human", "model"):
        for k in KS:
            c = T[kind][k]
            L.append(
                f"| {kind} | {k} (n={c['valid']['n']}) | {ci(c['d_in_source'])} | "
                f"{ci(c['d_absent_twin'])} | {ci(c['shared_in_twin'])} | {ci(c['valid'])} |"
            )
    L.append(
        f"\nWith the shared facts also met on the source (full construction): "
        f"{res['hn_full_construction']['valid']} of {res['hn_full_construction']['queries']}. "
        f"Task C source cells that differ from Task B+D: {res['taskC_vs_taskBD_source']}."
    )
    L.append(
        "\nInvalid for people: "
        + ("; ".join(f"{r['item']} {r['query_id']}" for r in res["hn_invalid"]) or "none")
        + "."
    )
    B = res["taskBD"]
    L += [
        "",
        "## 5. Task B+D: every fact on its source video",
        "",
        (
            f"{B['items']} sources, {B['facts']} facts ({B['rejected_facts']} rejected by at least "
            f"one judge), {B['queries']} queries. Labels: {B['labels']}; on rejected facts: "
            f"{B['rejected_labels']}."
        ),
        "",
        f"Facts the judges accepted, present for people: {ci(B['accepted_present'])} (n {B['accepted_present']['facts']}).",
        "",
        "Rejected facts present for people (judge error):",
        "",
        "| group | facts | present |",
        "|---|---|---|",
    ]
    L += [f"| {g} | {c['facts']} | {ci(c)} |" for g, c in B["rejected_present"].items()]
    c = B["per_fact"]
    L += [
        "",
        (
            f"Per fact, judges' accept vs human present: n {c['n']}, precision {f(c['precision'])}, "
            f"recall {f(c['recall'])}, kappa {f(c['kappa'])}."
        ),
        "",
        "Facts not marked present:",
        "",
    ]
    L += [
        f"- {r['item']} f{r['c']} ({r['human']}; judges: {r['by']}): {r['fact']}"
        for r in B["not_present"]
    ]
    L += [
        "",
        "Query acceptance (every fact present on the source), judges vs people:",
        "",
        "| set | k | queries | undecided | judges | people | judge-rejected, people accept | judge-accepted, people reject |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for g in SETS:
        for k in KS:
            c = B["acceptance"][g][k]
            L.append(
                f"| {g} | {k} | {c['queries']} | {c['undecided']} | {ci(c['judges'])} | "
                f"{ci(c['human'])} | {c['judge_rejected_human_accepts']} of {c['judge_rejected']} | "
                f"{c['judge_accepted_human_rejects']} |"
            )
    L += [
        "",
        "Queries people do not accept: "
        + "; ".join(
            f"{r['query_id']} ({'undecided' if r['human'] is None else 'rejected'}, judges "
            f"{'accept' if r['judges'] else 'reject'})"
            for r in B["human_rejected_queries"]
        )
        + ".",
        "",
    ]
    S = res["taskBD_strict"]
    L += [
        "",
        "### Strict variant: facts flagged borderline in a note counted absent",
        "",
        "Flagged: "
        + "; ".join(f"{r['item']} f{r['c']} ({r['label']}) {r['fact']}" for r in S["flagged"])
        + ".",
        "",
        "| measure | as labelled | strict |",
        "|---|---|---|",
        f"| rejected facts present | {ci(B['rejected_present']['all'])} | {ci(S['rejected_present']['all'])} |",
        f"| accepted facts present | {ci(B['accepted_present'])} | {ci(S['accepted_present'])} |",
    ]
    for g in SETS:
        for k in KS:
            b, s = B["acceptance"][g][k], S["acceptance"][g][k]
            L.append(
                f"| queries accepted, {g}, k {k} (judges {ci(b['judges'])}) | {ci(b['human'])} | "
                f"{ci(s['human'])} |"
            )
    L += [
        "",
    ]
    E = res.get("taskE")
    if E:
        V = E["vs_fact_judge"]
        L += [
            "",
            "## 5b. Task E: does the index description state the fact?",
            "",
            (
                f"{E['items']} sources, {E['facts']} facts (median description "
                f"{E['words']['median']:.0f} words). Labels: {E['labels']}."
            ),
            "",
            "| facts | n | stated | implied | absent |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {g} | {c['facts']} | {ci(c['stated'])} | {ci(c['implied'])} | {ci(c['absent'])} |"
            for g, c in E["by_fact"].items()
        ]
        L += [
            "",
            "Per query: every fact stated / none absent / mean share stated:",
            "",
            "| set | k | every fact stated | no fact absent | share stated |",
            "|---|---|---|---|---|",
        ]
        for g in SETS:
            for k in KS:
                c = E["per_query"][g][k]
                L.append(
                    f"| {g} | {k} | {ci(c['stated'])} | {ci(c['stated_or_implied'])} | "
                    f"{ci(c['share_stated'])} |"
                )
        L += [
            "",
            (
                f"Against the text-only fact judge (gpt-6-sol, natural facts, {V['matched']} matched): "
                f"human stated vs judge stated kappa {f(V['stated']['kappa'])} (precision "
                f"{f(V['stated']['precision'])}, recall {f(V['stated']['recall'])}); human stated or "
                f"implied vs judge stated kappa {f(V['stated_or_implied']['kappa'])} (recall "
                f"{f(V['stated_or_implied']['recall'])}). Cross-tab human -> judge: {V['crosstab']}."
            ),
        ]
        O = res.get("overlap_taskE")
        if O:
            L += [
                "",
                (
                    f"Task E overlap ({O['annotator']}): {O['n']} facts on {O['items']} sources, "
                    f"agreement {f(O['agreement'])}, kappa {f(O['kappa_3way'])} (3 labels), "
                    f"{f(O['kappa_stated'])} (stated vs rest); first {O['first']}, second {O['second']}."
                ),
            ]
            L += [
                f"- {d['item']} f{d['constraint']}: {d['fact']} ({d['first']} / {d['second']})"
                for d in O["disagreements"]
            ]
    L += [
        "",
        "## 6. Agreement between annotators",
        "",
    ]
    O = res.get("overlap_taskA")
    if O:
        L.append(f"Task A overlap ({O['annotator']} vs {A['annotator']}):\n")
        L += [
            "| cells | n | items | agreement | kappa (3 labels) | kappa (met vs rest) |",
            "|---|---|---|---|---|---|",
        ]
        for name in ("all", "off source video", "not pre-filled"):
            c = O[name]
            if c["n"]:
                L.append(
                    f"| {name} | {c['n']} | {c['items']} | {f(c['agreement'])} | "
                    f"{f(c['kappa_3way'])} | {f(c['kappa_met'])} |"
                )
        L += ["", "| item | video | fact | first | second |", "|---|---|---|---|---|"]
        L += [
            f"| {d['item']} | {d['video']} | {d['fact']} | {d['first']} | {d['second']} |"
            for d in O["all"]["disagreements"]
        ]
    O = res.get("overlap_taskBD")
    if O:
        L += [
            "",
            (
                f"Task B+D overlap ({O['annotator']}): {O['n']} facts on {O['items']} sources, "
                f"agreement {f(O['agreement'])}, kappa {f(O['kappa_3way'])}; labels first "
                f"{O['first']}, second {O['second']}."
            ),
        ]
        L += [
            f"- {d['item']} f{d['constraint']}: {d['fact']} ({d['first']} / {d['second']})"
            for d in O["disagreements"]
        ]
    O = res.get("overlap_taskC")
    if O:
        H, k = O["hn_validity"], O["kappa_met_ci"]
        L += [
            "",
            (
                f"Task C overlap ({O['annotator']} vs {A['annotator']}, every Task C item): "
                f"{O['n']} cells on {O['items']} items, agreement {f(O['agreement'])}, kappa "
                f"{f(O['kappa_3way'])} (3 labels), {f(O['kappa_met'])} (met vs rest; 95% CI "
                f"{f(k['ci_low'])} to {f(k['ci_high'])}, bootstrap over sources); labels first "
                f"{O['first']}, second {O['second']}."
            ),
            "",
            "| item | video | fact | first | second |",
            "|---|---|---|---|---|",
        ]
        L += [
            f"| {d['item']} | {d['video']} ({d['role']}) | f{d['constraint']}: {d['fact']} | "
            f"{d['first']} | {d['second']} |"
            for d in O["disagreements"]
        ]
        L += [
            "",
            (
                f"HN validity on the {H['queries']} Task C queries: valid {H['valid_first']} "
                f"under {A['annotator']}, {H['valid_second']} under {O['annotator']}; the two "
                f"agree on {H['agree']} of {H['queries']}."
            ),
        ]
        L += [
            f"- {d['item']} {d['query_id']}: valid {d['first']} / {d['second']}"
            for d in H["differ"]
        ]
    L += ["", "## Notes", ""]
    for who, notes in res["notes"].items():
        L += [f"- {who} {i}: {t.strip()}" for i, t in notes]
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--human", default="data/mc/full/human")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--tables-dir", type=Path, help="also copy the .tex tables here")
    args = ap.parse_args()

    human = Path(args.human)
    key = json.loads((human / "key.json").read_text())
    if key.get("plan") == "human_labeling_100src.md":
        if not all((human / f"{p}.csv").exists() for p in ("taskA", "taskBD", "taskC")):
            raise SystemExit(status_100src(human, key))
        print(status_100src(human, key).split("\n", 1)[1])
        res = analyze_100src(human, key)
        out = Path(args.out) if args.out != str(OUT) else OUT_100
        out.mkdir(parents=True, exist_ok=True)
        (out / "summary.json").write_text(json.dumps(res, indent=1))
        text = report_100src(res)
        (out / "report.md").write_text(text)
        (out / "tables").mkdir(exist_ok=True)
        for name, make in TABLES_100.items():
            (out / "tables" / name).write_text(make(res))
            if args.tables_dir:
                args.tables_dir.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(out / "tables" / name, args.tables_dir / name)
        print(text)
        return
    items = {it["item"]: it for it in key["tasks"]["taskA"]}
    who, labels, notes = read_labels(human / "taskA.csv", items)
    need = {
        (i, v, c)
        for i, it in items.items()
        for v in it["videos"]
        for c in range(1, len(it["facts"]) + 1)
    }
    assert set(labels) == need, f"taskA.csv: {len(need - set(labels))} labels missing"
    s5 = {ln["query_id"]: ln["trace"]["scores"] for ln in read_jsonl(S5_RUN)}

    qrels = {r["query_id"]: set(r["relevant"]) for r in read_jsonl(QRELS)}
    rows = query_rows(items, labels, s5, qrels)
    res = {
        "taskA": {
            "annotator": who,
            "items": len(items),
            "queries": sum(len(it["queries"]) for it in items.values()),
            "labels": len(labels),
            "per_constraint": per_constraint(items, labels, key["judges"]),
            "per_query_video": per_query_video(rows),
            "head_to_head": head_to_head(rows),
            "twin_validity": twin_validity(items, labels),
        },
        "notes": {who: notes},
    }
    if (human / "taskA_overlap.csv").exists():
        who2, second, notes2 = read_labels(human / "taskA_overlap.csv", items)
        res["overlap"] = {"annotator": who2} | overlap(items, labels, second)
        res["notes"][who2] = notes2
    if (human / "taskB.csv").exists():
        b_items = {it["item"]: it for it in key["tasks"]["taskB"]}
        who3, b_labels, notes3 = read_labels(human / "taskB.csv", b_items)
        need = {(i, "S", c) for i, it in b_items.items() for c in range(1, len(it["facts"]) + 1)}
        assert set(b_labels) == need, f"taskB.csv: {len(need - set(b_labels))} labels missing"
        eval_ids = {ln["query_id"] for p in RUNS.glob("*/S0.jsonl") for ln in read_jsonl(p)}
        dropped = sorted(i for i, it in b_items.items() if it["query_id"] not in eval_ids)
        kept = {i: it for i, it in b_items.items() if i not in dropped}
        res["taskB"] = {
            "annotator": who3,
            "drawn": len(b_items),
            "dropped_not_in_eval": [(i, b_items[i]["query_id"]) for i in dropped],
        } | task_b(kept, b_labels, eval_strata(eval_ids))
        res["notes"].setdefault(who3, []).extend(notes3)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    text = report(res)
    (out / "report.md").write_text(text)
    (out / "tables").mkdir(exist_ok=True)
    for name, make in TABLES.items():
        (out / "tables" / name).write_text(make(res))
        if args.tables_dir:
            args.tables_dir.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(out / "tables" / name, args.tables_dir / name)
    print(text)


if __name__ == "__main__":
    main()
