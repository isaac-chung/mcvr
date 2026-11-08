"""S6 trigger simulation (RQ3) on saved v2 pilot outputs: no API calls.

S6 always runs S4 (holistic rerank of S0's top 20 over descriptions) and pays for S5 (holistic
rerank of S0's top 10 over each candidate's own 0.5 fps frames) only when a cheap trigger fires.
The final ranking is S5 when triggered, else S4. Triggers use only what exists before frames are
paid for:

  s4_margin   S4 top-1 minus top-2 score (0-10)          fire when <= t
  s4_top      S4 top-1 score                             fire when <= t
  s2_unclear  share of S2 "unclear" verdicts on S4's top fire when >= t  (S2 cost added)
  k           constraints in the query                   fire when >= t
  s0_margin   S0 top-1 minus top-2 cosine                fire when <= t
  oracle      S5 nDCG@10 > S4 nDCG@10 (upper bound)      fire when -(gain) <= t

Cost per query (input tokens) comes from measured per-call costs: S4 and S2 description calls at
the set's measured tokens per call (cost.json), a frame call at its measured text overhead plus
389 tokens per image times the query's actual candidate frame count (S5 trace cost).

Queries: every query with both S4 and S5 (40 nested, 146 twin). The subsets are not random:
nested 40 and twin 40 are matched (source, k) pairs (10 per k); twin_both 120 are the twin queries
with both source and twin in S0's top 10 (14 overlap with twin 40).

    uv run python dev/trigger_dev.py
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from pathlib import Path

import numpy as np

from src.metrics import ndcg_at_k, recall_at_k
from src.systems.single import cluster_bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
PILOT = ROOT / "results/smoke_v2/pilot"
QRELS = ROOT / "data/mc/smoke_v2/judge/qrels_expanded.jsonl"
OUT = ROOT / "results/smoke_v2/trigger"
SETS = {"nested": "natural", "twin": "twin"}

# name: (direction, description)
TRIGGERS = {
    "s4_margin": ("le", "S4 top-1 minus top-2 score <= t"),
    "s4_top": ("le", "S4 top-1 score <= t"),
    "s2_unclear": ("ge", "S2 'unclear' share on S4's top-1 >= t"),
    "k": ("ge", "query k >= t"),
    "s0_margin": ("le", "S0 top-1 minus top-2 cosine <= t"),
    "oracle": ("le", "oracle: S5 nDCG@10 better than S4 (feature = -gain)"),
}
SUBSETS = ("nested", "twin_matched", "twin_both", "twin_all", "pooled_matched", "pooled_all")
FRACS = (0.8, 0.9)
CROSSFIT = ("twin_both", "twin_all", "pooled_all")  # subsets where S5 gains (CI > 0)


# ----------------------------------------------------------------------------- pure helpers


def margin(ranking: list) -> float:
    """Top-1 minus top-2 score of a [(doc, score), ...] ranking."""
    return float(ranking[0][1]) - float(ranking[1][1])


def unclear_share(verdicts: dict[str, list[str]], doc: str) -> float | None:
    v = verdicts.get(doc)
    if not v:
        return None
    return sum(1 for x in v if x == "unclear") / len(v)


def fires(value: float, t: float, direction: str) -> bool:
    return value <= t if direction == "le" else value >= t


def thresholds(values: list[float], direction: str) -> list[float]:
    """Never-fire first, then every distinct value, ordered so the share sent grows."""
    uniq = sorted(set(values))
    if direction == "le":
        return [-math.inf, *uniq]
    return [math.inf, *reversed(uniq)]


def kept(q_trig: float, q4: float, q5: float) -> float:
    """Share of S5's gain over S4 kept by the trigger (nan when S5 does not gain)."""
    gain = q5 - q4
    return (q_trig - q4) / gain if gain > 0 else math.nan


def sweep(rows: list[dict], feature: str, direction: str) -> list[dict]:
    """Cost-quality curve over thresholds. Each row needs feature, q4/q5 (nDCG@10), r4/r5 (R@1),
    c_base (always paid) and c5 (extra cost of frames)."""
    n = len(rows)
    q4 = sum(r["q4"] for r in rows) / n
    q5 = sum(r["q5"] for r in rows) / n
    r4 = sum(r["r4"] for r in rows) / n
    r5 = sum(r["r5"] for r in rows) / n
    pts = []
    for t in thresholds([r[feature] for r in rows], direction):
        trig = [fires(r[feature], t, direction) for r in rows]
        q = sum(r["q5"] if f else r["q4"] for r, f in zip(rows, trig, strict=True)) / n
        r1 = sum(r["r5"] if f else r["r4"] for r, f in zip(rows, trig, strict=True)) / n
        cost = sum(r["c_base"] + (r["c5"] if f else 0.0) for r, f in zip(rows, trig, strict=True))
        pts.append(
            {
                "t": t,
                "share": sum(trig) / n,
                "cost": cost / n,
                "ndcg10": q,
                "r1": r1,
                "kept_ndcg": kept(q, q4, q5),
                "kept_r1": kept(r1, r4, r5),
            }
        )
    return pts


def operating_point(pts: list[dict], frac: float) -> dict | None:
    """Cheapest point keeping >= frac of S5's nDCG@10 gain (ties: lower share)."""
    ok = [p for p in pts if not math.isnan(p["kept_ndcg"]) and p["kept_ndcg"] >= frac - 1e-12]
    return min(ok, key=lambda p: (p["cost"], p["share"])) if ok else None


def auc(xs: list[float], ys: list[float]) -> float:
    """Trapezoid area of y over x in [0, 1] (x sorted ascending; y nan -> skipped)."""
    pairs = sorted((x, y) for x, y in zip(xs, ys, strict=True) if not math.isnan(y))
    area = 0.0
    for (x0, y0), (x1, y1) in itertools.pairwise(pairs):
        area += (x1 - x0) * (y0 + y1) / 2
    return area


def auc_summary(pts: list[dict]) -> dict:
    """Area under gain-kept vs share sent, and vs normalised extra cost (random = 0.5; nan when
    S5 does not gain); excess: area of nDCG@10 above the random-trigger line, in nDCG units
    (random = 0; defined whatever the sign of S5's gain)."""
    c0, c1 = pts[0]["cost"], pts[-1]["cost"]
    span = c1 - c0 if c1 > c0 else 1.0
    ys = [p["kept_ndcg"] for p in pts]
    xs = [p["share"] for p in pts]
    q4, q5 = pts[0]["ndcg10"], pts[-1]["ndcg10"]
    nan = all(math.isnan(y) for y in ys)
    return {
        "auc_share": math.nan if nan else auc(xs, ys),
        "auc_cost": math.nan if nan else auc([(p["cost"] - c0) / span for p in pts], ys),
        "excess": auc(xs, [p["ndcg10"] - q4 for p in pts]) - (q5 - q4) / 2,
    }


def crossfit(
    rows: list[dict], feature: str, direction: str, frac: float, folds: int = 5, reps: int = 20
) -> dict:
    """Selection-honest operating point: split sources into folds, pick the cheapest threshold
    keeping >= frac on the other folds, apply it to the held-out fold; pool held-out decisions.
    Returns the mean over repetitions and the 2.5-97.5% spread across repetitions."""
    sources = sorted({r["source"] for r in rows})
    res = {"kept": [], "share": [], "cost": []}
    for rep in range(reps):
        rng = np.random.default_rng(rep)
        order = list(rng.permutation(len(sources)))
        fold = {sources[i]: j % folds for j, i in enumerate(order)}
        trig = {}
        for f in range(folds):
            train = [r for r in rows if fold[r["source"]] != f]
            op = operating_point(sweep(train, feature, direction), frac)
            t = op["t"] if op else (math.inf if direction == "le" else -math.inf)
            for r in rows:
                if fold[r["source"]] == f:
                    trig[r["query_id"]] = fires(r[feature], t, direction)
        n = len(rows)
        q = sum(r["q5"] if trig[r["query_id"]] else r["q4"] for r in rows) / n
        q4 = sum(r["q4"] for r in rows) / n
        q5 = sum(r["q5"] for r in rows) / n
        res["kept"].append(kept(q, q4, q5))
        res["share"].append(sum(trig.values()) / n)
        res["cost"].append(
            sum(r["c_base"] + (r["c5"] if trig[r["query_id"]] else 0.0) for r in rows) / n
        )
    out = {}
    for k, v in res.items():
        a = np.array([x for x in v if not math.isnan(x)])
        out[k] = (
            (float(a.mean()), *map(float, np.percentile(a, [2.5, 97.5])))
            if len(a)
            else (math.nan, math.nan, math.nan)
        )
    return out


def ratio_ci(
    num: list[float], den: list[float], clusters: list[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% cluster-bootstrap CI of sum(num) / sum(den), resampling source videos."""
    groups: dict[str, list[int]] = {}
    for i, c in enumerate(clusters):
        groups.setdefault(c, []).append(i)
    keys = list(groups)
    sn = np.array([sum(num[i] for i in groups[c]) for c in keys])
    sd = np.array([sum(den[i] for i in groups[c]) for c in keys])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(keys), size=(n_boot, len(keys)))
    d = sd[picks].sum(axis=1)
    r = np.where(d > 0, sn[picks].sum(axis=1) / np.where(d > 0, d, 1), np.nan)
    r = r[np.isfinite(r)]
    if not len(r):
        return math.nan, math.nan
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def point_cis(rows: list[dict], feature: str, direction: str, t: float) -> dict:
    trig = [fires(r[feature], t, direction) for r in rows]
    src = [r["source"] for r in rows]
    q = [r["q5"] if f else r["q4"] for r, f in zip(rows, trig, strict=True)]
    d = [qq - r["q4"] for qq, r in zip(q, rows, strict=True)]
    d5 = [r["q5"] - r["q4"] for r in rows]
    cost = [r["c_base"] + (r["c5"] if f else 0.0) for r, f in zip(rows, trig, strict=True)]
    return {
        "ndcg10_ci": cluster_bootstrap_ci(q, src),
        "gain_ci": cluster_bootstrap_ci(d, src),
        "kept_ci": ratio_ci(d, d5, src),
        "share_ci": cluster_bootstrap_ci([float(f) for f in trig], src),
        "cost_ci": cluster_bootstrap_ci(cost, src),
    }


# ----------------------------------------------------------------------------- data


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def by_qid(path: Path) -> dict[str, dict]:
    return {r["query_id"]: r for r in read_jsonl(path)}


def load_grades(rows: dict[str, dict]) -> dict[str, dict[str, int]]:
    """Expanded labels; the source always stays relevant (as pilot_verify.load_qrels)."""
    grades = {q: {r["source"]: 1} for q, r in rows.items()}
    for line in read_jsonl(QRELS):
        q = line["query_id"]
        if q in grades:
            for d in line.get("relevant", []):
                grades[q][d] = 1
    return grades


def rescore_at(first_stage: list[str], scores: dict[str, float], n: int) -> list:
    """Rank S0's top n by score (ties in S0 order), rest in S0 order (for the S4@10 context)."""
    head = sorted(range(n), key=lambda i: (-scores[first_stage[i]], i))
    return [(first_stage[i], 0.0) for i in head] + [(d, 0.0) for d in first_stage[n:]]


def cost_params(set_dir: Path, s5: dict[str, dict], tpi: float) -> dict:
    cj = json.loads((set_dir / "cost.json").read_text())
    calls = sum(r["cost"]["llm_calls"] for r in s5.values())
    toks = sum(r["cost"]["tokens_in"] for r in s5.values())
    imgs = sum(r["cost"]["images"] for r in s5.values())
    return {
        "s4_per_call": cj["S4"]["in_per_call"],
        "s2_per_call": cj["S2"]["in_per_call"],
        "frame_call_text": (toks - tpi * imgs) / calls,
        "tokens_per_image": tpi,
    }


def load_rows() -> tuple[list[dict], dict]:
    tpi = json.loads((PILOT / "frame_rate.json").read_text())["tokens_per_image"]
    matched = json.loads((PILOT / "frame_subset.json").read_text())
    both = set(json.loads((PILOT / "frame_subset_twin_both.json").read_text())["twin"])
    rows, params = [], {}
    for name, d in SETS.items():
        sd = PILOT / d
        s0, s2, s4, s5 = (by_qid(sd / f"{s}.jsonl") for s in ("S0", "S2", "S4", "S5"))
        qids = sorted(set(s4) & set(s5))
        grades = load_grades({q: s4[q] for q in qids})
        cp = cost_params(sd, {q: s5[q] for q in qids}, tpi)
        params[name] = cp
        for q in qids:
            a0, a4, a5 = s0[q]["ranking"], s4[q]["ranking"], s5[q]["ranking"]
            g = grades[q]
            top4 = a4[0][0]
            uc = unclear_share(s2[q]["trace"]["verdicts"], top4) if q in s2 else None
            c5 = s5[q]["cost"]
            frames = c5["llm_calls"] * cp["frame_call_text"] + cp["tokens_per_image"] * c5["images"]
            c4 = s4[q]["cost"]["llm_calls"] * cp["s4_per_call"]
            c2 = s2[q]["cost"]["llm_calls"] * cp["s2_per_call"] if q in s2 else math.nan
            tags = {name}
            if name == "nested":
                tags |= {"pooled_matched", "pooled_all"}
            else:
                tags |= {"twin_all", "pooled_all"}
                if q in matched["twin"]:
                    tags |= {"twin_matched", "pooled_matched"}
                if q in both:
                    tags.add("twin_both")
            s4_10 = rescore_at([d for d, _ in a0[:10]], s4[q]["trace"]["scores"], 10)
            q4, q5 = ndcg_at_k(a4, g, 10), ndcg_at_k(a5, g, 10)
            rows.append(
                {
                    "query_id": q,
                    "set": name,
                    "source": s4[q]["source"],
                    "k_": s4[q]["k"],
                    "tags": tags,
                    "q0": ndcg_at_k(a0, g, 10),
                    "q4": q4,
                    "q5": q5,
                    "q4_10": ndcg_at_k(s4_10, g, 10),
                    "r0": recall_at_k(a0, g, 1),
                    "r4": recall_at_k(a4, g, 1),
                    "r5": recall_at_k(a5, g, 1),
                    "s4_margin": margin(a4),
                    "s4_top": float(a4[0][1]),
                    "s2_unclear": uc if uc is not None else 0.0,
                    "k": float(s4[q]["k"]),
                    "s0_margin": margin(a0),
                    "oracle": -(q5 - q4),
                    "c4": c4,
                    "c2": c2,
                    "c5": frames,
                    "c5_recorded": float(c5["tokens_in"]),
                    "images": c5["images"],
                }
            )
    return rows, params


def with_cost(rows: list[dict], feature: str) -> list[dict]:
    """c_base: S4 always; S2 too when the trigger reads S2 verdicts."""
    extra = feature == "s2_unclear"
    return [{**r, "c_base": r["c4"] + (r["c2"] if extra else 0.0)} for r in rows]


# ----------------------------------------------------------------------------- report


def fmt_ci(ci: tuple[float, float], p: int = 3) -> str:
    return f"[{ci[0]:+.{p}f}, {ci[1]:+.{p}f}]"


def run() -> dict:
    rows, params = load_rows()
    OUT.mkdir(parents=True, exist_ok=True)
    curves, summary = [], {"params": params, "subsets": {}}
    for sub in SUBSETS:
        srows = [r for r in rows if sub in r["tags"]]
        n = len(srows)
        base = {
            "n": n,
            "sources": len({r["source"] for r in srows}),
            "S0": float(np.mean([r["q0"] for r in srows])),
            "S4": float(np.mean([r["q4"] for r in srows])),
            "S4@10": float(np.mean([r["q4_10"] for r in srows])),
            "S5": float(np.mean([r["q5"] for r in srows])),
            "S4_r1": float(np.mean([r["r4"] for r in srows])),
            "S5_r1": float(np.mean([r["r5"] for r in srows])),
            "S5-S4_ci": cluster_bootstrap_ci(
                [r["q5"] - r["q4"] for r in srows], [r["source"] for r in srows]
            ),
            "cost_S4": float(np.mean([r["c4"] for r in srows])),
            "cost_S4+S5": float(np.mean([r["c4"] + r["c5"] for r in srows])),
            "cost_S5_recorded": float(np.mean([r["c5_recorded"] for r in srows])),
            "cost_S5_model": float(np.mean([r["c5"] for r in srows])),
            "frames_per_query": float(np.mean([r["images"] for r in srows])),
        }
        trig_out = {}
        for feat, (direction, _) in TRIGGERS.items():
            frows = with_cost(srows, feat)
            pts = sweep(frows, feat, direction)
            for p in pts:
                curves.append({"subset": sub, "trigger": feat, **p})
            ops = {}
            for f in FRACS:
                op = operating_point(pts, f)
                if op is not None:
                    op = {**op, **point_cis(frows, feat, direction, op["t"])}
                ops[str(f)] = op
            oracle_pt = None
            if feat == "oracle":
                oracle_pt = [p for p in pts if p["t"] < 0]
                oracle_pt = oracle_pt[-1] if oracle_pt else None
                if oracle_pt is not None:
                    oracle_pt = {
                        **oracle_pt,
                        **point_cis(frows, feat, direction, oracle_pt["t"]),
                    }
            cf = (
                {str(f): crossfit(frows, feat, direction, f) for f in FRACS}
                if feat != "oracle" and sub in CROSSFIT
                else {}
            )
            trig_out[feat] = {
                "auc": auc_summary(pts),
                "ops": ops,
                "oracle_point": oracle_pt,
                "crossfit": cf,
            }
        summary["subsets"][sub] = {"base": base, "triggers": trig_out}
    summary["transfer"] = transfer(rows)
    with (OUT / "curves.csv").open("w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subset",
                "trigger",
                "t",
                "share",
                "cost",
                "ndcg10",
                "r1",
                "kept_ndcg",
                "kept_r1",
            ],
        )
        w.writeheader()
        for c in curves:
            w.writerow({k: (round(v, 6) if isinstance(v, float) else v) for k, v in c.items()})
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=_json))
    (OUT / "report.md").write_text(report(summary))
    return summary


def _json(o):
    if isinstance(o, set):
        return sorted(o)
    raise TypeError(type(o))


def apply_t(rows: list[dict], feat: str, t: float) -> dict:
    direction = TRIGGERS[feat][0]
    frows = with_cost(rows, feat)
    n = len(frows)
    trig = [fires(r[feat], t, direction) for r in frows]
    q = sum(r["q5"] if f else r["q4"] for r, f in zip(frows, trig, strict=True)) / n
    q4 = sum(r["q4"] for r in frows) / n
    q5 = sum(r["q5"] for r in frows) / n
    cost = sum(r["c_base"] + (r["c5"] if f else 0) for r, f in zip(frows, trig, strict=True)) / n
    return {"share": sum(trig) / n, "cost": cost, "ndcg10": q, "kept_ndcg": kept(q, q4, q5)}


def transfer(rows: list[dict]) -> list[dict]:
    """Thresholds picked on one subset (80% / 90% operating points) applied to another."""
    out = []
    pairs = (
        ("twin_all", "nested"),
        ("nested", "twin_all"),
        ("twin_matched", "nested"),
        ("twin_both", "twin_matched"),
        ("twin_matched", "twin_both"),
    )
    for tune, test in pairs:
        tr = [r for r in rows if tune in r["tags"]]
        te = [r for r in rows if test in r["tags"] and tune not in r["tags"]]
        for feat, (direction, _) in TRIGGERS.items():
            if feat == "oracle":
                continue
            pts = sweep(with_cost(tr, feat), feat, direction)
            for f in FRACS:
                op = operating_point(pts, f)
                if op is None:
                    continue
                out.append(
                    {
                        "tune": tune,
                        "test": test,
                        "n_test": len(te),
                        "trigger": feat,
                        "frac": f,
                        "t": op["t"],
                        "tune_share": op["share"],
                        **{f"test_{k}": v for k, v in apply_t(te, feat, op["t"]).items()},
                    }
                )
    return out


# Hand-written from the 2026-10-23 run; the tables below are regenerated on every run.
FINDINGS = """## Findings (preliminary: smoke corpus, one verifier, model-only labels)

- Cost: S4 is about 4.1k input tokens per query; S5 adds about 106k (10 frame calls, about 256
  frames per query at 389 tokens each), so always-S5 costs about 27x S4 and cost tracks the share
  of queries sent to frames almost one to one. The cost model reproduces the recorded S5 tokens to
  within 0.4%.
- S5's gain over S4 is clear only where a near-miss is in the pool: twin_both +.057 [+.019, +.101]
  (and twin_all / pooled_all, which it dominates). On the matched samples it is not: nested +.031
  [-.026, +.092], twin_matched -.003 [-.068, +.062]. There, a trigger has no reliable gain to keep
  and the cheapest policy is S4 alone; the nested operating points (ratios over a gain near 0)
  should not be read.
- Best trigger: S4's own confidence. s4_margin (top-1 minus top-2 score) has the highest AUC where
  S5 gains (twin_both .77, pooled_all .74; s4_top .74 / .72; s2_unclear .62 / .64; k .61 / .62;
  s0_margin .62 / .53; random .50). On twin_both, margin <= 2 sends 50% of queries to frames, costs
  54k tokens (49% of always-S5) and keeps .85 [.58, 1.08] of the gain; margin <= 3 sends 64%, 68k,
  keeps .99. Cross-fitted by source (threshold chosen on other folds): 80% target gives share .47,
  kept .77 (s4_top: .49, .85); 90% target gives share .58 to .59, kept .80 to .90.
- Oracle bound: S5 beats S4 on only about 20% of queries; sending only those costs about 25k
  tokens per query (pooled_all) and gives nDCG@10 .883, above always-S5 (.863), because S5 also
  loses on some queries. A real trigger is far from this (AUC .74 vs 1.44 pooled).
- What generalises: the ordering (S4 confidence > S2 unclear, k > S0 margin) holds on every subset
  where S5 gains, and the k and S0-margin triggers save little. Absolute thresholds do not transfer
  reliably: the margin threshold tuned on twins keeps the gain on nested, but s4_top and s2_unclear
  thresholds tuned on nested send 1-10% of twin queries and keep 0-14%. The twin_both evidence is
  conditional on both videos being in S0's top 10, which a deployed trigger cannot see.
"""


def report(s: dict) -> str:
    L = ["# S6 trigger simulation (RQ3), offline on saved v2 pilot outputs", "", FINDINGS]
    L += [
        (
            "Script `dev/trigger_dev.py`; curves `results/smoke_v2/trigger/curves.csv`; "
            "numbers `summary.json`. No API calls. Final ranking = S5 (frames, holistic, S0 top 10) "
            "when the trigger fires, else S4 (descriptions, holistic, S0 top 20). S4 is always paid; "
            "the S2 trigger also pays S2. Labels: expanded (`qrels_expanded.jsonl`, source always "
            "relevant). CIs: 95% cluster bootstrap over source videos (2,000 resamples). Thresholds "
            "are chosen in-sample, so operating points are optimistic; see the transfer table."
        ),
        "",
        (
            "Cost model (input tokens per query): S4 = calls x measured tokens per description call; "
            "frames = calls x measured text overhead per frame call + 389.4 x the query's actual "
            "frame count (all 10 candidates)."
        ),
        "",
        "| set | S4 per call | S2 per call | frame call text | tokens per image |",
        "|---|---|---|---|---|",
    ]
    for k, p in s["params"].items():
        L.append(
            f"| {k} | {p['s4_per_call']:.0f} | {p['s2_per_call']:.0f} | "
            f"{p['frame_call_text']:.0f} | {p['tokens_per_image']:.1f} |"
        )
    L += [
        "",
        "## Subsets (selection is not random)",
        "",
        "- nested: 40 matched (source, k) pairs, 10 per k (cap-limited sample).",
        "- twin_matched: the same 40 pairs on the twin set.",
        (
            "- twin_both: 120 twin queries with source and twin both in S0's top 10 "
            "(14 overlap with twin_matched). Selected on S0's pool, so S4 is already high."
        ),
        (
            "- twin_all = twin_matched + twin_both (146); pooled_matched = nested + twin_matched "
            "(80); pooled_all = 186."
        ),
        "",
        (
            "| subset | n | sources | S0 | S4 | S4@10 | S5 | S5-S4 [CI] | R@1 S4 / S5 | "
            "frames/q | tokens S4 | tokens S4+S5 | S5 recorded / model |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for sub, d in s["subsets"].items():
        b = d["base"]
        L.append(
            f"| {sub} | {b['n']} | {b['sources']} | {b['S0']:.3f} | {b['S4']:.3f} | "
            f"{b['S4@10']:.3f} | {b['S5']:.3f} | {b['S5'] - b['S4']:+.3f} {fmt_ci(b['S5-S4_ci'])} "
            f"| {b['S4_r1']:.3f} / {b['S5_r1']:.3f} | {b['frames_per_query']:.0f} | "
            f"{b['cost_S4']:,.0f} | {b['cost_S4+S5']:,.0f} | "
            f"{b['cost_S5_recorded']:,.0f} / {b['cost_S5_model']:,.0f} |"
        )
    L += [
        "",
        "## AUC summary",
        "",
        (
            "Cells: AUC share / AUC cost / excess. AUC = area under share of S5's nDCG@10 gain "
            "kept against share of queries sent to frames (or extra cost normalised to [0, 1]); "
            "random trigger = 0.50, can exceed 1 when the trigger beats always-S5; n/a when S5 "
            "does not gain. Excess = area of nDCG@10 above the random-trigger line, x 1000 "
            "(random = 0)."
        ),
        "",
        "| trigger | " + " | ".join(SUBSETS) + " |",
        "|---|" + "---|" * len(SUBSETS),
    ]
    for feat in TRIGGERS:
        cells = []
        for sub in SUBSETS:
            a = s["subsets"][sub]["triggers"][feat]["auc"]
            if math.isnan(a["auc_share"]):
                cells.append(f"n/a / n/a / {1000 * a['excess']:+.1f}")
            else:
                cells.append(
                    f"{a['auc_share']:.2f} / {a['auc_cost']:.2f} / {1000 * a['excess']:+.1f}"
                )
        L.append(f"| {feat} | " + " | ".join(cells) + " |")
    for f in FRACS:
        L += [
            "",
            f"## Operating points keeping >= {f:.0%} of S5's gain at the lowest cost",
            "",
            (
                "share = queries sent to frames; tokens = mean input tokens per query; kept = "
                "share of S5's nDCG@10 gain over S4 (and of its R@1 gain)."
            ),
            "",
            (
                "| subset | trigger | t | share [CI] | tokens [CI] | nDCG@10 [CI] | "
                "gain vs S4 [CI] | kept [CI] | kept R@1 |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for sub in SUBSETS:
            for feat in TRIGGERS:
                op = s["subsets"][sub]["triggers"][feat]["ops"][str(f)]
                if op is None:
                    L.append(f"| {sub} | {feat} | - | no point | | | | | |")
                    continue
                kr1 = "n/a" if math.isnan(op["kept_r1"]) else f"{op['kept_r1']:.2f}"
                L.append(
                    f"| {sub} | {feat} | {op['t']:.3g} | {op['share']:.2f} "
                    f"[{op['share_ci'][0]:.2f}, {op['share_ci'][1]:.2f}] | {op['cost']:,.0f} "
                    f"[{op['cost_ci'][0]:,.0f}, {op['cost_ci'][1]:,.0f}] | {op['ndcg10']:.3f} "
                    f"[{op['ndcg10_ci'][0]:.3f}, {op['ndcg10_ci'][1]:.3f}] | "
                    f"{fmt_ci(op['gain_ci'])} | {op['kept_ndcg']:.2f} "
                    f"[{op['kept_ci'][0]:.2f}, {op['kept_ci'][1]:.2f}] | {kr1} |"
                )
    L += [
        "",
        "## Cross-fitted operating points (threshold picked on other source folds)",
        "",
        (
            "5 folds by source, 20 repetitions; held-out decisions pooled. Mean over repetitions "
            "[2.5-97.5% spread across repetitions, not a CI]. Only subsets where S5 - S4 has a CI "
            "above 0."
        ),
        "",
        "| subset | trigger | target | share | tokens | kept |",
        "|---|---|---|---|---|---|",
    ]
    for sub in CROSSFIT:
        for feat in TRIGGERS:
            for f in FRACS:
                c = s["subsets"][sub]["triggers"][feat]["crossfit"].get(str(f))
                if not c:
                    continue
                L.append(
                    f"| {sub} | {feat} | {f:.0%} | {c['share'][0]:.2f} "
                    f"[{c['share'][1]:.2f}, {c['share'][2]:.2f}] | {c['cost'][0]:,.0f} | "
                    f"{c['kept'][0]:.2f} [{c['kept'][1]:.2f}, {c['kept'][2]:.2f}] |"
                )
    L += [
        "",
        "## Oracle point (S5 only where its nDCG@10 beats S4)",
        "",
        "| subset | share | tokens | nDCG@10 | kept |",
        "|---|---|---|---|---|",
    ]
    for sub in SUBSETS:
        op = s["subsets"][sub]["triggers"]["oracle"]["oracle_point"]
        if op:
            kk = "n/a" if math.isnan(op["kept_ndcg"]) else f"{op['kept_ndcg']:.2f}"
            L.append(
                f"| {sub} | {op['share']:.2f} | {op['cost']:,.0f} | {op['ndcg10']:.3f} | {kk} |"
            )
    L += [
        "",
        "## Transfer: threshold tuned on one subset, applied to a disjoint one",
        "",
        (
            "| tune -> test | n test | trigger | target | t | share tune / test | test tokens | "
            "test kept |"
        ),
        "|---|---|---|---|---|---|---|---|",
    ]
    for t in s["transfer"]:
        kk = "n/a" if math.isnan(t["test_kept_ndcg"]) else f"{t['test_kept_ndcg']:.2f}"
        L.append(
            f"| {t['tune']} -> {t['test']} | {t['n_test']} | {t['trigger']} | "
            f"{t['frac']:.0%} | {t['t']:.3g} | {t['tune_share']:.2f} / "
            f"{t['test_share']:.2f} | {t['test_cost']:,.0f} | {kk} |"
        )
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.parse_args()
    run()
    print((OUT / "report.md").read_text())


if __name__ == "__main__":
    main()
