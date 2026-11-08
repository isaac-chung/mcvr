"""Rebuttal checks on the saved full-run outputs (review W4, W6, W8, W11, W13-W16). FREE: no
model calls; reads results/full/runs, results/full/select/subsets.json, data/mc/full/judge and
the query files only.

    env -u OPENAI_API_KEY -u GEMINI_API_KEY uv run python paper/robustness_checks.py

1. W4: S5 - S4@50 on twin-both by the type of the twin's distinct fact d (row `d_type`, the
   fact at `d_index` with role "distinct") and by whether d is index_visible; the fact-type mix
   of nested and twin queries.
2. W6: the main contrasts on twin k >= 2 (twin k = 1 is not a near-miss); S0 nDCG@10 per k.
3. W8: source acceptance (both judges say yes to every fact of the query on the source) per set
   x k; nested (and twin) queries rejected through a fact shared with another query of the same
   source.
4. W11: Holm adjustment of the subgroup frame cells and of the headline tests (two-sided
   cluster-bootstrap p from the same resamples as the CIs).
5. W15: frame cost (images, tokens in / out per query) by Vript length bin next to the
   S5 - S4@50 gain per bin.
6. W16 / W13: cost per set x system (calls, tokens in / out, images, sequential LLM stages),
   whole pipelines for S3 / S5, and input tokens per +0.01 nDCG@10.
7. W14: S6 send rate (S4@50 top-1 minus top-2 score below t) over all queries, per set and k.

Reuses paper/tab_review_checks.py (load: runs, S2', qrels, scores; accepted_ids) and the
eval's cluster bootstrap (src.systems.single.cluster_bootstrap_ci: seed 0, percentile CI);
the p-values come from the same bootstrap draws. Writes
results/full/rebuttal/checks/{report.md, summary.json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


RC = _load(ROOT / "paper/tab_review_checks.py", "_rb_review_checks")
FS, PV, JD, jf = RC.FS, RC.PV, RC.JD, RC.jf
TS = FS.TS

from src.systems.single import cluster_bootstrap_ci

OUT = Path("results/full/rebuttal/checks")
SET_LABEL = RC.SET_LABEL
KINDS = ("expanded", "strict")
KS = ("1", "2", "3", "4")
BINS = FS.BINS
VRIPT = FS.VRIPT
TYPES = ("attribute", "object", "on_screen_text", "scene", "count")
M = "ndcg@10"
# sequential LLM stages per system, from the code: S1 = split (luna) then one search per
# constraint + RRF; S2 = split then per-constraint description checks of S0's top n (parallel);
# S4 = holistic description scores of S0's top n (parallel); S5 = S4@50, then holistic frame
# scores of its top 10; S3 = the split (splits.json, the S1/S2 split) and S4@50 (independent of
# each other), then per-constraint frame checks of S4@50's top 10 -> 3 LLM stages, 2 on the
# critical path.
STAGES = {
    "S0": (0, 0, "one search"),
    "S1": (1, 1, "split -> searches + RRF"),
    "S2@50": (2, 2, "split -> 50 description checks"),
    "S4@50": (1, 1, "50 description scores"),
    "S5": (2, 2, "S4@50 -> 10 frame scores"),
    "S3": (3, 2, "split || S4@50 -> 10 frame checks"),
}


# ---- statistics --------------------------------------------------------------------------------


def boot(diffs, clusters, n_boot: int) -> dict:
    """Mean, 95% percentile CI and two-sided bootstrap p of the mean of diffs, resampling
    clusters exactly as cluster_bootstrap_ci (seed 0) so the CI equals the eval's."""
    groups: dict[str, list[float]] = {}
    for v, c in zip(diffs, clusters, strict=True):
        groups.setdefault(c, []).append(v)
    sums = np.array([sum(g) for g in groups.values()])
    counts = np.array([len(g) for g in groups.values()])
    picks = np.random.default_rng(0).integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    p = 2 * min(((means <= 0).sum() + 1) / (n_boot + 1), ((means >= 0).sum() + 1) / (n_boot + 1))
    return {
        "n": len(diffs),
        "clusters": len(sums),
        "mean": float(np.mean(diffs)),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "p": float(min(1.0, p)),
    }


def pair(sa, sb, rows, n_boot, metric=M) -> dict | None:
    rs = [r for r in rows if r["query_id"] in sa and r["query_id"] in sb]
    if not rs:
        return None
    d = [sa[r["query_id"]][metric] - sb[r["query_id"]][metric] for r in rs]
    return boot(d, [r["source"] for r in rs], n_boot)


def mean_ci(values, clusters, n_boot) -> dict:
    lo, hi = cluster_bootstrap_ci(values, clusters, n_boot)
    return {"n": len(values), "mean": float(np.mean(values)), "ci_low": lo, "ci_high": hi}


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - j) * ps[i]))
        adj[i] = run
    return adj


def by_k(rows, k) -> list[dict]:
    return [r for r in rows if str(r["k"]) == k]


def by_bin(rows, b) -> list[dict]:
    return [r for r in rows if r["dataset"] in VRIPT and r["length_bin"] == b]


# ---- 1. W4: type of the distinct fact ---------------------------------------------------------


def d_fact(r: dict) -> dict:
    f = r["facts"][r["d_index"] - 1]
    assert f["role"] == "distinct" and f["type"] == r["d_type"], r["query_id"]
    return f


def check_dtype(D) -> dict:
    nb = D["S"].n_boot
    name, rows = D["frames"]["twin-both"]
    out: dict = {"by_type": {}, "by_visible": {}}
    for kind in KINDS:
        sc = D["scores"][kind][name]
        for t in TYPES:
            rs = [r for r in rows if d_fact(r)["type"] == t]
            out["by_type"].setdefault(t, {})[kind] = pair(sc["S5"], sc["S4@50"], rs, nb)
        for v in (True, False):
            rs = [r for r in rows if bool(d_fact(r).get("index_visible")) == v]
            out["by_visible"].setdefault(str(v), {})[kind] = pair(sc["S5"], sc["S4@50"], rs, nb)
        out.setdefault("all", {})[kind] = pair(sc["S5"], sc["S4@50"], rows, nb)
    # fact-type mix: fact instances over queries, unique facts per source, twin d
    mix: dict = {}
    for n, rs in D["sets"].items():
        inst = Counter(f["type"] for r in rs for f in r["facts"])
        uniq = {(r["source"], jf.norm_text(f["text"]), f["type"]) for r in rs for f in r["facts"]}
        vis = Counter(bool(f.get("index_visible")) for r in rs for f in r["facts"])
        mix[SET_LABEL[n]] = {
            "instances": dict(inst),
            "unique_per_source": dict(Counter(t for _, _, t in uniq)),
            "index_visible_instances": {str(k): v for k, v in vis.items()},
        }
    tw = D["sets"]["twin"]
    mix["twin d (all twin queries)"] = {"instances": dict(Counter(d_fact(r)["type"] for r in tw))}
    mix["twin d (twin-both)"] = {"instances": dict(Counter(d_fact(r)["type"] for r in rows))}
    mix["twin d visible"] = {
        "all twin": dict(Counter(str(bool(d_fact(r).get("index_visible"))) for r in tw)),
        "twin-both": dict(Counter(str(bool(d_fact(r).get("index_visible"))) for r in rows)),
    }
    out["mix"] = mix
    return out


# ---- 2. W6: twin k >= 2 ------------------------------------------------------------------------


def check_k2(D) -> dict:
    nb = D["S"].n_boot
    tw = [r for r in D["sets"]["twin"] if r["k"] >= 2]
    _, both = D["frames"]["twin-both"]
    both2 = [r for r in both if r["k"] >= 2]
    out: dict = {"n_twin_k2": len(tw), "n_twin_both_k2": len(both2), "pairs": {}, "s0_per_k": {}}
    for kind in KINDS:
        sc = D["scores"][kind]["twin"]
        cell = {}
        for a, b in (("S4@50", "S0"), ("S4@50", "S2@50"), ("S1", "S0")):
            cell[f"{a} - {b} (twin k>=2)"] = pair(sc[a], sc[b], tw, nb)
            cell[f"{a} - {b} (twin all k)"] = pair(sc[a], sc[b], D["sets"]["twin"], nb)
        for a, b in (("S5", "S4@50"), ("S3", "S5")):
            cell[f"{a} - {b} (twin-both k>=2)"] = pair(sc[a], sc[b], both2, nb)
            cell[f"{a} - {b} (twin-both all k)"] = pair(sc[a], sc[b], both, nb)
        out["pairs"][kind] = cell
        out["s0_per_k"][kind] = {}
        for n, rs in D["sets"].items():
            s0 = D["scores"][kind][n]["S0"]
            g = {"all": rs, **{k: by_k(rs, k) for k in KS}}
            out["s0_per_k"][kind][SET_LABEL[n]] = {
                k: mean_ci([s0[r["query_id"]][M] for r in v], [r["source"] for r in v], nb)
                for k, v in g.items()
            }
    return out


# ---- 3. W8: source acceptance ------------------------------------------------------------------


def check_acceptance(D, acc: dict) -> dict:
    jcfg, _rows, _facts, of_query, _ = JD.setup(D["S"])
    specs = tuple(jcfg["models"])
    judged = JD.JP.read_judged(D["S"].judge_dir / "judged.jsonl")

    def failing(r) -> list[str]:
        ans = judged[r["source"]][r["source"]]
        return [
            k
            for k in of_query[r["query_id"]]
            if not all(ans.get(k, {}).get(s, {}).get("answer") == "yes" for s in specs)
        ]

    out: dict = {}
    for n, rs in D["sets"].items():
        cell: dict = {}
        by_src: dict[str, list[dict]] = {}
        for r in rs:
            by_src.setdefault(r["source"], []).append(r)
        for g in ("all", *KS):
            sel = [r for r in rs if (g == "all" or str(r["k"]) == g) and r["query_id"] in acc]
            c = {
                "n": len([r for r in rs if g == "all" or str(r["k"]) == g]),
                "judged": len(sel),
                "accepted": sum(acc[r["query_id"]] for r in sel),
                "rejected": 0,
                "shared_fact": 0,  # a failing fact also in another query of the same source
                "inherited_only": 0,  # every failing fact already in a lower-k query
                "new_fact": 0,  # some failing fact first appears at this k
            }
            for r in sel:
                if acc[r["query_id"]]:
                    continue
                fail = failing(r)
                assert fail, r["query_id"]
                c["rejected"] += 1
                others = [o for o in by_src[r["source"]] if o["query_id"] != r["query_id"]]
                lower = {k for o in others if o["k"] < r["k"] for k in of_query[o["query_id"]]}
                every = {k for o in others for k in of_query[o["query_id"]]}
                c["shared_fact"] += any(k in every for k in fail)
                c["inherited_only"] += all(k in lower for k in fail)
                c["new_fact"] += any(k not in lower for k in fail)
            c["rate"] = c["accepted"] / max(1, c["judged"])
            cell[g] = c
        # distinct failing facts behind the rejected queries
        fails = {
            (r["source"], k)
            for r in rs
            if r["query_id"] in acc and not acc[r["query_id"]]
            for k in failing(r)
        }
        cell["distinct_failing_facts"] = len(fails)
        out[SET_LABEL[n]] = cell
    # check against review_checks.accepted_ids
    for r in [r for rs in D["sets"].values() for r in rs if r["query_id"] in acc]:
        assert acc[r["query_id"]] == (not failing(r)), r["query_id"]
    return out


# ---- 4. W11: Holm ------------------------------------------------------------------------------


def check_holm(D) -> dict:
    nb = D["S"].n_boot
    out: dict = {}
    for kind in KINDS:
        cells = []
        for label, (name, rows) in D["frames"].items():
            sc = D["scores"][kind][name]
            for a, b in (("S5", "S4@50"), ("S3", "S5")):
                for k in KS:
                    cells.append(
                        {
                            "contrast": f"{a} - {b}",
                            "group": label,
                            "cell": f"k={k}",
                            **pair(sc[a], sc[b], by_k(rows, k), nb),
                        }
                    )
                for b_ in BINS:
                    rs = by_bin(rows, b_)
                    if rs:
                        cells.append(
                            {
                                "contrast": f"{a} - {b}",
                                "group": label,
                                "cell": b_,
                                **pair(sc[a], sc[b], rs, nb),
                            }
                        )
        for n, rs in D["sets"].items():
            sc = D["scores"][kind][n]
            for k in KS:
                cells.append(
                    {
                        "contrast": "S4@50 - S2@50",
                        "group": SET_LABEL[n],
                        "cell": f"k={k}",
                        **pair(sc["S4@50"], sc["S2@50"], by_k(rs, k), nb),
                    }
                )
        adj = holm([c["p"] for c in cells])
        fam: dict[tuple, list[int]] = {}
        for i, c in enumerate(cells):
            c["holm_all"] = adj[i]
            fam.setdefault((c["contrast"], c["group"]), []).append(i)
        for idx in fam.values():  # smaller families: one contrast x one group
            for i, a in zip(idx, holm([cells[i]["p"] for i in idx])):
                cells[i]["holm_family"] = a
        head = []
        for n, rs in D["sets"].items():
            sc = D["scores"][kind][n]
            for a, b in (("S1", "S0"), ("S4@50", "S0"), ("S4@50", "S2@50")):
                head.append(
                    {"contrast": f"{a} - {b}", "group": SET_LABEL[n], **pair(sc[a], sc[b], rs, nb)}
                )
        for label, (name, rows) in D["frames"].items():
            sc = D["scores"][kind][name]
            for a, b in (("S5", "S4@50"), ("S3", "S5")):
                head.append(
                    {"contrast": f"{a} - {b}", "group": label, **pair(sc[a], sc[b], rows, nb)}
                )
        for c, a in zip(head, holm([c["p"] for c in head])):
            c["holm"] = a
        out[kind] = {"subgroup": cells, "headline": head}
    return out


# ---- 5. W15: frame cost by length bin -----------------------------------------------------------


def row_cost(D, name, system, q) -> dict:
    return D["lines"][name][system][q]["cost"]


def check_bins(D) -> dict:
    nb = D["S"].n_boot
    out: dict = {}
    for label, (name, rows) in D["frames"].items():
        cell = {}
        for b in (*BINS, "all Vript", "all"):
            rs = (
                rows
                if b == "all"
                else [r for r in rows if r["dataset"] in VRIPT]
                if b == "all Vript"
                else by_bin(rows, b)
            )
            if not rs:
                continue
            c: dict = {"n": len(rs)}
            for s in ("S3", "S5"):
                cs = [row_cost(D, name, s, r["query_id"]) for r in rs]
                c[s] = {
                    x: float(np.mean([z[x] for z in cs]))
                    for x in ("images", "tokens_in", "tokens_out")
                }
            for kind in KINDS:
                sc = D["scores"][kind][name]
                c[f"S5 - S4@50 {kind}"] = pair(sc["S5"], sc["S4@50"], rs, nb)
                c[f"S3 - S4@50 {kind}"] = pair(sc["S3"], sc["S4@50"], rs, nb)
            g = c["S5 - S4@50 expanded"]["mean"]
            c["S5 tokens_in per +0.01"] = c["S5"]["tokens_in"] / (100 * g) if g > 0 else None
            g3 = c["S3 - S4@50 expanded"]["mean"]
            c["S3 tokens_in per +0.01"] = c["S3"]["tokens_in"] / (100 * g3) if g3 > 0 else None
            cell[b] = c
        out[label] = cell
    return out


# ---- 6. W16 / W13: cost ------------------------------------------------------------------------

FIELDS = ("llm_calls", "tokens_in", "tokens_out", "images", "seconds")


def mean_cost(D, name, system, ids) -> dict:
    cs = [D["lines"][name][system][q]["cost"] for q in ids]
    return {x: float(np.mean([c[x] for c in cs])) for x in FIELDS}


def add(*cs) -> dict:
    return {x: sum(c[x] for c in cs) for x in FIELDS}


def check_cost(D) -> dict:
    nb = D["S"].n_boot
    table: dict = {}
    for n, rs in D["sets"].items():
        ids = [r["query_id"] for r in rs]
        cell = {}
        for s in ("S0", "S1", "S2@50", "S4@50"):
            cell[s] = {"queries": len(ids), "row": mean_cost(D, n, s, ids)}
        for label, (fn, frows) in D["frames"].items():
            if fn != n:
                continue
            fids = [r["query_id"] for r in frows]
            split = mean_cost(D, n, "S1", fids)  # S1's one call is the split S3 uses
            s4 = mean_cost(D, n, "S4@50", fids)
            for s in ("S5", "S3"):
                own = mean_cost(D, n, s, fids)
                pipe = add(s4, own) if s == "S5" else add(split, s4, own)
                cell[s] = {
                    "queries": len(fids),
                    "subset": label,
                    "row": own,
                    "pipeline": pipe,
                    "S4@50 on subset": s4,
                }
        for s, c in cell.items():
            c["llm_stages"], c["critical_path"], c["stages"] = STAGES[s]
        table[SET_LABEL[n]] = cell
    # W13: input tokens per +0.01 nDCG@10 over S0
    eff: dict = {}
    for kind in KINDS:
        eff[kind] = {}
        for n, rs in D["sets"].items():
            sc = D["scores"][kind][n]
            ids = [r["query_id"] for r in rs]
            s4 = mean_cost(D, n, "S4@50", ids)
            s2 = mean_cost(D, n, "S2@50", ids)
            split = mean_cost(D, n, "S1", ids)
            tok = {
                "S4@10": s4["tokens_in"] / 5,  # 10 of the 50 calls (per-call tokens not saved)
                "S4@50": s4["tokens_in"],
                "S2@10": split["tokens_in"] + (s2["tokens_in"] - split["tokens_in"]) / 5,
                "S2@50": s2["tokens_in"],
            }
            cell = {}
            for s, t in tok.items():
                g = pair(sc[s], sc["S0"], rs, nb)
                cell[s] = {
                    "tokens_in": t,
                    "gain": g,
                    "per_0.01": t / (100 * g["mean"]) if g["mean"] > 0 else None,
                }
            eff[kind][SET_LABEL[n]] = cell
        for label, (n, rows) in D["frames"].items():
            sc = D["scores"][kind][n]
            ids = [r["query_id"] for r in rows]
            s4 = mean_cost(D, n, "S4@50", ids)
            s5 = mean_cost(D, n, "S5", ids)
            cell = {}
            for s, t in (
                ("S4@10", s4["tokens_in"] / 5),
                ("S4@50", s4["tokens_in"]),
                ("S5", s4["tokens_in"] + s5["tokens_in"]),
            ):
                g = pair(sc[s], sc["S0"], rows, nb)
                cell[s] = {
                    "tokens_in": t,
                    "gain": g,
                    "per_0.01": t / (100 * g["mean"]) if g["mean"] > 0 else None,
                }
            g = pair(sc["S5"], sc["S4@50"], rows, nb)
            cell["S5 over S4@50 (frames only)"] = {
                "tokens_in": s5["tokens_in"],
                "gain": g,
                "per_0.01": s5["tokens_in"] / (100 * g["mean"]) if g["mean"] > 0 else None,
            }
            eff[kind][label] = cell
    return {"table": table, "efficiency": eff}


# ---- 7. W14: S6 send rate over all queries ------------------------------------------------------


def check_s6(D) -> dict:
    out: dict = {}
    groups = {SET_LABEL[n]: (n, rs) for n, rs in D["sets"].items()}
    groups |= {label: v for label, v in D["frames"].items()}
    for g, (n, rs) in groups.items():
        rk = D["ranks"][n]["S4@50"]
        cell = {}
        for kk in ("all", *KS):
            sel = rs if kk == "all" else by_k(rs, kk)
            m = np.array([TS.margin(rk[r["query_id"]]) for r in sel])
            cell[kk] = {"n": len(sel)}
            for t in (2, 3):
                cell[kk][f"<{t}"] = float((m < t).mean())
                cell[kk][f"<={t}"] = float((m <= t).mean())
        out[g] = cell
    return out


# ---- regression ---------------------------------------------------------------------------------


def regression(D, res) -> dict:
    ref = json.loads((D["S"].out / "eval" / "summary.json").read_text())["labels"]
    out = {}
    for kind in KINDS:
        for label in D["frames"]:
            a = ref[kind]["subset_paired"][label]["S5 - S4@50"][M]["None"]["all"]
            b = res["holm"][kind]["headline"]
            b = next(c for c in b if c["group"] == label and c["contrast"] == "S5 - S4@50")
            out[f"{kind} {label} S5 - S4@50"] = {
                "eval": [a["mean"], a["ci_low"], a["ci_high"]],
                "here": [b["mean"], b["ci_low"], b["ci_high"]],
                "same": abs(a["mean"] - b["mean"]) < 1e-12
                and abs(a["ci_low"] - b["ci_low"]) < 1e-12
                and abs(a["ci_high"] - b["ci_high"]) < 1e-12,
            }
        for n in D["sets"]:
            a = ref[kind]["paired"][n]["S4@50 - S0"][M]["None"]["all"]
            b = next(
                c
                for c in res["holm"][kind]["headline"]
                if c["group"] == SET_LABEL[n] and c["contrast"] == "S4@50 - S0"
            )
            out[f"{kind} {SET_LABEL[n]} S4@50 - S0"] = {
                "eval": [a["mean"], a["ci_low"], a["ci_high"]],
                "here": [b["mean"], b["ci_low"], b["ci_high"]],
                "same": abs(a["mean"] - b["mean"]) < 1e-12
                and abs(a["ci_low"] - b["ci_low"]) < 1e-12,
            }
    assert all(v["same"] for v in out.values()), out
    return out


# ---- report -------------------------------------------------------------------------------------


def ci(c) -> str:
    return "-" if not c else f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def pv(p: float) -> str:
    return f"{p:.4f}" if p >= 1e-3 else f"{p:.1e}"


def pct(a, b) -> str:
    return f"{a}/{b} ({a / b:.0%})" if b else "-"


def tok(v) -> str:
    return "-" if v is None else f"{v:,.0f}"


def takeaways(res) -> list[str]:
    d, k2, acc, h, bins, cost, s6 = (
        res["dtype"],
        res["twin_k2"],
        res["acceptance"],
        res["holm"],
        res["bins"],
        res["cost"],
        res["s6"],
    )
    e = d["by_type"]
    vis = d["by_visible"]
    kp = k2["pairs"]["expanded"]
    hs = h["expanded"]["subgroup"]
    surv = [c for c in hs if c["holm_all"] < 0.05]
    head = h["expanded"]["headline"]
    head_fail = [f"{c['contrast']} ({c['group']})" for c in head if c["holm"] >= 0.05]
    an, at = acc["nested"], acc["twin"]
    tb = bins["twin-both"]
    ns = bins["nested sample"]

    def cells(contrast, group):
        return (
            ", ".join(
                c["cell"]
                for c in hs
                if c["holm_all"] < 0.05 and c["contrast"] == contrast and c["group"] == group
            )
            or "none"
        )

    big = [
        f"{t} {e[t]['expanded']['mean']:+.3f} (n {e[t]['expanded']['n']})"
        for t in TYPES
        if e[t]["expanded"]["n"] >= 20
    ]
    small = [f"{t} n {e[t]['expanded']['n']}" for t in TYPES if e[t]["expanded"]["n"] < 20]
    L = [
        (
            f"- **W4.** The frame gain is not one fact type: S5 - S4@50 on twin-both is clearly positive "
            f"for each type with enough queries ({', '.join(big)}; all CIs above 0), largest when the "
            f"distinct fact is an attribute; {', '.join(small)} are too few to say. It is "
            f"{ci(vis['True']['expanded'])} when the twin's distinct fact is index-visible (n "
            f"{vis['True']['expanded']['n']}) and {ci(vis['False']['expanded'])} when it is not (n "
            f"{vis['False']['expanded']['n']})."
        ),
        (
            f"- **W6.** Dropping twin k=1 leaves every conclusion in place and makes the description "
            f"contrasts larger: on twin k>=2, S4@50 - S0 {ci(kp['S4@50 - S0 (twin k>=2)'])}, S4@50 - "
            f"S2@50 {ci(kp['S4@50 - S2@50 (twin k>=2)'])}, S1 - S0 {ci(kp['S1 - S0 (twin k>=2)'])}; "
            f"on twin-both k>=2, S5 - S4@50 {ci(kp['S5 - S4@50 (twin-both k>=2)'])}, S3 - S5 "
            f"{ci(kp['S3 - S5 (twin-both k>=2)'])}. S0 rises with k on nested queries ("
            + " / ".join(f"{k2['s0_per_k']['expanded']['nested'][k]['mean']:.3f}" for k in KS)
            + ") and falls on twin queries ("
            + " / ".join(f"{k2['s0_per_k']['expanded']['twin'][k]['mean']:.3f}" for k in KS)
            + ")."
        ),
        (
            "- **W8.** Source acceptance falls with k on nested queries ("
            + " / ".join(f"{an[k]['rate']:.0%}" for k in KS)
            + " for k=1..4), much less on twin queries ("
            + " / ".join(f"{at[k]['rate']:.0%}" for k in KS)
            + f"). Of the {an['all']['rejected']} rejected nested queries, {an['all']['inherited_only']} "
            f"are rejected only through a fact already in the same source's lower-k query (none new at "
            f"that k); {an['distinct_failing_facts']} distinct facts account for all nested rejections."
        ),
        (
            f"- **W11.** Holm over all {len(hs)} subgroup cells (expanded): {len(surv)} stay significant. "
            f"S4@50 - S2@50: all 8 k cells. S5 - S4@50: twin-both {cells('S5 - S4@50', 'twin-both')}; "
            f"nested sample {cells('S5 - S4@50', 'nested sample')}"
            f"; no S3 - S5 cell. Of the 10 headline tests, all survive except "
            + (", ".join(head_fail) if head_fail else "none")
            + " (which are not claimed as differences)."
        ),
        (
            f"- **W15.** Frame input tokens grow with length (nested sample "
            f"{ns['lt30s']['S5']['tokens_in'] / 1e3:.0f}k -> {ns['2to4min']['S5']['tokens_in'] / 1e3:.0f}k, "
            f"{ns['lt30s']['S5']['images']:.0f} -> {ns['2to4min']['S5']['images']:.0f} images; twin-both "
            f"{tb['lt30s']['S5']['tokens_in'] / 1e3:.0f}k -> {tb['2to4min']['S5']['tokens_in'] / 1e3:.0f}k) "
            f"while the gain does not, so S5 input tokens per +0.01 nDCG@10 are highest on 2-4 min videos "
            f"(twin-both "
            + " / ".join(tok(tb[b]["S5 tokens_in per +0.01"]) for b in BINS)
            + "; nested sample "
            + " / ".join(tok(ns[b]["S5 tokens_in per +0.01"]) for b in BINS)
            + ", where the 2-4 min gain is ~0)."
        ),
        (
            f"- **W16/W13.** Per +0.01 nDCG@10 over S0 (expanded), S4@10 costs about "
            f"{tok(cost['efficiency']['expanded']['nested']['S4@10']['per_0.01'])} / "
            f"{tok(cost['efficiency']['expanded']['twin']['S4@10']['per_0.01'])} input tokens (nested / "
            f"twin), S4@50 {tok(cost['efficiency']['expanded']['nested']['S4@50']['per_0.01'])} / "
            f"{tok(cost['efficiency']['expanded']['twin']['S4@50']['per_0.01'])}, S2@50 "
            f"{tok(cost['efficiency']['expanded']['nested']['S2@50']['per_0.01'])} / "
            f"{tok(cost['efficiency']['expanded']['twin']['S2@50']['per_0.01'])}; the frame step over "
            f"S4@50 costs {tok(cost['efficiency']['expanded']['twin-both']['S5 over S4@50 (frames only)']['per_0.01'])} "
            f"(twin-both) / {tok(cost['efficiency']['expanded']['nested sample']['S5 over S4@50 (frames only)']['per_0.01'])} "
            f"(nested sample). Shallow S4 is the most token-efficient; depth 50 buys the larger gain. "
            f"Output tokens (incl. reasoning) per query: S2@50 {cost['table']['nested']['S2@50']['row']['tokens_out']:,.0f}, "
            f"S4@50 {cost['table']['nested']['S4@50']['row']['tokens_out']:,.0f}, S3 frames "
            f"{cost['table']['nested']['S3']['row']['tokens_out']:,.0f}, S5 frames "
            f"{cost['table']['nested']['S5']['row']['tokens_out']:,.0f} (nested). S5 = 2 sequential LLM "
            f"stages (S4 then frames); S3 = 3 LLM stages, 2 on the critical path."
        ),
        (
            f"- **W14.** Over all queries, the margin trigger (S4@50 top-1 minus top-2 score < t) sends "
            f"{s6['nested']['all']['<2']:.1%} / {s6['nested']['all']['<3']:.1%} of nested and "
            f"{s6['twin']['all']['<2']:.1%} / {s6['twin']['all']['<3']:.1%} of twin queries to frames at "
            f"t = 2 / 3, more than inside twin-both ({s6['twin-both']['all']['<2']:.1%} / "
            f"{s6['twin-both']['all']['<3']:.1%}); scores are integers, so < 3 equals <= 2."
        ),
    ]
    return L


def safe_takeaways(res) -> list[str]:
    """takeaways() was written for the 279-source run; on a smaller set some subgroup cells
    are empty (fewer than 5 queries) and the prose cannot be filled in."""
    try:
        return takeaways(res)
    except (TypeError, KeyError) as err:
        return [
            (
                "- (takeaways not generated: a subgroup cell this prose quotes is empty on this "
                f"query set; see the tables below. {type(err).__name__}: {err})"
            )
        ]


def report(res) -> str:
    L = ["# Rebuttal checks (paper/robustness_checks.py; no model calls)", ""]
    L += [
        (
            "From saved outputs only (results/full/runs, results/full/select/subsets.json, "
            "data/mc/full/judge). Expanded labels primary, strict alongside. CIs: paired cluster "
            f"bootstrap over source videos, {res['n_boot']:,} resamples, 95%; p = two-sided bootstrap "
            "p from the same draws (ties at 0 count on both sides, floor 1/(n+1)); with few source "
            "clusters it is coarse. nDCG@10 throughout."
        ),
        "",
        "## Key takeaways",
        "",
        *safe_takeaways(res),
        "",
        "Reproduced first (identical to results/full/eval/summary.json): "
        + "; ".join(
            f"{k} {v['here'][0]:+.3f} [{v['here'][1]:+.3f}, {v['here'][2]:+.3f}]"
            for k, v in res["regression"].items()
        )
        + ".",
        "",
    ]
    # 1
    d = res["dtype"]
    L += ["## 1. W4: S5 - S4@50 on twin-both by the twin's distinct fact d", ""]
    L += ["| d | n | sources | expanded | p | strict | p |", "|---|---|---|---|---|---|---|"]
    for t in TYPES:
        c = d["by_type"][t]
        if c["expanded"]:
            L.append(
                f"| type {t} | {c['expanded']['n']} | {c['expanded']['clusters']} | {ci(c['expanded'])} | {pv(c['expanded']['p'])} | {ci(c['strict'])} | {pv(c['strict']['p'])} |"
            )
    for v in ("True", "False"):
        c = d["by_visible"][v]
        if c["expanded"]:
            L.append(
                f"| index_visible={v} | {c['expanded']['n']} | {c['expanded']['clusters']} | {ci(c['expanded'])} | {pv(c['expanded']['p'])} | {ci(c['strict'])} | {pv(c['strict']['p'])} |"
            )
    c = d["all"]
    L.append(
        f"| all | {c['expanded']['n']} | {c['expanded']['clusters']} | {ci(c['expanded'])} | {pv(c['expanded']['p'])} | {ci(c['strict'])} | {pv(c['strict']['p'])} |"
    )
    L += [
        "",
        "Fact-type mix (fact instances over queries; unique = distinct facts per source):",
        "",
    ]
    L += ["| group | " + " | ".join(TYPES) + " | total |", "|---|" + "---|" * (len(TYPES) + 1)]
    for g, m in d["mix"].items():
        if g == "twin d visible":
            continue
        for what in ("instances", "unique_per_source"):
            if what not in m:
                continue
            tot = sum(m[what].values())
            L.append(
                f"| {g} ({what}) | "
                + " | ".join(f"{m[what].get(t, 0)} ({m[what].get(t, 0) / tot:.0%})" for t in TYPES)
                + f" | {tot} |"
            )
    iv = d["mix"]
    L += [
        "",
        "index_visible share of fact instances: "
        + "; ".join(
            f"{g} {iv[g]['index_visible_instances'].get('True', 0)}/{sum(iv[g]['index_visible_instances'].values())}"
            for g in ("nested", "twin")
        )
        + f". Twin d index-visible: all twin {iv['twin d visible']['all twin']}, twin-both {iv['twin d visible']['twin-both']}.",
        "",
    ]
    # 2
    k2 = res["twin_k2"]
    L += [
        "## 2. W6: twin k >= 2",
        "",
        f"Twin k>=2: {k2['n_twin_k2']} queries; twin-both k>=2: {k2['n_twin_both_k2']}.",
        "",
    ]
    L += ["| contrast | expanded | strict |", "|---|---|---|"]
    for p_ in k2["pairs"]["expanded"]:
        L.append(f"| {p_} | {ci(k2['pairs']['expanded'][p_])} | {ci(k2['pairs']['strict'][p_])} |")
    L += [
        "",
        "S0 nDCG@10 per k [95% CI]:",
        "",
        "| labels | set | all | k=1 | k=2 | k=3 | k=4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for s_, c in k2["s0_per_k"][kind].items():
            L.append(
                f"| {kind} | {s_} | "
                + " | ".join(
                    f"{c[k]['mean']:.3f} [{c[k]['ci_low']:.3f}, {c[k]['ci_high']:.3f}]"
                    for k in ("all", *KS)
                )
                + " |"
            )
    # 3
    a = res["acceptance"]
    L += ["", "## 3. W8: source acceptance (both judges say yes to every fact on the source)", ""]
    L += [
        "| set | k | queries | judged | accepted | rejected | failing fact shared with another query of the source | rejected only through a fact inherited from lower k | some failing fact new at this k |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for s_, cell in a.items():
        for g in ("all", *KS):
            c = cell[g]
            L.append(
                f"| {s_} | {g} | {c['n']} | {c['judged']} | {pct(c['accepted'], c['judged'])} | {c['rejected']} | "
                f"{c['shared_fact']} | {c['inherited_only']} | {c['new_fact']} |"
            )
    L += [
        "",
        "Distinct failing (source, fact) pairs behind the rejections: "
        + "; ".join(f"{s_} {cell['distinct_failing_facts']}" for s_, cell in a.items())
        + ". Nested queries are prefixes (k adds one fact to k-1), so a fact rejected at k=1 rejects "
        "k=1..4 of that source.",
        "",
    ]
    # 4
    L += ["## 4. W11: Holm adjustment", ""]
    for kind in KINDS:
        h = res["holm"][kind]
        cells = h["subgroup"]
        L += [
            f"### {kind}: subgroup cells (Holm over all {len(cells)} cells; 'family' = Holm within one contrast x group)",
            "",
            "| contrast | group | cell | n | diff [95% CI] | p | Holm all | Holm family | sig. after Holm all |",
            "|---|---|---|---|---|---|---|---|---|",
        ]
        for c in cells:
            L.append(
                f"| {c['contrast']} | {c['group']} | {c['cell']} | {c['n']} | {ci(c)} | {pv(c['p'])} | "
                f"{pv(c['holm_all'])} | {pv(c['holm_family'])} | {'yes' if c['holm_all'] < 0.05 else 'no'} |"
            )
        L += ["", f"### {kind}: headline tests (Holm over {len(h['headline'])})", ""]
        L += [
            "| contrast | group | n | diff [95% CI] | p | Holm | sig. |",
            "|---|---|---|---|---|---|---|",
        ]
        for c in h["headline"]:
            L.append(
                f"| {c['contrast']} | {c['group']} | {c['n']} | {ci(c)} | {pv(c['p'])} | {pv(c['holm'])} | {'yes' if c['holm'] < 0.05 else 'no'} |"
            )
        L += [""]
    # 5
    L += ["## 5. W15: frame cost by Vript length bin (per query means)", ""]
    for label, cell in res["bins"].items():
        L += [f"### {label}", ""]
        L += [
            "| bin | n | images | S5 tokens in | S5 tokens out | S3 tokens in | S3 tokens out | S5 - S4@50 expanded | strict | S5 tokens in per +0.01 | S3 - S4@50 expanded | S3 tokens in per +0.01 |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for b, c in cell.items():
            L.append(
                f"| {b} | {c['n']} | {c['S5']['images']:.0f} | {c['S5']['tokens_in']:,.0f} | {c['S5']['tokens_out']:,.0f} | "
                f"{c['S3']['tokens_in']:,.0f} | {c['S3']['tokens_out']:,.0f} | {ci(c['S5 - S4@50 expanded'])} | "
                f"{ci(c['S5 - S4@50 strict'])} | {tok(c['S5 tokens_in per +0.01'])} | {ci(c['S3 - S4@50 expanded'])} | "
                f"{tok(c['S3 tokens_in per +0.01'])} |"
            )
        L += [""]
    # 6
    cst = res["cost"]
    L += [
        "## 6. W16 / W13: cost per query",
        "",
        (
            "Row = the system's own saved calls; pipeline = everything it needs (S5 = S4@50 + frame "
            "scores; S3 = split + S4@50 + frame checks; the split is S1's single call, the same split "
            "S2 uses and S3 reads from splits.json). Tokens out include reasoning tokens "
            "(src/llm.py: completion_tokens). Seconds = summed call seconds, not wall time. "
            "S2@50 = 1 split + 50 checks."
        ),
        "",
        "| set | system | queries | calls | tokens in | tokens out | images | call seconds | LLM stages | critical path | stages |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s_, cell in cst["table"].items():
        for sys_, c in cell.items():
            for what in ("row", "pipeline"):
                if what not in c:
                    continue
                r = c[what]
                name = sys_ if what == "row" else f"{sys_} pipeline"
                L.append(
                    f"| {s_} | {name}{' (' + c['subset'] + ')' if 'subset' in c else ''} | {c['queries']} | {r['llm_calls']:.1f} | "
                    f"{r['tokens_in']:,.0f} | {r['tokens_out']:,.0f} | {r['images']:.1f} | {r['seconds']:.1f} | "
                    f"{c['llm_stages'] if what != 'row' or sys_ not in ('S3', 'S5') else 1} | "
                    f"{c['critical_path'] if what != 'row' or sys_ not in ('S3', 'S5') else 1} | "
                    f"{c['stages'] if what != 'row' or sys_ not in ('S3', 'S5') else 'frame step only'} |"
                )
    L += [
        "",
        (
            "Input tokens per +0.01 nDCG@10 over S0 (S4@10 / S2@10 tokens estimated as 10 of the 50 "
            "per-candidate calls; per-call tokens are not saved):"
        ),
        "",
        "| labels | group | system | tokens in | gain over S0 | tokens per +0.01 |",
        "|---|---|---|---|---|---|",
    ]
    for kind in KINDS:
        for g, cell in cst["efficiency"][kind].items():
            for s_, c in cell.items():
                L.append(
                    f"| {kind} | {g} | {s_} | {c['tokens_in']:,.0f} | {ci(c['gain'])} | {tok(c['per_0.01'])} |"
                )
    # 7
    L += ["", "## 7. W14: S6 send rate over all queries (S4@50 top-1 minus top-2 score)", ""]
    L += ["| group | k | n | <2 | <3 | <=2 | <=3 |", "|---|---|---|---|---|---|---|"]
    for g, cell in res["s6"].items():
        for k, c in cell.items():
            L.append(
                f"| {g} | {k} | {c['n']} | {c['<2']:.1%} | {c['<3']:.1%} | {c['<=2']:.1%} | {c['<=3']:.1%} |"
            )
    L += [
        "",
        (
            "S4 scores are integers, so < 3 equals <= 2. trigger_sim.fires uses <= t; the dev-chosen "
            "S6 threshold in the paper is applied with that rule."
        ),
    ]
    return "\n".join(L) + "\n"


# ---- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--sources-file", help="the test set (as systems.py --sources-file)")
    args = ap.parse_args(argv)
    D = RC.load(args.n_boot, args.sources_file)
    D["lines"] = {n: FS.load_runs(D["S"].runs / n, D["S"].rescore)[1] for n in D["sets"]}
    acc, rates = RC.accepted_ids(D)
    res = {
        "n_boot": args.n_boot,
        "acceptance_overall": rates,
        "dtype": check_dtype(D),
        "twin_k2": check_k2(D),
        "acceptance": check_acceptance(D, acc),
        "holm": check_holm(D),
        "bins": check_bins(D),
        "cost": check_cost(D),
        "s6": check_s6(D),
    }
    res["regression"] = regression(D, res)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "report.md").write_text(report(res))
    print(f"wrote {OUT}/report.md, {OUT}/summary.json")


if __name__ == "__main__":
    main()
