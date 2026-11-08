"""Description-length ablation tables for the paper's appendix (Appendix "Description length").

    python paper/tab_length.py

Reads saved rankings only (no API calls): nested set, describe_v4 (results/smoke/pilot), v4b, v4c, v4d
(results/smoke/variant_<tag>/) on the queries all four have; j-condition set, v4 (results/smoke_v2/pilot/twin)
against v4c and v4d (results/smoke/variant_<tag>_twin/). Strict labels (source only), nDCG@10, paired
differences with 95% cluster-bootstrap intervals over sources (10,000 resamples, seed 0). Writes
results/full/figures/length_ablation.json and tables/length_*.tex (copied into results/paper/tables/).
S4 per-call tokens for v4c and v4d come from cost_S1S4.json (cost.json holds the later S2 run).
"""

from __future__ import annotations

import json
import statistics as st
from collections import defaultdict
from pathlib import Path

import numpy as np

from src.datasets.mc import read_jsonl
from src.metrics import ndcg_at_k

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "results/paper"
OUT = ROOT / "results/full/figures"
N_BOOT = 10_000

NESTED = {
    "v4": "results/smoke/pilot",
    "v4b": "results/smoke/variant_v4b",
    "v4c": "results/smoke/variant_v4c",
    "v4d": "results/smoke/variant_v4d",
}
TWIN = {
    "v4": "results/smoke_v2/pilot/twin",
    "v4c": "results/smoke/variant_v4c_twin",
    "v4d": "results/smoke/variant_v4d_twin",
}
PROMPTS = {"v4": "describe_v4", "v4b": "describe_v4b", "v4c": "describe_v4c", "v4d": "describe_v4d"}
SENTENCES = {"v4": "1--3", "v4b": "3--5", "v4c": "5--7", "v4d": "14--18"}
# output tokens per description as billed by the describer (index logs / docs); v4 from App. stats
OUT_TOKENS = {"v4": 57, "v4b": 89, "v4c": 126, "v4d": 249}


def load(path: str, system: str) -> dict[str, dict]:
    return {r["query_id"]: r for r in read_jsonl(ROOT / path / f"{system}.jsonl")}


def ndcg(rows: dict[str, dict]) -> dict[str, float]:
    return {
        q: ndcg_at_k([tuple(x) for x in r["ranking"]], {r["source"]: 1}, 10) for q, r in rows.items()
    }


class Boot:
    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def diff(self, a: dict[str, float], b: dict[str, float], src: dict[str, str], keep=None) -> dict:
        qs = [q for q in sorted(set(a) & set(b)) if keep is None or q in keep]
        by = defaultdict(list)
        for q in qs:
            by[src[q]].append(a[q] - b[q])
        groups = [np.array(v) for v in by.values()]
        mean = float(np.mean(np.concatenate(groups)))
        idx = self.rng.integers(0, len(groups), (N_BOOT, len(groups)))
        sums = np.array([g.sum() for g in groups])
        sizes = np.array([len(g) for g in groups])
        boots = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"n": len(qs), "sources": len(groups), "diff": mean, "lo": float(lo), "hi": float(hi)}


def words(tag: str) -> dict:
    counts = []
    pattern = "descriptions.jsonl" if tag == "v4" else f"descriptions_{PROMPTS[tag]}.jsonl"
    for f in sorted((ROOT / "data/mc").glob(f"*/{pattern}")):
        for line in open(f):
            text = (json.loads(line).get("description") or "").strip()
            if text:
                counts.append(len(text.split()))
    return {"n": len(counts), "mean": st.mean(counts), "median": st.median(counts)}


def per_call(tag: str, folder: str, system: str) -> float:
    name = "cost_S1S4.json" if (system == "S4" and tag in ("v4c", "v4d") and "twin" not in folder) else "cost.json"
    return json.load(open(ROOT / folder / name))[system]["in_per_call"]


def sg(x: float) -> str:
    return f"{x:+.3f}".replace("+0.", "+.").replace("-0.", "-.")


def cell(d: dict) -> str:
    return f"${sg(d['diff'])}$\\,{{\\scriptsize[${sg(d['lo'])}$, ${sg(d['hi'])}$]}}"


def num(x: float) -> str:
    return f"{x:.3f}".lstrip("0") if x < 1 else f"{x:.3f}"


def main() -> None:
    boot = Boot()
    res: dict = {"nested": {}, "twin": {}}

    # ---- nested: v4, v4b, v4c, v4d on common queries ----
    sc = {t: {s: ndcg(load(p, s)) for s in ("S0", "S2", "S4")} for t, p in NESTED.items()}
    src = {q: r["source"] for q, r in load(NESTED["v4"], "S0").items()}
    kq = {q: r["k"] for q, r in load(NESTED["v4"], "S0").items()}
    ids = set.intersection(*[set(sc[t][s]) for t in sc for s in sc[t]])
    sub = {t: {s: {q: v for q, v in sc[t][s].items() if q in ids} for s in sc[t]} for t in sc}
    res["nested"]["queries"] = len(ids)
    res["nested"]["sources"] = len({src[q] for q in ids})
    means = {t: {s: st.mean(sub[t][s].values()) for s in sub[t]} for t in sub}
    res["nested"]["means"] = means
    res["nested"]["words"] = {t: words(t) for t in NESTED}
    res["nested"]["tokens"] = {
        t: {s: per_call(t, p, s) for s in ("S2", "S4")} for t, p in NESTED.items()
    }
    gains = {}
    for t in ("v4b", "v4c", "v4d"):
        gains[t] = {s: boot.diff(sub[t][s], sub["v4"][s], src) for s in ("S0", "S2", "S4")}
        gains[t]["S4-S0"] = boot.diff(sub[t]["S4"], sub[t]["S0"], src)
        gains[t]["S4-S2"] = boot.diff(sub[t]["S4"], sub[t]["S2"], src)
        gap_v4 = {q: sub["v4"]["S4"][q] - sub["v4"]["S2"][q] for q in ids}
        gap_t = {q: sub[t]["S4"][q] - sub[t]["S2"][q] for q in ids}
        gains[t]["gap_narrowing"] = boot.diff(gap_v4, gap_t, src)
    gains["v4"] = {
        "S4-S0": boot.diff(sub["v4"]["S4"], sub["v4"]["S0"], src),
        "S4-S2": boot.diff(sub["v4"]["S4"], sub["v4"]["S2"], src),
    }
    gains["v4d-v4c"] = {s: boot.diff(sub["v4d"][s], sub["v4c"][s], src) for s in ("S0", "S2", "S4")}
    res["nested"]["gains"] = gains
    bykk = {}
    for t in ("v4c", "v4d"):
        for s in ("S2", "S4"):
            for k in (1, 2, 3, 4):
                keep = {q for q in ids if kq[q] == k}
                bykk[f"{t}/{s}/{k}"] = boot.diff(sub[t][s], sub["v4"][s], src, keep)
    res["nested"]["by_k"] = bykk

    # ---- j-condition: v4, v4c, v4d ----
    tw = {t: {s: ndcg(load(p, s)) for s in ("S0", "S1", "S4")} for t, p in TWIN.items()}
    tsrc = {q: r["source"] for q, r in load(TWIN["v4"], "S0").items()}
    tids = set.intersection(*[set(tw[t][s]) for t in tw for s in tw[t]])
    tsub = {t: {s: {q: v for q, v in tw[t][s].items() if q in tids} for s in tw[t]} for t in tw}
    res["twin"]["queries"] = len(tids)
    res["twin"]["sources"] = len({tsrc[q] for q in tids})
    res["twin"]["means"] = {t: {s: st.mean(tsub[t][s].values()) for s in tsub[t]} for t in tsub}
    res["twin"]["gains"] = {
        t: {s: boot.diff(tsub[t][s], tsub["v4"][s], tsrc) for s in ("S0", "S1", "S4")}
        for t in ("v4c", "v4d")
    }
    for t in ("v4", "v4c", "v4d"):
        res["twin"]["gains"].setdefault(t, {})["S4-S0"] = boot.diff(tsub[t]["S4"], tsub[t]["S0"], tsrc)
    res["twin"]["gains"]["v4d-v4c"] = {"S4": boot.diff(tsub["v4d"]["S4"], tsub["v4c"]["S4"], tsrc)}
    res["twin"]["tokens"] = {t: per_call(t, p, "S4") for t, p in TWIN.items()}

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "length_ablation.json").write_text(json.dumps(res, indent=1))

    # ---- LaTeX ----
    N, W, T = res["nested"], res["nested"]["words"], res["nested"]["tokens"]
    tags = ["v4", "v4b", "v4c", "v4d"]
    row = lambda label, vals: f"{label} & " + " & ".join(vals) + " \\\\\n"  # noqa: E731
    t1 = (
        "% Generated by paper/tab_length.py\n\\begin{table}[!ht]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\n"
        + row("", [f"\\texttt{{{t}}}" for t in tags])
        + "\\midrule\n"
        + row("Sentences asked", [SENTENCES[t] for t in tags])
        + row("Words, mean", [f"{W[t]['mean']:.0f}" for t in tags])
        + row("Describer output tokens", [str(OUT_TOKENS[t]) for t in tags])
        + row("S4 input tokens per call", [f"{T[t]['S4']:.0f}" for t in tags])
        + row("S2 input tokens per call", [f"{T[t]['S2']:.0f}" for t in tags])
        + "\\midrule\n"
        + "".join(row(f"{s} nDCG@10", [num(means[t][s]) for t in tags]) for s in ("S0", "S2", "S4"))
        + "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Description-length ablation on the nested development set ({N['queries']} queries from {N['sources']} sources; strict labels). \\texttt{{describe\\_v4}} is the prompt used elsewhere. Token counts are per rerank call (S0's top 20) and per description.}}\n"
        "\\label{tab:length}\n\\end{table}\n"
    )
    g = N["gains"]
    t2 = (
        "% Generated by paper/tab_length.py\n\\begin{table*}[!ht]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{5pt}\n\\begin{tabular}{lccc}\n\\toprule\n"
        + row("nDCG@10, minus \\texttt{v4}", [f"\\texttt{{{t}}}" for t in ("v4b", "v4c", "v4d")])
        + "\\midrule\n"
        + "".join(row(s, [cell(g[t][s]) for t in ("v4b", "v4c", "v4d")]) for s in ("S0", "S2", "S4"))
        + "\\midrule\n"
        + row("S4 $-$ S2 at this length", [cell(g[t]["S4-S2"]) for t in ("v4b", "v4c", "v4d")])
        + row("S4 $-$ S2 gap narrowed from \\texttt{v4}", [cell(g[t]["gap_narrowing"]) for t in ("v4b", "v4c", "v4d")])
        + row("S4 $-$ S0 at this length", [cell(g[t]["S4-S0"]) for t in ("v4b", "v4c", "v4d")])
        + "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{Paired differences for Table~\\ref{{tab:length}} (nested development set, {N['queries']} queries, strict labels; 95\\% cluster-bootstrap intervals over sources). At \\texttt{{v4}} the S4 $-$ S2 gap is {cell(g['v4']['S4-S2'])} and S4 $-$ S0 is {cell(g['v4']['S4-S0'])}. Between \\texttt{{v4c}} and \\texttt{{v4d}}, S2 changes by {cell(g['v4d-v4c']['S2'])} and S4 by {cell(g['v4d-v4c']['S4'])}.}}\n"
        "\\label{tab:length-gains}\n\\end{table*}\n"
    )
    bold = lambda d: (  # noqa: E731
        f"$\\mathbf{{{sg(d['diff'])}}}$" if d["lo"] > 0 or d["hi"] < 0 else f"${sg(d['diff'])}$"
    )
    bk = N["by_k"]
    t3 = (
        "% Generated by paper/tab_length.py\n\\begin{table}[!ht]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{4pt}\n\\begin{tabular}{lcccc}\n\\toprule\n"
        " & \\multicolumn{2}{c}{S2} & \\multicolumn{2}{c}{S4} \\\\\n\\cmidrule(lr){2-3}\\cmidrule(l){4-5}\n"
        "$k$ & \\texttt{v4c} & \\texttt{v4d} & \\texttt{v4c} & \\texttt{v4d} \\\\\n\\midrule\n"
        + "".join(
            f"{k} & " + " & ".join(bold(bk[f"{t}/{s}/{k}"]) for s in ("S2", "S4") for t in ("v4c", "v4d")) + " \\\\\n"
            for k in (1, 2, 3, 4)
        )
        + "\\bottomrule\n\\end{tabular}\n"
        "\\caption{nDCG@10 gain over \\texttt{v4} by number of constraints $k$ (nested development set). Bold: the 95\\% interval excludes zero.}\n"
        "\\label{tab:length-k}\n\\end{table}\n"
    )
    Tw = res["twin"]
    tg = Tw["gains"]
    tm = Tw["means"]
    t4 = (
        "% Generated by paper/tab_length.py\n\\begin{table}[!ht]\n\\centering\n\\small\n"
        "\\setlength{\\tabcolsep}{3pt}\n\\begin{tabular}{lccc}\n\\toprule\n"
        + row("", ["\\texttt{v4}", "\\texttt{v4c} $-$ \\texttt{v4}", "\\texttt{v4d} $-$ \\texttt{v4}"])
        + "\\midrule\n"
        + "".join(
            row(s, [num(tm["v4"][s]), cell(tg["v4c"][s]), cell(tg["v4d"][s])]) for s in ("S0", "S1", "S4")
        )
        + "\\bottomrule\n\\end{tabular}\n"
        f"\\caption{{The same comparison on the $j$-condition development set ({Tw['queries']} queries from {Tw['sources']} sources; strict labels). First column: nDCG@10 with \\texttt{{v4}}; others: paired differences with 95\\% intervals. S2 was not run. S4 $-$ S0 is {cell(tg['v4']['S4-S0'])}, {cell(tg['v4c']['S4-S0'])} and {cell(tg['v4d']['S4-S0'])} at \\texttt{{v4}}, \\texttt{{v4c}} and \\texttt{{v4d}}.}}\n"
        "\\label{tab:length-j}\n\\end{table}\n"
    )
    for name, tex in (("length_main", t1), ("length_gains", t2), ("length_k", t3), ("length_j", t4)):
        (OUT / f"{name}.tex").write_text(tex)
        (PAPER / "tables").mkdir(parents=True, exist_ok=True)
        (PAPER / f"tables/{name}.tex").write_text(tex)
    print(json.dumps({k: v for k, v in res.items()}, indent=1)[:6000])


if __name__ == "__main__":
    main()
