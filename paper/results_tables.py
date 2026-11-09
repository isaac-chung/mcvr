"""Table 1 of the AAAI-27 workshop paper: quality and cost per system.

    uv run python paper/results_tables.py [--labels expanded|strict] [--out PATH]

Reads results/full/eval/summary.json (every number comes from it; the paper's summary is the
test-set eval, `systems.py --stage eval --sources-file results/full/select/sources100.json`)
and writes a booktabs table (single column, \\small, no resizebox) to
results/paper/tables/table1.tex, plus a copy in results/full/figures/. With the primary labels it
also writes tables/table1_subsets.tex (the frame subsets). No API calls. Safe to rerun.

Layout (the review-W13 layout that was hand-edited into table1.tex before, now generated)
  Block 1, all queries: S0, S1, S2@50, S2g@50, S4@50.
  Block 2, frame systems on all queries: S4@50 (repeated), S3g, S5, S6 (summary keys
    "all nested" / "all near-miss" of subset_means; S6 = the S4@50 top-1 margin trigger at the
    dev threshold, i.e. margin <= 2, from summary["s6"]["test"]["all nested" / "all near-miss"],
    with its pooled share of queries sent from "all queries"). When the summary has no
    whole-set frame cells (an eval without --sources-file) the block falls back to the frame
    subsets and says so.
  Quality: mean nDCG@10 over all queries and at k = 4; R@1 over all queries; bold is best
    per block.
  Cost per query, pooled over both sets (weighted by queries, summary["costs"], which the eval
    restricts to the test set): "Base" = the description stage's calls and input tokens
    (thousands); a frame system's base is S4@50's (summary["frame_base"]). "Frame step" = the
    frame calls' input tokens (thousands) and images; S6's frame step is its recorded pooled
    tokens per query minus S4@50's, and its images the share sent times S5's.
  table1_subsets.tex: the old block 2 (S0, S4@50, S3, S3g, S5) on the nested Vript sample and
    the contested near-miss queries of the test set (summary keys "nested sample", "twin-both").
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results/full/eval/summary.json"
PAPER_OUT = ROOT / "results/paper/tables/table1.tex"
LOCAL_OUT = ROOT / "results/full/figures/table1.tex"

SETS = ("natural", "twin")  # natural = nested in the paper
WHOLE = {"natural": "all nested", "twin": "all near-miss"}  # frame systems on every query
SUBSETS = {"natural": "nested sample", "twin": "twin-both"}  # the frozen frame subsets
FULL_SYS = ["S0", "S1", "S2@50", "S2g@50", "S4@50"]
FRAME_SYS = ["S4@50", "S3g", "S5"]
SUBSET_SYS = ["S0", "S4@50", "S3", "S3g", "S5"]
S6_TARGET = "0.8"  # dev threshold that keeps >= 80% of the S5 gain on dev (margin <= 2)
LABEL = "tab:results"
OTHER = {
    True: "",
    False: (
        " Same layout as Table~\\ref{tab:results}, which uses expanded labels; S6 is evaluated "
        "on expanded labels only."
    ),
}

CAPTION = (
    "Quality and cost per query ({labels} labels): nDCG@10 over all queries and at $k=4$, "
    "R@1 over all queries; bold is best per block. $^a$Base: calls and input tokens "
    "(thousands) of the description stage, averaged over all {n_all:,} queries; S3g, S5 and S6 "
    "rerank S4@50's top 10, so their base is S4@50's. $^b$Frame step: the frame calls only "
    "(10 per query), averaged over {frame_scope}, where the lower block repeats S4@50; a frame "
    "system's total is base plus frame step.{s6_clause}{other}"
)
S6_CLAUSE = (
    " S6 is on the same rule as S3g and S5, averaged over all {n_s6:,} queries, although it "
    "sends only {share:.0f}\\% of them to frames (S4@50 top-1 margin $\\le {t:g}$; "
    "{calls:.0f} calls per query on average; images approximate)."
)
SUBSET_CAPTION = (
    "Frame systems on the frozen frame subsets of the test set ({labels} labels): the nested "
    "Vript sample ($n={n_nested}$) and the contested near-miss queries ($n={n_twin}$, source "
    "and near miss both in S4@50's top 10), the only queries S3 (per-constraint yes/no "
    "verifier on frames) was run on. nDCG@10 over all queries of the subset and at $k=4$, R@1 "
    "over all queries of the subset; bold is best per column. S3g, S5 (and S6) on every test "
    "query are in Table~\\ref{{tab:results}}."
)


def f3(x: float | None) -> str:
    """.444 style, no leading zero."""
    if x is None:
        return "--"
    return f"{x:.3f}".removeprefix("0")


def get(d: dict, *keys, default=None):
    for k in keys:
        if not isinstance(d, dict) or k not in d:
            return default
        d = d[k]
    return d


def metric(block: dict, system: str, metric_name: str, k: str | None) -> float | None:
    if k is None:
        return get(block, system, "None", "all", metric_name)
    return get(block, system, "k", k, metric_name)


def pooled(costs: dict, system: str, field: str) -> float | None:
    rows = [costs[s][system] for s in SETS if system in costs.get(s, {})]
    if not rows:
        return None
    n = sum(r["queries"] for r in rows)
    return sum(r[field] * r["queries"] for r in rows) / n


def bold_best(
    cells: list[list[str]], values: list[list[float | None]], cols: list[int], tol: float = 0.0
) -> None:
    """Bold the best value per column; with tol, also every value less than tol below it."""
    for c in cols:
        vals = [(v[c], i) for i, v in enumerate(values) if v[c] is not None]
        if not vals:
            continue
        best = max(v for v, _ in vals)
        for v, i in vals:
            if f3(v) == f3(best) or best - v < tol:
                cells[i][c] = f"\\textbf{{{cells[i][c]}}}"


def quality_row(blocks: dict, system: str, sets_keys: dict) -> list[float | None]:
    out = []
    for s in SETS:
        b = blocks[sets_keys[s]]
        out += [metric(b, system, "ndcg@10", None), metric(b, system, "ndcg@10", "4")]
    out += [metric(blocks[sets_keys[s]], system, "recall@1", None) for s in SETS]
    return out


def fmt(rows) -> str:
    return "\n".join(" & ".join(r) + " \\\\" for r in rows)


def main_table(d: dict, labels: str) -> str:
    L = d["labels"][labels]
    costs = d["costs"]
    full_blocks = {s: L["means"][s] for s in SETS}
    whole = all(WHOLE[s] in L["subset_means"] for s in SETS)
    frame_keys = WHOLE if whole else SUBSETS
    frame_blocks = {frame_keys[s]: L["subset_means"][frame_keys[s]] for s in SETS}
    n_full = {s: d["gate"]["queries"][s] for s in SETS}
    n_frame = {s: get(frame_blocks[frame_keys[s]], "S4@50", "None", "all", "n") for s in SETS}
    base_of = {"S3g": d.get("frame_base", {}).get("S5", "S4@50"), **d.get("frame_base", {})}
    base_s4 = base_of["S5"]

    def base_cells(system: str) -> list[str]:
        sysname = base_of.get(system, system)
        calls, tok = pooled(costs, sysname, "calls"), pooled(costs, sysname, "tokens_in")
        if calls is None:
            return ["--", "--"]
        return [f"{calls:.0f}", f"{tok / 1000:.1f}"]

    def frame_cells(system: str) -> list[str]:
        tok, img = pooled(costs, system, "tokens_in"), pooled(costs, system, "images")
        if tok is None:
            return ["--", "--"]
        return [f"{tok / 1000:.1f}", f"{img:.0f}"]

    # Block 1: all queries
    b1_vals = [quality_row(full_blocks, s, {k: k for k in SETS}) for s in FULL_SYS]
    b1_cells = [[f3(v) for v in row] for row in b1_vals]
    bold_best(b1_cells, b1_vals, list(range(6)))
    b1 = [[s] + c + base_cells(s) + ["--", "--"] for s, c in zip(FULL_SYS, b1_cells)]

    # Block 2: the frame systems (on every query when the eval has them there)
    b2_names, b2_vals, b2_cost = [], [], []
    for s in FRAME_SYS:
        if s not in frame_blocks[frame_keys["twin"]]:
            print(f"warning: {s} missing from the frame cells, skipped", file=sys.stderr)
            continue
        b2_names.append(s)
        b2_vals.append(quality_row(frame_blocks, s, frame_keys))
        b2_cost.append(base_cells(s) + (["--", "--"] if s in FULL_SYS else frame_cells(s)))
    s6 = get(d, "s6", "test")
    s6_clause = ""
    s6_keys = {"natural": "all nested", "twin": "all near-miss", "pooled": "all queries"}
    if not whole:
        s6_keys = {"natural": "nested sample", "twin": "twin-both", "pooled": "pooled"}
    if s6 and labels == d["primary_labels"]:  # S6 offline eval is on the primary labels only
        thr = {s: get(s6, s6_keys[s], "dev_threshold", S6_TARGET) for s in SETS}
        pooled_thr = get(s6, s6_keys["pooled"], "dev_threshold", S6_TARGET)
        if all(thr.values()) and pooled_thr:
            share = pooled_thr["share"]
            row = [thr["natural"]["ndcg10"], None, thr["twin"]["ndcg10"], None, None, None]
            step = (pooled_thr["cost"] - pooled(costs, base_s4, "tokens_in")) / 1000
            img = share * pooled(costs, "S5", "images")
            b2_names.append("S6")
            b2_vals.append(row)
            b2_cost.append(base_cells("S5") + [f"{step:.1f}", f"$\\approx${img:.0f}"])
            s6_clause = S6_CLAUSE.format(
                n_s6=get(s6, s6_keys["pooled"], "n"),
                share=100 * share,
                t=pooled_thr["t"],
                calls=10 * share,
            )
    b2_cells = [[f3(v) for v in row] for row in b2_vals]
    bold_best(b2_cells, b2_vals, list(range(6)))
    b2 = [[n] + c + k for n, c, k in zip(b2_names, b2_cells, b2_cost)]

    if whole:
        head2 = f"\\emph{{Frame systems, all queries}}$^b$ ($n={n_frame['natural']}$ / ${n_frame['twin']}$)"
        frame_scope = f"the same {sum(n_full.values()):,} queries"
    else:
        head2 = f"\\emph{{Frame subsets}}$^b$ ($n={n_frame['natural']}$ / ${n_frame['twin']}$ contested)"
        frame_scope = (
            f"the {sum(n_frame.values())} frame-subset queries ({n_frame['natural']} nested Vript "
            f"and {n_frame['twin']} contested)"
        )
    caption = CAPTION.format(
        n_all=sum(n_full.values()),
        labels=labels,
        frame_scope=frame_scope,
        s6_clause=s6_clause,
        other=OTHER[labels == d["primary_labels"]],
    )
    label = LABEL if labels == d["primary_labels"] else f"{LABEL}-{labels}"
    src = f"test set: {d['sources']} sources" if d.get("sources") else "all sources"
    return f"""% Generated by paper/results_tables.py from results/full/eval/summary.json ({labels} labels; {src}).
% Do not edit by hand: rerun the script after a new eval.
\\begin{{table}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{1pt}}
\\begin{{tabular}}{{@{{}}lccccccrrrr@{{}}}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Nested}} & \\multicolumn{{2}}{{c}}{{Near-miss}} & \\multicolumn{{2}}{{c}}{{R@1}} & \\multicolumn{{2}}{{c}}{{Base$^a$}} & \\multicolumn{{2}}{{c}}{{Frame step$^b$}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\\cmidrule(lr){{6-7}}\\cmidrule(lr){{8-9}}\\cmidrule(l){{10-11}}
System & all & $k{{=}}4$ & all & $k{{=}}4$ & Nest. & N-miss & Calls & \\#tok & \\#tok & Img. \\\\
\\midrule
\\multicolumn{{11}}{{@{{}}l}}{{\\emph{{All queries}} ($n={n_full["natural"]}$ / ${n_full["twin"]}$)}} \\\\
{fmt(b1)}
\\midrule
\\multicolumn{{11}}{{@{{}}l}}{{{head2}}} \\\\
{fmt(b2)}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{label}}}
\\end{{table}}
"""


def subsets_table(d: dict, labels: str) -> str | None:
    L = d["labels"][labels]
    if not all(SUBSETS[s] in L["subset_means"] for s in SETS):
        return None
    blocks = {SUBSETS[s]: L["subset_means"][SUBSETS[s]] for s in SETS}
    names, vals = [], []
    for s in SUBSET_SYS:
        if s not in blocks[SUBSETS["twin"]]:
            continue
        names.append(s)
        vals.append(quality_row(blocks, s, SUBSETS))
    cells = [[f3(v) for v in row] for row in vals]
    bold_best(cells, vals, list(range(6)))
    rows = [[n] + c for n, c in zip(names, cells)]
    n_sub = {s: get(blocks[SUBSETS[s]], "S0", "None", "all", "n") for s in SETS}
    caption = SUBSET_CAPTION.format(labels=labels, n_nested=n_sub["natural"], n_twin=n_sub["twin"])
    src = f"test set: {d['sources']} sources" if d.get("sources") else "all sources"
    return f"""% Generated by paper/results_tables.py from results/full/eval/summary.json ({labels} labels; {src}).
% Do not edit by hand: rerun the script after a new eval.
\\begin{{table}}[t]
\\centering
\\small
\\setlength{{\\tabcolsep}}{{3pt}}
\\begin{{tabular}}{{@{{}}lcccccc@{{}}}}
\\toprule
 & \\multicolumn{{2}}{{c}}{{Nested sample}} & \\multicolumn{{2}}{{c}}{{Contested}} & \\multicolumn{{2}}{{c}}{{R@1}} \\\\
\\cmidrule(lr){{2-3}}\\cmidrule(lr){{4-5}}\\cmidrule(l){{6-7}}
System & all & $k{{=}}4$ & all & $k{{=}}4$ & Nest. & Cont. \\\\
\\midrule
\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{Frame subsets}} ($n={n_sub["natural"]}$ / ${n_sub["twin"]}$ contested)}} \\\\
{fmt(rows)}
\\bottomrule
\\end{{tabular}}
\\caption{{{caption}}}
\\label{{{LABEL}-subsets}}
\\end{{table}}
"""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--summary", type=Path, default=SUMMARY)
    ap.add_argument("--labels", default=None, help="label set (default: summary's primary_labels)")
    ap.add_argument("--out", type=Path, default=PAPER_OUT)
    args = ap.parse_args()

    d = json.loads(args.summary.read_text())
    labels = args.labels or d["primary_labels"]
    tex = main_table(d, labels)
    outs = [args.out, LOCAL_OUT]
    if (
        labels != d["primary_labels"]
    ):  # never overwrite the paper's table with a secondary label set
        outs = [LOCAL_OUT.with_name(f"table1_{labels}.tex")] + (
            [args.out] if args.out != PAPER_OUT else []
        )
    for out in outs:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(tex)
        print(f"wrote {out}")
    if labels == d["primary_labels"]:
        sub = subsets_table(d, labels)
        if sub:
            for out in (
                args.out.with_name("table1_subsets.tex"),
                LOCAL_OUT.with_name("table1_subsets.tex"),
            ):
                out.write_text(sub)
                print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
