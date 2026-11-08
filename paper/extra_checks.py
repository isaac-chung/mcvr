"""Three small recomputations on the test set (the paper's `% TODO` lines) from saved outputs.
FREE: no model calls; reads results/full/runs, results/full/second, results/full/select and
data/mc/full only. Each writes results/full/todo_recompute/<item>/{summary.json, report.md}.

    uv run python paper/extra_checks.py collinearity --sources-file results/full/select/sources100.json
    uv run python paper/extra_checks.py spearman     --sources-file results/full/select/sources100.json
    uv run python paper/extra_checks.py corpus-part  --sources-file results/full/select/sources100.json

collinearity: systems.rate_collinearity (caption words per minute vs scenes per minute over the
  Vript sources, correlation and VIF) restricted to the drawn sources (appdx_03_stats_cost.tex).
spearman: the two verifiers' S4@50 scores (gpt-6-sol, results/full/runs/<set>/S4@50.jsonl, vs
  gpt-5.5, results/full/second/<set>/S4@50_gpt-5.5.jsonl) on the second-verifier subset: per
  query Spearman over the shared scored candidates (dev/second_verifier_dev.score_spearman,
  the definition behind docs/results_smoke.md "S4 Spearman"), mean with a cluster-bootstrap CI
  over sources, plus the same-top-1 share (appdx_01_systems.tex).
corpus-part: S5 - S4@50 by corpus part (dataset) on the contested near-miss subgroup (twin-both:
  source and HN video both in S4@50's top 10), expanded and strict labels, nDCG@10 and R@1
  (appdx_06_full_results.tex); review_checks.load as the rebuttal checks.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
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


RC = _load(ROOT / "paper/tab_review_checks.py", "_todo_review_checks")
FS = RC.FS
PV = FS.PV
SV = FS.SV  # dev/second_verifier_dev.py

OUT = Path("results/full/todo_recompute")
SECOND = Path("results/full/second")
VERIFIERS = {"S4@50": "gpt-6-sol", "S4@50_gpt-5.5": "gpt-5.5"}


def ci(c: dict) -> str:
    return f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def settings(args):
    S = FS.settings(argparse.Namespace(smoke=False, config=None, out=None, n_boot=args.n_boot))
    sets = FS.load_sets(S, FS.load_context(S, need_index=False))
    if args.sources_file:
        sets = FS.restrict_sets(S, sets, args.sources_file)
    return S, sets


# ---- collinearity ------------------------------------------------------------------------------


def collinearity(args) -> dict:
    S, _ = settings(args)
    res = FS.rate_collinearity(FS.source_meta(S), S.keep_sources)
    res["sources_file"] = args.sources_file
    a = res["all"]
    lines = [
        "# Caption words per minute vs scenes per minute (test set)",
        "",
        f"Sources: {a['sources']} Vript sources with a caption"
        + (f" of {args.sources_file}" if args.sources_file else "")
        + ".",
        "",
        "| group | sources | corr(words/min, scenes/min) | VIF with both in | words per scene p10 / p50 / p90 |",
        "|---|---|---|---|---|",
    ]
    for g in ("all", *FS.VRIPT):
        if g in res:
            c = res[g]
            lines.append(
                f"| {g} | {c['sources']} | {c['corr_words_scenes_per_min']:.3f} | "
                f"{c['vif_both_in']:.0f} | {' / '.join(f'{x:.0f}' for x in c['words_per_scene_p10_p50_p90'])} |"
            )
    return res, lines


# ---- spearman ----------------------------------------------------------------------------------


def scores_of(path: Path) -> dict[str, dict[str, float]]:
    """{qid: {doc: S4 score}} from a run's trace (the 50 scored candidates)."""
    return {
        line["query_id"]: {d: float(s) for d, s in line["trace"]["scores"].items()}
        for line in FS.read_jsonl(path)
    }


def spearman(args) -> dict:
    from scipy.stats import spearmanr

    S, sets = settings(args)
    sub = FS.read_subsets(S)
    res: dict = {"sources_file": args.sources_file, "verifiers": VERIFIERS, "sets": {}}
    lines = [
        "# Second verifier: Spearman of the two verifiers' S4@50 scores (test set)",
        "",
        "Per query: Spearman correlation of gpt-6-sol's and gpt-5.5's S4@50 scores over the "
        "candidates both scored (S0's top 50; queries with fewer than 3 shared candidates or a "
        "constant score list are counted, not averaged: dev/second_verifier_dev.score_spearman). "
        f"Mean with a 95% cluster-bootstrap CI over source videos ({args.n_boot:,} resamples); "
        "same top 1 = share of queries whose reranked top candidate is the same under both.",
        "",
        "| set | n (subset) | n (rho) | constant | mean rho | median | q25 | same top 1 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, rows in sets.items():
        a = scores_of(S.runs / name / "S4@50.jsonl")
        b = scores_of(SECOND / name / "S4@50_gpt-5.5.jsonl")
        ids = [q for q in sub["second"][name] if q in a and q in b]
        rs = [r for r in rows if r["query_id"] in set(ids)]
        qa = {q: a[q] for q in ids}
        qb = {q: b[q] for q in ids}
        agg = SV.score_spearman(qa, qb)
        per_q, src = [], []
        for r in rs:
            q = r["query_id"]
            docs = sorted(set(a[q]) & set(b[q]))
            x, y = [a[q][d] for d in docs], [b[q][d] for d in docs]
            if len(docs) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
                continue
            per_q.append(float(spearmanr(x, y).statistic))
            src.append(r["source"])
        boot = FS.P2.share_ci(per_q, src, args.n_boot)
        ra = FS.read_ranking(S.runs / name / "S4@50.jsonl")
        rb = FS.read_ranking(SECOND / name / "S4@50_gpt-5.5.jsonl")
        top1 = FS.P2.share_ci(
            [float(ra[r["query_id"]][0][0] == rb[r["query_id"]][0][0]) for r in rs],
            [r["source"] for r in rs],
            args.n_boot,
        )
        res["sets"][name] = {
            "n_subset": len(rs),
            "spearman": {**agg, "ci_low": boot["ci_low"], "ci_high": boot["ci_high"]},
            "same_top1": top1,
            "by_k": {
                str(k): SV.score_spearman(
                    {r["query_id"]: a[r["query_id"]] for r in rs if r["k"] == k},
                    {r["query_id"]: b[r["query_id"]] for r in rs if r["k"] == k},
                )
                for k in sorted({r["k"] for r in rs})
            },
        }
        lines.append(
            f"| {name} | {len(rs)} | {agg['n']} | {agg['constant']} | {agg['mean']:.3f} "
            f"[{boot['ci_low']:.3f}, {boot['ci_high']:.3f}] | {agg['median']:.3f} | {agg['q25']:.3f} | "
            f"{top1['rate']:.3f} [{top1['ci_low']:.3f}, {top1['ci_high']:.3f}] |"
        )
    return res, lines


# ---- corpus part on the contested subgroup -------------------------------------------------------


def corpus_part(args) -> dict:
    D = RC.load(args.n_boot, args.sources_file)
    name, rows = D["frames"]["twin-both"]
    res: dict = {"sources_file": args.sources_file, "subset": "twin-both (contested)", "n": len(rows)}
    lines = [
        "# S5 - S4@50 by corpus part on the contested near-miss subgroup (test set)",
        "",
        f"Contested = near-miss queries whose source and HN video are both in S4@50's top 10 "
        f"(n = {len(rows)}). Paired differences, 95% cluster-bootstrap CIs over source videos "
        f"({args.n_boot:,} resamples).",
        "",
        "| labels | metric | part | n | S5 - S4@50 |",
        "|---|---|---|---|---|",
    ]
    for kind in ("expanded", "strict"):
        sc = D["scores"][kind][name]
        res[kind] = {}
        for m in ("ndcg@10", "recall@1"):
            cells = PV.paired(sc["S5"], sc["S4@50"], rows, m, "dataset", args.n_boot)
            cells["all"] = PV.paired(sc["S5"], sc["S4@50"], rows, m, None, args.n_boot)["all"]
            res[kind][m] = cells
            for part, c in cells.items():
                lines.append(f"| {kind} | {m} | {part} | {c['n']} | {ci(c)} |")
    return res, lines


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("item", choices=("collinearity", "spearman", "corpus-part"))
    ap.add_argument("--sources-file", help="the test set, as systems.py --sources-file")
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--out", help=f"output folder (default {OUT}/<item>)")
    args = ap.parse_args(argv)
    fn = {"collinearity": collinearity, "spearman": spearman, "corpus-part": corpus_part}[args.item]
    res, lines = fn(args)
    out = Path(args.out) if args.out else OUT / args.item.replace("-", "_")
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"wrote {out}/summary.json and {out}/report.md")


if __name__ == "__main__":
    main()
