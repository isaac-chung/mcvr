"""Rebuttal W5: stronger forms of search-time splitting (S1) built offline. FREE: no model calls.

The reviewer's point: S1 is plain RRF over one search per split constraint, without the full
query as a list and without weights. Built here from the saved splits
(results/full/runs/<set>/splits.json), the description index (results/_indexes/<store>, the
full-run DESC index) and query / constraint embeddings read from the shared embedding cache
(results/_cache/embed.sqlite, keyed per (model, text)) only: a text that is not cached is never
sent to the API; its query is skipped and counted.

    env -u OPENAI_API_KEY -u GEMINI_API_KEY uv run python paper/fusion_variants.py

Systems (same RRF constant, 60, and the same list depth, 500 = configs/mc_full.yaml `depth`,
as S1 in pipeline/run_systems.py stage_retrieve / src.systems.decompose.fuse):

    S1          recomputed (check: must equal the saved S1)
    S1+Q        RRF over the per-constraint lists plus the full-query (S0) list
    S1+2Q       as S1+Q, the full-query list weighted 2
    S1+kQ       as S1+Q, the full-query list weighted k = the query's number of split
                constraints (S1+kQ = S1+Q when the split has one constraint)
    S1-mean     score sum: mean cosine over the constraints, over the whole corpus, top 500
    S1-mean+Q   mean cosine over the constraints and the full query

No weight is chosen on test: every variant is reported. Evaluation (shared with
paper/cross_encoder.py): nDCG@10 and R@1 by set (natural = the paper's "nested", twin)
and by k under strict (source only) and expanded labels (data/mc/full/judge/qrels_expanded.jsonl,
as pipeline/run_systems.py --stage eval), paired cluster-bootstrap CIs over sources (the repo's
pilot_verify.paired, 10,000 resamples), and for expanded labels the share of top-10 slots
whose (source, candidate) pair was never judged (the definition of paper/tab_review_checks.py).
The S1 - S0 and S4@50 - S0 cells are first recomputed from the saved runs and checked against
results/full/eval/summary.json.

Outputs: results/full/rebuttal/s1_variants/{report.md, summary.json, runs/<set>/<system>.jsonl}.
"""

from __future__ import annotations

import argparse
import base64
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JD = _load(ROOT / "pipeline/judge_labels.py", "_s1v_judge")
FS = JD.FS
PV = FS.PV

from src.cache import shared_cache
from src.embedders import MAX_INPUT_TOKENS, truncate_tokens
from src.fusion import rrf
from src.indexes.base import top_k
from src.systems.decompose import fuse

OUT = Path("results/full/rebuttal/s1_variants")
SETS = ("natural", "twin")
SET_LABEL = {"natural": "nested", "twin": "twin"}
KS = ("all", "1", "2", "3", "4")
METRICS = ("ndcg@10", "recall@1")
RRF_K = 60
VARIANTS = ("S1+Q", "S1+2Q", "S1+kQ", "S1-mean", "S1-mean+Q")
SANITY = {  # results/full/report.md, expanded labels, nDCG@10
    ("S1", "S0"): {"natural": -0.107, "twin": -0.111},
    ("S4@50", "S0"): {"natural": 0.200, "twin": 0.197},
}


# ---- shared evaluation (also used by paper/cross_encoder.py) ----------------------------


def load_eval(n_boot: int, sources_file: str | None = None) -> dict:
    """Settings, query rows, saved runs, both label sets and the judged pairs; with
    sources_file (the test set, as systems.py --sources-file) restricted to its queries."""
    S = FS.settings(argparse.Namespace(smoke=False, config=None, out=None, n_boot=n_boot))
    ctx = FS.load_context(S, need_index=False)
    sets = FS.load_sets(S, ctx)
    if sources_file:
        sets = FS.restrict_sets(S, sets, sources_file)
    ranks = {n: FS.load_runs(S.runs / n, ())[0] for n in sets}
    if S.keep:
        ranks = {
            n: {s: {q: v for q, v in r.items() if q in S.keep[n]} for s, r in ranks[n].items()}
            for n in sets
        }
    qrels = {
        "strict": {n: PV.load_qrels(None, rs) for n, rs in sets.items()},
        "expanded": {
            n: PV.load_qrels(S.judge_dir / "qrels_expanded.jsonl", rs) for n, rs in sets.items()
        },
    }
    judged = JD.JP.read_judged(S.judge_dir / "judged.jsonl")
    return {"S": S, "ctx": ctx, "sets": sets, "ranks": ranks, "qrels": qrels, "judged": judged}


def by_k(fn, rows: list[dict]) -> dict:
    """{"all" | k: fn(rows of that group)} (groups with no rows left out)."""
    out = {"all": fn(rows)}
    for k in KS[1:]:
        rs = [r for r in rows if str(r["k"]) == k]
        if rs:
            out[k] = fn(rs)
    return out


def unjudged(rank: dict, rows: list[dict], judged: dict, top: int = 10) -> dict:
    """Top-`top` slots whose (source, candidate) pair is not in judged.jsonl (counted as
    non-relevant under expanded labels), as paper/tab_review_checks.py counts them."""

    def f(rs):
        slots = miss = with_miss = at1 = 0
        for r in rs:
            head = [d for d, _ in rank[r["query_id"]][:top]]
            m = [d for d in head if d not in judged.get(r["source"], {})]
            slots, miss = slots + len(head), miss + len(m)
            with_miss += bool(m)
            at1 += bool(head) and head[0] in m
        return {
            "queries": len(rs),
            "slots": slots,
            "unjudged": miss,
            "share": miss / max(slots, 1),
            "queries_with_unjudged": with_miss,
            "unjudged_at_1": at1,
        }

    return by_k(f, rows)


def evaluate(E: dict, ranks: dict, systems: list[str], pairs: list[tuple], new: list[str]) -> dict:
    """Means and paired CIs (nDCG@10, R@1; all and per k) per label kind and set, restricted to
    the queries every listed system ranked; unjudged top-10 shares for `new` (and S0 / S1 /
    S2@50 / S4@50 for reference)."""
    n_boot = E["S"].n_boot
    res: dict = {"labels": {}, "unjudged": {}, "n": {}}
    rows_of = {}
    for name in SETS:
        have = set.intersection(*(set(ranks[name][s]) for s in systems))
        rows_of[name] = [r for r in E["sets"][name] if r["query_id"] in have]
        res["n"][name] = {"queries": len(rows_of[name]), "of": len(E["sets"][name])}
    for kind, qrels in E["qrels"].items():
        cell: dict = {"means": {}, "paired": {}}
        for name in SETS:
            rs = rows_of[name]
            sc = {
                s: PV.score_system(
                    {r["query_id"]: ranks[name][s][r["query_id"]] for r in rs}, qrels[name]
                )
                for s in systems
            }
            cell["means"][name] = {
                s: PV.mean_table(sc[s], rs, "k") | PV.mean_table(sc[s], rs, None) for s in systems
            }
            cell["paired"][name] = {
                f"{a} - {b}": {
                    m: PV.paired(sc[a], sc[b], rs, m, None, n_boot)
                    | PV.paired(sc[a], sc[b], rs, m, "k", n_boot)
                    for m in METRICS
                }
                for a, b in pairs
            }
        res["labels"][kind] = cell
    for name in SETS:
        res["unjudged"][name] = {
            s: unjudged(ranks[name][s], rows_of[name], E["judged"])
            for s in dict.fromkeys([*new, "S0", "S1", "S2@50", "S4@50"])
            if s in ranks[name]
        }
    return res


def sanity(E: dict) -> dict:
    """S1 - S0 and S4@50 - S0 (nDCG@10, all queries, expanded labels) from the saved runs, against
    results/full/eval/summary.json and the rounded values in results/full/report.md."""
    saved = json.loads((E["S"].out / "eval" / "summary.json").read_text())
    out = {}
    for (a, b), ref in SANITY.items():
        for name in SETS:
            rs, qr = E["sets"][name], E["qrels"]["expanded"][name]
            sa, sb = (PV.score_system(E["ranks"][name][s], qr) for s in (a, b))
            ours = PV.paired(sa, sb, rs, "ndcg@10", None, E["S"].n_boot)["all"]
            theirs = saved["labels"]["expanded"]["paired"][name][f"{a} - {b}"]["ndcg@10"]["None"][
                "all"
            ]
            ok = all(abs(ours[x] - theirs[x]) < 1e-9 for x in ("mean", "ci_low", "ci_high"))
            out[f"{a} - {b} / {name}"] = {
                "ours": ours,
                "summary_json": theirs,
                "report_md": ref[name],
                "match": ok and round(ours["mean"], 3) == ref[name],
            }
    if not all(v["match"] for v in out.values()):
        sys.exit(f"evaluation does not reproduce results/full/eval/summary.json: {out}")
    return out


def sanity_summary_only(E: dict) -> dict:
    """As sanity() against summary.json only (the report.md constants are the full set's)."""
    saved = json.loads((E["S"].out / "eval" / "summary.json").read_text())
    out = {}
    for a, b in SANITY:
        for name in SETS:
            rs, qr = E["sets"][name], E["qrels"]["expanded"][name]
            sa, sb = (PV.score_system(E["ranks"][name][s], qr) for s in (a, b))
            ours = PV.paired(sa, sb, rs, "ndcg@10", None, E["S"].n_boot)["all"]
            theirs = saved["labels"]["expanded"]["paired"][name][f"{a} - {b}"]["ndcg@10"]["None"][
                "all"
            ]
            ok = all(abs(ours[x] - theirs[x]) < 1e-9 for x in ("mean", "ci_low", "ci_high"))
            out[f"{a} - {b} / {name}"] = {"ours": ours, "summary_json": theirs, "match": ok}
    if not all(v["match"] for v in out.values()):
        sys.exit(f"evaluation does not reproduce results/full/eval/summary.json: {out}")
    return out


def load_saved_runs(E: dict) -> tuple[dict, dict]:
    """The variants' saved rankings (OUT/runs/<set>/<variant>.jsonl) in place of build()."""
    recs = {}
    for name in SETS:
        ranks, _ = FS.load_runs(OUT / "runs" / name, ())
        keep = E["S"].keep[name] if E["S"].keep else None
        recs[name] = {
            s: {q: {"ranking": r} for q, r in ranks[s].items() if keep is None or q in keep}
            for s in VARIANTS
            if s in ranks
        }
        missing = [s for s in VARIANTS if s not in recs[name]]
        if missing:
            sys.exit(f"{OUT}/runs/{name}: no saved run for {missing}; run without --from-saved")
    saved = OUT / "summary.json"  # the build stats of the run that wrote the saved rankings
    stats = {
        "index": "(saved runs)",
        "docs": 0,
        "zero_vectors": 0,
        "depth": "-",
        "rrf_k": RRF_K,
        "cache_misses": {"texts": 0},
        "skipped": {},
        "check": {},
    }
    if saved.exists():
        build = json.loads(saved.read_text()).get("build", {})
        if "index" in build:  # a real build's stats, not an earlier --from-saved summary
            stats |= build
    return recs, {**stats, "from_saved": True}


def f3(v: float) -> str:
    return f"{v:+.3f}" if v < 0 or v > 0 else "0.000"


def ci(d: dict) -> str:
    return f"{f3(d['mean'])} [{f3(d['ci_low'])}, {f3(d['ci_high'])}]"


def tables(res: dict, systems: list[str], pairs: list[tuple], new: list[str]) -> list[str]:
    """Markdown: means per set x k, paired CIs per set x k, unjudged shares."""
    md: list[str] = []
    for kind in ("expanded", "strict"):
        cell = res["labels"][kind]
        for m, label in (("ndcg@10", "nDCG@10"), ("recall@1", "R@1")):
            md += [
                f"### {label}, {kind} labels: means",
                "",
                "| system | " + " | ".join(f"{SET_LABEL[n]} {k}" for n in SETS for k in KS) + " |",
                "|---|" + "---:|" * (2 * len(KS)),
            ]
            for s in systems:
                vals = []
                for n in SETS:
                    t = cell["means"][n][s]
                    vals += [
                        f"{t[k if k == 'all' else int(k)][m]:.3f}"
                        if (k if k == "all" else int(k)) in t
                        else ""
                        for k in KS
                    ]
                md.append(f"| {s} | " + " | ".join(vals) + " |")
            md += [
                "",
                f"### {label}, {kind} labels: paired differences [95% CI]",
                "",
                "| pair | set | " + " | ".join(KS) + " |",
                "|---|---|" + "---:|" * len(KS),
            ]
            for a, b in pairs:
                for n in SETS:
                    p = cell["paired"][n][f"{a} - {b}"][m]
                    vals = [
                        ci(p[k if k == "all" else int(k)])
                        if (k if k == "all" else int(k)) in p
                        else ""
                        for k in KS
                    ]
                    md.append(f"| {a} − {b} | {SET_LABEL[n]} | " + " | ".join(vals) + " |")
            md.append("")
    md += [
        "### Unjudged top-10 slots (counted as non-relevant under expanded labels)",
        "",
        (
            "Share of top-10 slots whose (source, candidate) pair was never judged; in brackets the "
            "queries with at least one unjudged slot, and the queries whose top 1 is unjudged."
        ),
        "",
        "| system | set | " + " | ".join(KS) + " |",
        "|---|---|" + "---:|" * len(KS),
    ]
    for n in SETS:
        for s, t in res["unjudged"][n].items():
            vals = [
                f"{t[k]['share']:.1%} ({t[k]['queries_with_unjudged']}, {t[k]['unjudged_at_1']})"
                if k in t
                else ""
                for k in KS
            ]
            md.append(
                f"| {s}{' (new)' if s in new else ''} | {SET_LABEL[n]} | " + " | ".join(vals) + " |"
            )
    return md + [""]


def get(res: dict, kind: str, pair: str, name: str, m: str = "ndcg@10", k="all") -> dict:
    return res["labels"][kind]["paired"][name][pair][m][k]


def write_runs(out: Path, name: str, system: str, records: dict, rows_by_id: dict) -> None:
    PV.write_rankings(out / "runs" / name / f"{system}.jsonl", system, records, rows_by_id)


# ---- S1 variants -------------------------------------------------------------------------------


class CachedVectors:
    """Query / constraint vectors from results/_cache/embed.sqlite only (OpenAIEmbedder's
    namespace, truncation and normalisation); a miss returns None and is never fetched."""

    def __init__(self, spec: str, results_dir: str):
        kind, _, self.model = spec.partition(":")
        assert kind == "openai", spec
        self.cache = shared_cache("embed", results_dir)
        self.namespace = f"embed:openai:{self.model}"
        self.misses: set[str] = set()

    def get(self, text: str) -> np.ndarray | None:
        record = self.cache.get(self.namespace, truncate_tokens(text, MAX_INPUT_TOKENS))
        if record is None:
            self.misses.add(text)
            return None
        v = np.frombuffer(base64.b64decode(record["v"]), np.float32)
        return (v / max(float(np.linalg.norm(v)), 1e-9)).astype(np.float32)


def build(E: dict) -> tuple[dict, dict]:
    """({set: {system: {qid: record}}}, stats) for S0 / S1 recomputed and the variants."""
    S, ctx = E["S"], E["ctx"]
    store = ctx["store"]
    ids = json.loads((store / "DESC.ids.json").read_text())
    matrix = np.load(store / "DESC.npy")
    depth = min(S.depth, len(ids))
    vecs = CachedVectors(S.cfg["embedder"], S.cfg["results_dir"])
    zero = int((np.linalg.norm(matrix, axis=1) == 0).sum())
    out: dict = {}
    stats: dict = {
        "index": str(store),
        "docs": len(ids),
        "zero_vectors": zero,
        "depth": depth,
        "rrf_k": RRF_K,
        "skipped": {},
        "check": {},
    }
    for name in SETS:
        splits = json.loads((S.runs / name / "splits.json").read_text())
        s1_lines = {ln["query_id"]: ln for ln in FS.read_jsonl(S.runs / name / "S1.jsonl")}
        recs: dict = {s: {} for s in ("S0", "S1", *VARIANTS)}
        skipped = []
        for r in E["sets"][name]:
            q = r["query_id"]
            cons = splits[q]
            assert cons == s1_lines[q]["trace"]["constraints"], q
            qv = vecs.get(r["query"])
            cv = [vecs.get(c) for c in cons]
            if qv is None or any(v is None for v in cv):
                skipped.append(q)
                continue
            q_scores = matrix @ qv
            c_scores = [matrix @ v for v in cv]
            q_list = top_k(ids, q_scores, depth)
            c_lists = [top_k(ids, s, depth) for s in c_scores]
            n = len(cons)
            base_cost = s1_lines[q]["cost"]
            cost_q = {**base_cost, "searches": n + 1}
            trace = {"constraints": cons}
            recs["S0"][q] = {"ranking": q_list, "cost": {}, "trace": {}}
            recs["S1"][q] = {"ranking": fuse(c_lists, RRF_K), "cost": base_cost, "trace": trace}
            recs["S1+Q"][q] = {
                "ranking": rrf([*c_lists, q_list], RRF_K),
                "cost": cost_q,
                "trace": trace,
            }
            recs["S1+2Q"][q] = {
                "ranking": rrf([*c_lists, q_list], RRF_K, [1.0] * n + [2.0]),
                "cost": cost_q,
                "trace": trace | {"query_weight": 2},
            }
            recs["S1+kQ"][q] = {
                "ranking": rrf([*c_lists, q_list], RRF_K, [1.0] * n + [float(n)]),
                "cost": cost_q,
                "trace": trace | {"query_weight": n},
            }
            recs["S1-mean"][q] = {
                "ranking": top_k(ids, np.mean(c_scores, axis=0), depth),
                "cost": base_cost,
                "trace": trace,
            }
            recs["S1-mean+Q"][q] = {
                "ranking": top_k(ids, np.mean([*c_scores, q_scores], axis=0), depth),
                "cost": cost_q,
                "trace": trace,
            }
        stats["skipped"][name] = {"queries": len(skipped), "ids": skipped[:50]}
        # checks: recomputed S0 / S1 against the saved runs (top 10 ids; full lists)
        for s in ("S0", "S1"):
            saved = E["ranks"][name][s]
            top_same = sum(
                [d for d, _ in recs[s][q]["ranking"][:10]] == [d for d, _ in saved[q][:10]]
                for q in recs[s]
            )
            full_same = sum(
                [d for d, _ in recs[s][q]["ranking"]] == [d for d, _ in saved[q]] for q in recs[s]
            )
            stats["check"][f"{name}/{s}"] = {
                "queries": len(recs[s]),
                "top10_identical": top_same,
                "full_list_identical": full_same,
            }
        out[name] = recs
    stats["cache_misses"] = {"texts": len(vecs.misses), "examples": sorted(vecs.misses)[:20]}
    return out, stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--sources-file", help="the test set (as systems.py --sources-file)")
    ap.add_argument(
        "--from-saved",
        action="store_true",
        help="re-evaluate the saved variant runs (no embedding cache needed, runs not rewritten)",
    )
    args = ap.parse_args(argv)
    t0 = time.time()
    E = load_eval(args.n_boot, args.sources_file)
    check = sanity_summary_only(E) if args.sources_file else sanity(E)
    print("sanity: S1 - S0 and S4@50 - S0 reproduce results/full/eval/summary.json")
    recs, stats = load_saved_runs(E) if args.from_saved else build(E)
    print(json.dumps({k: stats[k] for k in ("skipped", "check", "cache_misses")}, indent=1))
    ranks = {n: dict(E["ranks"][n]) for n in SETS}
    for name in SETS:
        by_id = {r["query_id"]: r for r in E["sets"][name]}
        for s in VARIANTS:
            if not args.from_saved:
                write_runs(OUT, name, s, recs[name][s], by_id)
            ranks[name][s] = {q: rec["ranking"] for q, rec in recs[name][s].items()}
    systems = ["S0", "S1", *VARIANTS, "S2@50", "S4@50"]
    pairs = [*((v, "S1") for v in VARIANTS), *((v, "S0") for v in VARIANTS), ("S1", "S0")]
    res = evaluate(E, ranks, systems, pairs, list(VARIANTS))
    res |= {
        "sanity": check,
        "build": stats,
        "seconds": time.time() - t0,
        "n_boot": E["S"].n_boot,
        "systems": systems,
        "sources_file": args.sources_file,
        "from_saved": args.from_saved,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "report.md").write_text("\n".join(report(res, systems, pairs)))
    print(f"wrote {OUT}/report.md and {OUT}/summary.json ({time.time() - t0:.0f} s)")


def report(res: dict, systems: list[str], pairs: list[tuple]) -> list[str]:
    st = res["build"]
    g = lambda pair, n, kind="expanded": get(res, kind, pair, n)
    md = ["# W5: stronger search-time splitting (S1 variants), offline", "", "## Key takeaways", ""]
    md += [
        (
            "- Adding the full query as one more RRF list helps S1 (S1+Q − S1, nDCG@10, expanded: "
            f"nested {ci(g('S1+Q - S1', 'natural'))}, twin {ci(g('S1+Q - S1', 'twin'))}), but S1+Q "
            f"stays below the plain full-query search S0 (nested {ci(g('S1+Q - S0', 'natural'))}, "
            f"twin {ci(g('S1+Q - S0', 'twin'))})."
        ),
        "- Weighting the full-query list up moves the fusion towards S0 and closes part of the "
        "gap, never all of it: − S0 nested "
        + ", ".join(f"{v} {f3(g(f'{v} - S0', 'natural')['mean'])}" for v in VARIANTS[:3])
        + "; twin "
        + ", ".join(f"{v} {f3(g(f'{v} - S0', 'twin')['mean'])}" for v in VARIANTS[:3])
        + ". Score fusion (mean cosine) is no better than RRF without the query (S1-mean − S0 "
        f"nested {ci(g('S1-mean - S0', 'natural'))}, twin {ci(g('S1-mean - S0', 'twin'))}) and "
        f"the closest of all with it (S1-mean+Q − S0 nested {ci(g('S1-mean+Q - S0', 'natural'))}, "
        f"twin {ci(g('S1-mean+Q - S0', 'twin'))}). No weight was chosen on test; all are reported.",
        (
            "- Every variant's 95% CI over all queries lies below 0 against S0, in both sets, for "
            "nDCG@10 and R@1, under expanded and strict labels alike; the loss grows with k (at k = 1, "
            "where the split is usually the query itself, the variants tie or slightly beat S0). The RRF variants have "
            "about as many unjudged top-10 slots as the pooled systems, the mean-cosine ones somewhat "
            "more (unjudged table); since strict labels, which need no judging, give the same "
            "verdicts, the expanded labels are not what keeps the variants below S0."
        ),
        (
            "- None of the search-time variants comes near the description reranker S4@50 ("
            f"{res['sanity']['S4@50 - S0 / natural']['ours']['mean']:+.3f} nested / "
            f"{res['sanity']['S4@50 - S0 / twin']['ours']['mean']:+.3f} twin over S0): splitting at "
            "search time, with or without the full query and weights, is not where the gain comes from."
        ),
        "",
        "## Setup",
        "",
        (
            f"- Index {st['index']} ({st['docs']:,} videos, {st['zero_vectors']} zero vectors), list "
            f"depth {st['depth']}, RRF constant {st['rrf_k']}, as S1. Splits: "
            "results/full/runs/<set>/splits.json (S1's own)."
        ),
        "- Embeddings from results/_cache/embed.sqlite only (no API call). Cache misses: "
        f"{st['cache_misses']['texts']} texts; queries skipped: "
        + ", ".join(f"{SET_LABEL[n]} {v['queries']}" for n, v in st["skipped"].items())
        + ". Queries evaluated: "
        + ", ".join(f"{SET_LABEL[n]} {v['queries']} of {v['of']}" for n, v in res["n"].items())
        + ".",
        "- Check, recomputed against saved runs (top-10 identical / full list identical / "
        "queries): "
        + "; ".join(
            f"{k} {v['top10_identical']} / {v['full_list_identical']} / {v['queries']}"
            for k, v in st["check"].items()
        )
        + " (where full lists differ, it is at float32 near-ties deep in the list, from the "
        "batched search).",
        "- Eval check: S1 − S0 and S4@50 − S0 (nDCG@10, expanded) recomputed from the saved runs "
        "equal results/full/eval/summary.json to 1e-9 ("
        + "; ".join(f"{k} {v['ours']['mean']:+.3f}" for k, v in res["sanity"].items())
        + ").",
        (
            "- S1+kQ: k = the query's number of split constraints (S1+kQ = S1+Q for one-constraint "
            "splits). S1-mean: mean cosine over the constraints over the whole corpus; S1-mean+Q adds "
            "the full query as one more term."
        ),
        (
            f"- 95% CIs: paired cluster bootstrap over source videos, {res['n_boot']:,} resamples "
            "(pilot_verify.paired). natural = the paper's nested set."
        ),
        "",
        "## Tables",
        "",
    ]
    return md + tables(res, systems, pairs, list(VARIANTS))


if __name__ == "__main__":
    main()
