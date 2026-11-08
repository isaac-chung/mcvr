"""Rebuttal W12: a cheap local reranker. A cross-encoder (BAAI/bge-reranker-v2-m3, or
--model BAAI/bge-reranker-base) scores (query, description) for S0's top 50 of every test query
and reranks them; the rest of S0's list follows (src.systems.base.merge_reranked, as S4@50).
FREE: no API call; the model weights come from Hugging Face once.

    env -u OPENAI_API_KEY -u GEMINI_API_KEY uv run python paper/cross_encoder.py \
        [--model BAAI/bge-reranker-v2-m3] [--device mps] [--batch 64] [--max-length 512]
    # --limit N: score the first N queries per set only and print the time estimate

Inputs: the queries (data/mc/full/queries.jsonl, queries_twin.jsonl, as loaded by
pipeline/run_systems.py), S0's saved rankings (results/full/runs/<set>/S0.jsonl) and the index
descriptions (data/mc/<dataset>/descriptions_fps05.jsonl, the texts S2 / S4 read). A candidate
with an empty description (zero vector in the index) gets the lowest score (ties: S0 order).

Evaluation as paper/fusion_variants.py (shared code): nDCG@10 and R@1 by set and k, strict and
expanded labels, paired cluster-bootstrap CIs over sources (10,000 resamples) for CE@50 - S0,
S4@50 - CE@50 and S2@50 - CE@50, and the share of CE@50's top-10 slots never judged.

Outputs: results/full/rebuttal/cross_encoder/{report.md, summary.json, runs/<set>/CE@50.jsonl}
(the trace holds S0's top 50 and the cross-encoder scores; cost.seconds is the scoring time per
query on this machine).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SV = _load(ROOT / "paper/fusion_variants.py", "_ce_s1_variants")
FS, PV = SV.FS, SV.PV

from src.systems.base import merge_reranked
from src.systems.verify import rank_by_score

OUT = Path("results/full/rebuttal/cross_encoder")
DEPTH = 50
NAME = f"CE@{DEPTH}"


def device_of(want: str | None) -> str:
    import torch

    if want:
        return want
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def score_all(model, E: dict, limit: int | None, batch: int) -> tuple[dict, dict]:
    """({set: {qid: record}}, timing): S0's top DEPTH per query reranked by the cross-encoder."""
    desc = E["ctx"]["descriptions"]
    out: dict = {}
    timing = {"pairs": 0, "empty_descriptions": 0, "seconds": 0.0}
    for name in SV.SETS:
        rows = E["sets"][name][:limit] if limit else E["sets"][name]
        s0 = E["ranks"][name]["S0"]
        pairs, spans = [], {}
        for r in rows:
            first = s0[r["query_id"]][:DEPTH]
            spans[r["query_id"]] = (len(pairs), first)
            pairs += [(r["query"], desc.get(d, "")) for d, _ in first]
        start = time.perf_counter()
        scores = model.predict(
            [p for p in pairs], batch_size=batch, show_progress_bar=True, convert_to_numpy=True
        )
        seconds = time.perf_counter() - start
        timing["seconds"] += seconds
        timing["pairs"] += len(pairs)
        per = seconds / max(1, len(rows))
        floor = float(scores.min()) - 1.0 if len(scores) else 0.0
        recs = {}
        for r in rows:
            q = r["query_id"]
            i, first = spans[q]
            sc = []
            for j, (d, _) in enumerate(first):
                empty = not desc.get(d, "").strip()
                timing["empty_descriptions"] += empty
                sc.append(floor if empty else float(scores[i + j]))
            recs[q] = {
                "ranking": merge_reranked(s0[q], rank_by_score(first, sc)),
                "cost": {
                    "searches": 1,
                    "llm_calls": 0,
                    "tokens_in": 0,
                    "tokens_out": 0,
                    "seconds": per,
                    "images": 0,
                    "model_version": "",
                },
                "trace": {
                    "n": DEPTH,
                    "first_stage": [d for d, _ in first],
                    "scores": {d: round(s, 6) for (d, _), s in zip(first, sc)},
                },
            }
        out[name] = recs
        print(f"{name}: {len(rows)} queries, {len(pairs)} pairs, {seconds:.0f} s", flush=True)
    return out, timing


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="BAAI/bge-reranker-v2-m3")
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--fp16", action="store_true", help="half precision weights")
    ap.add_argument("--limit", type=int, help="first N queries per set only (timing estimate)")
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument(
        "--report-only", action="store_true", help="rewrite report.md from summary.json"
    )
    ap.add_argument(
        "--sources-file",
        help="keep only the queries of these sources ({'sources': [...]}), as systems.py",
    )
    ap.add_argument(
        "--from-saved",
        action="store_true",
        help=f"no scoring: re-evaluate the saved runs/<set>/{NAME}.jsonl (with --sources-file, "
        "on the test set) against the current qrels",
    )
    ap.add_argument("--out", help=f"output folder (default {OUT}; runs are read from {OUT})")
    args = ap.parse_args(argv)
    out = Path(args.out) if args.out else OUT
    if args.report_only:
        res = json.loads((out / "summary.json").read_text())
        pairs = [(NAME, "S0"), ("S4@50", NAME), ("S2@50", NAME), ("S4@50", "S0")]
        (out / "report.md").write_text("\n".join(report(res, res["systems"], pairs)))
        return
    if args.from_saved:
        evaluate_saved(args, out)
        return

    import sentence_transformers
    import torch
    from sentence_transformers import CrossEncoder

    t0 = time.time()
    E = SV.load_eval(args.n_boot)
    check = SV.sanity(E)
    device = device_of(args.device)
    start = time.perf_counter()
    kwargs = {"torch_dtype": torch.float16} if args.fp16 else {}
    model = CrossEncoder(args.model, device=device, max_length=args.max_length, model_kwargs=kwargs)
    load_s = time.perf_counter() - start
    print(f"{args.model} on {device}, loaded in {load_s:.0f} s", flush=True)
    recs, timing = score_all(model, E, args.limit, args.batch)
    n_q = sum(len(v) for v in recs.values())
    total_q = sum(len(E["sets"][n]) for n in SV.SETS)
    est = timing["seconds"] / max(1, n_q) * total_q
    print(f"scored {n_q} queries in {timing['seconds']:.0f} s; all {total_q}: ~{est / 60:.1f} min")
    if args.limit:
        return
    ranks = {n: dict(E["ranks"][n]) for n in SV.SETS}
    for name in SV.SETS:
        by_id = {r["query_id"]: r for r in E["sets"][name]}
        SV.write_runs(OUT, name, NAME, recs[name], by_id)
        ranks[name][NAME] = {q: rec["ranking"] for q, rec in recs[name].items()}
    systems = ["S0", "S1", NAME, "S2@50", "S4@50"]
    pairs = [(NAME, "S0"), ("S4@50", NAME), ("S2@50", NAME), ("S4@50", "S0")]
    res = SV.evaluate(E, ranks, systems, pairs, [NAME])
    run = {
        "model": args.model,
        "device": device,
        "dtype": "float16" if args.fp16 else "float32",
        "max_length": args.max_length,
        "batch": args.batch,
        "depth": DEPTH,
        "load_seconds": load_s,
        "scoring_seconds": timing["seconds"],
        "pairs": timing["pairs"],
        "queries": n_q,
        "ms_per_query": 1000 * timing["seconds"] / max(1, n_q),
        "empty_descriptions": timing["empty_descriptions"],
        "machine": f"{platform.machine()} {platform.platform()}",
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
    }
    res |= {
        "sanity": check,
        "run": run,
        "seconds": time.time() - t0,
        "n_boot": E["S"].n_boot,
        "systems": systems,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "report.md").write_text("\n".join(report(res, systems, pairs)))
    print(f"wrote {OUT}/report.md and {OUT}/summary.json")


def evaluate_saved(args, out: Path) -> None:
    """The saved CE@50 rankings re-evaluated (no model), restricted to --sources-file."""
    t0 = time.time()
    E = SV.load_eval(args.n_boot, args.sources_file)
    check = SV.sanity_summary_only(E) if args.sources_file else SV.sanity(E)
    saved = json.loads((OUT / "summary.json").read_text())
    ranks = {n: dict(E["ranks"][n]) for n in SV.SETS}
    for name in SV.SETS:
        ce = FS.read_ranking(OUT / "runs" / name / f"{NAME}.jsonl")
        ranks[name][NAME] = {q: r for q, r in ce.items() if q in {x["query_id"] for x in E["sets"][name]}}
    systems = ["S0", "S1", NAME, "S2@50", "S4@50"]
    pairs = [(NAME, "S0"), ("S4@50", NAME), ("S2@50", NAME), ("S4@50", "S0")]
    res = SV.evaluate(E, ranks, systems, pairs, [NAME])
    res |= {
        "sanity": check,
        "run": {**saved["run"], "from_saved": str(OUT / "runs")},
        "sources_file": args.sources_file,
        "seconds": time.time() - t0,
        "n_boot": E["S"].n_boot,
        "systems": systems,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    res = json.loads((out / "summary.json").read_text())  # per-k keys as strings, as --report-only
    (out / "report.md").write_text("\n".join(report(res, systems, pairs)))
    print(f"wrote {out}/report.md and {out}/summary.json")


def report(res: dict, systems: list[str], pairs: list[tuple]) -> list[str]:
    g = lambda pair, n, kind="expanded", m="ndcg@10": SV.get(res, kind, pair, n, m)
    ci, run = SV.ci, res["run"]
    unj = {n: res["unjudged"][n][NAME]["all"]["share"] for n in SV.SETS}
    unj4 = {n: res["unjudged"][n]["S4@50"]["all"]["share"] for n in SV.SETS}
    md = ["# W12: a local cross-encoder reranker on S0's top 50", "", "## Key takeaways", ""]
    md += [
        f"- A local cross-encoder ({run['model']}, no API call, {run['ms_per_query']:.0f} ms per "
        f"query on {run['device']}) reranking S0's top 50 changes nDCG@10 (expanded) by nested "
        f"{ci(g(f'{NAME} - S0', 'natural'))}, twin {ci(g(f'{NAME} - S0', 'twin'))} over S0: no "
        f"overall gain. It helps single-constraint queries (k = 1: nested "
        f"{ci(SV.get(res, 'expanded', f'{NAME} - S0', 'natural', 'ndcg@10', '1'))}, twin "
        f"{ci(SV.get(res, 'expanded', f'{NAME} - S0', 'twin', 'ndcg@10', '1'))}) and hurts at "
        f"k = 4 (nested {ci(SV.get(res, 'expanded', f'{NAME} - S0', 'natural', 'ndcg@10', '4'))}, "
        f"twin {ci(SV.get(res, 'expanded', f'{NAME} - S0', 'twin', 'ndcg@10', '4'))}), the "
        "multi-constraint case the paper is about.",
        f"- The LLM description reranker S4@50 stays above it: S4@50 − CE@50 nested "
        f"{ci(g(f'S4@50 - {NAME}', 'natural'))}, twin {ci(g(f'S4@50 - {NAME}', 'twin'))}; the "
        f"constraint verifier S2@50 − CE@50 nested {ci(g(f'S2@50 - {NAME}', 'natural'))}, twin "
        f"{ci(g(f'S2@50 - {NAME}', 'twin'))}.",
        f"- Strict labels (source only) tell the same story: CE@50 − S0 nested "
        f"{ci(g(f'{NAME} - S0', 'natural', 'strict'))}, twin "
        f"{ci(g(f'{NAME} - S0', 'twin', 'strict'))}; S4@50 − CE@50 nested "
        f"{ci(g(f'S4@50 - {NAME}', 'natural', 'strict'))}, twin "
        f"{ci(g(f'S4@50 - {NAME}', 'twin', 'strict'))}.",
        f"- CE@50 was not in the judge pool: {unj['natural']:.1%} (nested) / {unj['twin']:.1%} "
        f"(twin) of its top-10 slots are unjudged and count as non-relevant under expanded labels "
        f"(S4@50: {unj4['natural']:.1%} / {unj4['twin']:.1%}), so its expanded-label scores are a "
        "lower bound; the strict-label comparison is unaffected.",
        "",
        "## Setup",
        "",
        f"- Model {run['model']} (sentence-transformers {run['sentence_transformers']} CrossEncoder, "
        f"torch {run['torch']}, {run['dtype']}, max length {run['max_length']}, batch "
        f"{run['batch']}), device {run['device']} ({run['machine']}).",
        f"- {run['queries']:,} queries, {run['pairs']:,} (query, description) pairs; scoring wall "
        f"time {run['scoring_seconds'] / 60:.1f} min ({run['ms_per_query']:.0f} ms per query), "
        f"model load {run['load_seconds']:.0f} s. Empty descriptions in the top 50 (scored last): "
        f"{run['empty_descriptions']}.",
        "- Reranks S0's top 50 by score (ties in S0 order), then S0's tail, exactly as S4@50 is "
        "merged. Descriptions: data/mc/<dataset>/descriptions_fps05.jsonl, the texts S2 / S4 read.",
        "- Eval check: S1 − S0 and S4@50 − S0 (nDCG@10, expanded) recomputed from the saved runs "
        "equal results/full/eval/summary.json to 1e-9 ("
        + "; ".join(f"{k} {v['ours']['mean']:+.3f}" for k, v in res["sanity"].items())
        + ").",
        f"- 95% CIs: paired cluster bootstrap over source videos, {res['n_boot']:,} resamples "
        "(pilot_verify.paired). natural = the paper's nested set.",
        "",
        "## Tables",
        "",
    ]
    return md + SV.tables(res, systems, pairs, [NAME])


if __name__ == "__main__":
    main()
