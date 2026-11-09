"""Rebuttal W12: a cheap local reranker on S0's top 50 of every test query; the rest of S0's list
follows (src.systems.base.merge_reranked, as S4@50). FREE: no API call; the model weights
come from Hugging Face once. Three kinds (--kind):

- ce (CE@50): a cross-encoder (BAAI/bge-reranker-v2-m3, or --model BAAI/bge-reranker-base)
  scores each (query, description) pair.
- qwen (CE@50, in cross_encoder_qwen/): the same with Qwen/Qwen3-VL-Reranker-2B (bfloat16, the
  model's default instruction).
- pw (PW@50): the same with lightonai/LightOn-rerank-PW-2B, the pointwise sibling of the
  LightOn listwise model (bfloat16, its bundled prompt; score logit(Yes) - logit(No)).
- lw (LW@50): a generative listwise reranker (lightonai/LightOn-rerank-LW-0.8B) ranks 4
  descriptions per prompt by generating a permutation, with the model card's sliding window
  (window 4, stride 2, bottom to top); a window whose output does not parse keeps its order.
- late (LI@50): a late-interaction encoder (perplexity-ai/pplx-embed-v2-late-0.6b) scores each
  description by MaxSim against the query.

    env -u OPENAI_API_KEY -u GEMINI_API_KEY uv run python pipeline/cross_encoders.py \
        [--kind ce|qwen|pw|lw|late] [--model ...] [--device mps] [--batch 64] [--max-length 512] \
        [--sources-file results/full/select/sources100.json]
    # --limit N: score the first N queries per set only and print the time estimate

Inputs: the queries (data/mc/full/queries.jsonl, queries_twin.jsonl, as loaded by
pipeline/run_systems.py), S0's saved rankings (results/full/runs/<set>/S0.jsonl) and the index
descriptions (data/mc/<dataset>/descriptions_fps05.jsonl, the texts S2 / S4 read). A candidate
with an empty description (zero vector in the index) gets the lowest score (ties: S0 order).

Evaluation as paper/fusion_variants.py (shared code): nDCG@10 and R@1 by set and k, strict and
expanded labels, paired cluster-bootstrap CIs over sources (10,000 resamples) for <X>@50 - S0,
S4@50 - <X>@50 and S2@50 - <X>@50, and the share of <X>@50's top-10 slots never judged.

Outputs: results/full/rebuttal/<cross_encoder | cross_encoder_qwen | pointwise_pw | listwise_lw | late_pplx>/{report.md, summary.json,
runs/<set>/<X>@50.jsonl} (the trace holds S0's top 50 and the scores; for lw the score is minus
the final position; cost.seconds is the scoring time per query on this machine).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import re
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

DEPTH = 50
# kind: (system prefix, default model, output folder, label in the report)
KINDS = {
    "ce": ("CE", "BAAI/bge-reranker-v2-m3", "cross_encoder", "cross-encoder"),
    "qwen": ("CE", "Qwen/Qwen3-VL-Reranker-2B", "cross_encoder_qwen", "cross-encoder"),
    "pw": ("PW", "lightonai/LightOn-rerank-PW-2B", "pointwise_pw", "pointwise reranker"),
    "lw": ("LW", "lightonai/LightOn-rerank-LW-0.8B", "listwise_lw", "listwise reranker"),
    "late": (
        "LI",
        "perplexity-ai/pplx-embed-v2-late-0.6b",
        "late_pplx",
        "late-interaction encoder",
    ),
}
# set in main() from --kind; the defaults are the cross-encoder run the paper reports
NAME = f"CE@{DEPTH}"
OUT = Path("results/full/rebuttal/cross_encoder")
LABEL = "cross-encoder"

# the LightOn-rerank-LW model card's prompt and output format
LW_PROMPT = "<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
LW_PERM = re.compile(r"\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]\s*>\s*\[(\d)\]")
LW_WINDOW, LW_STRIDE = 4, 2


def device_of(want: str | None) -> str:
    import torch

    if want:
        return want
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def ce_scorer(model, batch: int):
    """Cross-encoder: one score per (query, description) pair."""

    def score(queries: list[str], cands: list[list[str]]) -> tuple[list[list[float]], dict]:
        pairs = [(q, d) for q, ds in zip(queries, cands) for d in ds]
        flat = model.predict(pairs, batch_size=batch, show_progress_bar=True, convert_to_numpy=True)
        out, i = [], 0
        for ds in cands:
            out.append([float(x) for x in flat[i : i + len(ds)]])
            i += len(ds)
        return out, {}

    return score


def late_scorer(model, batch: int):
    """Late interaction: MaxSim of the query's token vectors against each description's; each
    distinct description is encoded once."""

    def score(queries: list[str], cands: list[list[str]]) -> tuple[list[list[float]], dict]:
        uniq = list(dict.fromkeys(d for ds in cands for d in ds))
        docs = model.encode_document(uniq, batch_size=batch, show_progress_bar=True)
        at = dict(zip(uniq, docs))
        qs = model.encode_query(queries, batch_size=batch)
        out = [
            [float(x) for x in model.similarity([q], [at[d] for d in ds])[0]]
            for q, ds in zip(qs, cands)
        ]
        return out, {"documents_encoded": len(uniq)}

    return score


def lw_windows(n: int) -> list[tuple[int, int]]:
    """The model card's sliding window over n candidates: (start, end) from the bottom up."""
    positions = list(range(max(0, n - LW_WINDOW), -1, -LW_STRIDE))
    if positions and positions[-1] != 0:
        positions.append(0)
    spans = []
    for pos in positions:
        end = min(pos + LW_WINDOW, n)
        start = max(0, end - LW_WINDOW)
        if end - start >= 2:
            spans.append((start, end))
    return spans


def lw_scorer(model, processor, batch: int):
    """Listwise: the sliding window of the model card, run in lockstep over all queries so each
    window step is one batch of prompts. Score = minus the final position."""
    tok = processor.tokenizer
    tok.padding_side = "left"

    def prompt(query: str, docs: list[str]) -> str:
        body = "\n".join(f"[{i + 1}]: {d}" for i, d in enumerate(docs))
        user = (
            f"Query: {query}\n\nRank these passages from most to least relevant:\n{body}\n\n"
            "Ranking:"
        )
        return LW_PROMPT.format(user=user)

    def generate(prompts: list[str]) -> list[str]:
        texts = []
        for i in range(0, len(prompts), batch):
            inputs = processor(text=prompts[i : i + batch], return_tensors="pt", padding=True)
            inputs = inputs.to(model.device)
            out = model.generate(
                **inputs, max_new_tokens=30, do_sample=False, pad_token_id=tok.eos_token_id
            )
            texts += processor.batch_decode(
                out[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
            )
        return texts

    def score(queries: list[str], cands: list[list[str]]) -> tuple[list[list[float]], dict]:
        orders = [list(range(len(ds))) for ds in cands]
        spans = [lw_windows(len(ds)) for ds in cands]
        info = {"windows": 0, "unparsed": 0}
        for step in range(max((len(s) for s in spans), default=0)):
            jobs = []
            for qi, (q, ds) in enumerate(zip(queries, cands)):
                if step >= len(spans[qi]):
                    continue
                start, end = spans[qi][step]
                docs = [ds[i] for i in orders[qi][start:end]]
                docs += [docs[-1]] * (LW_WINDOW - len(docs))  # pad as the model card says
                jobs.append((qi, start, end, prompt(q, docs)))
            texts = generate([j[3] for j in jobs])
            for (qi, start, end, _), text in zip(jobs, texts):
                info["windows"] += 1
                m = LW_PERM.search(text)
                perm = [int(g) - 1 for g in m.groups()] if m else []
                if sorted(perm) != list(range(LW_WINDOW)):
                    info["unparsed"] += 1
                    continue
                window = orders[qi][start:end]
                orders[qi][start:end] = [window[j] for j in perm if j < end - start]
            print(f"window step {step + 1}: {len(jobs)} prompts", flush=True)
        out = []
        for order in orders:
            sc = [0.0] * len(order)
            for pos, i in enumerate(order):
                sc[i] = -float(pos)
            out.append(sc)
        return out, info

    return score


def load_scorer(kind: str, args, device: str):
    """(scorer, versions) for --kind."""
    import torch

    if kind in ("ce", "qwen", "pw"):
        from sentence_transformers import CrossEncoder

        kwargs = {"torch_dtype": torch.float16} if args.fp16 else {}
        if kind in ("qwen", "pw"):
            kwargs = {"torch_dtype": torch.bfloat16}
        model = CrossEncoder(
            args.model, device=device, max_length=args.max_length, model_kwargs=kwargs
        )
        return ce_scorer(model, args.batch)
    if kind == "late":
        from sentence_transformers import MultiVectorEncoder

        kwargs = {"torch_dtype": torch.float16} if args.fp16 else {}
        model = MultiVectorEncoder(args.model, device=device, model_kwargs=kwargs)
        return late_scorer(model, args.batch)
    from transformers import AutoModelForImageTextToText, AutoProcessor

    model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.bfloat16)
    model = model.to(device).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    return lw_scorer(model, processor, args.batch)


def score_all(scorer, E: dict, limit: int | None) -> tuple[dict, dict]:
    """({set: {qid: record}}, timing): S0's top DEPTH per query reranked by the scorer."""
    desc = E["ctx"]["descriptions"]
    out: dict = {}
    timing = {"pairs": 0, "empty_descriptions": 0, "seconds": 0.0}
    for name in SV.SETS:
        rows = E["sets"][name][:limit] if limit else E["sets"][name]
        s0 = E["ranks"][name]["S0"]
        firsts = [s0[r["query_id"]][:DEPTH] for r in rows]
        start = time.perf_counter()
        scores, info = scorer(
            [r["query"] for r in rows], [[desc.get(d, "") for d, _ in first] for first in firsts]
        )
        seconds = time.perf_counter() - start
        for k, v in info.items():
            timing[k] = timing.get(k, 0) + v
        n_pairs = sum(len(f) for f in firsts)
        timing["seconds"] += seconds
        timing["pairs"] += n_pairs
        per = seconds / max(1, len(rows))
        floor = min((min(s) for s in scores if s), default=1.0) - 1.0
        recs = {}
        for r, first, raw in zip(rows, firsts, scores):
            sc = []
            for (d, _), s in zip(first, raw):
                empty = not desc.get(d, "").strip()
                timing["empty_descriptions"] += empty
                sc.append(floor if empty else s)
            recs[r["query_id"]] = {
                "ranking": merge_reranked(s0[r["query_id"]], rank_by_score(first, sc)),
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
        print(f"{name}: {len(rows)} queries, {n_pairs} pairs, {seconds:.0f} s", flush=True)
    return out, timing


def main(argv: list[str] | None = None) -> None:
    global NAME, OUT, LABEL
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--kind", choices=sorted(KINDS), default="ce")
    ap.add_argument("--model", help="default: the kind's model (see the docstring)")
    ap.add_argument("--device")
    ap.add_argument("--batch", type=int, default=64, help="pairs, documents or lw prompts")
    ap.add_argument("--max-length", type=int, default=512, help="ce only")
    ap.add_argument("--fp16", action="store_true", help="half precision weights (ce, late)")
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
        help="no scoring: re-evaluate the saved runs/<set>/<X>@50.jsonl (with --sources-file, "
        "on the test set) against the current qrels",
    )
    ap.add_argument(
        "--config", help="--from-saved: labels of this config (default configs/mc_full.yaml)"
    )
    ap.add_argument("--out", help="output folder (default by kind; runs are read from the default)")
    args = ap.parse_args(argv)
    prefix, default_model, folder, LABEL = KINDS[args.kind]
    NAME, OUT = f"{prefix}@{DEPTH}", Path("results/full/rebuttal") / folder
    args.model = args.model or default_model
    out = Path(args.out) if args.out else OUT
    pairs = [(NAME, "S0"), ("S4@50", NAME), ("S2@50", NAME), ("S4@50", "S0")]
    if args.report_only:
        res = json.loads((out / "summary.json").read_text())
        (out / "report.md").write_text("\n".join(report(res, res["systems"], pairs)))
        return
    if args.from_saved:
        evaluate_saved(args, out)
        return

    import sentence_transformers
    import torch
    import transformers

    t0 = time.time()
    E = SV.load_eval(args.n_boot, args.sources_file)
    check = SV.sanity_summary_only(E) if args.sources_file else SV.sanity(E)
    device = device_of(args.device)
    start = time.perf_counter()
    scorer = load_scorer(args.kind, args, device)
    load_s = time.perf_counter() - start
    print(f"{args.model} on {device}, loaded in {load_s:.0f} s", flush=True)
    recs, timing = score_all(scorer, E, args.limit)
    n_q = sum(len(v) for v in recs.values())
    total_q = sum(len(E["sets"][n]) for n in SV.SETS)
    est = timing["seconds"] / max(1, n_q) * total_q
    print(f"scored {n_q} queries in {timing['seconds']:.0f} s; all {total_q}: ~{est / 60:.1f} min")
    if args.kind == "lw":
        print(f"lw windows {timing['windows']}, unparsed {timing['unparsed']}")
    if args.limit:
        return
    ranks = {n: dict(E["ranks"][n]) for n in SV.SETS}
    for name in SV.SETS:
        by_id = {r["query_id"]: r for r in E["sets"][name]}
        SV.write_runs(out, name, NAME, recs[name], by_id)
        ranks[name][NAME] = {q: rec["ranking"] for q, rec in recs[name].items()}
    systems = ["S0", "S1", NAME, "S2@50", "S4@50"]
    res = SV.evaluate(E, ranks, systems, pairs, [NAME])
    if args.kind in ("lw", "qwen", "pw"):
        dtype = "bfloat16"
    else:
        dtype = "float16" if args.fp16 else "float32"
    run = {
        "kind": args.kind,
        "model": args.model,
        "device": device,
        "dtype": dtype,
        "max_length": args.max_length if args.kind in ("ce", "qwen", "pw") else None,
        "batch": args.batch,
        "depth": DEPTH,
        "load_seconds": load_s,
        "scoring_seconds": timing["seconds"],
        "pairs": timing["pairs"],
        "queries": n_q,
        "ms_per_query": 1000 * timing["seconds"] / max(1, n_q),
        "empty_descriptions": timing["empty_descriptions"],
        "lw_windows": timing.get("windows"),
        "lw_unparsed": timing.get("unparsed"),
        "machine": f"{platform.machine()} {platform.platform()}",
        "torch": torch.__version__,
        "sentence_transformers": sentence_transformers.__version__,
        "transformers": transformers.__version__,
    }
    res |= {
        "sanity": check,
        "run": run,
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


def evaluate_saved(args, out: Path) -> None:
    """The saved <X>@50 rankings re-evaluated (no model), restricted to --sources-file."""
    t0 = time.time()
    E = SV.load_eval(args.n_boot, args.sources_file, args.config)
    if args.config:  # other labels on purpose: the check reports the shift instead of stopping
        check = SV.sanity_summary_only(E, strict=False)
    else:
        check = SV.sanity_summary_only(E) if args.sources_file else SV.sanity(E)
    saved = json.loads((OUT / "summary.json").read_text())
    ranks = {n: dict(E["ranks"][n]) for n in SV.SETS}
    for name in SV.SETS:
        ce = FS.read_ranking(OUT / "runs" / name / f"{NAME}.jsonl")
        ranks[name][NAME] = {
            q: r for q, r in ce.items() if q in {x["query_id"] for x in E["sets"][name]}
        }
    systems = ["S0", "S1", NAME, "S2@50", "S4@50"]
    pairs = [(NAME, "S0"), ("S4@50", NAME), ("S2@50", NAME), ("S4@50", "S0")]
    res = SV.evaluate(E, ranks, systems, pairs, [NAME])
    res |= {
        "sanity": check,
        "run": {**saved["run"], "from_saved": str(OUT / "runs")},
        "sources_file": args.sources_file,
        "label_dir": str(E["S"].judge_dir),
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
    md = [f"# W12: a local {LABEL} on S0's top 50", "", "## Key takeaways", ""]
    md += [
        f"- A local {LABEL} ({run['model']}, no API call, {run['ms_per_query']:.0f} ms per "
        f"query on {run['device']}) reranking S0's top 50 changes nDCG@10 (expanded) by nested "
        f"{ci(g(f'{NAME} - S0', 'natural'))}, twin {ci(g(f'{NAME} - S0', 'twin'))} over S0. "
        f"By k: k = 1 nested "
        f"{ci(SV.get(res, 'expanded', f'{NAME} - S0', 'natural', 'ndcg@10', '1'))}, twin "
        f"{ci(SV.get(res, 'expanded', f'{NAME} - S0', 'twin', 'ndcg@10', '1'))}; "
        f"k = 4 nested {ci(SV.get(res, 'expanded', f'{NAME} - S0', 'natural', 'ndcg@10', '4'))}, "
        f"twin {ci(SV.get(res, 'expanded', f'{NAME} - S0', 'twin', 'ndcg@10', '4'))}.",
        f"- S4@50 − {NAME} nested {ci(g(f'S4@50 - {NAME}', 'natural'))}, twin "
        f"{ci(g(f'S4@50 - {NAME}', 'twin'))}; S2@50 − {NAME} nested "
        f"{ci(g(f'S2@50 - {NAME}', 'natural'))}, twin {ci(g(f'S2@50 - {NAME}', 'twin'))}.",
        f"- Strict labels (source only): {NAME} − S0 nested "
        f"{ci(g(f'{NAME} - S0', 'natural', 'strict'))}, twin "
        f"{ci(g(f'{NAME} - S0', 'twin', 'strict'))}; S4@50 − {NAME} nested "
        f"{ci(g(f'S4@50 - {NAME}', 'natural', 'strict'))}, twin "
        f"{ci(g(f'S4@50 - {NAME}', 'twin', 'strict'))}.",
        f"- {NAME} was not in the judge pool: {unj['natural']:.1%} (nested) / {unj['twin']:.1%} "
        f"(twin) of its top-10 slots are unjudged and count as non-relevant under expanded labels "
        f"(S4@50: {unj4['natural']:.1%} / {unj4['twin']:.1%}), so its expanded-label scores are a "
        "lower bound; the strict-label comparison is unaffected.",
        "",
        "## Setup",
        "",
        f"- Model {run['model']} ({LABEL}; sentence-transformers "
        f"{run['sentence_transformers']}, torch {run['torch']}, {run['dtype']}"
        + (f", max length {run['max_length']}" if run.get("max_length") else "")
        + f", batch {run['batch']}), device {run['device']} ({run['machine']}).",
        f"- {run['queries']:,} queries, {run['pairs']:,} (query, description) pairs; scoring wall "
        f"time {run['scoring_seconds'] / 60:.1f} min ({run['ms_per_query']:.0f} ms per query), "
        f"model load {run['load_seconds']:.0f} s. Empty descriptions in the top 50 (scored last): "
        f"{run['empty_descriptions']}.",
    ]
    if run.get("kind") == "lw":
        md.append(
            f"- Listwise: windows of {LW_WINDOW}, stride {LW_STRIDE}, bottom to top, one pass; "
            f"{run['lw_windows']:,} windows, {run['lw_unparsed']} unparsed (kept their order)."
        )
    md += [
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
