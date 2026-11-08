"""Describe and index the smoke videos (docs/plan.md, "Smoke and dev set"). PAID: Gemini + OpenAI.

For every video in data/mc/smoke/sample.json: save 16 frames (data/mc/<dataset>/frames16/; or
the config's frame_policy, e.g. 0.5 fps in configs/mc_smoke_fps05.yaml, frames_fps0.5/),
describe them with describe_v4 on gemini-3.5-flash-lite, embed the descriptions with
text-embedding-3-small and build the DESC index in results/_indexes/<hash>/. All calls go
through the shared caches (results/_cache/vlm.sqlite, embed.sqlite), so a rerun pays only for
what is new. Describing runs on a thread pool with a rate limit on paid calls. Writes
data/mc/<dataset>/descriptions.jsonl entries and, in the index folder, build.json (settings,
calls, tokens, images, seconds) and manifest.json.

    uv run --env-file .env python dev/build_index_dev.py      # the whole sample (lead only)
    uv run --env-file .env python dev/build_index_dev.py --ids <global id> --store <dir>  # check
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import yaml

from src.cache import shared_cache
from src.datasets.mc import (
    McCorpus,
    RateLimit,
    RateLimitedVision,
    corpus_descriptions,
    describe_corpus,
    description_index,
    index_options,
    index_settings,
    index_store,
)
from src.embedders import make_embedder
from src.llm import make_vision
from src.manifest import write_manifest
from src.types import Cost


def cost_summary(cost: Cost) -> dict:
    return {
        "calls": cost.llm_calls,
        "tokens_in": cost.tokens_in,
        "tokens_out": cost.tokens_out,
        "images": cost.images,
        "api_seconds": round(cost.seconds, 1),
        "model_versions": [v for v in cost.model_version.split(",") if v],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_smoke.yaml")
    parser.add_argument("--ids", nargs="*", help="global ids to index instead of the sample")
    parser.add_argument("--store", help="index folder (default results/_indexes/<hash>/)")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    ids = args.ids or sample["videos"]
    corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], ids)

    vlm = make_vision(cfg["describer"], shared_cache("vlm", cfg["results_dir"]))
    limited = RateLimitedVision(vlm.inner, RateLimit(cfg["describe_per_minute"]))
    vlm.inner = limited
    options = index_options(cfg)  # frame policy, labels, descriptions file (default: 16 frames)
    start = time.perf_counter()
    records = describe_corpus(
        corpus,
        vlm,
        cfg["describe_prompt"],
        cfg["describer"],
        cfg["n_frames"],
        cfg["workers"],
        filename=options["filename"],
        frame_policy=options["frame_policy"],
        labels=options["labels"],
    )
    describe_seconds = time.perf_counter() - start
    recorded = sum((Cost(**r["cost"]) for r in records.values()), Cost())  # cached calls incl.
    paid = limited.cost  # calls made in this run
    empty = [d for d, r in records.items() if not r["description"]]

    prompt_name = Path(cfg["describe_prompt"]).name
    descriptions = corpus_descriptions(corpus, prompt_name, options["filename"])
    settings = index_settings(
        corpus.doc_ids,
        cfg["embedder"],
        cfg["describer"],
        cfg["describe_prompt"],
        cfg["n_frames"],
        options["frame_policy"],
        options["labels"],
    )
    store = Path(args.store) if args.store else index_store(cfg["results_dir"], settings)
    embedder = make_embedder(cfg["embedder"], cache=shared_cache("embed", cfg["results_dir"]))
    start = time.perf_counter()
    index = description_index(corpus, descriptions, embedder, store)
    embed_seconds = time.perf_counter() - start

    build = {
        "settings": settings,
        "describe": {
            "videos": len(records),
            "frames": sum(r["n_frames"] for r in records.values()),
            "descriptions_file": options["filename"],
            "empty_descriptions": empty,
            "wall_seconds": round(describe_seconds, 1),
            # every description's cost as first paid (offline cost of the index)
            "recorded": cost_summary(recorded),
            # what this run paid (cache misses only)
            "paid_this_run": cost_summary(paid),
        },
        "embed": {
            "docs": len(index.doc_ids),
            "paid_calls": embedder.cost.llm_calls,
            "paid_tokens": embedder.cost.tokens_in,
            "wall_seconds": round(embed_seconds, 1),
            "model_versions": sorted(embedder.versions),
        },
    }
    (store / "build.json").write_text(json.dumps(build, indent=1) + "\n")
    write_manifest(
        store,
        models={"describer": cfg["describer"], "embedder": cfg["embedder"]},
        seeds={"sample_seed": sample["seed"]},
        cost=recorded + embedder.cost,
        extra={"step": "smoke index", "settings": settings, "config_file": args.config},
    )
    print(json.dumps({"store": str(store), **build["describe"], "embed": build["embed"]}, indent=1))


if __name__ == "__main__":
    main()
