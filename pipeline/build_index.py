"""Full-run index (docs/full_run_plan.md, stages P0 frames and G1). PAID (describe, embed).

Generalises dev/build_index_dev.py to the full corpus (data/mc/full/sample.json "videos", all
8,603). Steps, each its own subcommand:

- frames (free): save the index frames (config frame_policy, 0.5 fps, floor 8) of every video
  into data/mc/<dataset>/frames_fps0.5/<video_id>/; existing folders are reused. Writes
  results/full/index/frames.json.
- check (free): the G1 go/no-go (plan 7.2): frames exist for every video, the smoke
  descriptions are cache hits, the projected new Gemini input (frames of uncached videos x
  1,125 tokens, the smoke rate) is within 10% of the plan's 164.4M and under caps.describe.
  Writes results/full/index/check.json.
- canary (paid, 10 calls): describe 10 uncached videos (seed 1, spread over the datasets);
  every returned modelVersion must be the describer's model and no description empty.
- describe (paid): describe every video (describe_v4, labelled frames, default media
  resolution) in chunks, through the shared cache (results/_cache/vlm.sqlite), with a rate
  limit on paid calls and a hard stop before caps.describe new input tokens. A failed call
  (rate limit after the client's own retries, network) is not cached: it is left empty,
  waited out (minutes) and retried in later passes. An empty answer that is cached (a safety
  block) is retried once with a separate cache namespace ("vision-retry1:<spec>").
- embed (paid): embed the descriptions with text-embedding-3-small (inputs capped at
  embed_max_tokens) and build the DESC index in results/_indexes/<hash>/ (fingerprinted; a
  store holding another index raises). Writes build.json and manifest.json there and
  results/full/index/{spend,build,manifest}.json.

    uv run python pipeline/build_index.py frames [--procs 12]
    uv run python pipeline/build_index.py check
    uv run --env-file .env python pipeline/build_index.py canary
    uv run --env-file .env python pipeline/build_index.py describe
    uv run --env-file .env python pipeline/build_index.py embed
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import threading
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    read_jsonl,
    split_global_id,
)
from src.embedders import make_embedder
from src.frames import FramePolicy, frame_labels
from src.llm import make_vision, parse_spec
from src.manifest import write_manifest
from src.types import Cost

TOKENS_PER_FRAME = 1125  # smoke: 13.19M input for 11,722 images, text included (plan 4.1)
PLAN_DESCRIBE_INPUT = 164.4e6  # plan 4.2
GEMINI_PRICE = (0.30, 2.50)  # $ per 1M input / output (plan 4.2)
CHUNK = 400


def out_dir(cfg: dict) -> Path:
    path = Path(cfg["gate_out"]) / "index"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cost_summary(cost: Cost) -> dict:
    return {
        "calls": cost.llm_calls,
        "tokens_in": cost.tokens_in,
        "tokens_out": cost.tokens_out,
        "images": cost.images,
        "api_seconds": round(cost.seconds, 1),
        "model_versions": [v for v in cost.model_version.split(",") if v],
    }


def gemini_dollars(cost: Cost) -> float:
    return round(
        cost.tokens_in / 1e6 * GEMINI_PRICE[0] + cost.tokens_out / 1e6 * GEMINI_PRICE[1], 2
    )


# ---- frames ---------------------------------------------------------------------------------

_CORPUS: dict = {}


def _extract(job: tuple[str, list[str], str, str]) -> tuple[str, int, str]:
    root, datasets, gid, policy_raw = job
    corpus = _CORPUS.get("c")
    if corpus is None:
        corpus = _CORPUS["c"] = McCorpus.pool(root, datasets)
    policy = FramePolicy.from_config(json.loads(policy_raw))
    try:
        return gid, len(corpus.frames(gid, policy=policy)), ""
    except Exception as exc:  # noqa: BLE001 (report, do not stop the pool)
        return gid, 0, f"{type(exc).__name__}: {exc}"


def cmd_frames(cfg: dict, ids: list[str], procs: int) -> None:
    policy = index_options(cfg)["frame_policy"]
    raw = json.dumps(cfg["frame_policy"])
    start = time.perf_counter()
    total, errors, done = 0, {}, 0
    jobs = [(cfg["data_root"], cfg["datasets"], gid, raw) for gid in ids]
    with ProcessPoolExecutor(procs) as pool:
        for fut in as_completed([pool.submit(_extract, j) for j in jobs]):
            gid, n, err = fut.result()
            done += 1
            total += n
            if err or n == 0:
                errors[gid] = err or "no frames"
            if done % 500 == 0:
                elapsed = time.perf_counter() - start
                print(f"{done}/{len(ids)} videos, {total} frames, {elapsed:.0f} s", flush=True)
    summary = {
        "policy": policy.settings(),
        "videos": len(ids),
        "frames": total,
        "errors": errors,
        "wall_seconds": round(time.perf_counter() - start, 1),
    }
    (out_dir(cfg) / "frames.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "errors"} | {"n_errors": len(errors)})
    )


# ---- the describe call, as describe_corpus makes it -----------------------------------------


class Setup:
    """The corpus, the cached describer and what a describe call sends for a video."""

    def __init__(self, cfg: dict, ids: list[str]):
        self.cfg = cfg
        self.corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], ids)
        self.options = index_options(cfg)
        self.policy: FramePolicy = self.options["frame_policy"]
        self.prompt_path = cfg["describe_prompt"]
        self.prompt_text = Path(self.prompt_path).read_text()
        self.vlm = make_vision(cfg["describer"], shared_cache("vlm", cfg["results_dir"]))

    def frame_count(self, gid: str) -> int:
        record = self.corpus.frames_dir(gid, policy=self.policy) / "frames.json"
        return len(json.loads(record.read_text())["saved"]) if record.exists() else 0

    def call(self, gid: str) -> tuple[str, list[bytes], list[str]]:
        """(prompt, frames, labels) of describe_frames for this video (as describe_corpus)."""
        frames = self.corpus.frames(gid, policy=self.policy)
        times = self.corpus.frame_times(gid, policy=self.policy)
        labels = frame_labels(times) if self.options["labels"] else None
        return self.prompt_text.replace("{n}", str(len(frames))), frames, labels

    def is_cached(self, gid: str) -> bool:
        prompt, frames, labels = self.call(gid)
        return bool(frames) and self.vlm.is_cached(prompt, frames, labels)

    def records(self) -> dict[str, dict]:
        """Description records of the corpus videos keyed by global id (descriptions file)."""
        out, tables = {}, {}
        for gid in self.corpus.doc_ids:
            folder, vid = self.corpus.locate(gid)
            if folder not in tables:
                path = folder / self.options["filename"]
                tables[folder] = (
                    {r["video_id"]: r for r in read_jsonl(path)} if path.exists() else {}
                )
            if vid in tables[folder]:
                out[gid] = tables[folder][vid]
        return out


class CapExceeded(RuntimeError):
    pass


class GuardedVision(RateLimitedVision):
    """Rate limit on paid calls, plus a hard stop before the stage cap on new input tokens."""

    def __init__(self, inner, limit: RateLimit, cap: float, already: int = 0):
        super().__init__(inner, limit)
        self.cap, self.already = cap, already
        self.reserved = 0
        self.errors: list[str] = []

    def describe(self, prompt: str, images: list[bytes], **kwargs) -> tuple[str, Cost]:
        estimate = TOKENS_PER_FRAME * len(images)
        with self._lock:
            projected = self.already + self.cost.tokens_in + self.reserved + estimate
            if projected > self.cap:
                raise CapExceeded(
                    f"next call would pass the describe cap ({projected:,.0f} > {self.cap:,.0f})"
                )
            self.reserved += estimate
        try:
            return super().describe(prompt, images, **kwargs)
        finally:
            with self._lock:
                self.reserved -= estimate


class Tolerant:
    """Wraps the cached describer: a failed call returns "" without being cached (so a later
    pass retries it) and is counted; a cap stop is re-raised."""

    def __init__(self, cached):
        self.cached = cached
        self.failures: list[str] = []
        self._lock = threading.Lock()

    def describe(self, prompt, images, labels=None):
        try:
            return self.cached.describe(prompt, images, labels=labels)
        except CapExceeded:
            raise
        except Exception as exc:  # noqa: BLE001 (not cached; retried in a later pass)
            with self._lock:
                self.failures.append(f"{type(exc).__name__}: {str(exc)[:200]}")
            return "", Cost()


def spend_file(cfg: dict) -> Path:
    return out_dir(cfg) / "spend.json"


def load_spend(cfg: dict) -> dict:
    path = spend_file(cfg)
    return json.loads(path.read_text()) if path.exists() else {}


def save_spend(cfg: dict, spend: dict) -> None:
    spend_file(cfg).write_text(json.dumps(spend, indent=1) + "\n")


def add_spend(cfg: dict, stage: str, cost: Cost, seconds: float, extra: dict | None = None) -> dict:
    """Accumulate paid cost of a stage across runs (spend.json[stage])."""
    spend = load_spend(cfg)
    old = spend.get(stage, {})
    total = Cost(**old.get("cost", {})) + cost
    spend[stage] = {
        "cost": total.to_dict(),
        **cost_summary(total),
        "wall_seconds": round(old.get("wall_seconds", 0.0) + seconds, 1),
        "runs": old.get("runs", 0) + 1,
        **(extra or {}),
    }
    if stage.startswith("describe"):
        spend[stage]["gemini_usd"] = gemini_dollars(total)
    save_spend(cfg, spend)
    return spend


def paid_describe_tokens(cfg: dict) -> int:
    spend = load_spend(cfg)
    return sum(v["cost"]["tokens_in"] for k, v in spend.items() if k.startswith("describe"))


# ---- check ----------------------------------------------------------------------------------


def cmd_check(cfg: dict, sample: dict) -> dict:
    setup = Setup(cfg, sample["videos"])
    ids = setup.corpus.doc_ids
    counts = {gid: setup.frame_count(gid) for gid in ids}
    missing = [g for g, n in counts.items() if n == 0]
    smoke = json.loads((Path(cfg["data_root"]) / "smoke" / "sample.json").read_text())["videos"]
    records = setup.records()
    smoke_nonempty = [g for g in smoke if records.get(g, {}).get("description")]
    smoke_hits = [g for g in smoke_nonempty if counts.get(g) and setup.is_cached(g)]
    cached = set()
    for gid in ids:
        if counts[gid] and setup.is_cached(gid):
            cached.add(gid)
    uncached = [g for g in ids if g not in cached]
    new_frames = sum(counts[g] for g in uncached)
    projection = new_frames * TOKENS_PER_FRAME
    cap = float(cfg["caps"]["describe"])
    by_ds: dict[str, dict] = {}
    for gid in ids:
        ds = split_global_id(gid)[0]
        cell = by_ds.setdefault(ds, {"videos": 0, "frames": 0, "uncached": 0, "uncached_frames": 0})
        cell["videos"] += 1
        cell["frames"] += counts[gid]
        if gid not in cached:
            cell["uncached"] += 1
            cell["uncached_frames"] += counts[gid]
    result = {
        "videos": len(ids),
        "frames": sum(counts.values()),
        "videos_without_frames": missing,
        "smoke_videos": len(smoke),
        "smoke_nonempty_descriptions": len(smoke_nonempty),
        "smoke_cache_hits": len(smoke_hits),
        "cached_videos": len(cached),
        "uncached_videos": len(uncached),
        "uncached_frames": new_frames,
        "projected_new_input": projection,
        "plan_input": PLAN_DESCRIBE_INPUT,
        "projection_vs_plan": round(projection / PLAN_DESCRIBE_INPUT - 1, 4),
        "cap": cap,
        "by_dataset": by_ds,
    }
    result["go"] = {
        "frames_all": not missing,
        "smoke_cache_hits_499": len(smoke_hits) >= 499,
        "projection_within_10pct": abs(result["projection_vs_plan"]) <= 0.10,
        "projection_under_cap": projection < cap,
    }
    (out_dir(cfg) / "check.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "videos_without_frames"}, indent=1))
    print(f"videos without frames: {len(missing)} {missing[:5]}")
    return result


# ---- describe -------------------------------------------------------------------------------


def describe_ids(setup: Setup, ids: list[str], guard: GuardedVision) -> tuple[dict, list[str]]:
    """describe_corpus over ids, tolerant of failed calls; returns (records, failed ids)."""
    tolerant = Tolerant(setup.vlm)
    records = describe_corpus(
        setup.corpus,
        tolerant,
        setup.prompt_path,
        setup.cfg["describer"],
        setup.cfg["n_frames"],
        setup.cfg["workers"],
        video_ids=ids,
        filename=setup.options["filename"],
        frame_policy=setup.policy,
        labels=setup.options["labels"],
    )
    failed = [g for g in ids if not records[g]["description"] and not setup.is_cached(g)]
    if tolerant.failures:
        print(f"  {len(tolerant.failures)} failed calls, e.g. {tolerant.failures[0]}", flush=True)
    return records, failed


def make_guard(setup: Setup, cfg: dict) -> GuardedVision:
    guard = GuardedVision(
        setup.vlm.inner,
        RateLimit(cfg["describe_per_minute"]),
        float(cfg["caps"]["describe"]),
        already=paid_describe_tokens(cfg),
    )
    setup.vlm.inner = guard
    return guard


def cmd_canary(cfg: dict, sample: dict, n: int = 10) -> None:
    setup = Setup(cfg, sample["videos"])
    rng = random.Random(cfg["seed"])
    by_ds: dict[str, list[str]] = {}
    for gid in setup.corpus.doc_ids:
        by_ds.setdefault(split_global_id(gid)[0], []).append(gid)
    picked: list[str] = []
    pools = {ds: rng.sample(v, len(v)) for ds, v in by_ds.items()}
    while len(picked) < n:
        for ds in cfg["datasets"]:
            while pools[ds]:
                gid = pools[ds].pop()
                if not setup.is_cached(gid):
                    picked.append(gid)
                    break
            if len(picked) == n:
                break
    guard = make_guard(setup, cfg)
    start = time.perf_counter()
    records, failed = describe_ids(setup, picked, guard)
    seconds = time.perf_counter() - start
    add_spend(cfg, "describe", guard.cost, seconds)
    model = parse_spec(cfg["describer"])[1]
    versions = {r["model_version"] for r in records.values()}
    empty = [g for g, r in records.items() if not r["description"]]
    result = {
        "ids": picked,
        "model_versions": sorted(versions),
        "empty": empty,
        "failed": failed,
        "paid": cost_summary(guard.cost),
        "frames": [r["n_frames"] for r in records.values()],
        "tokens_per_frame": round(guard.cost.tokens_in / max(guard.cost.images, 1), 1),
        "go": versions == {model} and not empty and not failed,
        "descriptions": {g: r["description"] for g, r in records.items()},
    }
    (out_dir(cfg) / "canary.json").write_text(json.dumps(result, indent=1) + "\n")
    print(json.dumps(result, indent=1))


def cmd_describe(cfg: dict, sample: dict, max_passes: int = 6) -> None:
    setup = Setup(cfg, sample["videos"])
    guard = make_guard(setup, cfg)
    ids = setup.corpus.doc_ids
    start = time.perf_counter()
    have = setup.records()
    todo = [g for g in ids if not have.get(g, {}).get("description")]
    print(f"{len(todo)} of {len(ids)} videos without a description", flush=True)
    wait = 120.0
    stopped = ""
    try:
        for pass_no in range(1, max_passes + 1):
            failed_all: list[str] = []
            for i in range(0, len(todo), CHUNK):
                chunk = todo[i : i + CHUNK]
                _, failed = describe_ids(setup, chunk, guard)
                failed_all += failed
                elapsed = time.perf_counter() - start
                print(
                    f"pass {pass_no}: {min(i + CHUNK, len(todo))}/{len(todo)}, failed {len(failed)}"
                    f", paid {guard.cost.llm_calls} calls / {guard.cost.tokens_in:,} in / "
                    f"{guard.cost.tokens_out:,} out, {elapsed:.0f} s",
                    flush=True,
                )
                if len(failed) > len(chunk) // 4:  # mostly errors: rate limited or offline
                    print(f"  backing off {wait:.0f} s", flush=True)
                    time.sleep(wait)
                    wait = min(wait * 2, 900.0)
                else:
                    wait = 120.0
            todo = failed_all
            if not todo:
                break
            print(f"pass {pass_no} done; {len(todo)} failed calls to retry after 300 s", flush=True)
            time.sleep(300)
    except CapExceeded as exc:
        stopped = str(exc)
        print(f"STOP: {stopped}", flush=True)
    seconds = time.perf_counter() - start

    # cached empty answers (safety blocks): retry once, under a separate cache namespace
    records = setup.records()
    empty = [g for g in ids if g in records and not records[g]["description"]]
    retried: dict[str, str] = {}
    retry_cost = Cost()
    if not stopped:
        retry_cache = shared_cache("vlm", cfg["results_dir"])
        for gid in empty:
            if not setup.is_cached(gid):
                continue  # a failed call, not a block
            prompt, frames, labels = setup.call(gid)
            payload = setup.vlm._payload(prompt, frames, labels)

            def call(prompt=prompt, frames=frames, labels=labels) -> dict:
                text, cost = guard.describe(prompt, frames, labels=labels)
                return {"text": text, "cost": cost.to_dict()}

            try:
                rec = retry_cache.get_or_compute(f"vision-retry1:{cfg['describer']}", payload, call)
            except CapExceeded as exc:
                stopped = str(exc)
                break
            except Exception as exc:  # noqa: BLE001
                print(f"  retry of {gid} failed: {exc}", flush=True)
                continue
            retry_cost += Cost(**rec["cost"])
            if rec["text"].strip():
                retried[gid] = rec["text"].strip()
        if retried:  # write them into the descriptions file, marked as retries
            by_folder: dict[Path, dict[str, str]] = {}
            for gid, text in retried.items():
                folder, vid = setup.corpus.locate(gid)
                by_folder.setdefault(folder, {})[vid] = text
            for folder, new in by_folder.items():
                path = folder / setup.options["filename"]
                rows = {r["video_id"]: r for r in read_jsonl(path)}
                for vid, text in new.items():
                    rows[vid] = {**rows[vid], "description": text, "retry1": True}
                path.write_text("".join(json.dumps(rows[v]) + "\n" for v in sorted(rows)))
    records = setup.records()
    still_empty = [g for g in ids if not records.get(g, {}).get("description")]
    extra = {
        "videos": len(ids),
        "described": len(ids) - len(still_empty),
        "empty_after_retry": still_empty,
        "blocked_first_try": empty,
        "recovered_by_retry": sorted(retried),
        "stopped": stopped,
    }
    add_spend(cfg, "describe", guard.cost, seconds, extra)
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in extra.items()}))
    print(json.dumps(load_spend(cfg)["describe"], indent=1))


# ---- embed and index ------------------------------------------------------------------------


def cmd_embed(cfg: dict, sample: dict, config_file: str) -> None:
    setup = Setup(cfg, sample["videos"])
    corpus, options = setup.corpus, setup.options
    records = setup.records()
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
    store = index_store(cfg["results_dir"], settings)
    embedder = make_embedder(
        cfg["embedder"],
        cache=shared_cache("embed", cfg["results_dir"]),
        max_tokens=cfg["embed_max_tokens"],
    )
    start = time.perf_counter()
    index = description_index(corpus, descriptions, embedder, store)
    embed_seconds = time.perf_counter() - start
    if embedder.cost.tokens_in > float(cfg["caps"]["embed"]):
        print("WARNING: embed spend over the cap", flush=True)
    add_spend(
        cfg, "embed", embedder.cost, embed_seconds, {"truncated_inputs": len(embedder.truncated)}
    )

    recorded = sum((Cost(**r["cost"]) for r in records.values()), Cost())
    spend = load_spend(cfg)
    empty = [g for g in corpus.doc_ids if not descriptions.get(g)]
    build = {
        "settings": settings,
        "store": str(store),
        "describe": {
            "videos": len(records),
            "frames": sum(r["n_frames"] for r in records.values()),
            "descriptions_file": options["filename"],
            "empty_descriptions": empty,
            "retried": sorted(g for g, r in records.items() if r.get("retry1")),
            "recorded": cost_summary(recorded),  # every description's cost as first paid
            "paid_full_run": spend.get("describe", {}),  # cache misses in this stage
        },
        "embed": {
            "docs": len(index.doc_ids),
            "paid_calls": embedder.cost.llm_calls,
            "paid_tokens": embedder.cost.tokens_in,
            "truncated_inputs": len(embedder.truncated),
            "wall_seconds": round(embed_seconds, 1),
            "model_versions": sorted(embedder.versions),
        },
    }
    for folder in (store, out_dir(cfg)):
        (folder / "build.json").write_text(json.dumps(build, indent=1) + "\n")
        write_manifest(
            folder,
            models={"describer": cfg["describer"], "embedder": cfg["embedder"]},
            seeds={"sample_seed": sample["seed"]},
            cost=recorded + embedder.cost,
            extra={
                "step": "full index (G1)",
                "settings": settings,
                "config_file": config_file,
                "store": str(store),
            },
        )
    print(json.dumps({"store": str(store), "embed": build["embed"], "empty": len(empty)}, indent=1))


def main() -> None:
    logging.basicConfig(level=logging.WARNING)
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=["frames", "check", "canary", "describe", "embed"])
    parser.add_argument("--config", default="configs/mc_full.yaml")
    parser.add_argument("--procs", type=int, default=12)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    if args.step == "frames":
        cmd_frames(cfg, sample["videos"], args.procs)
    elif args.step == "check":
        cmd_check(cfg, sample)
    elif args.step == "canary":
        cmd_canary(cfg, sample)
    elif args.step == "describe":
        cmd_describe(cfg, sample)
    else:
        cmd_embed(cfg, sample, args.config)


if __name__ == "__main__":
    main()
