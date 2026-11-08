"""W5 (run_plan_w5_longdesc_20261106.md): long descriptions (describe_v4d, 14-18
sentences) of the frame-pool videos only, on the index's frames. PAID (describe).

Same describer, frames and call as the 44-word index (pipeline/build_index.py describe):
gemini:gemini-3.5-flash-lite, default sampling settings, the saved 0.5 fps frames (floor 8),
"Frame i of n (t = ... s):" labels; only the prompt (prompts/describe_v4d.txt) and the output
file change. Writes data/mc/<dataset>/<--filename> (default descriptions_fps05_v4d.jsonl);
never descriptions_fps05.jsonl, never results/_indexes/, no embed step. Ledger:
results/full/long_desc/spend.json, cap 75M new input tokens (plan section 7).

The video list is the union of the top 10 of S4@50 and of S0 over the queries of
--sources-file (the 722 test queries: 3,470 videos), or --ids FILE (a JSON list of global ids).

    uv run python pipeline/long_descriptions.py project                       # free: counts, $
    uv run --env-file .env python pipeline/long_descriptions.py describe       # the full run
    # dry run: 30 videos into a dry-run file; drift canary with describe_v4 in a fresh namespace
    uv run --env-file .env python pipeline/long_descriptions.py describe --ids IDS.json \
        --filename descriptions_fps05_v4d_w5dryrun.jsonl --stage-name long_desc_dryrun
    uv run --env-file .env python pipeline/long_descriptions.py describe --ids IDS.json \
        --prompt prompts/describe_v4.txt --namespace w5-drift-canary \
        --filename descriptions_fps05_v4_w5canary.jsonl --stage-name long_desc_canary_v4
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from src.datasets.mc import RateLimit, read_jsonl

ROOT = Path(__file__).resolve().parents[1]
CONFIG = Path("configs/mc_full.yaml")
PROMPT = Path("prompts/describe_v4d.txt")
FILENAME = "descriptions_fps05_v4d.jsonl"
CAP = 85e6  # plan section 7: 75M, raised to 85M after the dry run (projection 71.8M left ~4% headroom)
TOTAL_USD_CAP = 44.0
OUT_TOKENS_DEV = 249  # v4d output tokens per video, dev ablation (plan section 3)


def load_index_script():
    spec = importlib.util.spec_from_file_location("_full_index", ROOT / "pipeline/build_index.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


IX = load_index_script()


class Namespaced:
    """Wraps the uncached describer so its cache key carries an extra param (a fresh namespace
    for the drift canary); the request sent is unchanged."""

    def __init__(self, inner, tag: str):
        self.inner = inner
        self.params = {**(getattr(inner, "params", None) or {}), "namespace": tag}

    def describe(self, prompt, images, labels=None):
        return self.inner.describe(prompt, images, labels=labels)


def pool_union(cfg: dict, sources_file: Path, top: int = 10) -> tuple[list[str], dict]:
    """Union of the top `top` of S4@50 and S0 over the queries of the drawn sources."""
    keep = set(json.loads(sources_file.read_text())["sources"])
    runs = Path(cfg["gate_out"]) / "runs"
    by_pool: dict[str, set] = {"S4@50": set(), "S0": set()}
    pairs: dict[str, set] = {"S4@50": set(), "S0": set()}
    queries = 0
    for name in ("natural", "twin"):
        for pool in by_pool:
            for line in read_jsonl(runs / name / f"{pool}.jsonl"):
                if line["source"] not in keep:
                    continue
                queries += pool == "S0"
                docs = [d for d, _ in line["ranking"][:top]]
                by_pool[pool] |= set(docs)
                pairs[pool] |= {(line["query_id"], d) for d in docs}
    union = sorted(by_pool["S4@50"] | by_pool["S0"])
    counts = {
        "queries": queries,
        "videos_s4pool": len(by_pool["S4@50"]),
        "videos_s0pool": len(by_pool["S0"]),
        "videos_overlap": len(by_pool["S4@50"] & by_pool["S0"]),
        "videos_union": len(union),
        "pairs_s4pool": len(pairs["S4@50"]),
        "pairs_s0pool": len(pairs["S0"]),
        "pairs_union": len(pairs["S4@50"] | pairs["S0"]),
        "pairs_shared": len(pairs["S4@50"] & pairs["S0"]),
    }
    return union, counts


def setup_for(args, ids: list[str]):
    cfg = yaml.safe_load(CONFIG.read_text())
    cfg = {**cfg, "describe_prompt": str(args.prompt), "descriptions_file": args.filename}
    if args.filename == yaml.safe_load(CONFIG.read_text())["descriptions_file"]:
        sys.exit("refusing to write the index's descriptions file")
    setup = IX.Setup(cfg, ids)
    if args.namespace:
        setup.vlm.inner = Namespaced(setup.vlm.inner, args.namespace)
    return cfg, setup


def ledger_path(cfg: dict) -> Path:
    return Path(cfg["gate_out"]) / "long_desc" / "spend.json"


def ledger(cfg: dict) -> dict:
    path = ledger_path(cfg)
    return json.loads(path.read_text()) if path.exists() else {"cap": CAP, "entries": []}


def add_entry(cfg: dict, entry: dict) -> None:
    data = ledger(cfg)
    data["entries"].append(entry)
    path = ledger_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")


def all_spend(cfg: dict) -> float:
    """Input tokens in every results/full/*/spend.json and the build ledger (as systems.py)."""
    total = 0.0
    for path in [*Path(cfg["gate_out"]).glob("*/spend.json"), Path(cfg["out_dir"]) / "spend.json"]:
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        if "entries" in data:
            total += sum(e["tokens_in"] for e in data["entries"])
        elif "totals" in data:
            total += sum(step.get("tokens_in", 0) for step in data["totals"].values())
        else:
            total += sum(v.get("tokens_in", 0) for v in data.values() if isinstance(v, dict))
    return total


def video_ids(args, cfg) -> tuple[list[str], dict]:
    if args.ids:
        ids = json.loads(Path(args.ids).read_text())
        return ids, {"videos": len(ids), "from": args.ids}
    return pool_union(cfg, Path(args.sources_file))


def project(setup, ids: list[str]) -> dict:
    frames = {g: setup.frame_count(g) for g in ids}
    uncached = [g for g in ids if frames[g] and not setup.is_cached(g)]
    new_frames = sum(frames[g] for g in uncached)
    tokens_in = new_frames * IX.TOKENS_PER_FRAME
    tokens_out = len(uncached) * OUT_TOKENS_DEV
    return {
        "videos": len(ids),
        "without_frames": [g for g in ids if not frames[g]],
        "frames": sum(frames.values()),
        "uncached_videos": len(uncached),
        "uncached_frames": new_frames,
        "projected_in": tokens_in,
        "projected_out": tokens_out,
        "projected_usd": round(
            tokens_in / 1e6 * IX.GEMINI_PRICE[0] + tokens_out / 1e6 * IX.GEMINI_PRICE[1], 2
        ),
    }


def cmd_project(args) -> None:
    cfg = yaml.safe_load(CONFIG.read_text())
    ids, counts = video_ids(args, cfg)
    _, setup = setup_for(args, ids)
    proj = project(setup, ids)
    print(json.dumps({**counts, **{k: v for k, v in proj.items() if k != "without_frames"}}))
    print(f"videos without frames: {len(proj['without_frames'])} {proj['without_frames'][:5]}")
    spent = sum(e["tokens_in"] for e in ledger(cfg)["entries"])
    print(f"ledger {ledger_path(cfg)}: {spent:,} spent of cap {CAP:,.0f}")


def cmd_describe(args) -> None:
    cfg0 = yaml.safe_load(CONFIG.read_text())
    ids, _ = video_ids(args, cfg0)
    cfg, setup = setup_for(args, ids)
    proj = project(setup, ids)
    spent = sum(e["tokens_in"] for e in ledger(cfg)["entries"])
    print(
        f"[{args.stage_name}] {len(ids)} videos, {proj['uncached_videos']} to pay, projection "
        f"{proj['projected_in']:,} in (${proj['projected_usd']}); ledger {spent:,} of cap {CAP:,.0f}",
        flush=True,
    )
    if spent + proj["projected_in"] > CAP:
        sys.exit(f"[{args.stage_name}] stop: the long_desc cap would be passed")
    everything = all_spend(cfg)
    print(f"all stages {everything:,.0f} + projection of total cap {float(cfg['total_cap']):,.0f}")
    if everything + proj["projected_in"] > float(cfg["total_cap"]):
        sys.exit(f"[{args.stage_name}] stop: the total cap would be passed")
    guard = IX.GuardedVision(
        setup.vlm.inner, RateLimit(cfg["describe_per_minute"]), CAP, already=spent
    )
    setup.vlm.inner = guard
    start = time.perf_counter()
    todo = list(ids)
    stopped = ""
    try:
        for pass_no in range(1, args.passes + 1):
            failed_all: list[str] = []
            for i in range(0, len(todo), IX.CHUNK):
                chunk = todo[i : i + IX.CHUNK]
                _, failed = IX.describe_ids(setup, chunk, guard)
                failed_all += failed
                print(
                    f"pass {pass_no}: {min(i + IX.CHUNK, len(todo))}/{len(todo)}, failed "
                    f"{len(failed)}, paid {guard.cost.llm_calls} calls / {guard.cost.tokens_in:,}"
                    f" in / {guard.cost.tokens_out:,} out, {time.perf_counter() - start:.0f} s",
                    flush=True,
                )
            todo = failed_all
            if not todo:
                break
            if pass_no < args.passes:
                print(f"{len(todo)} failed calls; retry after {args.wait:.0f} s", flush=True)
                time.sleep(args.wait)
    except IX.CapExceeded as exc:
        stopped = str(exc)
        print(f"STOP: {stopped}", flush=True)
    records = setup.records()
    empty = [g for g in ids if not records.get(g, {}).get("description")]
    retried: list[str] = []
    if empty and not stopped:  # cached empty answers (blocks): retry once in a fresh namespace
        setup.vlm.inner = Namespaced(guard, f"{args.namespace or ''}retry1")
        try:
            IX.describe_ids(setup, empty, guard)
        except IX.CapExceeded as exc:
            stopped = str(exc)
        setup.vlm.inner = guard
        records = setup.records()
        retried = [g for g in empty if records.get(g, {}).get("description")]
        empty = [g for g in empty if g not in set(retried)]
    versions = sorted({records[g].get("model_version", "") for g in ids if g in records})
    cost = guard.cost
    entry = {
        "stage": args.stage_name,
        "tokens_in": int(cost.tokens_in),
        "tokens_out": int(cost.tokens_out),
        "calls": cost.llm_calls,
        "images": int(cost.images),
        "usd": IX.gemini_dollars(cost),
        "when": datetime.now(UTC).isoformat(timespec="seconds"),
        "videos": len(ids),
        "failed_after_passes": len(todo),
        "empty": empty,
        "recovered_by_retry": retried,
        "model_versions": versions,
        "prompt": Path(args.prompt).name,
        "filename": args.filename,
        "namespace": args.namespace,
        "projected_in": proj["projected_in"],
        "stopped": stopped,
        "wall_seconds": round(time.perf_counter() - start, 1),
    }
    add_entry(cfg, entry)
    print(json.dumps({k: (len(v) if k == "empty" else v) for k, v in entry.items()}))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=("project", "describe"))
    ap.add_argument("--sources-file", default="results/full/select/sources100.json")
    ap.add_argument("--ids", help="JSON list of global video ids instead of the pool union")
    ap.add_argument("--prompt", type=Path, default=PROMPT)
    ap.add_argument("--filename", default=FILENAME, help="descriptions file in each dataset dir")
    ap.add_argument("--namespace", help="extra cache-key param: a fresh namespace (drift canary)")
    ap.add_argument("--stage-name", default="long_desc", help="ledger entry name")
    ap.add_argument("--passes", type=int, default=4, help="retry passes for failed calls")
    ap.add_argument("--wait", type=float, default=120.0, help="seconds before a retry pass")
    args = ap.parse_args(argv)
    {"project": cmd_project, "describe": cmd_describe}[args.cmd](args)


if __name__ == "__main__":
    main()
