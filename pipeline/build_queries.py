"""G2 of the full run (docs/full_run_plan.md 2.1, 4, 7.2): the query build for the 280 test
sources, with the smoke v2 pipeline unchanged (`dev/build_queries_dev.py`: facts_mc_v2 mining,
twin_facts_v1, ground_v3 local grounding with Gemini LOW + gpt-6-astra agreement, query_mc_v2,
index-visibility check on the source's 0.5 fps index frames).

PAID (Gemini, gpt-6-astra, gpt-6-luna). What differs from the smoke runner:
- sources: data/mc/full/sample.json "query_sources"; outputs in configs/mc_full.yaml out_dir;
- twin neighbours: caption-embedding neighbours over every captioned corpus video (not only a
  sample), never from the source's own YouTube / source video (`datasets.mc.source_key`);
- go/no-go checks (plan 7.2 G2) before any paid call: test sources disjoint from the dev
  sources and their YouTube ids, derived length bins, caption embedding inputs under the
  8,191-token limit, index frames present for the sources;
- `--canary N`: N stratified sources, compared with the smoke per-source cost (x 1.35) and
  projected to all sources; nothing else is written;
- a spend ledger of the calls actually paid (data/mc/full/spend.json), and a hard stop at the
  stage cap (caps.build, new input tokens over Gemini + astra + luna) counted across runs;
- per-source errors (rate limits, transient failures) do not stop the run: failed sources are
  retried after a wait, from the cache, in up to --rounds passes;
- `--fix-twins`: sources without a k = 4 twin query get max_b_passes 5 (plan 7.1); all others
  come from the cache unchanged.

    uv run --env-file .env python pipeline/build_queries.py --check            # free: go/no-go
    uv run --env-file .env python pipeline/build_queries.py --estimate         # free
    uv run --env-file .env python pipeline/build_queries.py --canary 5
    uv run --env-file .env python pipeline/build_queries.py
    uv run --env-file .env python pipeline/build_queries.py --fix-twins
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import constraints, grounding_v2, queries_mc, twins
from src.cache import shared_cache
from src.datasets.mc import load_captions, load_videos, source_key
from src.frames import FramePolicy, saved_frames
from src.grounding_v2 import FrameStore, Grounder
from src.llm import make_llm
from src.types import Cost

CONFIG = Path("configs/mc_full.yaml")
BUILD_MODELS = (
    "gemini:gemini-3.5-flash-lite",
    "openai:gpt-6-astra",
    "openai:gpt-6-luna@low",
)
# smoke v2 build, input tokens per source (results/smoke_v2/build_report.md; mining from the
# 16-frame run), and the factor for the longer full-run sources (plan 4.2)
SMOKE_PER_SOURCE = {
    "gemini:gemini-3.5-flash-lite": 32.0e3,
    "openai:gpt-6-astra": 35.0e3,
    "openai:gpt-6-luna@low": 13.7e3,
}
LONG_FACTOR = 1.35
CANARY_TOLERANCE = 0.30
EMBED_TOKEN_LIMIT = 8191
FIX_B_PASSES = 5
TWIN_K4_MIN = 150
QUERIES_MIN = 900


def load_build_v2():
    path = ROOT / "dev" / "build_queries_dev.py"
    spec = importlib.util.spec_from_file_location("build_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


B2 = load_build_v2()


# ---- spend ledger and the stage cap ------------------------------------------------------------


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Ledger:
    """Cost of the calls actually paid in this run, per model spec."""

    def __init__(self):
        self.by_spec: dict[str, Cost] = defaultdict(Cost)
        self._lock = threading.Lock()

    def add(self, spec: str, cost: Cost) -> None:
        with self._lock:
            self.by_spec[spec] = self.by_spec[spec] + cost

    def tokens_in(self, specs=BUILD_MODELS) -> int:
        return sum(self.by_spec[s].tokens_in for s in specs if s in self.by_spec)

    def to_dict(self) -> dict:
        return {s: c.to_dict() for s, c in sorted(self.by_spec.items())}


class CapBudget(B2.Budget):
    """The smoke projection guard plus a hard stop on actual paid input tokens: prior (earlier
    runs of this stage) + this run's ledger must stay under cap."""

    def __init__(self, cap: float, prior: float, ledger: Ledger, specs=BUILD_MODELS):
        super().__init__(cap - prior)
        self.cap, self.prior, self.ledger, self.specs = cap, prior, ledger, specs

    def charge(self, tokens: float, what: str) -> None:
        actual = self.prior + self.ledger.tokens_in(self.specs)
        if actual + tokens > self.cap:
            raise B2.BudgetExceeded(
                f"{what}: actual paid input {actual:,.0f} + {tokens:,.0f} would pass the stage "
                f"cap {self.cap:,.0f}"
            )
        super().charge(tokens, what)


class PaidLLM(B2.GuardedLLM):
    def __init__(self, inner, budget, spec, ledger: Ledger):
        super().__init__(inner, budget, spec)
        self.ledger = ledger

    def complete(self, prompt: str):
        paid = not self.is_cached(prompt)
        if paid:
            self.budget.charge(B2.text_tokens(prompt), self.spec)
        text, cost = self.inner.complete(prompt)
        if paid:
            self.ledger.add(self.spec, cost)
        return text, cost


class PaidVision(B2.GuardedVision):
    def __init__(self, inner, budget, spec, ledger: Ledger):
        super().__init__(inner, budget, spec)
        self.ledger = ledger

    def describe(self, prompt: str, images: list[bytes], labels=None):
        probe = getattr(self.inner, "is_cached", None)
        paid = probe is None or not probe(prompt, images, labels)
        if paid:
            self.budget.charge(
                B2.text_tokens(prompt)
                + B2.image_tokens(self.spec, len(images), self.media_resolution),
                self.spec,
            )
        text, cost = self.inner.describe(prompt, images, labels=labels)
        if paid:
            self.ledger.add(self.spec, cost)
        return text, cost


def read_spend(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {"runs": []}


def prior_tokens(spend: dict, stage: str = "build", specs=BUILD_MODELS) -> float:
    return sum(
        c["tokens_in"]
        for run in spend["runs"]
        if run["stage"] == stage
        for s, c in run["paid"].items()
        if s in specs
    )


def record_spend(path: Path, stage: str, what: str, ledger: Ledger, extra: dict | None = None):
    """Append one run to spend.json and recompute the totals per stage and model."""
    spend = read_spend(path)
    spend["runs"].append(
        {"stage": stage, "what": what, "date": now(), "paid": ledger.to_dict(), **(extra or {})}
    )
    totals: dict = defaultdict(lambda: defaultdict(Cost))
    for run in spend["runs"]:
        for spec, c in run["paid"].items():
            totals[run["stage"]][spec] = totals[run["stage"]][spec] + Cost(**c)
    spend["totals"] = {
        stage: {
            **{s: c.to_dict() for s, c in sorted(by.items())},
            "tokens_in": sum(c.tokens_in for c in by.values()),
            "tokens_out": sum(c.tokens_out for c in by.values()),
        }
        for stage, by in totals.items()
    }
    gem = [c for by in totals.values() for s, c in by.items() if s.startswith("gemini:")]
    gin, gout = sum(c.tokens_in for c in gem), sum(c.tokens_out for c in gem)
    spend["gemini_usd"] = {
        "tokens_in": gin,
        "tokens_out": gout,
        "usd": round(gin * 0.30e-6 + gout * 2.50e-6, 2),
        "prices": "$0.30 / $2.50 per 1M in / out (third-party listing, plan 4.2)",
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spend, indent=1))
    return spend


# ---- data and checks ---------------------------------------------------------------------------


def youtube_id(video) -> str | None:
    """The YouTube id a corpus unit comes from (None for CaReBench and TikTok)."""
    key = source_key(video)
    if key.startswith("yt:"):
        return key[3:]
    if video.dataset == "vript_short" and (
        video.source == "youtube" or (not video.source and len(video.video_id) == 11)
    ):  # TikTok ids are long numbers
        return video.video_id
    return None


def captioned_pool(root: str, datasets: list[str]) -> list[str]:
    """Every corpus video with a caption (twin candidates), as global ids."""
    pool = []
    for dataset in datasets:
        folder = Path(root) / dataset
        captions = load_captions(folder)
        pool += [f"{dataset}/{v.video_id}" for v in load_videos(folder) if v.video_id in captions]
    return pool


def embed_tokens(texts: list[str]) -> list[int]:
    import tiktoken

    enc = tiktoken.get_encoding("cl100k_base")
    return [len(enc.encode(t, disallowed_special=())) for t in texts]


def go_no_go(cfg: dict, sources: list[str], pool: list[str], store: FrameStore) -> dict:
    """Plan 7.2 G2 checks that need no paid call."""
    root = cfg["data_root"]
    dev = json.loads(Path(cfg["dev_sample"]).read_text())["query_sources"]
    out: dict = {"n_sources": len(sources), "n_pool": len(pool), "n_dev": len(dev)}

    def meta(gid):
        _, video, _ = B2.lookup(root, gid)
        return video

    test_v = {g: meta(g) for g in sources}
    dev_v = {g: meta(g) for g in dev}
    out["shared_ids"] = sorted(set(sources) & set(dev))
    test_src = {source_key(v) for v in test_v.values()}
    dev_src = {source_key(v) for v in dev_v.values()}
    out["shared_source_keys"] = sorted(test_src & dev_src)
    test_yt = {youtube_id(v) for v in test_v.values()} - {None}
    dev_yt = {youtube_id(v) for v in dev_v.values()} - {None}
    out["shared_youtube_ids"] = sorted(test_yt & dev_yt)
    out["duplicate_sources"] = len(sources) - len(set(sources))
    strata: dict = defaultdict(int)
    for g, v in test_v.items():
        strata[f"{g.split('/')[0]}/{v.length_bin}"] += 1
    out["strata"] = dict(sorted(strata.items()))
    out["bins_ok"] = all(
        v.length_bin in ("lt30s", "30to60s", "1to2min", "2to4min")
        for g, v in test_v.items()
        if g.startswith("vript")
    )
    texts = [twins.embedding_text(B2.lookup(root, g)[2]) for g in pool]
    tokens = embed_tokens(texts)
    out["caption_embed_tokens"] = {
        "max": max(tokens),
        "total": sum(tokens),
        "over_limit": sum(t > EMBED_TOKEN_LIMIT for t in tokens),
        "truncated_at_chars": sum(
            len(" ".join(B2.lookup(root, g)[2].caption.split())) > twins.EMBED_CHARS for g in pool
        ),
    }
    missing = []
    for g, v in test_v.items():
        dataset = g.split("/")[0]
        if not (store.index_dir(dataset, v) / "frames.json").exists():
            missing.append(g)
    out["index_frames_missing"] = missing
    out["pass"] = not (
        out["shared_ids"]
        or out["shared_source_keys"]
        or out["shared_youtube_ids"]
        or out["duplicate_sources"]
        or not out["bins_ok"]
        or out["caption_embed_tokens"]["over_limit"]
    )
    return out


def extract_index_frames(cfg: dict, gids: list[str], store: FrameStore, workers: int = 8):
    """0.5 fps index frames for the given videos via `frames.saved_frames` (reused when saved,
    never overwritten)."""
    root = cfg["data_root"]
    p = store.index_policy

    def one(gid):
        dataset, video, _ = B2.lookup(root, gid)
        out = store.index_dir(dataset, video)
        if (out / "frames.json").exists():
            return 0
        saved_frames(store.video_path(dataset, video), p.n, out, fps=p.fps, max_n=p.max_n)
        return 1

    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(one, gids))


def canary_sources(sources: list[str], root: str, n: int, seed: int) -> list[str]:
    """n sources, one per group in turn (Shot2Story, CaReBench, Vript short, Vript long twice),
    seeded."""
    rng = random.Random(seed)
    groups = defaultdict(list)
    for g in sources:
        groups[g.split("/")[0]].append(g)
    turn = ["shot2story", "carebench", "vript_short", "vript_long", "vript_long"]
    picked: list[str] = []
    bins_used: set = set()
    while len(picked) < n:
        dataset = turn[len(picked) % len(turn)]
        options = [g for g in groups[dataset] if g not in picked]
        if dataset == "vript_long":  # two different long bins
            fresh = [g for g in options if B2.lookup(root, g)[1].length_bin not in bins_used]
            options = fresh or options
        choice = rng.choice(sorted(options))
        bins_used.add(B2.lookup(root, choice)[1].length_bin)
        picked.append(choice)
    return picked


# ---- the run -----------------------------------------------------------------------------------


def make_context(cfg: dict, budget, ledger: Ledger):
    llm = PaidLLM(
        make_llm(constraints.LLM_SPEC, shared_cache("llm")), budget, constraints.LLM_SPEC, ledger
    )
    gspecs = tuple(cfg["grounders"])
    media = cfg.get("grounding_media_resolution")
    models = {s: PaidVision(B2.make_grounder_model(s, media), budget, s, ledger) for s in gspecs}
    lf = cfg.get("local_frames", {})
    store = FrameStore(
        cfg["data_root"],
        lf.get("max_side", grounding_v2.MAX_SIDE),
        index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8)),
    )
    fallback = FramePolicy.from_config(cfg["fallback_policy"])
    grounder = Grounder(
        models,
        grounding_v2.GROUND_PROMPT.read_text(),
        store,
        n_local=lf.get("n", grounding_v2.N_LOCAL),
        fallback=fallback,
        min_offset=lf.get("min_offset_s", grounding_v2.MIN_OFFSET_S),
    )
    templates = {
        "facts": constraints.FACTS_PROMPT.read_text(),
        "query": queries_mc.QUERY_PROMPT.read_text(),
        "twin": twins.TWIN_PROMPT.read_text(),
        "ground": grounding_v2.GROUND_PROMPT.read_text(),
    }
    return B2.Context(cfg, llm, grounder, templates)


def run_sources(sources, one, workers: int, rounds: int, wait_s: float, log=print):
    """{gid: result} and {gid: last error}. Failed sources are retried (from the cache) after
    a growing wait; a BudgetExceeded stops everything."""
    results: dict = {}
    errors: dict = {}
    todo = list(sources)
    for attempt in range(rounds + 1):
        with ThreadPoolExecutor(workers) as ex:
            futures = {ex.submit(one, g): g for g in todo}
            for i, fut in enumerate(as_completed(futures), 1):
                g = futures[fut]
                try:
                    results[g] = fut.result()
                    errors.pop(g, None)
                except B2.BudgetExceeded:
                    ex.shutdown(wait=False, cancel_futures=True)
                    raise
                except Exception as error:  # noqa: BLE001 - retried in the next round
                    errors[g] = f"{type(error).__name__}: {error}"[:500]
                    log(f"  source failed ({g}): {errors[g][:200]}")
                if i % 20 == 0:
                    log(f"  round {attempt}: {i}/{len(todo)} done, {len(errors)} failed")
        todo = [g for g in sources if g not in results]
        if not todo or attempt == rounds:
            break
        wait = wait_s * (attempt + 1)
        log(f"round {attempt}: {len(todo)} sources failed; waiting {wait:.0f} s, then retrying")
        time.sleep(wait)
    return results, errors


def write_outputs(cfg, out: Path, sources, results, ctx, gspecs, extra: dict):
    records = [results[g][0] for g in sources if g in results]
    natural = [row for g in sources if g in results for row in results[g][1]]
    twin_rows = [row for g in sources if g in results for row in results[g][2]]
    twin_records = [results[g][3] for g in sources if g in results]
    out.mkdir(parents=True, exist_ok=True)

    def dump(name, rows):
        with open(out / name, "w") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in rows)

    dump("constraints.jsonl", records)
    dump("queries.jsonl", natural)
    dump("queries_twin.jsonl", twin_rows)
    dump("twins.jsonl", twin_records)
    B2.write_csv(
        B2.twin_audit_rows(twin_records, records), B2.TWIN_AUDIT_FIELDS, out / "twin_audit.csv"
    )
    B2.write_csv(
        B2.audit_rows(records, cfg["data_root"], seed="full"),
        B2.AUDIT_FIELDS,
        out / "grounding_audit.csv",
    )
    st = B2.stats(records, natural, twin_rows, gspecs)
    (out / "build_stats.json").write_text(json.dumps(st, indent=1))
    from src.manifest import write_manifest

    total = sum(
        (Cost(**c) for r in records for by in r["cost"].values() for c in by.values()), Cost()
    )
    write_manifest(
        out,
        models={
            "mine": constraints.LLM_SPEC,
            "write": queries_mc.LLM_SPEC,
            "twin": twins.TWIN_SPEC,
            "ground": list(gspecs),
            "index_check": list(ctx.index_specs),
            "embed": cfg["twins"]["embedder"],
            "grounding_media_resolution": cfg.get("grounding_media_resolution"),
            "prompts": {
                "facts": constraints.FACTS_PROMPT.name,
                "ground": grounding_v2.GROUND_PROMPT.name,
                "query": queries_mc.QUERY_PROMPT.name,
                "twin": twins.TWIN_PROMPT.name,
            },
        },
        cost=total,
        extra=extra,
    )
    return records, natural, twin_rows, twin_records


def yield_check(natural: list[dict], twin_rows: list[dict]) -> dict:
    per_k = {
        name: {k: sum(r["k"] == k for r in rows) for k in range(1, 5)}
        for name, rows in (("natural", natural), ("twin", twin_rows))
    }
    return {
        "natural": len(natural),
        "twin": len(twin_rows),
        "per_k": per_k,
        "pass_queries": len(natural) >= QUERIES_MIN and len(twin_rows) >= QUERIES_MIN,
        "pass_twin_k4": per_k["twin"][4] >= TWIN_K4_MIN,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--check", action="store_true", help="go/no-go checks only (free)")
    parser.add_argument("--estimate", action="store_true", help="projection only (free)")
    parser.add_argument("--canary", type=int, default=0, help="run N stratified sources only")
    parser.add_argument("--fix-twins", action="store_true", help="max_b_passes 5 where no k=4 twin")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=4, help="retry rounds for failed sources")
    parser.add_argument("--wait", type=float, default=300.0, help="seconds before retry round 1")
    parser.add_argument(
        "--extract-frames",
        action="store_true",
        help="extract missing 0.5 fps index frames of the sources",
    )
    args = parser.parse_args(argv)
    started = time.time()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = cfg["data_root"]
    out = Path(cfg["out_dir"])
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    sources = sample["query_sources"][: args.limit]
    pool = captioned_pool(root, cfg["datasets"])
    spend_path = out / "spend.json"
    cap = float(cfg["caps"]["build"])
    prior = prior_tokens(read_spend(spend_path))
    ledger = Ledger()
    budget = CapBudget(cap, prior, ledger)
    ctx = make_context(cfg, budget, ledger)
    gspecs = tuple(cfg["grounders"])
    store = ctx.grounder.store

    if args.extract_frames:
        n = extract_index_frames(cfg, sources, store)
        print(f"extracted index frames for {n} sources")
    checks = go_no_go(cfg, sources, pool, store)
    print(json.dumps({k: v for k, v in checks.items() if k != "index_frames_missing"}, indent=1))
    print(f"index frames missing for {len(checks['index_frames_missing'])} sources")
    out.mkdir(parents=True, exist_ok=True)
    (out / "gonogo.json").write_text(json.dumps(checks, indent=1))
    if not checks["pass"]:
        raise SystemExit("go/no-go failed (see gonogo.json)")
    if args.check:
        return
    from src.embedders import make_embedder

    embedder = make_embedder(cfg["twins"]["embedder"])
    est = B2.estimate(cfg, sources, pool, ctx, embedder)
    B2.print_estimate(est, len(sources))
    print(f"stage cap {cap:,.0f}; paid by earlier build runs {prior:,.0f}")
    if args.estimate:
        return
    if prior + est["total_in"] > cap:
        raise SystemExit(f"projection {est['total_in']:,.0f} + prior {prior:,.0f} > cap {cap:,.0f}")

    run_sources_list = sources
    if args.canary:
        run_sources_list = canary_sources(sources, root, args.canary, cfg.get("seed", 1))
        canary_est = B2.estimate(cfg, run_sources_list, pool, ctx, embedder)
        print("canary sources:", run_sources_list)
    missing = sorted(set(checks["index_frames_missing"]) & set(run_sources_list))
    if missing:
        raise SystemExit(f"index frames missing for {len(missing)} sources: use --extract-frames")

    neighbours = B2.neighbours_for(cfg, run_sources_list, pool, embedder)
    embed_cost = getattr(embedder, "cost", Cost())
    print(f"caption embeddings paid: {embed_cost.to_dict()}")

    passes: dict[str, int] = {}
    if args.fix_twins:
        rows = list(B2.read_jsonl(out / "queries_twin.jsonl"))
        with_k4 = {r["source"] for r in rows if r["k"] == 4}
        passes = {g: FIX_B_PASSES for g in sources if g not in with_k4}
        print(f"fix-twins: {len(passes)} sources without a k = 4 twin get max_b_passes 5")

    def one(gid: str):
        local = ctx
        if gid in passes:
            local = copy.copy(ctx)
            local.max_b_passes = passes[gid]
        result = B2.build_source(gid, neighbours.get(gid, []), local)
        result[3]["max_b_passes"] = local.max_b_passes
        return result

    for g in run_sources_list:
        B2.lookup(root, g)
    stage_what = "canary" if args.canary else ("fix_twins" if args.fix_twins else "main")
    try:
        results, errors = run_sources(run_sources_list, one, args.workers, args.rounds, args.wait)
    finally:
        embed_ledger = Ledger()
        if embed_cost.llm_calls:
            embed_ledger.add(cfg["twins"]["embedder"], embed_cost)
            record_spend(spend_path, "embed", f"build captions ({stage_what})", embed_ledger)
        record_spend(
            spend_path,
            "build",
            stage_what,
            ledger,
            {
                "projected_paid": budget.spent,
                "calls_paid": budget.calls,
                "wall_s": round(time.time() - started),
            },
        )
    print(f"paid this run: {json.dumps(ledger.to_dict(), indent=1)}")
    if errors:
        print(f"{len(errors)} sources still failing: {errors}")

    if args.canary:
        n = len(run_sources_list)
        per = {s: ledger.by_spec[s].tokens_in / n for s in BUILD_MODELS}
        expected = {s: SMOKE_PER_SOURCE[s] * LONG_FACTOR for s in BUILD_MODELS}
        ratio = {s: per[s] / expected[s] for s in BUILD_MODELS}
        scale = ledger.tokens_in() / max(canary_est["total_in"], 1)
        report = {
            "sources": run_sources_list,
            "per_source_in": per,
            "expected_per_source_in": expected,
            "ratio": ratio,
            "within_30pct": {s: abs(r - 1) <= CANARY_TOLERANCE for s, r in ratio.items()},
            "canary_estimate_in": canary_est["total_in"],
            "actual_over_estimate": scale,
            "projection_all_in_by_mean": sum(per.values()) * len(sources),
            "projection_all_in_by_estimate": est["total_in"] * scale,
            "cap": cap,
            "errors": errors,
        }
        report["pass"] = (
            all(report["within_30pct"].values())
            and max(report["projection_all_in_by_mean"], report["projection_all_in_by_estimate"])
            + prior
            + ledger.tokens_in()
            < cap
        )
        (out / "canary.json").write_text(json.dumps(report, indent=1))
        print(json.dumps(report, indent=1))
        return

    records, natural, twin_rows, _ = write_outputs(
        cfg,
        out,
        sources,
        results,
        ctx,
        gspecs,
        {
            "config": args.config,
            "run": stage_what,
            "failed_sources": errors,
            "max_b_passes_raised": sorted(passes),
        },
    )
    yc = yield_check(natural, twin_rows)
    (out / "yield_check.json").write_text(json.dumps(yc, indent=1))
    print(json.dumps(yc, indent=1))
    print(f"{len(records)} sources, {len(natural)} natural and {len(twin_rows)} twin queries")
    print(f"wall clock {time.time() - started:,.0f} s")
    import build_queries_report as build_report

    build_report.main(["--config", args.config])


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    main()
