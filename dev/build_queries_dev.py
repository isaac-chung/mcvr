"""Query pipeline v2 on the smoke set: natural nested queries and twin queries, grounded on local
frames by two models, each rendered as a paraphrase and as a template (docs/plan.md 4.3-4.4).

PAID (Gemini, OpenAI). Run --estimate first; every paid step is guarded by --max-input-tokens.

Per query source A (data/mc/smoke/sample.json "query_sources"):
1. mine facts from A's caption (facts_mc_v2, gpt-6-luna; cached from the smoke run) and select
   candidates (`constraints.select_candidates`);
2. twins: top-5 caption-embedding neighbours B from other source videos (text-embedding-3-small
   over captions; mining only), one gpt-6-luna call per (A, B) (`prompts/twin_facts_v1.txt`);
3. ground the natural candidates and every twin fact on A's local frames (`grounding_v2`,
   ground_v3, Gemini + gpt-6-astra, both must say yes), one call per scene window and model;
4. natural queries: nested k = 1..4 from the confirmed candidates (`constraints.nested_order`),
   written with query_mc_v2 (gpt-6-luna);
5. twin queries: best B (`twins.choose_twin`: shared facts yes on B, d no on B from both
   models), Query_k = (k - 1 shared) + d, written with query_mc_v2;
6. index visibility: one Gemini call on A's index frames (0.5 fps) over the natural candidates
   and the twin facts used ("the index dropped it" statistic);
7. template renders ("video showing: X; Y; Z") of every fact set.

Writes to the config's out_dir (data/mc/smoke_v2/): constraints.jsonl (per source: mined facts,
candidates, per-model local answers, agreement, index visibility), queries.jsonl (natural),
queries_twin.jsonl, twins.jsonl (per source: neighbours, pair facts, B passes, the chosen twin),
grounding_audit.csv and twin_audit.csv (audit by model agreement: every disagreement plus a
seeded sample of agreements; no human check), build_report.md, build_stats.json, manifest.json.

    uv run --env-file .env python dev/build_queries_dev.py --estimate        # no paid call
    uv run --env-file .env python dev/build_queries_dev.py --max-input-tokens 20000000
    uv run --env-file .env python dev/build_queries_dev.py --only twins      # = build_twins.py
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import yaml

from src import constraints, grounding_v2, queries_mc, twins
from src.cache import shared_cache
from src.datasets.mc import (
    Caption,
    Video,
    load_captions,
    load_videos,
    read_jsonl,
    source_key,
    split_global_id,
)
from src.frames import FramePolicy
from src.grounding_v2 import FrameStore, Grounder, add_costs, agreement_label
from src.llm import CachedVision, make_llm
from src.types import Cost

CONFIG = Path("configs/mc_smoke_v2.yaml")
N_AUDIT = 40
# input tokens per frame, measured: Gemini ~1,100-1,190 per 768-px frame (describe on the smoke
# set; live check 2026-10-23: 7,133 for 6 frames + prompt), gpt-6-astra ~440 per 768-px frame at
# high detail (live check: 2,940 for 6 frames + prompt)
TOKENS_PER_IMAGE = {"gemini": 1150, "openai": 450}
# Gemini 3 per-image budget by media resolution (docs: 280 / 560 / 1,120 plus a little overhead)
GEMINI_TOKENS_BY_RESOLUTION = {
    "MEDIA_RESOLUTION_LOW": 300,
    "MEDIA_RESOLUTION_MEDIUM": 580,
    "MEDIA_RESOLUTION_HIGH": 1150,
    "MEDIA_RESOLUTION_ULTRA_HIGH": 2300,
}
TOKENS_PER_WORD = 1.35
OUT_TOKENS = {"mine": 1400, "twin": 1500, "write": 310, "ground": 400}  # per call, from smoke


# ---- budget ------------------------------------------------------------------------------------


class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """Projected input tokens of paid (uncached) calls; a call that would pass max raises."""

    def __init__(self, max_tokens: float | None):
        self.max, self.spent, self.calls = max_tokens, 0.0, 0
        self._lock = threading.Lock()

    def charge(self, tokens: float, what: str) -> None:
        with self._lock:
            if self.max is not None and self.spent + tokens > self.max:
                raise BudgetExceeded(
                    f"{what}: projected input tokens {self.spent + tokens:,.0f} would pass "
                    f"--max-input-tokens {self.max:,.0f}"
                )
            self.spent += tokens
            self.calls += 1


def text_tokens(text: str) -> float:
    return len(text.split()) * TOKENS_PER_WORD


def image_tokens(spec: str, n: int, media_resolution: str | None = None) -> float:
    """Projected input tokens of n images; media_resolution (Gemini only) as in
    `src.llm.media_level`."""
    kind = spec.split(":", 1)[0]
    if kind == "gemini" and media_resolution:
        from src.llm import media_level

        return n * GEMINI_TOKENS_BY_RESOLUTION[media_level(media_resolution)]
    return n * TOKENS_PER_IMAGE.get(kind, 1100)


def media_resolution_of(model) -> str | None:
    """The Gemini media resolution of a (cached, guarded) vision model, None when unset."""
    for _ in range(3):
        if getattr(model, "media_resolution", None):
            return model.media_resolution
        model = getattr(model, "inner", None)
    return None


class GuardedLLM:
    def __init__(self, inner, budget: Budget, spec: str):
        self.inner, self.budget, self.spec = inner, budget, spec

    def is_cached(self, prompt: str) -> bool:
        return getattr(self.inner, "is_cached", lambda p: False)(prompt)

    def complete(self, prompt: str):
        if not self.is_cached(prompt):
            self.budget.charge(text_tokens(prompt), self.spec)
        return self.inner.complete(prompt)


class GuardedVision:
    def __init__(self, inner, budget: Budget, spec: str):
        self.inner, self.budget, self.spec = inner, budget, spec
        self.media_resolution = media_resolution_of(inner)

    def describe(self, prompt: str, images: list[bytes], labels=None):
        probe = getattr(self.inner, "is_cached", None)
        if probe is None or not probe(prompt, images, labels):
            self.budget.charge(
                text_tokens(prompt) + image_tokens(self.spec, len(images), self.media_resolution),
                self.spec,
            )
        return self.inner.describe(prompt, images, labels=labels)


def make_grounder_model(spec: str, media_resolution: str | None = None):
    """A cached vision model that accepts frame labels: Gemini, or OpenAI through
    `systems.verify.LabelledOpenAIVision` (default temperature; max 2,500 output tokens
    without reasoning effort). media_resolution (config `grounding_media_resolution`, e.g.
    "low") applies to Gemini only; unset keeps today's requests and cache keys."""
    from src.llm import make_vision, parse_spec
    from src.systems.verify import LabelledOpenAIVision

    kind, model, effort = parse_spec(spec)
    if kind == "openai":
        inner = LabelledOpenAIVision(model, effort)
        # the labelled path has no temperature fallback, and gpt-6-astra takes only the default
        inner.send_temperature = False
        if not effort:
            inner.max_tokens = 2500
        return CachedVision(inner, shared_cache("vlm"), spec)
    return make_vision(spec, shared_cache("vlm"), media_resolution=media_resolution)


# ---- data --------------------------------------------------------------------------------------


@cache
def dataset_tables(root: str, dataset: str) -> tuple[dict[str, Video], dict[str, Caption]]:
    folder = Path(root) / dataset
    return {v.video_id: v for v in load_videos(folder)}, load_captions(folder)


def lookup(root: str, gid: str) -> tuple[str, Video, Caption]:
    dataset, video_id = split_global_id(gid)
    videos, captions = dataset_tables(root, dataset)
    return dataset, videos[video_id], captions[video_id]


def twin_pool(cfg: dict, sample: dict) -> list[str]:
    """Videos that can be twins: the sample videos with a description in the index's
    descriptions file (all sample videos when that file is missing)."""
    name = cfg.get("descriptions_file", "descriptions.jsonl")
    have: dict[str, set[str]] = {}
    pool = []
    for gid in sample["videos"]:
        dataset, video_id = split_global_id(gid)
        if dataset not in have:
            path = Path(cfg["data_root"]) / dataset / name
            have[dataset] = (
                {r["video_id"] for r in read_jsonl(path) if r.get("description")}
                if path.exists()
                else None
            )
        if have[dataset] is None or video_id in have[dataset]:
            pool.append(gid)
    return pool


def neighbours_for(
    cfg: dict, sources: list[str], pool: list[str], embedder, cached_only: bool = False
) -> dict | None:
    """Caption-embedding neighbours of each source in the pool. cached_only: None unless every
    caption embedding is already in the embed cache (no paid call)."""
    root = cfg["data_root"]
    ids = list(dict.fromkeys([*pool, *sources]))
    texts = [twins.embedding_text(lookup(root, g)[2]) for g in ids]
    if cached_only:
        cache_ = getattr(embedder, "cache", None)
        if cache_ is None or any(cache_.get(embedder.namespace, t) is None for t in texts):
            return None
    matrix = embedder.encode(texts, "doc")
    vectors = dict(zip(ids, matrix, strict=True))
    source_of = {g: source_key(lookup(root, g)[1]) for g in ids}
    return twins.nearest_neighbours(
        sources, pool, vectors, source_of, cfg.get("twins", {}).get("neighbours", 5)
    )


# ---- one source --------------------------------------------------------------------------------


class Context:
    """Models, prompts and options shared by all sources."""

    def __init__(self, cfg, llm, grounder: Grounder, templates, natural=True, twin=True):
        self.cfg, self.llm, self.grounder, self.templates = cfg, llm, grounder, templates
        self.root = cfg["data_root"]
        self.natural, self.twin = natural, twin
        tw = cfg.get("twins", {})
        raw = tw.get("absence_policy")
        self.absence = FramePolicy.from_config(raw) if raw else twins.ABSENCE_POLICY
        self.max_b_passes = tw.get("max_b_passes", twins.MAX_B_PASSES)
        self.index_specs = tuple(cfg.get("index_check", grounding_v2.INDEX_CHECK))
        self.media_resolution = cfg.get("grounding_media_resolution")


def keyed_candidates(candidates: list[dict]) -> list[dict]:
    return [{**f, "key": f"n{f['id']}"} for f in candidates]


def build_source(gid: str, neighbours: list[tuple[str, float]], ctx: Context):
    """(constraints record, natural rows, twin rows, twin record, costs by stage) for one
    source video."""
    dataset, video, caption = lookup(ctx.root, gid)
    specs = tuple(ctx.grounder.models)
    costs: dict[str, dict[str, Cost]] = defaultdict(dict)
    source = {"source": gid, "dataset": dataset, "length_bin": video.length_bin}

    # 1. mine and select (cached)
    facts, n_scenes, cost, _ = constraints.mine_facts(ctx.llm, caption, ctx.templates["facts"])
    costs["mine"][constraints.LLM_SPEC] = cost
    candidates = keyed_candidates(constraints.select_candidates(facts, gid)) if ctx.natural else []

    # 2. twin facts per pair
    pairs = []
    if ctx.twin:
        for rank, (b, sim) in enumerate(neighbours, 1):
            _b_dataset, b_video, b_caption = lookup(ctx.root, b)
            n_b = len(grounding_v2.scene_windows(b_caption, b_video.duration_s))
            mined = twins.mine_pair(
                ctx.llm, ctx.templates["twin"], caption, b_caption, n_scenes, n_b
            )
            add_costs(costs["twin_mine"], {twins.TWIN_SPEC: Cost(**mined["cost"])})
            pairs.append({"b": b, "rank": rank, "similarity": sim, **mined})
    taken = {twins.norm_text(f["text"]): f["key"] for f in candidates}
    a_twin_facts = twins.a_side_facts(pairs, taken)

    # 3. local grounding on A: natural candidates and twin facts together
    to_ground = candidates + a_twin_facts
    local, cost = ctx.grounder.local(dataset, video, caption, to_ground)
    costs["ground_local"] = add_costs(costs["ground_local"], cost)
    confirmed_keys = {k for k, r in local.items() if r["confirmed"]}

    # 4. natural nested queries
    natural_rows, written_log = [], {}
    confirmed = [f for f in candidates if f["key"] in confirmed_keys]
    for f in confirmed:
        f["grounded_frame"] = None
        f["grounded_time"] = local[f["key"]]["answers"].get(specs[0], {}).get("time")
    text_dropped: list[dict] = []
    order = constraints.nested_order(confirmed, dropped=text_dropped) if ctx.natural else []
    for k, subset in queries_mc.nested_sets(order).items():
        written = queries_mc.write_query(ctx.llm, subset, ctx.templates["query"])
        add_costs(costs["write"], {queries_mc.LLM_SPEC: written["cost"]})
        written_log[str(k)] = {x: written[x] for x in ("query", "valid", "attempts", "missing")}
        if written["valid"]:
            row = queries_mc.query_row(source, k, subset, written)
            row["set"] = "natural"
            row["query_template"] = queries_mc.render_template(subset)
            for out, f in zip(row["facts"], subset, strict=True):
                out["key"], out["grounded_time"] = f["key"], f["grounded_time"]
            natural_rows.append(row)

    # 5. twin
    twin_rows, twin_record = [], {"source": gid, "neighbours": neighbours}
    chosen = None
    if ctx.twin:

        def run_b(pair, shared, distinct):
            b_dataset, b_video, b_caption = lookup(ctx.root, pair["b"])
            record, cost = twins.b_pass(
                ctx.grounder, b_dataset, b_video, b_caption, shared, distinct, ctx.absence
            )
            add_costs(costs["ground_twin_b"], cost)
            return record

        chosen, log = twins.choose_twin(pairs, confirmed_keys, run_b, ctx.max_b_passes)
        twin_record.update(
            pairs=[{k: v for k, v in p.items() if k != "raw"} for p in pairs],
            log=log,
            chosen=None,
        )
        if chosen:
            t_order = twins.twin_order(chosen["d"], chosen["valid_shared"])
            twin_record["chosen"] = {
                "b": chosen["b"],
                "rank": chosen["rank"],
                "similarity": chosen["similarity"],
                "d": chosen["d"],
                "order": [f["a_key"] for f in t_order],
            }
            for k, (subset, pos) in twins.twin_sets(t_order, gid).items():
                written = queries_mc.write_query(ctx.llm, subset, ctx.templates["query"])
                add_costs(costs["write"], {queries_mc.LLM_SPEC: written["cost"]})
                if written["valid"]:
                    row = twins.twin_row(source, k, subset, pos, chosen, written)
                    row["query_template"] = queries_mc.render_template(subset)
                    twin_rows.append(row)

    # 6. index visibility of the natural candidates and the twin facts used
    check = list(candidates)
    if chosen:
        check += [
            {"key": f["a_key"], "text": f["text"]}
            for f in [chosen["d"], *chosen["valid_shared"]]
            if f["a_key"] not in {c["key"] for c in check}
        ]
    index, cost = ctx.grounder.index_check(dataset, video, check, ctx.index_specs)
    costs["ground_index"] = add_costs(costs["ground_index"], cost)
    for row in natural_rows:
        for f in row["facts"]:
            f["index_visible"] = index[f["key"]]["visible"]
    for row in twin_rows:
        for f in row["facts"]:
            f["index_visible"] = index.get(f["a_key"], {}).get("visible")

    keyed = {f["key"]: f for f in to_ground}
    record = {
        **source,
        "video_id": video.video_id,
        "duration_s": video.duration_s,
        "caption_origin": caption.caption_origin,
        "n_scenes": n_scenes,
        "facts": facts,
        "candidates": [f["id"] for f in candidates],
        "grounding": [
            {
                "key": key,
                "text": keyed[key]["text"],
                "type": keyed[key].get("type"),
                "role": "natural" if key.startswith("n") else "twin",
                **rec,
                "agreement": agreement_label(rec["answers"], specs),
                "index": index.get(key),
                "index_dropped": (rec["confirmed"] and key in index and not index[key]["visible"]),
            }
            for key, rec in local.items()
        ],
        "grounded": [f["id"] for f in confirmed],
        "dropped_by_text_rule": [f["id"] for f in text_dropped],
        "selection": [f["id"] for f in order],
        "queries": written_log,
        "twin": (twin_record.get("chosen") or {}).get("b"),
        "cost": {
            stage: {spec: c.to_dict() for spec, c in by.items()} for stage, by in costs.items()
        },
        "models": {
            "mine": constraints.LLM_SPEC,
            "write": queries_mc.LLM_SPEC,
            "twin": twins.TWIN_SPEC,
            "ground": list(specs),
            "index_check": list(ctx.index_specs),
        },
        "prompts": {
            "facts": constraints.FACTS_PROMPT.name,
            "ground": grounding_v2.GROUND_PROMPT.name,
            "query": queries_mc.QUERY_PROMPT.name,
            "twin": twins.TWIN_PROMPT.name,
        },
    }
    return record, natural_rows, twin_rows, twin_record, costs


# ---- estimate ----------------------------------------------------------------------------------


def estimate(cfg: dict, sources: list[str], pool: list[str], ctx: Context, embedder=None) -> dict:
    """Projected input / output tokens per stage without any paid call. Exact where the inputs
    are known offline (mining from the cache, A's local frame plan over all its scenes, index
    frame counts); twin calls use the real neighbours when caption embeddings are cached, else
    the five pool videos with the most similar caption length; B passes assume the first
    `assume_b_passes` neighbours are grounded."""
    root, specs = ctx.root, tuple(ctx.grounder.models)
    store = ctx.grounder.store
    tw = cfg.get("twins", {})
    n_nb = tw.get("neighbours", 5)
    passes = float(tw.get("assume_b_passes", 2))
    stages: dict[str, dict[str, float]] = defaultdict(lambda: {"in": 0.0, "out": 0.0, "calls": 0})

    def add(stage, tokens_in, tokens_out, calls=1):
        stages[stage]["in"] += tokens_in
        stages[stage]["out"] += tokens_out
        stages[stage]["calls"] += calls

    neighbours = None
    if embedder is not None and ctx.twin:
        neighbours = neighbours_for(cfg, sources, pool, embedder, cached_only=True)
        if neighbours is None:
            print("(caption embeddings not cached: twin pairs estimated by caption length)")
    lengths = {g: len(lookup(root, g)[2].caption.split()) for g in pool}
    probe = getattr(ctx.llm, "is_cached", lambda p: False)
    ground_template = ctx.templates["ground"]
    for gid in sources:
        dataset, video, caption = lookup(root, gid)
        text, _n_scenes = constraints.scene_caption(caption)
        prompt = ctx.templates["facts"].replace("{caption}", text)
        if not probe(prompt):
            add("mine", text_tokens(prompt), OUT_TOKENS["mine"])
        windows = grounding_v2.scene_windows(caption, video.duration_s)
        idx_times = store.index_times(dataset, video)
        n_local = sum(
            len(
                grounding_v2.local_times(
                    w, idx_times, video.duration_s, ctx.grounder.n_local, ctx.grounder.fallback
                )
            )
            for w in set(windows.values())
        )
        n_facts = 10 + (25 if ctx.twin else 0)
        for spec in specs:
            calls = len(set(windows.values()))
            prompt_in = text_tokens(ground_template) * calls + n_facts * 15
            add(
                "ground_local",
                prompt_in + image_tokens(spec, n_local, ctx.media_resolution),
                OUT_TOKENS["ground"] * calls,
                calls,
            )
        for spec in ctx.index_specs:
            add(
                "ground_index",
                text_tokens(ground_template)
                + image_tokens(spec, len(idx_times), ctx.media_resolution),
                OUT_TOKENS["ground"],
            )
        add("write", 8 * 180, 8 * OUT_TOKENS["write"], 8)
        if not ctx.twin:
            continue
        if neighbours is not None:
            nbs = [b for b, _ in neighbours[gid]]
        else:
            own = lengths.get(gid, len(caption.caption.split()))
            others = [g for g in pool if g != gid]
            nbs = sorted(others, key=lambda g: abs(lengths[g] - own))[:n_nb]
        for b in nbs:
            _, _, b_caption = lookup(root, b)
            prompt = twins.twin_prompt(ctx.templates["twin"], caption, b_caption)
            if not probe(prompt):
                add("twin_mine", text_tokens(prompt), OUT_TOKENS["twin"])
        for b in nbs[: round(passes)]:
            b_dataset, b_video, b_caption = lookup(root, b)
            b_windows = grounding_v2.scene_windows(b_caption, b_video.duration_s)
            scenes = sorted(b_windows)[:3]
            times = twins.b_times(ctx.grounder, b_dataset, b_video, b_caption, scenes, ctx.absence)
            for spec in specs:
                add(
                    "ground_twin_b",
                    text_tokens(ground_template)
                    + 150
                    + image_tokens(spec, len(times), ctx.media_resolution),
                    OUT_TOKENS["ground"],
                )
    total_in = sum(s["in"] for s in stages.values())
    total_out = sum(s["out"] for s in stages.values())
    return {"stages": dict(stages), "total_in": total_in, "total_out": total_out}


def print_estimate(est: dict, n_sources: int) -> None:
    print(f"projected cost for {n_sources} sources (uncached calls only, rough):")
    for stage, s in est["stages"].items():
        print(
            f"  {stage:14s} calls {s['calls']:6.0f}  in {s['in'] / 1e6:7.2f}M  "
            f"out {s['out'] / 1e6:6.2f}M"
        )
    print(f"  total          in {est['total_in'] / 1e6:.2f}M  out {est['total_out'] / 1e6:.2f}M")


# ---- reports and audits ------------------------------------------------------------------------

AUDIT_FIELDS = (
    "source",
    "key",
    "role",
    "fact",
    "type",
    "window",
    "agreement",
    "index_visible",
    "gemini_answer",
    "gemini_evidence",
    "gemini_time",
    "openai_answer",
    "openai_evidence",
    "openai_time",
    "frame_dir",
)
TWIN_AUDIT_FIELDS = (
    "source",
    "twin",
    "role",
    "fact",
    "type",
    "on_a_gemini",
    "on_a_openai",
    "on_b_gemini",
    "on_b_openai",
    "on_b_evidence_gemini",
    "on_b_evidence_openai",
    "b_caption_evidence",
)


def _by_kind(answers: dict, kind: str) -> dict:
    return next((a for s, a in answers.items() if s.startswith(kind + ":")), {})


def audit_rows(records: list[dict], root: str, n: int = N_AUDIT, seed: str = "smoke_v2") -> list:
    """Every grounding decision the two models disagree on, plus a seeded sample of n
    agreements."""
    rows = []
    for r in records:
        frame_dir = str(Path(root) / r["dataset"] / grounding_v2.LOCAL_FOLDER / r["video_id"])
        for g in r["grounding"]:
            gem, oai = _by_kind(g["answers"], "gemini"), _by_kind(g["answers"], "openai")
            rows.append(
                {
                    "source": r["source"],
                    "key": g["key"],
                    "role": g["role"],
                    "fact": g["text"],
                    "type": g.get("type"),
                    "window": g["window"]["kind"],
                    "agreement": g["agreement"],
                    "index_visible": (g.get("index") or {}).get("visible"),
                    "gemini_answer": gem.get("answer"),
                    "gemini_evidence": gem.get("evidence"),
                    "gemini_time": gem.get("time"),
                    "openai_answer": oai.get("answer"),
                    "openai_evidence": oai.get("evidence"),
                    "openai_time": oai.get("time"),
                    "frame_dir": frame_dir,
                }
            )
    disagree = [x for x in rows if x["agreement"] == "disagree"]
    agree = [x for x in rows if x["agreement"] != "disagree"]
    rng = random.Random(constraints.seed_for(seed, "grounding_audit"))
    return disagree + sorted(rng.sample(agree, min(n, len(agree))), key=lambda x: x["source"])


def twin_audit_rows(twin_records: list[dict], records: list[dict]) -> list[dict]:
    """For each chosen twin: d and the shared facts with both models' answers on A and on B."""
    local = {r["source"]: {g["key"]: g for g in r["grounding"]} for r in records}
    rows = []
    for t in twin_records:
        chosen = t.get("chosen")
        if not chosen:
            continue
        entry = next((e for e in t["log"] if e["b"] == chosen["b"]), {})
        b_pass = entry.get("b_pass", {})
        pair = next(p for p in t["pairs"] if p["b"] == chosen["b"])
        facts = {f["a_key"]: f for f in pair["shared"] + pair["distinct"]}
        for key in chosen["order"]:
            role = "distinct" if key == chosen["d"]["a_key"] else "shared"
            on_a = local[t["source"]].get(key, {}).get("answers", {})
            on_b = b_pass.get(role, {}).get(key, {})
            rows.append(
                {
                    "source": t["source"],
                    "twin": chosen["b"],
                    "role": role,
                    "fact": facts[key]["text"],
                    "type": facts[key]["type"],
                    "on_a_gemini": _by_kind(on_a, "gemini").get("answer"),
                    "on_a_openai": _by_kind(on_a, "openai").get("answer"),
                    "on_b_gemini": _by_kind(on_b, "gemini").get("answer"),
                    "on_b_openai": _by_kind(on_b, "openai").get("answer"),
                    "on_b_evidence_gemini": _by_kind(on_b, "gemini").get("evidence"),
                    "on_b_evidence_openai": _by_kind(on_b, "openai").get("evidence"),
                    "b_caption_evidence": facts[key].get("b_evidence", ""),
                }
            )
    return rows


def write_csv(rows: list[dict], fields: tuple[str, ...], path: Path) -> None:
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def yield_table(records: list[dict], rows: list[dict], k_max: int = constraints.K_MAX) -> list:
    """Markdown rows: per group (all, dataset, length bin) the number of sources and the share
    with a query at each k."""
    have = defaultdict(set)
    for row in rows:
        have[row["source"]].add(row["k"])
    groups = defaultdict(list)
    for r in records:
        for name in ("all", f"dataset={r['dataset']}", f"bin={r['length_bin']}"):
            groups[name].append(r["source"])
    lines = ["| group | n | " + " | ".join(f"k={k}" for k in range(1, k_max + 1)) + " |"]
    lines.append("|---|---|" + "---|" * k_max)
    for name in sorted(groups, key=lambda g: (g != "all", g)):
        srcs = groups[name]
        cells = [f"{sum(k in have[s] for s in srcs) / len(srcs):.2f}" for k in range(1, k_max + 1)]
        lines.append(f"| {name} | {len(srcs)} | " + " | ".join(cells) + " |")
    return lines


def stats(records: list[dict], natural: list[dict], twin_rows: list[dict], specs) -> dict:
    items = [g for r in records for g in r["grounding"]]
    for g in items:
        g.setdefault("type", None)
    out: dict = {
        "n_sources": len(records),
        "n_natural": len(natural),
        "n_twin": len(twin_rows),
        "windows": dict(sorted(_count(g["window"]["kind"] for g in items).items())),
        "agreement": dict(sorted(_count(g["agreement"] for g in items).items())),
        "confirmed": sum(g["confirmed"] for g in items),
        "n_decisions": len(items),
    }
    if len(specs) == 2:
        out["kappa_by_type"] = grounding_v2.agreement_table(items, specs[0], specs[1], by="type")
    dropped = defaultdict(lambda: [0, 0])
    for r in records:
        for g in r["grounding"]:
            if g["confirmed"] and g.get("index") is not None:
                for name in ("all", f"dataset={r['dataset']}", f"bin={r['length_bin']}"):
                    dropped[name][0] += 1
                    dropped[name][1] += bool(g["index_dropped"])
    out["index_dropped"] = {
        name: {"confirmed": n, "dropped": d, "share": d / n if n else None}
        for name, (n, d) in sorted(dropped.items(), key=lambda kv: (kv[0] != "all", kv[0]))
    }
    out["d_types"] = dict(sorted(_count(r["d_type"] for r in twin_rows if r["k"] == 1).items()))
    return out


def _count(values) -> dict:
    out: dict = {}
    for v in values:
        out[v] = out.get(v, 0) + 1
    return out


def cost_lines(records: list[dict]) -> list[str]:
    total: dict[tuple[str, str], Cost] = defaultdict(Cost)
    for r in records:
        for stage, by in r["cost"].items():
            for spec, c in by.items():
                total[(stage, spec)] = total[(stage, spec)] + Cost(**c)
    lines = [
        "| stage | model | calls | images | tokens in | tokens out |",
        "|---|---|---|---|---|---|",
    ]
    for (stage, spec), c in sorted(total.items()):
        lines.append(
            f"| {stage} | {spec} | {c.llm_calls} | {c.images} | {c.tokens_in:,} | {c.tokens_out:,} |"
        )
    return lines


def report(records, natural, twin_rows, st) -> str:
    lines = ["# Query pipeline v2: smoke build", ""]
    lines += [
        (
            f"Sources {st['n_sources']}, natural queries {st['n_natural']}, "
            f"twin queries {st['n_twin']}. Recorded cost includes cached calls."
        ),
        "",
    ]
    lines += ["## Natural yield (share of sources with a query at k)", ""]
    lines += yield_table(records, natural) + [""]
    lines += ["## Twin yield (share of sources with a twin query at k)", ""]
    lines += yield_table(records, twin_rows) + [""]
    lines += ["## Grounding", "", f"Windows: {st['windows']}", f"Agreement: {st['agreement']}", ""]
    if "kappa_by_type" in st:
        lines += [
            "| type | n | raw agreement | kappa | kappa (yes vs not) | yes A | yes B |",
            "|---|---|---|---|---|---|---|",
        ]
        for t, s in st["kappa_by_type"].items():
            kap = "-" if s["kappa"] is None else f"{s['kappa']:.2f}"
            kyes = "-" if s["kappa_yes"] is None else f"{s['kappa_yes']:.2f}"
            lines.append(
                f"| {t} | {s['n']} | {s['raw_agreement']:.2f} | {kap} | {kyes} | "
                f"{s['yes_a']:.2f} | {s['yes_b']:.2f} |"
            )
        lines.append("")
    lines += ["## Index dropped it (confirmed locally, not visible in the index frames)", ""]
    lines += ["| group | confirmed | dropped | share |", "|---|---|---|---|"]
    for name, s in st["index_dropped"].items():
        share = "-" if s["share"] is None else f"{s['share']:.2f}"
        lines.append(f"| {name} | {s['confirmed']} | {s['dropped']} | {share} |")
    lines += ["", f"Twin d types (k = 1): {st['d_types']}", "", "## Cost", ""]
    lines += cost_lines(records)
    return "\n".join(lines) + "\n"


# ---- main --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(CONFIG))
    parser.add_argument("--limit", type=int, default=None, help="first N sources only")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--only", choices=("all", "natural", "twins"), default="all")
    parser.add_argument("--estimate", action="store_true", help="print the projection and stop")
    parser.add_argument("--max-input-tokens", type=float, default=None)
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = cfg["data_root"]
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    sources = sample["query_sources"][: args.limit]
    pool = twin_pool(cfg, sample)
    max_tokens = args.max_input_tokens or cfg.get("max_input_tokens")
    budget = Budget(max_tokens)

    llm = GuardedLLM(make_llm(constraints.LLM_SPEC, shared_cache("llm")), budget, "llm")
    gspecs = tuple(cfg.get("grounders", grounding_v2.GROUNDERS))
    media = cfg.get("grounding_media_resolution")
    models = {s: GuardedVision(make_grounder_model(s, media), budget, s) for s in gspecs}
    lf = cfg.get("local_frames", {})
    store = FrameStore(
        root,
        lf.get("max_side", grounding_v2.MAX_SIDE),
        index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8)),
    )
    fallback = (
        FramePolicy.from_config(cfg["fallback_policy"])
        if cfg.get("fallback_policy")
        else grounding_v2.FALLBACK_POLICY
    )
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
    ctx = Context(cfg, llm, grounder, templates, args.only != "twins", args.only != "natural")
    from src.embedders import make_embedder

    embedder = make_embedder(cfg.get("twins", {}).get("embedder", twins.EMBED_SPEC))

    est = estimate(cfg, sources, pool, ctx, embedder if args.estimate else None)
    print_estimate(est, len(sources))
    if args.estimate:
        return
    if max_tokens is not None and est["total_in"] > max_tokens:
        raise SystemExit(
            f"projected {est['total_in']:,.0f} input tokens > --max-input-tokens {max_tokens:,.0f}"
        )

    neighbours = neighbours_for(cfg, sources, pool, embedder) if ctx.twin else {}
    print(
        f"caption embeddings paid: {embedder.cost.to_dict() if hasattr(embedder, 'cost') else '-'}"
    )

    def one(gid: str):
        return build_source(gid, neighbours.get(gid, []), ctx)

    for g in sources:
        lookup(root, g)
    with ThreadPoolExecutor(args.workers or cfg.get("workers", 8)) as pool_ex:
        results = list(pool_ex.map(one, sources))

    out = Path(cfg["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    records = [r[0] for r in results]
    natural = [row for r in results for row in r[1]]
    twin_rows = [row for r in results for row in r[2]]
    twin_records = [r[3] for r in results]
    suffix = "" if args.only == "all" else f"_{args.only}"
    with open(out / f"constraints{suffix}.jsonl", "w") as handle:
        handle.writelines(json.dumps(r) + "\n" for r in records)
    if ctx.natural:
        with open(out / "queries.jsonl", "w") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in natural)
    if ctx.twin:
        with open(out / "queries_twin.jsonl", "w") as handle:
            handle.writelines(json.dumps(r) + "\n" for r in twin_rows)
        with open(out / "twins.jsonl", "w") as handle:
            handle.writelines(json.dumps(t) + "\n" for t in twin_records)
        write_csv(twin_audit_rows(twin_records, records), TWIN_AUDIT_FIELDS, out / "twin_audit.csv")
    write_csv(audit_rows(records, root), AUDIT_FIELDS, out / f"grounding_audit{suffix}.csv")
    st = stats(records, natural, twin_rows, gspecs)
    (out / f"build_stats{suffix}.json").write_text(json.dumps(st, indent=1))
    (out / f"build_report{suffix}.md").write_text(report(records, natural, twin_rows, st))
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
            "embed": twins.EMBED_SPEC,
            "grounding_media_resolution": media,
        },
        cost=total,
        extra={"config": args.config, "only": args.only, "budget_projected_paid": budget.spent},
    )
    print(f"{len(records)} sources, {len(natural)} natural and {len(twin_rows)} twin queries")
    print(f"paid calls projected {budget.spent:,.0f} input tokens over {budget.calls} calls")
    print(report(records, natural, twin_rows, st))


if __name__ == "__main__":
    main()
