"""Smoke pilot on the v2 query sets (natural nested and twin), 0.5 fps index. PAID.

Systems as in dev/verifiers_dev.py (whose functions and the src.systems modules
are reused unchanged) over the 0.5 fps describe_v4 index (results/_indexes/79d1955c, 499
videos): S0 single search; S1 split + one search per constraint + RRF; S2 / S4 per-constraint /
holistic checks of S0's top 20 against the 0.5 fps descriptions (rescored at top 10 from the
same verdicts as S2@10 / S4@10); S3 / S5 the same against the candidate's own 0.5 fps index
frames (data/mc/<dataset>/frames_fps0.5/), labelled "Frame i of n (t = ... s)", no frame cap,
top 10, on a seeded subset of (source, k) pairs present in both sets, balanced over k and
spread over length bins (proportional, at least one per bin where available). Verifier
`openai:gpt-6-sol@low`, splitter `openai:gpt-6-luna@low`; all calls go through the shared caches.

Stages (each paid stage prints its projection of new input tokens first and stops when the
running total in <out>/spend.json plus the projection would pass --cap):

    uv run --env-file .env python dev/run_systems_dev.py --stage retrieve   # S0, split, S1
    uv run --env-file .env python dev/run_systems_dev.py --stage desc       # S2, S4
    uv run --env-file .env python dev/run_systems_dev.py --stage frames --measure 2
    uv run --env-file .env python dev/run_systems_dev.py --stage frames --reserve 25e6
    uv run --env-file .env python dev/run_systems_dev.py --stage frames \
        --query-ids results/smoke_v2/pilot/frame_subset_twin_both.json --stage-cap 30e6
    uv run python dev/run_systems_dev.py --stage record --what judge --tokens-in N ...
    uv run python dev/run_systems_dev.py --stage eval   # strict + judge qrels, report
    uv run --env-file .env python dev/run_systems_dev.py --stage eval-ids \
        --query-ids results/smoke_v2/pilot/frame_subset_twin_both.json   # conditional twin set

`--query-ids FILE` ({set: [query ids]}) runs S3 / S5 on exactly those queries instead of the
seeded subset: the new rankings are merged into <set>/S3.jsonl and S5.jsonl (lines of other
queries kept), frame_subset.json is left as it is, and with --stage-cap the stage stops when its
own projection passes that cap (the overall --cap is not applied).

Rankings: results/smoke_v2/pilot/<set>/<system>.jsonl (full depth, with trace). Report:
results/smoke_v2/pilot_report.md (with <out>/notes.md, if present, at the top).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import yaml

from src.cache import shared_cache
from src.datasets.mc import (
    McCorpus,
    corpus_descriptions,
    description_index,
    index_options,
    index_settings,
    index_store,
    read_jsonl,
)
from src.embedders import make_embedder
from src.frames import FramePolicy
from src.grounding import frame_labels
from src.grounding_v2 import agreement_table
from src.llm import CachedVision, make_llm
from src.manifest import write_manifest
from src.systems.decompose import SPLIT_PROMPT, SPLIT_SPEC, QuerySplitter, fuse
from src.systems.rerank import (
    DESC_EVIDENCE,
    FRAMES_EVIDENCE,
    RERANK_PROMPT,
    DescriptionScore,
    FrameScore,
    HolisticReranker,
    fill,
)
from src.systems.single import search_all
from src.systems.verify import (
    DESC_PROMPT,
    FRAMES_PROMPT,
    VERIFIER_SPEC,
    ConstraintVerifier,
    DescriptionCheck,
    FrameCheck,
    LabelledOpenAIVision,
    numbered,
    saved_frame_source,
)
from src.types import Cost, Query

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """A sibling script as a module (read-only use of its functions)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / "dev" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


PV = load_script("verifiers_dev")

CONFIG = Path("configs/mc_smoke_v2.yaml")
OUT = Path("results/smoke_v2/pilot")
REPORT = Path("results/smoke_v2/pilot_report.md")
JUDGE_DIR = Path("data/mc/smoke_v2/judge")
SETS = {"natural": "queries.jsonl", "twin": "queries_twin.jsonl"}
CAP = 55e6
CHARS_PER_TOKEN = 3.6  # conservative text-token estimate for projections
TOKENS_PER_IMAGE = 400.0  # gpt-6-sol, 768-px frame at high detail (smoke: ~345-390); --measure
TEXT_PER_CALL = 450.0  # prompt tokens of an S3 / S5 call besides the frames (for --measure)
VRIPT = ("vript_short", "vript_long")
METRICS = PV.METRICS  # recall@1, recall@5, ndcg@10, mrr
ALL_SYSTEMS = ("S0", "S1", "S2", "S4", "S2@10", "S4@10")
SUB_SYSTEMS = ("S0", "S2@10", "S4@10", "S3", "S5")
ALL_PAIRS = (
    ("S1", "S0"),
    ("S2", "S0"),
    ("S4", "S0"),
    ("S2", "S4"),
    ("S2@10", "S0"),
    ("S4@10", "S0"),
)
SUB_PAIRS = (("S3", "S2@10"), ("S5", "S4@10"), ("S3", "S5"), ("S3", "S0"), ("S5", "S0"))
GAIN_ALL = ("S1", "S2", "S4", "S2@10", "S4@10")
GAIN_SUB = ("S2@10", "S4@10", "S3", "S5")


# ---- data -------------------------------------------------------------------------------------


def load_index(cfg: dict):
    """The 0.5 fps description index of the smoke corpus (videos with an empty description
    left out), as in the caption-seeded query check (not in this repo)."""
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    options = index_options(cfg)
    corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], sample["videos"])
    descriptions = corpus_descriptions(
        corpus, Path(cfg["describe_prompt"]).name, options["filename"]
    )
    missing = sorted(d for d in corpus.doc_ids if not descriptions.get(d))
    if missing:
        kept = [d for d in corpus.doc_ids if d not in set(missing)]
        corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], kept)
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
    embedder = make_embedder(cfg["embedder"], cache=shared_cache("embed", cfg["results_dir"]))
    index = description_index(corpus, descriptions, embedder, store)
    return corpus, descriptions, missing, settings, store, embedder, index, sample, options


def load_sets(out_dir: Path, missing: list[str]) -> dict[str, list[dict]]:
    """{set: query rows} without queries whose source is not in the index."""
    drop = set(missing)
    return {
        name: [r for r in read_jsonl(out_dir / file) if r["source"] not in drop]
        for name, file in SETS.items()
    }


# ---- the frame subset -------------------------------------------------------------------------


def allocate(avail: dict[str, int], n: int) -> dict[str, int]:
    """n split over bins: at least one per bin that has any, the rest proportional to the bin
    sizes (largest shortfall first, ties in bin order), never more than a bin has."""
    bins = sorted(avail)
    n = min(n, sum(avail.values()))
    alloc = {b: 0 for b in bins}
    for b in bins:
        if avail[b] and sum(alloc.values()) < n:
            alloc[b] = 1
    total = sum(avail.values())
    while sum(alloc.values()) < n:
        open_bins = [b for b in bins if alloc[b] < avail[b]]
        best = max(open_bins, key=lambda b: (n * avail[b] / total - alloc[b], -bins.index(b)))
        alloc[best] += 1
    return alloc


def matched_subset(sets: dict[str, list[dict]], total: int, seed: int = 0) -> list[tuple[str, int]]:
    """About total (source, k) pairs that have a query in every set, balanced over k
    (pilot_verify.per_k_counts) and spread over length bins within each k (`allocate`), each
    bin drawn by a seeded shuffle of its sorted pairs."""
    keys = None
    for rows in sets.values():
        have = {(r["source"], r["k"]) for r in rows}
        keys = have if keys is None else keys & have
    bin_of = {(r["source"], r["k"]): r["length_bin"] for rows in sets.values() for r in rows}
    pseudo = [{"k": k} for _, k in sorted(keys)]
    out = []
    for k, n in PV.per_k_counts(pseudo, total).items():
        by_bin: dict[str, list[tuple[str, int]]] = {}
        for key in sorted(x for x in keys if x[1] == k):
            by_bin.setdefault(bin_of[key], []).append(key)
        for b, m in allocate({b: len(v) for b, v in by_bin.items()}, n).items():
            pool = list(by_bin[b])
            random.Random(f"{seed}|k{k}|{b}").shuffle(pool)
            out += pool[:m]
    return sorted(out)


def subset_query_ids(rows: list[dict], keys: list[tuple[str, int]]) -> list[str]:
    wanted = set(keys)
    return sorted(r["query_id"] for r in rows if (r["source"], r["k"]) in wanted)


def read_query_ids(path: Path, sets: dict[str, list[dict]]) -> dict[str, list[str]]:
    """{set: sorted query ids} from a JSON file ({set: [ids]}, sets may be missing); every id
    must be a query of its set."""
    given = json.loads(Path(path).read_text())
    unknown = sorted(set(given) - set(sets))
    if unknown:
        sys.exit(f"--query-ids: unknown sets {unknown}")
    out = {}
    for name, rows in sets.items():
        have = {r["query_id"] for r in rows}
        ids = sorted(set(given.get(name, [])))
        missing = [q for q in ids if q not in have]
        if missing:
            sys.exit(f"--query-ids: {len(missing)} ids not in the {name} set, e.g. {missing[:3]}")
        out[name] = ids
    return out


# ---- spend ledger -----------------------------------------------------------------------------


class Ledger:
    """Running total of paid input tokens of this pilot (<out>/spend.json)."""

    def __init__(self, path: Path, cap: float):
        self.path, self.cap = path, cap
        self.entries = json.loads(path.read_text())["entries"] if path.exists() else []

    @property
    def total(self) -> float:
        return float(sum(e["tokens_in"] for e in self.entries))

    def check(
        self, stage: str, projected: float, reserve: float = 0.0, stage_cap: float | None = None
    ) -> None:
        if stage_cap is not None:
            print(
                f"[{stage}] projection: {projected:,.0f} new input tokens; stage cap "
                f"{stage_cap:,.0f} (overall cap not applied); running total {self.total:,.0f}",
                flush=True,
            )
            if projected > stage_cap:
                sys.exit(f"[{stage}] stop: this stage would exceed its cap")
            return
        after = self.total + projected + reserve
        print(
            f"[{stage}] projection: {projected:,.0f} new input tokens; running total "
            f"{self.total:,.0f}; reserve {reserve:,.0f}; after {after:,.0f} of cap {self.cap:,.0f}",
            flush=True,
        )
        if after > self.cap:
            sys.exit(f"[{stage}] stop: running total would exceed the cap")

    def add(self, stage: str, cost: Cost, calls: int, extra: dict | None = None) -> None:
        self.entries.append(
            {
                "stage": stage,
                "tokens_in": int(cost.tokens_in),
                "tokens_out": int(cost.tokens_out),
                "calls": calls,
                "images": int(cost.images),
                "when": datetime.now(UTC).isoformat(timespec="seconds"),
                **(extra or {}),
            }
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"cap": self.cap, "entries": self.entries}, indent=1))
        print(
            f"[{stage}] paid {cost.tokens_in:,} in / {cost.tokens_out:,} out; total {self.total:,.0f}"
        )


def text_tokens(text: str) -> float:
    return len(text) / CHARS_PER_TOKEN


# ---- models -----------------------------------------------------------------------------------


class Models:
    """Splitter, text verifier and frame verifier as in pilot_verify (same cache keys), with
    the paid calls metered."""

    def __init__(self, cfg: dict):
        self.errors: list[str] = []
        self.paid: dict[str, list] = {"luna": [], "sol_text": [], "sol_vision": []}
        cache = shared_cache("llm", cfg["results_dir"])
        split_cached = PV.more_retries(make_llm(SPLIT_SPEC, cache))
        split_cached.inner = PV.Metered(split_cached.inner, self.paid["luna"])
        self.split_cached = split_cached
        self.splitter = QuerySplitter(
            PV.SafeCalls(split_cached, self.errors), SPLIT_PROMPT.read_text()
        )
        verifier_cached = PV.more_retries(make_llm(VERIFIER_SPEC, cache))
        verifier_cached.inner = PV.Metered(verifier_cached.inner, self.paid["sol_text"])
        self.verifier_cached = verifier_cached
        self.verifier = PV.SafeCalls(verifier_cached, self.errors)
        vlm_inner = PV.Metered(
            LabelledOpenAIVision(VERIFIER_SPEC.split(":", 1)[1].split("@")[0], "low"),
            self.paid["sol_vision"],
        )
        self.vlm_cached = PV.more_retries(
            CachedVision(vlm_inner, shared_cache("vlm", cfg["results_dir"]), VERIFIER_SPEC)
        )
        self.vlm = PV.SafeCalls(self.vlm_cached, self.errors)

    def paid_cost(self) -> tuple[Cost, int]:
        logs = [c for log in self.paid.values() for c in log]
        return sum(logs, Cost()), len(logs)


# ---- stages -----------------------------------------------------------------------------------


def s0_records(index, rows: list[dict], depth: int) -> dict[str, dict]:
    queries = [Query(r["query_id"], r["query"], r["source"]) for r in rows]
    rankings, cost = search_all(index, queries, depth)
    per = cost.seconds / max(1, len(queries))
    return {
        q: {"ranking": r, "cost": Cost(searches=1, seconds=per).to_dict()}
        for q, r in rankings.items()
    }


def run_splits(models: Models, rows: list[dict], workers: int) -> dict[str, tuple]:
    with ThreadPoolExecutor(workers) as pool:
        found = pool.map(lambda r: models.splitter(r["query"]), rows)
        return dict(zip([r["query_id"] for r in rows], found, strict=True))


def stage_retrieve(cfg, ctx, sets, args, ledger: Ledger, models: Models) -> None:
    template = SPLIT_PROMPT.read_text()
    rows = [r for rs in sets.values() for r in rs]
    new = [
        r
        for r in rows
        if not models.split_cached.is_cached(template.replace("{query}", r["query"]))
    ]
    proj = sum(text_tokens(template.replace("{query}", r["query"])) for r in new)
    proj += sum(text_tokens(r["query"]) * 4 for r in rows)  # query + constraint embeddings
    print(f"splits to pay: {len(new)} of {len(rows)}")
    ledger.check("retrieve", proj)
    index, depth = ctx["index"], len(ctx["corpus"].doc_ids)
    embed_before = ctx["embedder"].cost.tokens_in
    for name, rs in sets.items():
        out = Path(args.out) / name
        by_id = {r["query_id"]: r for r in rs}
        s0 = s0_records(index, rs, depth)
        splits = run_splits(models, rs, args.workers * args.inner)
        subs = [
            Query(f"{r['query_id']}#c{i}", c, r["source"])
            for r in rs
            for i, c in enumerate(splits[r["query_id"]][0], 1)
        ]
        start = time.perf_counter()
        sub_rank, _sub_cost = search_all(index, subs, depth)
        per_search = (time.perf_counter() - start) / max(1, len(subs))
        s1 = {}
        for r in rs:
            cons, cost = splits[r["query_id"]]
            lists = [sub_rank[f"{r['query_id']}#c{i}"] for i in range(1, len(cons) + 1)]
            s1[r["query_id"]] = {
                "ranking": fuse(lists),
                "cost": (cost + Cost(searches=len(cons), seconds=per_search * len(cons))).to_dict(),
                "trace": {"constraints": cons},
            }
        PV.write_rankings(out / "S0.jsonl", "S0", s0, by_id)
        PV.write_rankings(out / "S1.jsonl", "S1", s1, by_id)
        (out / "splits.json").write_text(
            json.dumps({q: c for q, (c, _) in splits.items()}, indent=1)
        )
        print(f"{name}: S0, S1 on {len(rs)} queries")
    paid, calls = models.paid_cost()
    embed = Cost(tokens_in=ctx["embedder"].cost.tokens_in - embed_before)
    ledger.add("retrieve", paid + embed, calls, {"embedding_tokens": embed.tokens_in})


def read_s0(out: Path) -> dict:
    return {
        line["query_id"]: [(d, s) for d, s in line["ranking"]]
        for line in read_jsonl(out / "S0.jsonl")
    }


def read_splits(out: Path) -> dict[str, list[str]]:
    return json.loads((out / "splits.json").read_text())


def desc_projection(
    models: Models, sets, args, descriptions, n2: int | None = None, n4: int | None = None
) -> tuple[float, int]:
    """New input tokens and calls of S2 and S4 (prompts built as the checkers build them), on
    S0's top n2 (S2) and n4 (S4); both default to args.n. n2 = 0 or n4 = 0 leaves a system out."""
    n2 = args.n if n2 is None else n2
    n4 = args.n if n4 is None else n4
    desc_t, rerank_t = DESC_PROMPT.read_text(), RERANK_PROMPT.read_text()
    tokens, calls = 0.0, 0
    for name, rs in sets.items():
        out = Path(args.out) / name
        s0, splits = read_s0(out), read_splits(out)
        for r in rs:
            for i, (d, _) in enumerate(s0[r["query_id"]][: max(n2, n4)]):
                desc = descriptions.get(d, "") or "(no description)"
                p2 = (
                    desc_t.replace("{query}", r["query"])
                    .replace("{constraints}", numbered(splits[r["query_id"]]))
                    .replace("{description}", desc)
                )
                p4 = fill(rerank_t, r["query"], DESC_EVIDENCE.replace("{description}", desc))
                for p in [p for p, n in ((p2, n2), (p4, n4)) if i < n]:
                    if not models.verifier_cached.is_cached(p):
                        tokens += text_tokens(p)
                        calls += 1
    return tokens, calls


def merge_records(path: Path, records: dict[str, dict]) -> dict[str, dict]:
    """The ranking records already saved at path (if any), with records replacing those of the
    same queries; saved order first, new queries after."""
    merged = {}
    if path.exists():
        for line in read_jsonl(path):
            merged[line["query_id"]] = {
                "ranking": line["ranking"],
                "cost": line["cost"],
                "trace": line.get("trace", {}),
            }
    merged.update(records)
    return merged


def rerank_set(
    models,
    name,
    rs,
    plan,
    args,
    ledger_stage: str,
    merge: bool = False,
    first: dict | None = None,
    cost_file: str = "cost.json",
) -> dict:
    """Run each planned system on its queries; with merge, the new rankings are merged into the
    saved <system>.jsonl and their cost goes to cost.json under "<system> (<stage>)". first:
    the first-stage rankings to rerank ({query_id: ranking}; default <out>/<set>/S0.jsonl).
    cost_file: the per-set cost file to update instead of cost.json."""
    out = Path(args.out) / name
    s0 = read_s0(out) if first is None else first
    by_id = {r["query_id"]: r for r in rs}
    costs = json.loads((out / cost_file).read_text()) if (out / cost_file).exists() else {}
    for system, (reranker, qids, n) in plan.items():
        if not qids:
            continue
        queries = [Query(q, by_id[q]["query"], by_id[q]["source"]) for q in qids]
        print(f"{name} {system}: {len(queries)} queries x top-{n}", flush=True)
        start = time.perf_counter()
        records = PV.rerank_all(reranker, queries, s0, n, args.workers, f"{name} {system}")
        wall = time.perf_counter() - start
        path = out / f"{system}.jsonl"
        PV.write_rankings(path, system, merge_records(path, records) if merge else records, by_id)
        key = f"{system} ({ledger_stage})" if merge else system
        costs[key] = {**PV.per_call(records), "wall_seconds": wall, "stage": ledger_stage}
        print(f"  {costs[key]}", flush=True)
    costs["n_errors"] = len(models.errors)
    costs["errors"] = models.errors[:50]
    (out / cost_file).write_text(json.dumps(costs, indent=1))
    return costs


def stage_desc(cfg, ctx, sets, args, ledger: Ledger, models: Models) -> None:
    proj, calls = desc_projection(models, sets, args, ctx["descriptions"])
    print(f"S2 + S4 calls to pay: {calls}")
    ledger.check("desc", proj)
    desc_t, rerank_t = DESC_PROMPT.read_text(), RERANK_PROMPT.read_text()
    for name, rs in sets.items():
        qids = [r["query_id"] for r in rs]
        plan = {
            "S2": (
                ConstraintVerifier(
                    "S2",
                    models.splitter,
                    DescriptionCheck(models.verifier, desc_t, ctx["descriptions"]),
                    args.inner,
                ),
                qids,
                args.n,
            ),
            "S4": (
                HolisticReranker(
                    "S4",
                    DescriptionScore(models.verifier, rerank_t, ctx["descriptions"]),
                    args.inner,
                ),
                qids,
                args.n,
            ),
        }
        rerank_set(models, name, rs, plan, args, "desc")
    paid, n = models.paid_cost()
    ledger.add("desc", paid, n, {"errors": len(models.errors)})


def frame_source(ctx):
    policy = ctx["options"]["frame_policy"] or FramePolicy(ctx["cfg"]["n_frames"])
    corpus = ctx["corpus"]
    return saved_frame_source(lambda d: corpus.frames_dir(d, policy=policy))


def frame_projection(
    models, sets, qids_by_set, args, frames, per_image: float, first: dict | None = None,
    systems: tuple[str, ...] = ("S3", "S5"),
) -> tuple[float, int, int]:
    """New input tokens, calls and images of S3 and S5 (or the given systems) on the given
    queries. first: {set: {query_id: ranking}} to rerank (default <out>/<set>/S0.jsonl)."""
    frames_t, rerank_t = FRAMES_PROMPT.read_text(), RERANK_PROMPT.read_text()
    cache: dict[str, tuple] = {}

    def load(doc):
        if doc not in cache:
            cache[doc] = frames(doc)
        return cache[doc]

    tokens, calls, images = 0.0, 0, 0
    for name, rs in sets.items():
        out = Path(args.out) / name
        s0 = read_s0(out) if first is None else first[name]
        splits = read_splits(out)
        by_id = {r["query_id"]: r for r in rs}
        for q in qids_by_set[name]:
            r = by_id[q]
            for d, _ in s0[q][: args.n_frames_systems]:
                imgs, times = load(d)
                labels = frame_labels(times)
                p3 = (
                    frames_t.replace("{n}", str(len(imgs)))
                    .replace("{query}", r["query"])
                    .replace("{constraints}", numbered(splits[q]))
                )
                p5 = fill(rerank_t, r["query"], FRAMES_EVIDENCE.replace("{n}", str(len(imgs))))
                for p in [p for p, s in ((p3, "S3"), (p5, "S5")) if s in systems]:
                    if not models.vlm_cached.is_cached(p, imgs, labels):
                        tokens += text_tokens(p) + 12 * len(imgs) + per_image * len(imgs)
                        calls += 1
                        images += len(imgs)
    return tokens, calls, images


def stage_frames(cfg, ctx, sets, args, ledger: Ledger, models: Models) -> None:
    out = Path(args.out)
    frames = frame_source(ctx)
    rate_file = out / "frame_rate.json"
    per_image = (
        json.loads(rate_file.read_text())["tokens_per_image"]
        if rate_file.exists()
        else TOKENS_PER_IMAGE
    )
    seed = ctx["sample"]["seed"]

    def qids_for(total: int) -> dict[str, list[str]]:
        keys = matched_subset(sets, total, seed)
        return {name: subset_query_ids(rs, keys) for name, rs in sets.items()}

    total = args.subset_total
    if args.query_ids:
        qids = read_query_ids(Path(args.query_ids), sets)
        stage = args.stage_name or "frames-ids"
    elif args.measure:
        keys = matched_subset(sets, args.subset_total, seed)
        keys = keys[:: max(1, len(keys) // args.measure)][: args.measure]
        qids = {name: subset_query_ids(rs, keys) for name, rs in sets.items()}
        stage = "frames-measure"
    else:
        qids = qids_for(total)
        stage = "frames"
    proj, calls, images = frame_projection(models, sets, qids, args, frames, per_image)
    print(
        f"S3 + S5: {sum(map(len, qids.values()))} queries, {calls} calls, {images:,} images to "
        f"pay at {per_image:.0f} tokens per image"
    )
    while (
        not args.measure
        and not args.query_ids
        and ledger.total + proj + args.reserve > ledger.cap
        and total > 4
    ):
        total -= 4
        qids = qids_for(total)
        proj, calls, images = frame_projection(models, sets, qids, args, frames, per_image)
        print(f"  shrink subset to {total} (source, k) pairs: {proj:,.0f} tokens")
    if args.project_only:
        print(f"subset {total}: {proj:,.0f} tokens, {calls} calls, {images:,} images")
        return
    ledger.check(stage, proj, args.reserve, args.stage_cap)
    frames_t, rerank_t = FRAMES_PROMPT.read_text(), RERANK_PROMPT.read_text()
    for name, rs in sets.items():
        plan = {
            "S3": (
                ConstraintVerifier(
                    "S3", models.splitter, FrameCheck(models.vlm, frames_t, frames), args.inner
                ),
                qids[name],
                args.n_frames_systems,
            ),
            "S5": (
                HolisticReranker("S5", FrameScore(models.vlm, rerank_t, frames), args.inner),
                qids[name],
                args.n_frames_systems,
            ),
        }
        if args.measure:
            measure_out = out / "measure" / name
            measure_out.mkdir(parents=True, exist_ok=True)
            for src in ("S0.jsonl", "splits.json"):
                (measure_out / src).write_text((out / name / src).read_text())
            margs = argparse.Namespace(**{**vars(args), "out": str(out / "measure")})
            rerank_set(models, name, rs, plan, margs, stage)
        else:
            rerank_set(models, name, rs, plan, args, stage, merge=bool(args.query_ids))
    paid, n = models.paid_cost()
    vision = sum(models.paid["sol_vision"], Cost())
    n_vision = len(models.paid["sol_vision"])
    if vision.images:
        # about TEXT_PER_CALL prompt tokens per call and ~12 per frame label
        text_part = TEXT_PER_CALL * n_vision + 12 * vision.images
        rate = max(0.0, vision.tokens_in - text_part) / vision.images
        print(
            f"measured: {vision.tokens_in:,} in for {vision.images:,} images -> {rate:.0f} per image"
        )
        if args.measure:
            rate_file.write_text(json.dumps({"tokens_per_image": rate, "calls": n_vision}))
    if args.query_ids:
        extra = {
            "errors": len(models.errors),
            "query_ids": str(args.query_ids),
            "queries": {name: len(v) for name, v in qids.items()},
            "projected": round(proj),
        }
        ledger.add(stage, paid, n, extra)
        return
    ledger.add(stage, paid, n, {"errors": len(models.errors), "subset_pairs": total})
    if not args.measure:
        (out / "frame_subset.json").write_text(json.dumps(qids, indent=1))


# ---- evaluation -------------------------------------------------------------------------------


def diff_of_means_ci(
    a: list[float],
    a_cl: list[str],
    b: list[float],
    b_cl: list[str],
    n_boot: int = 2000,
    seed: int = 0,
) -> tuple[float, float, float]:
    """mean(a) - mean(b) with a 95% CI from one bootstrap over the clusters of both (source
    videos are resampled jointly, so a source's queries in both sets move together)."""
    clusters = sorted(set(a_cl) | set(b_cl))
    pos = {c: i for i, c in enumerate(clusters)}
    sa, na, sb, nb = (np.zeros(len(clusters)) for _ in range(4))
    for v, c in zip(a, a_cl, strict=True):
        sa[pos[c]] += v
        na[pos[c]] += 1
    for v, c in zip(b, b_cl, strict=True):
        sb[pos[c]] += v
        nb[pos[c]] += 1
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(clusters), size=(n_boot, len(clusters)))
    with np.errstate(invalid="ignore", divide="ignore"):
        diffs = sa[picks].sum(1) / na[picks].sum(1) - sb[picks].sum(1) / nb[picks].sum(1)
    lo, hi = np.nanpercentile(diffs, [2.5, 97.5])
    return float(np.mean(a) - np.mean(b)), float(lo), float(hi)


def gain_difference(scores: dict, sets: dict, system: str, metric: str, ids: dict | None) -> dict:
    """Per k and overall: (twin gain over S0) - (natural gain over S0) for system."""
    out = {}
    for group in ("all", 1, 2, 3, 4):
        vals = {}
        for name in ("twin", "natural"):
            sc, s0 = scores[name].get(system), scores[name].get("S0")
            if sc is None or s0 is None:
                return {}
            keep = [
                r
                for r in sets[name]
                if r["query_id"] in sc
                and (ids is None or r["query_id"] in ids[name])
                and (group == "all" or r["k"] == group)
            ]
            vals[name] = (
                [sc[r["query_id"]][metric] - s0[r["query_id"]][metric] for r in keep],
                [r["source"] for r in keep],
            )
        if not vals["twin"][0] or not vals["natural"][0]:
            continue
        m, lo, hi = diff_of_means_ci(*vals["twin"], *vals["natural"])
        out[str(group)] = {
            "n_twin": len(vals["twin"][0]),
            "n_natural": len(vals["natural"][0]),
            "gain_twin": float(np.mean(vals["twin"][0])),
            "gain_natural": float(np.mean(vals["natural"][0])),
            "mean": m,
            "ci_low": lo,
            "ci_high": hi,
        }
    return out


def twin_win_rate(rankings: dict, rows: list[dict], ids: set | None = None) -> dict:
    """Per k and overall: share of twin queries where the twin ranks above the source."""
    groups: dict = {}
    for r in rows:
        q = r["query_id"]
        if q not in rankings or (ids is not None and q not in ids):
            continue
        rank = {d: i for i, (d, _) in enumerate(rankings[q])}
        big = len(rank) + 1
        win = rank.get(r["twin"], big) < rank.get(r["source"], big)
        for g in ("all", str(r["k"])):
            groups.setdefault(g, []).append(win)
    return {g: {"n": len(v), "rate": float(np.mean(v))} for g, v in groups.items()}


def cost_table(out: Path) -> dict:
    """Per system: mean per query of calls, tokens in/out, images, seconds (summed call seconds)
    and searches, from the saved ranking lines (cached calls count with their recorded cost)."""
    table = {}
    for path in sorted(out.glob("S*.jsonl")):
        costs = [Cost(**line["cost"]) for line in read_jsonl(path)]
        n = max(1, len(costs))
        tot = sum(costs, Cost())
        table[path.stem] = {
            "queries": len(costs),
            "calls": tot.llm_calls / n,
            "tokens_in": tot.tokens_in / n,
            "tokens_out": tot.tokens_out / n,
            "images": tot.images / n,
            "seconds": tot.seconds / n,
            "searches": tot.searches / n,
        }
    return table


def load_judge_qrels(rows: list[dict], judge_dir: Path, kind: str) -> dict | None:
    path = judge_dir / f"qrels_{kind}.jsonl"
    return PV.load_qrels(path, rows) if path.exists() else None


def fact_agreement_by_k(judge_dir: Path, sets: dict, rankings_by_set: dict, top: int = 10) -> dict:
    """Gemini x astra agreement on the facts of each query over the judged candidates in that
    query's pool (union of the systems' top `top`), per k and per set."""
    path = judge_dir / "judged.jsonl"
    if not path.exists():
        return {}
    from src.judge_frames import JUDGES, source_facts

    judged: dict[str, dict[str, dict]] = {}
    for line in read_jsonl(path):
        judged.setdefault(line["source"], {})[line["candidate"]] = line["answers"]
    out = {}
    for name, rows in sets.items():
        items = []
        by_source: dict[str, list[dict]] = {}
        all_rows = [r for rs in sets.values() for r in rs]
        for r in all_rows:
            by_source.setdefault(r["source"], []).append(r)
        for r in rows:
            _, of_query = source_facts(by_source[r["source"]])
            pool = {
                d
                for ranks in rankings_by_set[name].values()
                for d, _ in ranks.get(r["query_id"], [])[:top]
            }
            for d in pool:
                answers = judged.get(r["source"], {}).get(d)
                if answers is None or d == r["source"]:
                    continue
                for key in of_query[r["query_id"]]:
                    items.append({"answers": answers.get(key, {}), "k": r["k"]})
        out[name] = agreement_table(items, JUDGES[0], JUDGES[1], by="k")
    return out


def evaluate(sets: dict, out: Path, judge_dir: Path) -> dict:
    systems = {name: PV.read_rankings(out / name) for name in sets}
    subset = {name: set(systems[name]["S3"]) if "S3" in systems[name] else set() for name in sets}
    if (out / "frame_subset.json").exists():  # the seeded subset (S3 may hold more, --query-ids)
        seeded = json.loads((out / "frame_subset.json").read_text())
        subset = {name: subset[name] & set(seeded.get(name, [])) for name in sets}
    result: dict = {"labels": {}, "subset_sizes": {n: len(s) for n, s in subset.items()}}
    for kind in ("strict", "expanded"):
        qrels = {
            name: (
                PV.load_qrels(None, rows)
                if kind == "strict"
                else load_judge_qrels(rows, judge_dir, "expanded")
            )
            for name, rows in sets.items()
        }
        if any(q is None for q in qrels.values()):
            continue
        res: dict = {"means": {}, "subset_means": {}, "paired": {}, "gain_diff": {}}
        scores = {}
        for name, rows in sets.items():
            sc = {s: PV.score_system(r, qrels[name]) for s, r in systems[name].items()}
            scores[name] = sc
            vript = [r for r in rows if r["dataset"] in VRIPT]
            sub_rows = [r for r in rows if r["query_id"] in subset[name]]
            res["means"][name] = {
                s: {
                    "overall": PV.mean_table(sc[s], rows, None),
                    "k": PV.mean_table(sc[s], rows, "k"),
                    "dataset": PV.mean_table(sc[s], rows, "dataset"),
                    "length_bin_vript": PV.mean_table(sc[s], vript, "length_bin"),
                }
                for s in ALL_SYSTEMS
                if s in sc
            }
            res["subset_means"][name] = {
                s: {
                    "overall": PV.mean_table(sc[s], sub_rows, None),
                    "k": PV.mean_table(sc[s], sub_rows, "k"),
                    "dataset": PV.mean_table(sc[s], sub_rows, "dataset"),
                    "length_bin_vript": PV.mean_table(
                        sc[s], [r for r in sub_rows if r["dataset"] in VRIPT], "length_bin"
                    ),
                }
                for s in SUB_SYSTEMS
                if s in sc and sub_rows
            }
            pr = {}
            for pairs, rs in ((ALL_PAIRS, rows), (SUB_PAIRS, sub_rows)):
                for a, b in pairs:
                    if a not in sc or b not in sc or not rs:
                        continue
                    pr[f"{a} - {b}"] = {
                        "on": "all" if pairs is ALL_PAIRS else "subset",
                        **{
                            m: {
                                "overall": PV.paired(sc[a], sc[b], rs, m),
                                "k": PV.paired(sc[a], sc[b], rs, m, "k"),
                                "dataset": PV.paired(sc[a], sc[b], rs, m, "dataset"),
                                "length_bin_vript": PV.paired(
                                    sc[a],
                                    sc[b],
                                    [r for r in rs if r["dataset"] in VRIPT],
                                    m,
                                    "length_bin",
                                ),
                            }
                            for m in METRICS
                        },
                    }
            res["paired"][name] = pr
        for m in ("ndcg@10", "recall@1"):
            res["gain_diff"][m] = {
                **{s: gain_difference(scores, sets, s, m, None) for s in GAIN_ALL},
                **{f"{s} (subset)": gain_difference(scores, sets, s, m, subset) for s in GAIN_SUB},
            }
        result["labels"][kind] = res
    twin_rows = sets["twin"]
    result["twin_win"] = {
        s: twin_win_rate(r, twin_rows) for s, r in systems["twin"].items() if s in ALL_SYSTEMS
    }
    result["twin_win_subset"] = {
        s: twin_win_rate(r, twin_rows, subset["twin"])
        for s, r in systems["twin"].items()
        if s in SUB_SYSTEMS and subset["twin"]
    }
    result["costs"] = {name: cost_table(out / name) for name in sets}
    result["headroom"] = {
        name: {
            "top20": PV.headroom(systems[name]["S0"], rows, 20),
            "top10_subset": PV.headroom(systems[name]["S0"], rows, 10, subset[name])
            if subset[name]
            else {},
        }
        for name, rows in sets.items()
    }
    stats_path = judge_dir / "judge_stats.json"
    if stats_path.exists():
        result["judge"] = json.loads(stats_path.read_text())
        result["judge_fact_agreement_by_k"] = fact_agreement_by_k(judge_dir, sets, systems)
    return result


# ---- conditional twin evaluation on an explicit query-id set (--stage eval-ids) ---------------

IDS_SYSTEMS = ("S0", "S2@10", "S4@10", "S3", "S5")
IDS_PAIRS = (("S3", "S2@10"), ("S5", "S4@10"), ("S3", "S5"), ("S4@10", "S0"), ("S5", "S0"))
IDS_METRICS = ("ndcg@10", "recall@1", "mrr")
MATCH_PROMPT = Path("prompts/split_match_v1.txt")
PHRASE_MIN, PHRASE_MARGIN = 0.5, 0.15


def d_by_phrase(row: dict, constraints: list[str]) -> list[int] | None:
    """[index of the parsed constraint that states d] by token F1 of the query's stored phrase
    for d (or d's text) against each constraint; None (ambiguous) when the best F1 is below
    PHRASE_MIN, beats the runner-up by less than PHRASE_MARGIN, or another fact of the query
    matches that constraint at least as well."""
    if not constraints:
        return None
    di = row["d_index"]
    phrase = row.get("phrases", {}).get(str(di), row["d"])

    def score(fact_text: str, phrase_text: str, c: str) -> float:
        return max(PV.token_overlap(phrase_text, c), PV.token_overlap(fact_text, c))

    sc = [score(row["d"], phrase, c) for c in constraints]
    best = max(sc)
    i = sc.index(best)
    runner = sorted(sc, reverse=True)[1] if len(sc) > 1 else 0.0
    others = [
        score(f["text"], row.get("phrases", {}).get(str(j), f["text"]), constraints[i])
        for j, f in enumerate(row["facts"], 1)
        if j != di
    ]
    if best < PHRASE_MIN or best - runner < PHRASE_MARGIN or any(o >= best for o in others):
        return None
    return [i]


def d_by_match(text: str, d_index: int, n_cons: int) -> list[int]:
    """Constraint indices a split_match_v1 answer gives for fact d_index: the fact's own
    "constraints" list, else the constraints whose "fact" is d_index."""
    from src.constraints import as_int, parse_json

    data = parse_json(text)
    data = data if isinstance(data, dict) else {}
    found: set[int] = set()
    for pos, item in enumerate(data.get("facts") or [], 1):
        if isinstance(item, dict) and as_int(item.get("fact"), pos) == d_index:
            for c in item.get("constraints") or []:
                c = as_int(c, 0)
                if 1 <= c <= n_cons:
                    found.add(c - 1)
    if not found:
        for pos, item in enumerate(data.get("constraints") or [], 1):
            fact = item.get("fact") if isinstance(item, dict) else None
            if fact is not None and as_int(fact, 0) == d_index:
                c = as_int(item.get("constraint"), pos)
                if 1 <= c <= n_cons:
                    found.add(c - 1)
    return sorted(found)


def d_verdict(verdicts: list[str] | None, idx: list[int]) -> str:
    """ "no" if any d constraint is answered no, "yes" if all are yes, else "unclear"
    ("missing" when the video has no verdicts or d maps to no constraint)."""
    if verdicts is None or not idx:
        return "missing"
    got = [verdicts[i] if i < len(verdicts) else "unclear" for i in idx]
    if "no" in got:
        return "no"
    return "yes" if all(v == "yes" for v in got) else "unclear"


def share_ci(values: list[float], clusters: list[str], n_boot: int = 2000) -> dict:
    from src.systems.single import cluster_bootstrap_ci

    if not values:
        return {"n": 0}
    lo, hi = cluster_bootstrap_ci(values, clusters, n_boot)
    return {"n": len(values), "rate": float(np.mean(values)), "ci_low": lo, "ci_high": hi}


def by_k(rows: list[dict], value, n_boot: int = 2000) -> dict:
    """{"all"|k: share_ci} of value(row) (a float, or None to skip) over the rows."""
    out = {}
    for g in ("all", 1, 2, 3, 4):
        rs = [r for r in rows if g == "all" or r["k"] == g]
        vals = [(value(r), r["source"]) for r in rs]
        vals = [(v, c) for v, c in vals if v is not None]
        if vals:
            out[str(g)] = share_ci([v for v, _ in vals], [c for _, c in vals], n_boot)
    return out


def ids_evaluate(
    rows: list[dict], systems: dict, qrels: dict, traces: dict, d_map: dict[str, dict]
) -> dict:
    """Means by k, paired differences, twin-above-source rates and d verdicts on the given rows
    (all with S3 and S5). qrels: {kind: qrels}; traces: {"S2"|"S3": {query_id: trace}};
    d_map: {query_id: {"idx": [constraint indices], "how": "phrase"|"luna"}}."""
    res: dict = {"n": len(rows), "labels": {}}
    for kind, qr in qrels.items():
        sc = {
            s: PV.score_system({q: systems[s][q] for q in qr if q in systems[s]}, qr)
            for s in IDS_SYSTEMS
        }
        res["labels"][kind] = {
            "means": {
                s: {
                    "all": PV.mean_table(sc[s], rows, None)["all"],
                    **{str(k): v for k, v in PV.mean_table(sc[s], rows, "k").items()},
                }
                for s in IDS_SYSTEMS
            },
            "paired": {
                f"{a} - {b}": {
                    m: {
                        **PV.paired(sc[a], sc[b], rows, m),
                        **{str(k): v for k, v in PV.paired(sc[a], sc[b], rows, m, "k").items()},
                    }
                    for m in IDS_METRICS
                }
                for a, b in IDS_PAIRS
            },
        }

    def above(s):
        def f(r):
            rank = {d: i for i, (d, _) in enumerate(systems[s][r["query_id"]])}
            big = len(rank) + 1
            return float(rank.get(r["twin"], big) < rank.get(r["source"], big))

        return f

    res["twin_above_source"] = {s: by_k(rows, above(s)) for s in IDS_SYSTEMS}
    res["d"] = {"mapping": count_of(d_map[r["query_id"]]["how"] for r in rows)}
    for s, tr in traces.items():
        verdict = {
            r["query_id"]: {
                who: d_verdict(
                    tr[r["query_id"]]["verdicts"].get(r[who]), d_map[r["query_id"]]["idx"]
                )
                for who in ("twin", "source")
            }
            for r in rows
        }
        res["d"][s] = {
            "twin_rejects": by_k(
                rows, lambda r, v=verdict: float(v[r["query_id"]]["twin"] == "no")
            ),
            "twin_accepts": by_k(
                rows, lambda r, v=verdict: float(v[r["query_id"]]["twin"] == "yes")
            ),
            "source_accepts": by_k(
                rows, lambda r, v=verdict: float(v[r["query_id"]]["source"] == "yes")
            ),
            "source_rejects": by_k(
                rows, lambda r, v=verdict: float(v[r["query_id"]]["source"] == "no")
            ),
            "both": by_k(
                rows,
                lambda r, v=verdict: float(
                    v[r["query_id"]]["twin"] == "no" and v[r["query_id"]]["source"] == "yes"
                ),
            ),
            "verdict_pairs": count_of(f"{v['twin']}|{v['source']}" for v in verdict.values()),
        }
    return res


def count_of(items) -> dict:
    out: dict[str, int] = {}
    for x in items:
        out[x] = out.get(x, 0) + 1
    return dict(sorted(out.items()))


def rate_cell(d: dict) -> str:
    return f"{d['rate']:.3f} [{d['ci_low']:.2f}, {d['ci_high']:.2f}]" if d.get("n") else "-"


def ids_report(ev: dict, title: str, meta: dict) -> str:
    groups = ("all", "1", "2", "3", "4")
    lines = [f"# {title}", "", *meta.get("notes", []), ""]
    ns = ev["labels"][next(iter(ev["labels"]))]["means"]["S0"]
    lines += ["Queries: " + ", ".join(f"{g}: {ns[g]['n']}" for g in groups if g in ns), ""]
    for kind, res in ev["labels"].items():
        lines += [f"## {kind} labels: means by k", ""]
        for m in IDS_METRICS:
            lines += [
                f"### {m}",
                "",
                "| system | all | k=1 | k=2 | k=3 | k=4 |",
                "|---|---|---|---|---|---|",
            ]
            for s in IDS_SYSTEMS:
                cells = [f3(res["means"][s].get(g, {}).get(m)) for g in groups]
                lines.append(f"| {s} | " + " | ".join(cells) + " |")
            lines.append("")
        lines += [
            f"## {kind} labels: paired differences (95% CI, cluster bootstrap over sources)",
            "",
        ]
        lines += [
            "| pair | metric | all | k=1 | k=2 | k=3 | k=4 |",
            "|---|---|---|---|---|---|---|",
        ]
        for pair, by_m in res["paired"].items():
            for m in IDS_METRICS:
                cells = [ci(by_m[m][g]) if g in by_m[m] else "-" for g in groups]
                lines.append(f"| {pair} | {m} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## Twin ranked above the source (rate [95% CI])", ""]
    lines += ["| system | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
    for s, t in ev["twin_above_source"].items():
        lines.append(f"| {s} | " + " | ".join(rate_cell(t.get(g, {})) for g in groups) + " |")
    lines += ["", "## Verdicts on the distinguishing fact d (S2: descriptions, S3: frames)", ""]
    lines += [f"d mapped to the parsed constraint by: {ev['d']['mapping']}", ""]
    lines += ["| system | measure | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|---|"]
    for s in ("S2", "S3"):
        if s not in ev["d"]:
            continue
        for key, label in (
            ("twin_rejects", "twin: d = no"),
            ("twin_accepts", "twin: d = yes"),
            ("source_accepts", "source: d = yes"),
            ("source_rejects", "source: d = no"),
            ("both", "twin no and source yes"),
        ):
            t = ev["d"][s][key]
            lines.append(
                f"| {s} | {label} | " + " | ".join(rate_cell(t.get(g, {})) for g in groups) + " |"
            )
    lines.append("")
    for s in ("S2", "S3"):
        if s in ev["d"]:
            lines.append(f"{s} verdict pairs (twin|source): {ev['d'][s]['verdict_pairs']}")
    lines.append("")
    return "\n".join(lines)


def stage_eval_ids(cfg, ctx, sets, args, ledger: Ledger, models: Models) -> None:
    """Conditional evaluation on --query-ids (twin set): strict and expanded labels, d verdicts
    (d mapped by the stored phrase; luna split_match_v1 where the phrase match is ambiguous)."""
    out = Path(args.out)
    ids = read_query_ids(Path(args.query_ids), sets)["twin"]
    rows = [r for r in sets["twin"] if r["query_id"] in set(ids)]
    systems = PV.read_rankings(out / "twin")
    lacking = [
        r["query_id"] for r in rows if any(r["query_id"] not in systems[s] for s in IDS_SYSTEMS)
    ]
    if lacking:
        sys.exit(f"eval-ids: {len(lacking)} queries lack a system ranking, e.g. {lacking[:3]}")
    traces = {
        s: {line["query_id"]: line["trace"] for line in read_jsonl(out / "twin" / f"{s}.jsonl")}
        for s in ("S2", "S3")
    }
    template = MATCH_PROMPT.read_text()
    d_map, ambiguous = {}, []
    for r in rows:
        cons = traces["S3"][r["query_id"]]["constraints"]
        idx = d_by_phrase(r, cons)
        if idx is None:
            ambiguous.append(r)
        else:
            d_map[r["query_id"]] = {"idx": idx, "how": "phrase"}
    proj = sum(
        text_tokens(
            template.replace("{facts}", numbered([f["text"] for f in r["facts"]])).replace(
                "{constraints}", numbered(traces["S3"][r["query_id"]]["constraints"])
            )
        )
        for r in ambiguous
    )
    ledger.check("eval-ids-match", proj, stage_cap=args.stage_cap or 1e6)
    matcher = PV.SafeCalls(models.split_cached, models.errors)
    for r in ambiguous:
        cons = traces["S3"][r["query_id"]]["constraints"]
        prompt = template.replace("{facts}", numbered([f["text"] for f in r["facts"]])).replace(
            "{constraints}", numbered(cons)
        )
        text, _ = matcher.complete(prompt)
        d_map[r["query_id"]] = {
            "idx": d_by_match(text, r["d_index"], len(cons)),
            "how": "luna",
            "constraints": cons,
            "answer": text,
        }
    if ambiguous:
        paid, n = models.paid_cost()
        ledger.add("eval-ids-match", paid, n, {"errors": len(models.errors)})
    qrels = {
        "strict": PV.load_qrels(None, rows),
        "expanded": PV.load_qrels(Path(args.judge_dir) / "qrels_expanded.jsonl", rows),
    }
    ev = ids_evaluate(rows, systems, qrels, traces, d_map)
    ev["d"]["luna"] = {
        q: {k: v for k, v in m.items() if k != "answer"}
        for q, m in d_map.items()
        if m["how"] == "luna"
    }
    stem = Path(args.query_ids).stem.removeprefix("frame_subset_")
    (out / f"{stem}_summary.json").write_text(json.dumps(ev, indent=1, default=str))
    notes = [
        (
            f"Query ids: `{args.query_ids}` ({len(rows)} twin queries). Systems: S0, S2@10 and "
            "S4@10 (S2 / S4 verdicts and scores on S0's top 10), S3 and S5 (frames, top 10). "
            f"Labels: strict (source only) and expanded (`{args.judge_dir}/qrels_expanded.jsonl`)."
        ),
        "",
        (
            "d mapping: the query's stored phrase for d against the parsed constraints (token F1 "
            f">= {PHRASE_MIN}, margin {PHRASE_MARGIN}, no other fact as close); ambiguous ones by "
            f"gpt-6-luna@low with `{MATCH_PROMPT.name}`. A d spread over several constraints is "
            "rejected if any is no and accepted if all are yes."
        ),
    ]
    (out / f"{stem}_report.md").write_text(
        ids_report(ev, f"Conditional twin set: {stem}", {"notes": notes})
    )
    print(f"wrote {out}/{stem}_report.md and {stem}_summary.json")


# ---- report -----------------------------------------------------------------------------------


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def ci(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def means_block(
    title: str, means: dict, systems: tuple, metrics=("ndcg@10", "recall@1", "recall@5", "mrr")
) -> list[str]:
    lines = [f"#### {title}", "", "| system | group | n | " + " | ".join(metrics) + " |"]
    lines.append("|---" * (3 + len(metrics)) + "|")
    for s in systems:
        if s not in means:
            continue
        for key in ("overall", "k", "dataset", "length_bin_vript"):
            for g, st in means[s].get(key, {}).items():
                label = g if key != "length_bin_vript" else f"Vript {g}"
                lines.append(
                    f"| {s} | {label} | {st['n']} | "
                    + " | ".join(f3(st[m]) for m in metrics)
                    + " |"
                )
    return [*lines, ""]


def k_grid(means: dict, systems: tuple, metric: str = "ndcg@10") -> list[str]:
    head = "| system | all | k=1 | k=2 | k=3 | k=4 |"
    lines = [head, "|---|---|---|---|---|---|"]
    for s in systems:
        if s not in means:
            continue
        cells = [f3(means[s]["overall"].get("all", {}).get(metric))]
        for k in (1, 2, 3, 4):
            cells.append(f3(means[s]["k"].get(k, {}).get(metric)))
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def paired_block(paired: dict) -> list[str]:
    lines = [
        "| pair | on | group | n | nDCG@10 | R@1 | R@5 | MRR |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for pair, p in paired.items():
        for key in ("overall", "k", "dataset", "length_bin_vript"):
            for g, d in p["ndcg@10"][key].items():
                label = g if key != "length_bin_vript" else f"Vript {g}"
                cells = " | ".join(
                    ci(p[m][key][g]) for m in ("ndcg@10", "recall@1", "recall@5", "mrr")
                )
                lines.append(f"| {pair} | {p['on']} | {label} | {d['n']} | {cells} |")
    return [*lines, ""]


def gain_block(gd: dict) -> list[str]:
    lines = [
        "| system | group | n twin / natural | gain twin | gain natural | twin - natural [95% CI] |",
        "|---|---|---|---|---|---|",
    ]
    for s, by in gd.items():
        for g, d in by.items():
            lines.append(
                f"| {s} | {g} | {d['n_twin']} / {d['n_natural']} | {d['gain_twin']:+.3f} | "
                f"{d['gain_natural']:+.3f} | {ci(d)} |"
            )
    return [*lines, ""]


def report(ev: dict, sets: dict, notes: str, meta: dict) -> str:
    lines = ["# Smoke pilot v2: S0-S5 on the natural nested and twin sets (0.5 fps index)", ""]
    if notes:
        lines += [notes.strip(), ""]
    lines += [
        "## Setup (generated)",
        "",
        (
            f"- Index `{meta['store']}` ({meta['n_docs']} videos; left out for empty descriptions: "
            f"{meta['missing']}). Queries: natural {len(sets['natural'])}, twin {len(sets['twin'])} "
            "(paraphrased text; queries of the left-out source dropped)."
        ),
        (
            f"- Verifier `{VERIFIER_SPEC}`, splitter `{SPLIT_SPEC}`; prompts split_v1, verify_desc_v1, "
            "verify_frames_v1, rerank_v1. S2/S4 on S0's top 20 (S2@10/S4@10 rescored from the same "
            "verdicts and scores); S3/S5 on S0's top 10, the candidate's own 0.5 fps index frames, no cap."
        ),
        (
            f"- Frame subset: {ev['subset_sizes']} queries ((source, k) pairs present in both sets, "
            "balanced over k, spread over length bins)."
        ),
        "- Length bins are analysed within Vript only (Vript short + Vript long).",
        (
            "- CIs: 95%, cluster bootstrap over source videos (2,000 resamples). Twin - natural gain "
            "differences resample source videos jointly over both sets."
        ),
        "",
    ]
    for kind, res in ev["labels"].items():
        lines += [f"## Results, {kind} labels", ""]
        for name in ("natural", "twin"):
            lines += [f"### {name}: nDCG@10 by k (all queries)", ""]
            lines += k_grid(res["means"][name], ALL_SYSTEMS)
            if res["subset_means"].get(name):
                lines += [f"### {name}: nDCG@10 by k (frame subset)", ""]
                lines += k_grid(res["subset_means"][name], SUB_SYSTEMS)
        lines += ["### Twin minus natural: gain over S0 (nDCG@10)", ""]
        lines += gain_block(res["gain_diff"]["ndcg@10"])
        lines += ["### Twin minus natural: gain over S0 (R@1)", ""]
        lines += gain_block(res["gain_diff"]["recall@1"])
        for name in ("natural", "twin"):
            lines += [f"### {name}: paired differences", ""]
            lines += paired_block(res["paired"][name])
        for name in ("natural", "twin"):
            lines += means_block(f"{name}: means, all queries", res["means"][name], ALL_SYSTEMS)
            if res["subset_means"].get(name):
                lines += means_block(
                    f"{name}: means, frame subset", res["subset_means"][name], SUB_SYSTEMS
                )
    lines += ["## Twin win rate (twin ranked above the source)", ""]
    lines += ["| system | on | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|---|"]
    for on, table in (("all", ev["twin_win"]), ("subset", ev["twin_win_subset"])):
        for s in ALL_SYSTEMS if on == "all" else SUB_SYSTEMS:
            if s not in table:
                continue
            cells = [
                f"{table[s][g]['rate']:.3f} ({table[s][g]['n']})" if g in table[s] else "-"
                for g in ("all", "1", "2", "3", "4")
            ]
            lines.append(f"| {s} | {on} | " + " | ".join(cells) + " |")
    lines += ["", "## Rerank headroom (positive in S0's top N but not first, strict)", ""]
    lines += ["| set | N | group | n | share |", "|---|---|---|---|---|"]
    for name, h in ev["headroom"].items():
        for label, table in (("20", h["top20"]), ("10 (subset)", h["top10_subset"])):
            for g, d in table.items():
                lines.append(f"| {name} | {label} | {g} | {d['n']} | {d['share']:.3f} |")
    lines += ["", "## Online cost per query (recorded cost of every call, cached or not)", ""]
    lines += [
        "| set | system | queries | calls | tokens in | tokens out | images | seconds (sum of calls) | searches |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for name, table in ev["costs"].items():
        for s, c in table.items():
            lines.append(
                f"| {name} | {s} | {c['queries']} | {c['calls']:.1f} | {c['tokens_in']:,.0f} | "
                f"{c['tokens_out']:,.0f} | {c['images']:.1f} | {c['seconds']:.2f} | {c['searches']:.1f} |"
            )
    if "judge" in ev:
        j = ev["judge"]
        lines += ["", "## Frame judge (Gemini LOW + gpt-6-astra, both must agree)", ""]
        lines += [
            f"Pools: {j['pools']}. Sanity: {j['sanity']}. Extra positives: {j['extra_positives']}.",
            "",
        ]
        lines += ["### Agreement on facts, by fact type", ""]
        lines += [
            "| type | n | raw agreement | kappa | kappa (yes vs not) | yes Gemini | yes astra |"
        ]
        lines += ["|---|---|---|---|---|---|---|"]
        for t, s in j["agreement"]["facts"].items():
            lines.append(
                f"| {t} | {s['n']} | {s['raw_agreement']:.3f} | {f3(s['kappa'])} | "
                f"{f3(s['kappa_yes'])} | {s['yes_a']:.3f} | {s['yes_b']:.3f} |"
            )
        lines += ["", "### Agreement on query facts over each query's pooled candidates, by k", ""]
        lines += [
            "| set | k | n | raw agreement | kappa | kappa (yes vs not) |",
            "|---|---|---|---|---|---|",
        ]
        for name, table in ev.get("judge_fact_agreement_by_k", {}).items():
            for k, s in table.items():
                lines.append(
                    f"| {name} | {k} | {s['n']} | {s['raw_agreement']:.3f} | {f3(s['kappa'])} | {f3(s['kappa_yes'])} |"
                )
        lines += ["", "### Agreement on relevance decisions (all facts yes), by k (both sets)", ""]
        lines += ["| k | n | raw agreement | kappa | relevant Gemini | relevant astra | both |"]
        lines += ["|---|---|---|---|---|---|---|"]
        for k, s in j["agreement"]["relevance"].items():
            lines.append(
                f"| {k} | {s['n']} | {s['raw_agreement']:.3f} | {f3(s['kappa'])} | {s['relevant_a']} | "
                f"{s['relevant_b']} | {s['relevant_both']} |"
            )
        lines += ["", f"Verdict pairs (Gemini|astra): {j['agreement']['verdicts']}", ""]
        lines += ["### Unjudged extras (k >= 3), estimated missed positives", ""]
        lines += ["| k | queries | sampled | relevant | rate | unjudged | est. missed / query |"]
        lines += ["|---|---|---|---|---|---|---|"]
        for k, c in j["false_negatives"].items():
            lines.append(
                f"| {k} | {c['queries']} | {c['judged']} | {c['relevant']} | {f3(c['rate'])} | "
                f"{c['unjudged']} | {f3(c['est_missed_per_query'])} |"
            )
        lines += ["", f"Judge cost: {j['cost']}", ""]
    lines += ["## Spend (paid input tokens of this pilot)", ""]
    lines += [
        "| stage | calls | images | tokens in | tokens out | when |",
        "|---|---|---|---|---|---|",
    ]
    for e in meta["spend"]:
        lines.append(
            f"| {e['stage']} | {e['calls']:,} | {e.get('images', 0):,} | {e['tokens_in']:,} | "
            f"{e['tokens_out']:,} | {e['when']} |"
        )
    total = sum(e["tokens_in"] for e in meta["spend"])
    lines += [f"| total | | | {total:,} | {sum(e['tokens_out'] for e in meta['spend']):,} | |", ""]
    return "\n".join(lines)


def stage_eval(cfg, ctx, sets, args, ledger: Ledger) -> None:
    out = Path(args.out)
    ev = evaluate(sets, out, Path(args.judge_dir))
    (out / "summary.json").write_text(json.dumps(ev, indent=1, default=str))
    notes = (out / "notes.md").read_text() if (out / "notes.md").exists() else ""
    meta = {
        "store": str(ctx["store"]),
        "n_docs": len(ctx["corpus"].doc_ids),
        "missing": ctx["missing"],
        "spend": ledger.entries,
    }
    Path(args.report).write_text(report(ev, sets, notes, meta))
    write_manifest(
        out,
        models={
            "verifier": VERIFIER_SPEC,
            "splitter": SPLIT_SPEC,
            "embedder": cfg["embedder"],
            "describer": cfg["describer"],
        },
        seeds={"sample_seed": ctx["sample"]["seed"], "subset_seed": ctx["sample"]["seed"]},
        cost=Cost(tokens_in=int(ledger.total)),
        extra={
            "step": "smoke pilot v2 (natural + twin, 0.5 fps index)",
            "index_settings": ctx["settings"],
            "subset_sizes": ev["subset_sizes"],
            "prompts": [p.name for p in (SPLIT_PROMPT, DESC_PROMPT, FRAMES_PROMPT, RERANK_PROMPT)],
        },
    )
    print(f"wrote {args.report} and {out}/summary.json")


# ---- main -------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument(
        "--stage",
        required=True,
        choices=("retrieve", "desc", "frames", "record", "eval", "eval-ids"),
    )
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--judge-dir", default=str(JUDGE_DIR))
    ap.add_argument("--cap", type=float, default=CAP)
    ap.add_argument("--reserve", type=float, default=0.0, help="tokens kept for later stages")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--n-frames-systems", type=int, default=10)
    ap.add_argument("--subset-total", type=int, default=76, help="(source, k) pairs per set")
    ap.add_argument("--measure", type=int, default=0, help="frames: cost check on N pairs")
    ap.add_argument("--project-only", action="store_true")
    ap.add_argument(
        "--query-ids",
        help="frames: JSON {set: [query ids]} to run S3 / S5 on (merged into the saved rankings)",
    )
    ap.add_argument(
        "--stage-cap",
        type=float,
        default=None,
        help="stop when this stage's projection passes it (replaces the overall --cap check)",
    )
    ap.add_argument("--stage-name", help="ledger name of the stage (default frames-ids)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--what", help="record: stage name")
    ap.add_argument("--tokens-in", type=int, default=0)
    ap.add_argument("--tokens-out", type=int, default=0)
    ap.add_argument("--calls", type=int, default=0)
    ap.add_argument("--images", type=int, default=0)
    args = ap.parse_args(argv)
    ledger = Ledger(Path(args.out) / "spend.json", args.cap)
    if args.stage == "record":
        cost = Cost(tokens_in=args.tokens_in, tokens_out=args.tokens_out, images=args.images)
        ledger.add(args.what or "external", cost, args.calls)
        return
    cfg = yaml.safe_load(Path(args.config).read_text())
    corpus, descriptions, missing, settings, store, embedder, index, sample, options = load_index(
        cfg
    )
    ctx = {
        "cfg": cfg,
        "corpus": corpus,
        "descriptions": descriptions,
        "missing": missing,
        "settings": settings,
        "store": store,
        "embedder": embedder,
        "index": index,
        "sample": sample,
        "options": options,
    }
    sets = load_sets(Path(cfg["out_dir"]), missing)
    if args.stage == "eval":
        stage_eval(cfg, ctx, sets, args, ledger)
        return
    models = Models(cfg)
    if args.stage == "eval-ids" and not args.query_ids:
        sys.exit("eval-ids needs --query-ids")
    stages = {
        "retrieve": stage_retrieve,
        "desc": stage_desc,
        "frames": stage_frames,
        "eval-ids": stage_eval_ids,
    }
    stages[args.stage](cfg, ctx, sets, args, ledger, models)


if __name__ == "__main__":
    main()
