"""Full run: S0 / gate, G3 (S1, S2@50, S4@50, second verifier), G4 (S3, S5), repeats and E
(docs/full_run_plan.md sections 1-3, 5.3, 7.2, decisions 1-6 approved as their defaults).

Thin layer over the smoke code, which it imports and reuses unchanged: the systems in
src/systems/*, and the stage functions of dev/run_systems_dev.py (Models, s0_records,
run_splits, desc_projection, rerank_set, frame_projection, frame_source, Ledger),
verifiers_dev.py (rerank_all, write_rankings, scoring, paired CIs), second_verifier_dev.py
(FixedSplitter, project, verdict agreement) and trigger_dev.py (sweep, operating points,
cross-fit). Prompts, model specs and therefore cache keys are those of the smoke pilot. New
here: config-driven paths, depth 50, the frame-subset rules (twin-both, decision 3 fallback,
the 150-query Vript nested sample), repeats with fresh cache keys, per-stage caps from
configs/mc_full.yaml `caps`, canaries, retry passes, and the evaluation pieces (full-corpus
gate, 10,000-resample cells, per-bin loss statistics, length regression, S6 offline).

Stages (every paid stage prints its projection of new input tokens first and stops when the
stage's spend.json total (plus stages sharing its cap) plus the projection would pass its cap,
or all results/full/*/spend.json plus the projection would pass total_cap):

    uv run --env-file .env python pipeline/run_systems.py --stage retrieve --s0-only  # free gate
    uv run --env-file .env python pipeline/run_systems.py --stage retrieve            # + split, S1
    uv run python pipeline/run_systems.py --stage select      # freeze subsets (decision 3 aware)
    uv run --env-file .env python pipeline/run_systems.py --stage desc --project-only
    uv run --env-file .env python pipeline/run_systems.py --stage desc         # S2@50, S4@50
    uv run python pipeline/run_systems.py --stage select      # again if it said "pending"
    uv run --env-file .env python pipeline/run_systems.py --stage second       # gpt-5.5
    uv run --env-file .env python pipeline/run_systems.py --stage frames       # S3, S5
    uv run --env-file .env python pipeline/run_systems.py --stage repeats      # decision 5
    uv run python pipeline/run_systems.py --stage eval        # results/full/report.md

Second frame verifier (a separately reported check, never written into results/full/runs):

    uv run --env-file .env python pipeline/run_systems.py --stage frames \
        --frame-verifier gemini:gemini-3.8-flash@low --frame-media-resolution medium \
        --frame-stage frame_verifier_gemini38 [--canary-only]

runs S5 only, with the same prompt, frames, labels and pool, and writes
results/full/<frame-stage>/{runs/<set>/S5_<model>-<resolution>.jsonl, spend.json,
manifest.json, frame_rate_<model>-<resolution>.json}, under its own cap (--frame-cap).

S5 on S0's top 10 (W4, run_plan_w4_s0pool_20261106.md), the 722 test queries:

    uv run --env-file .env python pipeline/run_systems.py --stage frames --frame-pool s0top10 \
        --run-suffix _s0top10 --sources-file results/full/select/sources100.json [--project-only]

writes results/full/runs/<set>/{S5_s0top10.jsonl, cost_s0top10.json} and results/full/
s0pool_frames/{spend.json, manifest.json}; --query-ids / --limit-queries narrow it (dry run).

S4 on long descriptions (W5, run_plan_w5_longdesc_20261106.md): S4L = rerank_v1 on
DESC evidence from --desc-file (describe_v4d, written by pipeline/long_descriptions.py), gpt-6-sol@low,
on the top 10 of --frame-pool s4at50 (S4@50) or s0top10 (S0), the 722 test queries:

    uv run --env-file .env python pipeline/run_systems.py --stage s4long --frame-pool s4at50 \
        --desc-file descriptions_fps05_v4d.jsonl --sources-file results/full/select/sources100.json

writes runs/<set>/{S4L_s4pool<suffix>.jsonl, cost_S4L_s4pool<suffix>.json} (S4L_s0pool for
s0top10; --run-dir puts them elsewhere, e.g. a dry run) and results/full/s4long/{spend.json,
manifest.json} under its own cap; --query-ids / --limit-queries narrow it.

`--sources-file PATH` (a draw written by pipeline/draw_test_set.py, {"sources": [...]}) keeps
only the queries of the listed sources after load_sets; every frozen subset of subsets.json
(frames, twin_both, nested_sample, second, repeats), the cost tables, the gate and the repeat
comparisons are then intersected with the kept queries. The paper's eval:

    uv run python pipeline/run_systems.py --stage eval --sources-file results/full/select/sources100.json

`--smoke` points everything at the smoke run (configs/mc_smoke_v2.yaml, data/mc/smoke_v2, the
499-video index; outputs to results/full_smoke/) with the smoke pilot's depths (S2 at top 20,
S4 at top 50 as in depth50_dev.py, second verifier at top 20), the pilot's nested frame subset
and twin_both_min 0, so every call is a cache hit; caps are --smoke-cap (default 200k). The
eval stage then also writes smoke_check.md against the pilot, deep-pool and twin-both summaries.

Outputs: results/full/runs/<set>/<system>.jsonl (+ splits.json, cost.json), results/full/<stage>/
{spend.json, manifest.json, ...}, results/full/select/subsets.json (frozen), results/full/report.md
and results/full/eval/summary.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import re
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from src.cache import shared_cache
from src.constraints import parse_json
from src.datasets.mc import (
    McCorpus,
    corpus_descriptions,
    description_index,
    index_options,
    index_settings,
    index_store,
    load_captions,
    read_jsonl,
)
from src.embedders import make_embedder
from src.llm import CachedVision, GeminiVision, make_llm, media_level, parse_spec
from src.manifest import write_manifest
from src.systems.base import merge_reranked
from src.systems.decompose import SPLIT_PROMPT, SPLIT_SPEC
from src.systems.rerank import (
    DESC_EVIDENCE,
    RERANK_PROMPT,
    DescriptionScore,
    FrameScore,
    HolisticReranker,
    fill,
    parse_score,
)
from src.systems.verify import (
    DESC_PROMPT,
    FRAMES_PROMPT,
    VERIFIER_SPEC,
    ConstraintVerifier,
    DescriptionCheck,
    FrameCheck,
    LabelledOpenAIVision,
    aggregate_grades,
    rank_by_score,
)
from src.types import Cost, Query

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """A smoke script as a module (read-only reuse of its functions)."""
    spec = importlib.util.spec_from_file_location(
        f"_full_{name}", ROOT / "dev" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


P2 = load_script("run_systems_dev")
PV = P2.PV
SV = load_script("second_verifier_dev")
TS = load_script("trigger_dev")

FULL_CONFIG, SMOKE_CONFIG = Path("configs/mc_full.yaml"), Path("configs/mc_smoke_v2.yaml")
SETS = ("natural", "twin")  # natural = the nested set (queries.jsonl)
VRIPT = P2.VRIPT
BINS = ("lt30s", "30to60s", "1to2min", "2to4min")
N_BOOT = 10_000
RATE_REF = 389.3885  # results/smoke_v2/pilot/frame_rate.json
DEV = Path("results/smoke_v2")  # dev outputs (S6 threshold)
DEV_DATA = Path("data/mc/smoke_v2")
# stages that share one cap in configs/mc_full.yaml `caps`
CAP_OF = {
    "retrieve": "desc_systems",
    "desc": "desc_systems",
    "second": "second_verifier",
    "frames": "frames",
    "repeats": "repeats",
}
STAGES_OF_CAP = {c: [s for s, k in CAP_OF.items() if k == c] for c in set(CAP_OF.values())}


# ---- settings ---------------------------------------------------------------------------------


def settings(args) -> SimpleNamespace:
    """Paths, depths and caps for the full run, or for the smoke reproduction (--smoke)."""
    full = yaml.safe_load(FULL_CONFIG.read_text())
    if not args.smoke:
        cfg = yaml.safe_load(Path(args.config or FULL_CONFIG).read_text())
        d = cfg["rerank_depth"]
        S = SimpleNamespace(
            cfg=cfg,
            out=Path(args.out or cfg["gate_out"]),
            s2=d,
            s4=d,
            second=d,
            rescore=(10,),
            twin_both_min=cfg["twin_both_min"],
            nested_ids=None,
            caps=cfg["caps"],
            total_cap=cfg["total_cap"],
        )
    else:
        cfg = {**full, **yaml.safe_load(Path(args.config or SMOKE_CONFIG).read_text())}
        # the smoke pilot's depths, so every description call is a cache hit
        S = SimpleNamespace(
            cfg=cfg,
            out=Path(args.out or "results/full_smoke"),
            s2=20,
            s4=50,
            second=20,
            rescore=(10, 20),
            twin_both_min=0,
            nested_ids=json.loads((DEV / "pilot/frame_subset.json").read_text())["natural"],
            caps={k: args.smoke_cap for k in full["caps"]},
            total_cap=10 * args.smoke_cap,
        )
    assert cfg["verifier"] == VERIFIER_SPEC and cfg["splitter"] == SPLIT_SPEC
    S.smoke, S.runs, S.top = args.smoke, S.out / "runs", cfg["frame_top"]
    S.keep = None  # --sources-file: {set: kept query ids} (restrict_sets)
    S.keep_sources = None
    S.depth = cfg["depth"]
    S.judge_dir = Path(cfg["judge"]["out_dir"])
    S.n_boot = args.n_boot
    S.S2, S.S4 = f"S2@{S.s2}", f"S4@{S.s4}"
    S.desc_file = getattr(args, "desc_file", None)  # s4long: the long descriptions (W5)
    S.desc_prompt = getattr(args, "desc_prompt", None)
    return S


def load_context(S, need_index: bool = True) -> dict:
    """As pilot_v2.load_index (sources with an empty description left out), but the index must
    already exist (pipeline/build_index.py builds and pays for it)."""
    cfg = S.cfg
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    options = index_options(cfg)
    corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], sample["videos"])
    descriptions = corpus_descriptions(
        corpus, Path(cfg["describe_prompt"]).name, options["filename"]
    )
    missing = sorted(d for d in corpus.doc_ids if not descriptions.get(d))

    def settings_for(doc_ids):
        return index_settings(
            doc_ids,
            cfg["embedder"],
            cfg["describer"],
            cfg["describe_prompt"],
            cfg["n_frames"],
            options["frame_policy"],
            options["labels"],
        )

    # The smoke index leaves out videos with an empty description; the full index
    # (pipeline/build_index.py) keeps them as zero vectors. Use whichever store exists.
    settings_ = settings_for(corpus.doc_ids)
    store = index_store(cfg["results_dir"], settings_)
    if missing and not store.exists():
        corpus = McCorpus.pool(
            cfg["data_root"], cfg["datasets"], [d for d in corpus.doc_ids if d not in set(missing)]
        )
        settings_ = settings_for(corpus.doc_ids)
        store = index_store(cfg["results_dir"], settings_)
    ctx = {
        "cfg": cfg,
        "corpus": corpus,
        "descriptions": descriptions,
        "missing": missing,
        "settings": settings_,
        "store": store,
        "sample": sample,
        "options": options,
    }
    if getattr(S, "desc_file", None):  # --desc-file: a second description set (S4L), never DESC
        if S.desc_file == options["filename"]:
            sys.exit(f"--desc-file {S.desc_file} is the index's description file")
        ctx["descriptions_long"] = corpus_descriptions(
            corpus, Path(S.desc_prompt).name, S.desc_file
        )
    if need_index:
        if not (store / "DESC.npy").exists():
            sys.exit(f"no index at {store}: build it with pipeline/build_index.py first")
        embedder = make_embedder(cfg["embedder"], cache=shared_cache("embed", cfg["results_dir"]))
        ctx["embedder"] = embedder
        ctx["index"] = description_index(corpus, descriptions, embedder, store)
    return ctx


def load_sets(S, ctx) -> dict[str, list[dict]]:
    return P2.load_sets(Path(S.cfg["out_dir"]), ctx["missing"])


def restrict_sets(S, sets: dict, path: Path | str) -> dict:
    """--sources-file: the query sets restricted to the drawn sources (every query of a drawn
    source). Sets S.keep = {set: kept query ids} and S.keep_sources, which read_subsets, the
    cost tables, the gate and the repeat comparisons honour."""
    draw = json.loads(Path(path).read_text())
    keep = set(draw["sources"])
    out = {n: [r for r in rs if r["source"] in keep] for n, rs in sets.items()}
    S.keep = {n: {r["query_id"] for r in rs} for n, rs in out.items()}
    S.keep_sources = keep
    S.sources_file = str(path)
    print(
        f"[sources-file] {len(keep)} sources (seed {draw.get('seed')}): queries "
        f"{ {n: len(rs) for n, rs in out.items()} } of { {n: len(rs) for n, rs in sets.items()} }"
    )
    return out


def restrict_ids(obj, keep: dict):
    """subsets.json with every per-set id list intersected with keep[set] (order kept)."""
    if isinstance(obj, dict):
        return {
            k: (
                [q for q in v if q in keep[k]]
                if k in keep and isinstance(v, list)
                else restrict_ids(v, keep)
            )
            for k, v in obj.items()
        }
    return obj


def rargs(S, args, out: Path | None = None) -> argparse.Namespace:
    """The args object pilot_v2's stage functions read (out, workers, inner, n)."""
    return argparse.Namespace(
        out=str(out or S.runs),
        workers=args.workers,
        inner=args.inner,
        n=S.s4,
        n_frames_systems=S.top,
    )


# ---- spend --------------------------------------------------------------------------------------


def ledger_total(path: Path) -> float:
    """Input tokens in a spend.json: this script's ledgers ({"entries": [...]}) and the index
    (pipeline/build_index.py) and build (pipeline/build_queries.py) ledgers, which use other formats."""
    if not path.exists():
        return 0.0
    data = json.loads(path.read_text())
    if "entries" in data:
        return float(sum(e["tokens_in"] for e in data["entries"]))
    if "totals" in data:  # build ledger: per-step totals, each with an overall tokens_in
        return float(sum(step.get("tokens_in", 0) for step in data["totals"].values()))
    return float(sum(step.get("tokens_in", 0) for step in data.values() if isinstance(step, dict)))


def all_spend(S) -> float:
    """Input tokens spent by every full-run stage, the query build included."""
    ledgers = list(S.out.glob("*/spend.json"))
    if getattr(S, "cfg", None):
        ledgers.append(Path(S.cfg["out_dir"]) / "spend.json")
    return sum(ledger_total(p) for p in ledgers)


class Ledger(P2.Ledger):
    """pilot_v2.Ledger at <out>/<stage>/spend.json, checked against the stage's cap (shared with
    the other stages of that cap) and against total_cap over every <out>/*/spend.json."""

    def __init__(self, S, stage: str, cap_key: str | None = None):
        self.cap_key = cap_key or CAP_OF[stage]
        super().__init__(S.out / stage / "spend.json", float(S.caps[self.cap_key]))
        self.S, self.stage = S, stage

    def check(self, stage: str, projected: float, *_, **__) -> None:
        shared = sum(
            ledger_total(self.S.out / s / "spend.json")
            for s in STAGES_OF_CAP.get(self.cap_key, [self.stage])
            if s != self.stage
        )
        everything = all_spend(self.S)
        after = self.total + shared + projected
        print(
            f"[{stage}] projection: {projected:,.0f} new input tokens; cap {self.cap_key} "
            f"{self.cap:,.0f}: spent {self.total + shared:,.0f}, after {after:,.0f}; all stages "
            f"{everything:,.0f} + projection of total cap {float(self.S.total_cap):,.0f}",
            flush=True,
        )
        if after > self.cap:
            sys.exit(f"[{stage}] stop: the {self.cap_key} cap would be passed")
        if everything + projected > float(self.S.total_cap):
            sys.exit(f"[{stage}] stop: the total cap would be passed")


def versions(paths) -> Cost:
    """Summed recorded cost of the ranking lines in paths (model_version = returned versions)."""
    total = Cost()
    for p in paths:
        if Path(p).exists():
            for line in read_jsonl(p):
                total += Cost(**line["cost"])
    return total


def manifest(S, stage: str, models: dict, cost: Cost, paths, extra: dict | None = None) -> None:
    recorded = versions(paths)
    write_manifest(
        S.out / stage,
        models=models,
        seeds={"sample_seed": 1, "nested_sample_seed": 1, "second_subset_seed": 0},
        cost=Cost(**{**cost.to_dict(), "model_version": recorded.model_version}),
        extra={
            "stage": stage,
            "smoke": S.smoke,
            "recorded_cost": recorded.to_dict(),
            **(extra or {}),
        },
    )


def passes(run, errors: list, n: int, wait: float, label: str) -> int:
    """run() up to n times while calls failed (failed calls are not cached, so a pass pays only
    for them); waits `wait` seconds between passes (rate limits). Returns errors left."""
    for i in range(n):
        errors.clear()
        run()
        if not errors:
            return 0
        print(f"[{label}] pass {i + 1}: {len(errors)} failed calls, e.g. {errors[0][:160]}")
        if i + 1 < n:
            time.sleep(wait * (i + 1))
    return len(errors)


class Watch:
    """Wraps a text model (outside its cache) and logs answers that do not parse."""

    def __init__(self, inner, ok, log: list):
        self.inner, self.ok, self.log = inner, ok, log

    def complete(self, prompt: str):
        text, cost = self.inner.complete(prompt)
        if text and not self.ok(text):
            self.log.append(text[:200])
        return text, cost


def s2_parses(text: str) -> bool:
    data = parse_json(text)
    return isinstance(data, dict) and ("constraints" in data or "verdicts" in data)


def s4_parses(text: str) -> bool:
    return parse_score(text) >= 0


class Reseeded:
    """Wraps an uncached model so its cache key carries params {"repeat": seed}
    (src.llm.cache_payload): fresh calls for a run-to-run repeat, same request."""

    def __init__(self, inner, seed: int):
        self.inner, self.params = inner, {"repeat": seed}

    def __getattr__(self, name):
        return getattr(self.__dict__["inner"], name)

    def complete(self, prompt: str):
        return self.inner.complete(prompt)

    def describe(self, prompt: str, images: list[bytes], labels=None):
        return self.inner.describe(prompt, images, labels=labels)


# ---- rankings -----------------------------------------------------------------------------------


def read_ranking(path: Path) -> dict:
    return {line["query_id"]: [(d, s) for d, s in line["ranking"]] for line in read_jsonl(path)}


def rescore(line: dict, s0, top: int):
    """S2/S4 on S0's top `top` from the saved verdicts / scores (as pilot_verify.read_rankings)."""
    first = line["trace"]["first_stage"][:top]
    if "verdicts" in line["trace"]:
        scores = [sum(a == "yes" for a in line["trace"]["verdicts"][d]) for d in first]
    else:
        scores = [line["trace"]["scores"][d] for d in first]
    return merge_reranked(s0, rank_by_score([(d, 0.0) for d in first], scores))


def rescore_graded(line: dict, first: list, top: int, how: str = "mean") -> list:
    """A graded verifier (S2g / S3g, Tier 1) on the first `top` candidates, rescored from the
    saved per-constraint grades (as pipeline/run_graded_systems.py rescore_graded)."""
    head = line["trace"]["first_stage"][:top]
    scores = [aggregate_grades(line["trace"]["grades"][d], how) for d in head]
    return merge_reranked(first, rank_by_score([(d, 0.0) for d in head], scores))


def load_runs(folder: Path, rescore_at=(10,)) -> tuple[dict, dict]:
    """({system: {qid: ranking}}, {system: {qid: line}}) of folder/S*.jsonl, plus S2@r / S4@r
    rescored from S2@n / S4@n (n > r) for r in rescore_at, and S2g@n / S2g@r rescored from the
    saved grades (mean) when S2g@n.jsonl (Tier 1) is there."""
    ranks, lines = {}, {}
    for path in sorted(folder.glob("S*.jsonl")):
        lines[path.stem] = {line["query_id"]: line for line in read_jsonl(path)}
        ranks[path.stem] = {
            q: [(d, s) for d, s in ln["ranking"]] for q, ln in lines[path.stem].items()
        }
    for stem in list(ranks):
        m = re.fullmatch(r"(S2|S4|S2g)@(\d+)", stem)
        if not m or "S0" not in ranks:
            continue
        if m[1] == "S2g":
            ranks[stem] = {
                q: rescore_graded(ln, ranks["S0"][q], int(m[2])) for q, ln in lines[stem].items()
            }
        for r in rescore_at:
            if r < int(m[2]):
                ranks[f"{m[1]}@{r}"] = {
                    q: (rescore_graded if m[1] == "S2g" else rescore)(ln, ranks["S0"][q], r)
                    for q, ln in lines[stem].items()
                }
    return ranks, lines


def cost_table(folder: Path, keep: set | None) -> dict:
    """pilot_v2.cost_table restricted to the kept queries (--sources-file)."""
    if keep is None:
        return P2.cost_table(folder)
    table = {}
    for path in sorted(folder.glob("S*.jsonl")):
        costs = [Cost(**ln["cost"]) for ln in read_jsonl(path) if ln["query_id"] in keep]
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


def both_in_top(ranks: dict, rows: list[dict], top: int) -> list[str]:
    """Twin queries with both the source and the twin in the ranking's top `top`."""
    out = []
    for r in rows:
        head = {d for d, _ in ranks.get(r["query_id"], [])[:top]}
        if r["source"] in head and r.get("twin") in head:
            out.append(r["query_id"])
    return sorted(out)


def nested_sample(rows: list[dict], n: int, seed: int = 1) -> list[str]:
    """n Vript queries balanced over (length bin, k): an even share per cell (cells short of
    queries give the rest to the others, round robin in cell order), each cell drawn by a seeded
    shuffle of its sorted ids. Uses the query rows only, not any ranking."""
    cells: dict[tuple, list[str]] = {}
    for r in rows:
        if r["dataset"] in VRIPT:
            cells.setdefault((r["length_bin"], r["k"]), []).append(r["query_id"])
    keys = sorted(cells, key=lambda c: (BINS.index(c[0]) if c[0] in BINS else 9, c[1]))
    for c in keys:
        cells[c] = sorted(cells[c])
        random.Random(f"{seed}|{c[0]}|k{c[1]}").shuffle(cells[c])
    alloc = {c: 0 for c in keys}
    n = min(n, sum(len(v) for v in cells.values()))
    while sum(alloc.values()) < n:
        for c in keys:
            if alloc[c] < len(cells[c]) and sum(alloc.values()) < n:
                alloc[c] += 1
    return sorted(q for c in keys for q in cells[c][: alloc[c]])


# ---- stage: retrieve (S0, the full-corpus gate, split, S1) ------------------------------------


def recall_table(rows: list[dict], s0: dict, ns=(10, 20, 50), n_boot: int = N_BOOT) -> dict:
    """Source recall@n in S0 per group (all, k, dataset, Vript bin), cluster-bootstrap CIs."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        if r["query_id"] not in s0:
            continue
        keys = ["all", f"k={r['k']}", f"dataset={r['dataset']}"]
        if r["dataset"] in VRIPT:
            keys.append(f"Vript {r['length_bin']}")
        for g in keys:
            groups.setdefault(g, []).append(r)
    out = {}
    for g, rs in groups.items():
        out[g] = {}
        for n in ns:
            hits = [float(r["source"] in {d for d, _ in s0[r["query_id"]][:n]}) for r in rs]
            out[g][f"recall@{n}"] = P2.share_ci(hits, [r["source"] for r in rs], n_boot)
    order = ("all", "k=", "dataset=", "Vript ")
    return dict(
        sorted(
            out.items(),
            key=lambda kv: (
                next(i for i, p in enumerate(order) if kv[0].startswith(p)),
                BINS.index(kv[0][6:]) if kv[0].startswith("Vript ") and kv[0][6:] in BINS else 0,
                kv[0],
            ),
        )
    )


def gate_report(gate: dict) -> list[str]:
    lines = ["| set | group | n | recall@10 | recall@20 | recall@50 |", "|---|---|---|---|---|---|"]
    for name, table in gate["recall"].items():
        for g, cell in table.items():
            cells = " | ".join(P2.rate_cell(cell[f"recall@{n}"]) for n in (10, 20, 50))
            lines.append(f"| {name} | {g} | {cell['recall@10']['n']} | {cells} |")
    return lines


def stage_retrieve(S, ctx, sets, args) -> None:
    ledger = Ledger(S, "retrieve")
    depth = min(S.depth, len(ctx["corpus"].doc_ids))
    embed_before = ctx["embedder"].cost.tokens_in
    s0s = {}
    for name, rs in sets.items():
        by_id = {r["query_id"]: r for r in rs}
        s0 = P2.s0_records(ctx["index"], rs, depth)
        PV.write_rankings(S.runs / name / "S0.jsonl", "S0", s0, by_id)
        s0s[name] = read_ranking(S.runs / name / "S0.jsonl")
        print(f"{name}: S0 on {len(rs)} queries, depth {depth}")
    counts = {
        n: {str(k): sum(r["k"] == k for r in rs) for k in (1, 2, 3, 4)} for n, rs in sets.items()
    }
    gate = {
        "corpus": len(ctx["corpus"].doc_ids),
        "depth": depth,
        "queries": {n: len(rs) for n, rs in sets.items()},
        "queries_by_k": counts,
        "checks": {
            "queries_per_set_ge_900": {n: len(rs) >= 900 for n, rs in sets.items()},
            "k4_per_set_ge_150": {n: counts[n]["4"] >= 150 for n in sets},
        },
        "twin_both_s0_top10": len(both_in_top(s0s["twin"], sets["twin"], S.top)),
        "recall": {n: recall_table(rs, s0s[n], n_boot=S.n_boot) for n, rs in sets.items()},
    }
    (S.out / "retrieve").mkdir(parents=True, exist_ok=True)
    (S.out / "retrieve" / "gate.json").write_text(json.dumps(gate, indent=1))
    md = [
        "# Full-corpus gate: source recall in S0's top 10 / 20 / 50",
        "",
        (
            f"Corpus {gate['corpus']}, depth {depth}. Queries {gate['queries']}, by k {counts}. "
            f"Checks: {gate['checks']}. Twin-both (S0 top 10): {gate['twin_both_s0_top10']} "
            f"(twin_both_min {S.twin_both_min})."
        ),
        "",
        *gate_report(gate),
        "",
    ]
    (S.out / "retrieve" / "gate.md").write_text("\n".join(md))
    print("\n".join(md[:3]))
    embed = Cost(tokens_in=ctx["embedder"].cost.tokens_in - embed_before)
    if args.s0_only:
        ledger.add("S0", embed, 0, {"embedding_tokens": embed.tokens_in})
        return
    models = P2.Models(S.cfg)
    template = SPLIT_PROMPT.read_text()
    rows = [r for rs in sets.values() for r in rs]
    new = [
        r
        for r in rows
        if not models.split_cached.is_cached(template.replace("{query}", r["query"]))
    ]
    proj = sum(P2.text_tokens(template.replace("{query}", r["query"])) for r in new)
    print(f"splits to pay: {len(new)} of {len(rows)}")
    ledger.check("retrieve", proj)
    if args.project_only:
        return
    if (
        new or args.canary_always
    ) and not args.no_canary:  # canary: 10 splits, none may fall back to the whole query
        new = new or rows
        found = P2.run_splits(models, new[: args.canary], args.workers)
        bad = [
            q
            for r in new[: args.canary]
            for q in [r["query_id"]]
            if found[q][0] == [" ".join(r["query"].split())] and r["k"] > 1
        ]
        if models.errors or bad:
            sys.exit(f"[retrieve] canary failed: errors {models.errors[:2]}, unparsed {bad}")
        print(f"[retrieve] canary ok ({len(found)} splits)")

    def run():
        for name, rs in sets.items():
            by_id = {r["query_id"]: r for r in rs}
            splits = P2.run_splits(models, rs, args.workers * args.inner)
            subs = [
                Query(f"{r['query_id']}#c{i}", c, r["source"])
                for r in rs
                for i, c in enumerate(splits[r["query_id"]][0], 1)
            ]
            sub_rank, _ = P2.search_all(ctx["index"], subs, depth)
            s1 = {}
            for r in rs:
                cons, cost = splits[r["query_id"]]
                lists = [sub_rank[f"{r['query_id']}#c{i}"] for i in range(1, len(cons) + 1)]
                s1[r["query_id"]] = {
                    "ranking": P2.fuse(lists),
                    "cost": (cost + Cost(searches=len(cons))).to_dict(),
                    "trace": {"constraints": cons},
                }
            PV.write_rankings(S.runs / name / "S1.jsonl", "S1", s1, by_id)
            (S.runs / name / "splits.json").write_text(
                json.dumps({q: c for q, (c, _) in splits.items()}, indent=1)
            )

    left = passes(run, models.errors, args.passes, args.wait, "retrieve")
    paid, calls = models.paid_cost()
    embed = Cost(tokens_in=ctx["embedder"].cost.tokens_in - embed_before)
    ledger.add(
        "retrieve", paid + embed, calls, {"embedding_tokens": embed.tokens_in, "errors": left}
    )
    manifest(
        S,
        "retrieve",
        {"splitter": SPLIT_SPEC, "embedder": S.cfg["embedder"]},
        paid,
        [S.runs / n / "S1.jsonl" for n in sets],
        {"index": str(ctx["store"]), "gate": gate["checks"]},
    )


# ---- stage: select (frozen subsets) -------------------------------------------------------------


def stage_select(S, ctx, sets, args) -> None:
    out = S.out / "select"
    path = out / "subsets.json"
    if path.exists() and not args.force:
        print(
            f"{path} is frozen (use --force to redraw): {summary_of(json.loads(path.read_text()))}"
        )
        return
    s0 = {n: read_ranking(S.runs / n / "S0.jsonl") for n in sets}
    both_s0 = both_in_top(s0["twin"], sets["twin"], S.top)
    pool = "S0" if len(both_s0) >= S.twin_both_min else S.S4
    twin_both = both_s0
    if pool != "S0":  # decision 3: S3/S5 on S4@50's top 10 for both sets
        missing = [n for n in sets if not (S.runs / n / f"{pool}.jsonl").exists()]
        if missing:
            out.mkdir(parents=True, exist_ok=True)
            (out / "pending.json").write_text(
                json.dumps({"twin_both_s0": len(both_s0), "pool": pool, "needs": missing}, indent=1)
            )
            print(
                f"twin-both on S0's top {S.top}: {len(both_s0)} < {S.twin_both_min}: the frame "
                f"pool is {pool}'s top {S.top} (decision 3); run --stage desc, then select again"
            )
            return
        twin_both = both_in_top(
            read_ranking(S.runs / "twin" / f"{pool}.jsonl"), sets["twin"], S.top
        )
    drawn = nested_sample(sets["natural"], S.cfg["nested_frame_sample"], seed=1)
    nested = sorted(S.nested_ids) if S.nested_ids is not None else drawn
    second = {
        n: PV.subset_balanced(
            [r for r in rs if r["query_id"] in s0[n]], S.cfg["second_verifier_subset"], 0
        )
        for n, rs in sets.items()
    }
    s5_rep = sorted(twin_both)
    random.Random("repeat|S5|0").shuffle(s5_rep)
    sub = {
        "pool": pool,
        "top": S.top,
        "twin_both_s0": len(both_s0),
        "twin_both_min": S.twin_both_min,
        "twin_both": {"twin": twin_both},
        "nested_sample": {"natural": nested},
        "nested_sample_drawn": {"natural": drawn},
        "frames": {"natural": nested, "twin": twin_both},
        "second": second,
        "repeats": {"desc": second, "S5": {"twin": sorted(s5_rep[:100])}},
    }
    out.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sub, indent=1))
    for key in ("frames", "second"):
        (out / f"{key}_subset.json").write_text(json.dumps(sub[key], indent=1))
    (out / "pending.json").unlink(missing_ok=True)
    print(f"wrote {path}: {summary_of(sub)}")


def summary_of(sub: dict) -> dict:
    return {
        "pool": sub["pool"],
        "twin_both_s0": sub["twin_both_s0"],
        **{
            k: {n: len(v) for n, v in sub[k].items()}
            for k in ("twin_both", "nested_sample", "second")
        },
        "repeats_S5": len(sub["repeats"]["S5"]["twin"]),
    }


def read_subsets(S) -> dict:
    """The frozen subsets, intersected with the kept queries under --sources-file."""
    path = S.out / "select" / "subsets.json"
    if not path.exists():
        sys.exit(f"{path} missing: run --stage select first")
    sub = json.loads(path.read_text())
    return restrict_ids(sub, S.keep) if S.keep else sub


# ---- stage: desc (S2@50, S4@50; @10 rescored at eval) -----------------------------------------


def desc_plan(models, descriptions, S, args, log, s2_n, s4_n, splitter=None):
    desc_t, rerank_t = DESC_PROMPT.read_text(), RERANK_PROMPT.read_text()
    v2, v4 = Watch(models.verifier, s2_parses, log), Watch(models.verifier, s4_parses, log)
    plan = {}
    if s2_n:
        plan["S2"] = (
            ConstraintVerifier(
                "S2",
                splitter or models.splitter,
                DescriptionCheck(v2, desc_t, descriptions),
                args.inner,
            ),
            s2_n,
        )
    if s4_n:
        plan["S4"] = (
            HolisticReranker("S4", DescriptionScore(v4, rerank_t, descriptions), args.inner),
            s4_n,
        )
    return plan


def run_plan(models, sets, plan, qids, first, ra, suffix: str, stage: str) -> None:
    """pilot_v2.rerank_set on {S2@n<suffix>, S4@n<suffix>} for the given query ids per set."""
    for name, rs in sets.items():
        p = {f"{s}@{n}{suffix}": (rr, qids[name], n) for s, (rr, n) in plan.items()}
        P2.rerank_set(models, name, rs, p, ra, stage, first=first[name])


def stage_desc(S, ctx, sets, args) -> None:
    ledger = Ledger(S, "desc")
    for n in sets:
        if not (S.runs / n / "splits.json").exists():
            sys.exit("run --stage retrieve (splits) first")
    models = P2.Models(S.cfg)
    ra = rargs(S, args)
    proj, calls = P2.desc_projection(models, sets, ra, ctx["descriptions"], S.s2, S.s4)
    print(f"S2@{S.s2} + S4@{S.s4} calls to pay: {calls}")
    ledger.check("desc", proj)
    if args.project_only:
        return
    first = {n: read_ranking(S.runs / n / "S0.jsonl") for n in sets}
    log: list = []
    plan = desc_plan(models, ctx["descriptions"], S, args, log, S.s2, S.s4)
    if (
        calls or args.canary_always
    ) and not args.no_canary:  # canary: 10 queries, no parse failures
        ids = canary_ids(sets, args.canary)
        run_plan(
            models, sets, plan, ids, first, rargs(S, args, S.out / "desc" / "canary"), "", "canary"
        )
        if log or models.errors:
            sys.exit(f"[desc] canary failed: unparsed {log[:2]}, errors {models.errors[:2]}")
        print("[desc] canary ok")
    all_ids = {n: [r["query_id"] for r in rs] for n, rs in sets.items()}
    left = passes(
        lambda: run_plan(models, sets, plan, all_ids, first, ra, "", "desc"),
        models.errors,
        args.passes,
        args.wait,
        "desc",
    )
    paid, n = models.paid_cost()
    ledger.add("desc", paid, n, {"errors": left, "unparsed": len(log)})
    manifest(
        S,
        "desc",
        {"verifier": VERIFIER_SPEC, "splitter": SPLIT_SPEC},
        paid,
        [S.runs / s / f"{x}.jsonl" for s in sets for x in (S.S2, S.S4)],
        {"depths": {"S2": S.s2, "S4": S.s4}, "prompts": [DESC_PROMPT.name, RERANK_PROMPT.name]},
    )


def canary_ids(sets, n: int) -> dict[str, list[str]]:
    """About n query ids, split over the sets, drawn by a seeded shuffle."""
    out = {}
    for i, (name, rs) in enumerate(sets.items()):
        ids = sorted(r["query_id"] for r in rs)
        random.Random(f"canary|{name}").shuffle(ids)
        out[name] = sorted(ids[: n // len(sets) + (i < n % len(sets))])
    return out


# ---- stage: second (gpt-5.5 on the frozen subset) ----------------------------------------------


def fixed_splitter(S, sets, ids) -> dict:
    out = {}
    for name, rs in sets.items():
        splits = json.loads((S.runs / name / "splits.json").read_text())
        keep = [r for r in rs if r["query_id"] in set(ids[name])]
        out[name] = SV.FixedSplitter(SV.constraints_by_text(keep, splits))
    return out


def text_models(spec: str, S, reseed: int | None = None):
    """(models-like namespace, errors) for one text verifier, metered, optionally reseeded."""
    paid, errors = [], []
    cached = make_llm(spec, shared_cache("llm", S.cfg["results_dir"]))
    if spec.startswith("openai:"):
        cached = PV.more_retries(cached)
    inner = cached.inner if reseed is None else Reseeded(cached.inner, reseed)
    cached.inner = PV.Metered(inner, paid)
    m = SimpleNamespace(
        verifier_cached=cached,
        verifier=PV.SafeCalls(cached, errors),
        errors=errors,
        paid={"text": paid},
    )
    m.paid_cost = lambda: (sum(paid, Cost()), len(paid))
    return m


def subset_sets(sets, ids) -> dict:
    return {n: [r for r in rs if r["query_id"] in set(ids.get(n, []))] for n, rs in sets.items()}


def run_text_subset(S, ctx, sets, args, stage, m, ids, suffix, n2, n4, models_spec) -> None:
    ledger = Ledger(S, stage)
    sub = subset_sets(sets, ids)
    ra = rargs(S, args)
    proj, calls = P2.desc_projection(m, sub, ra, ctx["descriptions"], n2, n4)
    print(f"[{stage}] {sum(map(len, sub.values()))} queries, {calls} calls to pay")
    ledger.check(stage, proj)
    if args.project_only:
        return
    first = {n: read_ranking(S.runs / n / "S0.jsonl") for n in sets}
    log: list = []
    splitters = fixed_splitter(S, sets, ids)
    out = rargs(S, args, S.out / stage)
    for name in sets:
        plan = desc_plan(m, ctx["descriptions"], S, args, log, n2, n4, splitters[name])
        one = {name: sub[name]}
        if (calls or args.canary_always) and not args.no_canary and name == SETS[0]:
            can = {name: canary_ids(one, args.canary)[name]}
            run_plan(
                m, one, plan, can, first, rargs(S, args, S.out / stage / "canary"), suffix, "canary"
            )
            if log or m.errors:
                sys.exit(f"[{stage}] canary failed: unparsed {log[:2]}, errors {m.errors[:2]}")
        ids_one = {name: [r["query_id"] for r in sub[name]]}
        passes(
            lambda one=one, plan=plan, ids_one=ids_one: run_plan(
                m, one, plan, ids_one, first, out, suffix, stage
            ),
            m.errors,
            args.passes,
            args.wait,
            f"{stage} {name}",
        )
    paid, n = m.paid_cost()
    ledger.add(stage, paid, n, {"errors": len(m.errors), "unparsed": len(log)})
    manifest(
        S,
        stage,
        models_spec,
        paid,
        sorted((S.out / stage).glob("*/S*.jsonl")),
        {"depths": {"S2": n2, "S4": n4}},
    )


def stage_second(S, ctx, sets, args) -> None:
    sub = read_subsets(S)
    spec = S.cfg["second_verifier"]
    m = text_models(spec, S)
    name = spec.split(":", 1)[1].split("@")[0]
    run_text_subset(
        S,
        ctx,
        sets,
        args,
        "second",
        m,
        sub["second"],
        f"_{name}",
        S.second,
        S.second,
        {"second_verifier": spec},
    )


# ---- stage: frames (S3, S5) and repeats -------------------------------------------------------


class VisionModels(SimpleNamespace):
    def paid_cost(self):
        logs = [c for log in self.paid.values() for c in log]
        return sum(logs, Cost()), len(logs)


def vision_models(S, reseed: int | None = None):
    """The pilot's frame verifier (same cache keys as pilot_v2.Models), optionally reseeded."""
    paid, errors = [], []
    raw = LabelledOpenAIVision(VERIFIER_SPEC.split(":", 1)[1].split("@")[0], "low")
    raw.client = raw.client.with_options(max_retries=10, timeout=300)
    inner = raw if reseed is None else Reseeded(raw, reseed)
    cached = CachedVision(
        PV.Metered(inner, paid), shared_cache("vlm", S.cfg["results_dir"]), VERIFIER_SPEC
    )
    return VisionModels(
        vlm_cached=cached,
        vlm=PV.SafeCalls(cached, errors),
        errors=errors,
        paid={"sol_vision": paid},
    )


GEMINI_IMAGE_TOKENS = {None: 1100.0, "MEDIA_RESOLUTION_LOW": 264.0}  # canary 0, 2026-10-24
GEMINI_IMAGE_TOKENS_DEFAULT = {"MEDIA_RESOLUTION_MEDIUM": 560.0, "MEDIA_RESOLUTION_HIGH": 1120.0}


def frame_slug(spec: str, media_resolution: str | None) -> str:
    """'gemini:gemini-3.8-flash@low', 'medium' -> 'gemini-3.8-flash-medium' (default: unset)."""
    level = media_level(media_resolution)
    res = level.removeprefix("MEDIA_RESOLUTION_").lower() if level else "default"
    return f"{parse_spec(spec)[1]}-{res}"


def gemini_vision_models(S, spec: str, media_resolution: str | None):
    """A second frame verifier (Gemini) in the shape of vision_models: metered inside the vlm
    cache; media_resolution is sent per image and is part of the cache key when set."""
    kind, model, effort = parse_spec(spec)
    if kind != "gemini":
        raise ValueError(f"--frame-verifier supports gemini specs only, not {spec!r}")
    paid, errors = [], []
    inner = GeminiVision(model, effort, media_resolution=media_resolution)
    cached = CachedVision(PV.Metered(inner, paid), shared_cache("vlm", S.cfg["results_dir"]), spec)
    return VisionModels(
        vlm_cached=cached,
        vlm=PV.SafeCalls(cached, errors),
        errors=errors,
        paid={"frame_verifier": paid},
        spec=spec,
        media_resolution=inner.media_resolution,
    )


def frame_plan(m, frames, S, args, splitters=None, systems=("S3", "S5")) -> dict:
    out = {}
    if "S3" in systems:
        out["S3"] = lambda name: ConstraintVerifier(
            "S3", splitters[name], FrameCheck(m.vlm, FRAMES_PROMPT.read_text(), frames), args.inner
        )
    if "S5" in systems:
        out["S5"] = lambda name: HolisticReranker(
            "S5", FrameScore(m.vlm, RERANK_PROMPT.read_text(), frames), args.inner
        )
    return out


def run_frames(
    S,
    ctx,
    sets,
    args,
    stage,
    m,
    qids,
    systems,
    suffix="",
    out_dir=None,
    verifier=None,
    pool=None,
    cost_file="cost.json",
    rate_ref=RATE_REF,
) -> None:
    """verifier (second frame verifier): {"spec", "media_resolution", "rate", "rate_file"}; the
    canary then reports its measured rate (no tolerance check) and writes rate_file, and
    --canary-only stops after it. pool: the first-stage run to rerank (default the frozen
    subsets.json pool); cost_file: the per-set cost file (default cost.json); rate_ref: the
    canary's reference tokens per image (default the smoke rate)."""
    sets = {n: rs for n, rs in sets.items() if n in qids}  # repeats run S5 on the twin set only
    ledger = Ledger(S, stage)
    sub = read_subsets(S)
    pool = pool or sub["pool"]
    first = {n: read_ranking(S.runs / n / f"{pool}.jsonl") for n in sets}
    frames = P2.frame_source(ctx)
    ra = rargs(S, args)
    rate_file = S.out / "frames" / "frame_rate.json"
    per_image = (
        json.loads(rate_file.read_text())["tokens_per_image"] if rate_file.exists() else RATE_REF
    )
    if verifier:
        rate_file = verifier["rate_file"]
        per_image = (
            json.loads(rate_file.read_text())["tokens_per_image"]
            if rate_file.exists()
            else verifier["rate"]
        )
    proj, calls, images = P2.frame_projection(m, sets, qids, ra, frames, per_image, first, systems)
    print(
        f"[{stage}] pool {pool} top {S.top}: {sum(map(len, qids.values()))} queries, {calls} "
        f"calls, {images:,} images to pay at {per_image:.1f} tokens per image"
    )
    try:
        ledger.check(stage, proj)
    except SystemExit as stop:
        if not (verifier and args.canary_only):
            raise
        print(f"{stop} (--canary-only: the canary still runs, the main run would not)")
    if args.project_only:
        return
    splitters = fixed_splitter(S, sets, qids) if "S3" in systems else None
    plan = frame_plan(m, frames, S, args, splitters, systems)
    out = rargs(S, args, out_dir)

    def run(ids, ra_):
        for name, rs in sets.items():
            p = {f"{s}{suffix}": (make(name), ids.get(name, []), S.top) for s, make in plan.items()}
            P2.rerank_set(
                m, name, rs, p, ra_, stage, merge=True, first=first[name], cost_file=cost_file
            )

    if (
        calls or args.canary_always
    ) and not args.no_canary:  # canary: 2 queries, tokens per image at most 2% above the reference
        can = {
            n: v
            for n, v in canary_ids(
                {n: [r for r in sets[n] if r["query_id"] in set(qids.get(n, []))] for n in sets},
                args.canary_frames,
            ).items()
        }
        canary_dir = S.out / stage / "canary"
        run(can, rargs(S, args, canary_dir))
        # recorded cost of the canary's calls (paid now, or as recorded when they were paid)
        vision = sum(
            (
                Cost(**ln["cost"])
                for n in sets
                for s in plan
                for ln in read_jsonl(canary_dir / n / f"{s}{suffix}.jsonl")
                if ln["query_id"] in set(can.get(n, []))
            ),
            Cost(),
        )
        if vision.images:
            n_calls = vision.llm_calls
            # text part: the canary's own prompts (chars / 3.6) + 12 per frame label, as projected
            never = SimpleNamespace(vlm_cached=SimpleNamespace(is_cached=lambda *a: False))
            text, _, _ = P2.frame_projection(never, sets, can, ra, frames, 0.0, first, systems)
            rate = (vision.tokens_in - text) / vision.images
            print(
                f"[{stage}] canary: {vision.tokens_in:,} in, {vision.tokens_out:,} out for "
                f"{vision.images:,} images in {n_calls} calls -> {rate:.1f} per image (smoke "
                f"{rate_ref:.1f}), {vision.tokens_out / n_calls:.1f} out per call; versions "
                f"{vision.model_version}"
            )
            if verifier:
                if m.errors:
                    sys.exit(f"[{stage}] canary failed: errors {m.errors[:2]}")
                rate_file.parent.mkdir(parents=True, exist_ok=True)
                rate_file.write_text(
                    json.dumps(
                        {
                            "tokens_per_image": rate,
                            "calls": n_calls,
                            "images": vision.images,
                            "tokens_in": vision.tokens_in,
                            "tokens_out": vision.tokens_out,
                            "out_per_call": vision.tokens_out / n_calls,
                            "model_version": vision.model_version,
                            "verifier": verifier["spec"],
                            "media_resolution": verifier["media_resolution"],
                        },
                        indent=1,
                    )
                )
            elif (rate - rate_ref) / rate_ref > args.rate_tol or m.errors:  # cost guard: high only
                sys.exit(f"[{stage}] canary failed: rate {rate:.1f}, errors {m.errors[:2]}")
            if stage == "frames":
                rate_file.parent.mkdir(parents=True, exist_ok=True)
                rate_file.write_text(json.dumps({"tokens_per_image": rate, "calls": n_calls}))
    if verifier and args.canary_only:
        paid, n = m.paid_cost()
        ledger.add(
            stage,
            paid,
            n,
            {"canary_only": True, "verifier": verifier["spec"], **verifier_meta(verifier)},
        )
        print(f"[{stage}] --canary-only: stopping before the main run")
        return
    left = passes(lambda: run(qids, out), m.errors, args.passes, args.wait, stage)
    paid, n = m.paid_cost()
    ledger.add(
        stage,
        paid,
        n,
        {
            "errors": left,
            "pool": pool,
            "queries": {k: len(v) for k, v in qids.items()},
            "projected": round(proj),
            **({"verifier": verifier["spec"], **verifier_meta(verifier)} if verifier else {}),
        },
    )
    manifest(
        S,
        stage,
        {"verifier": verifier["spec"] if verifier else VERIFIER_SPEC},
        paid,
        [Path(out.out) / n_ / f"{s}{suffix}.jsonl" for n_ in sets for s in systems],
        {
            "pool": pool,
            "top": S.top,
            "prompts": [FRAMES_PROMPT.name, RERANK_PROMPT.name],
            **(verifier_meta(verifier) if verifier else {}),
        },
    )


def verifier_meta(verifier: dict) -> dict:
    return {
        "frame_verifier": verifier["spec"],
        "media_resolution": verifier["media_resolution"] or "API default (unset)",
        "systems_label": verifier["label"],
        "note": "second frame verifier, reported separately; not a substitute for gpt-6-sol",
    }


def stage_frames(S, ctx, sets, args) -> None:
    sub = read_subsets(S)
    if args.frame_pool == "s0top10":
        if args.frame_verifier:
            sys.exit("--frame-pool s0top10 runs the gpt-6-sol S5 only, not --frame-verifier")
        stage_frames_s0pool(S, ctx, sets, args)
        return
    if args.frame_verifier:
        stage_frame_verifier(S, ctx, sets, args, sub)
        return
    run_frames(S, ctx, sets, args, "frames", vision_models(S), sub["frames"], ("S3", "S5"))


S0POOL_STAGE = "s0pool_frames"
S0POOL_CAP = 35e6  # run_plan_w4_s0pool_20261106.md: S5 only, $71 / 35M input


def stage_frames_s0pool(S, ctx, sets, args) -> None:
    """W4: S5 on S0's top 10 (a pool chosen by neither system) for every query of the sets
    (with --sources-file results/full/select/sources100.json: the 722 test queries), with the
    model, prompt, frames and labels of S5, so the pairs already scored on S4@50's top 10 are
    cache hits. Writes runs/<set>/S5<run-suffix>.jsonl and runs/<set>/cost<run-suffix>.json
    (never S5.jsonl or cost.json); ledger <out>/s0pool_frames/spend.json under its own cap.
    --query-ids FILE ({set: [query ids]}) and --limit-queries N (first N per set) narrow the
    queries (dry run); the canary reference is the recorded frames rate."""
    if not args.run_suffix:
        sys.exit("--frame-pool s0top10 needs --run-suffix (S5.jsonl is the S4@50 pool's run)")
    stage = S0POOL_STAGE
    CAP_OF[stage] = stage  # its own cap
    STAGES_OF_CAP[stage] = [stage]
    S.caps = {**S.caps, stage: S0POOL_CAP}
    qids = {n: [r["query_id"] for r in rs] for n, rs in sets.items()}
    if args.query_ids:
        keep = json.loads(Path(args.query_ids).read_text())
        qids = {n: [q for q in v if q in set(keep.get(n, []))] for n, v in qids.items()}
    if args.limit_queries is not None:
        qids = {n: v[: args.limit_queries] for n, v in qids.items()}
    rate_file = S.out / "frames" / "frame_rate.json"
    rate_ref = json.loads(rate_file.read_text())["tokens_per_image"]
    run_frames(
        S,
        ctx,
        sets,
        args,
        stage,
        vision_models(S),
        qids,
        ("S5",),
        args.run_suffix,
        pool="S0",
        cost_file=f"cost{args.run_suffix}.json",
        rate_ref=rate_ref,
    )


S4LONG_STAGE = "s4long"
S4LONG_CAP = 5.5e6  # run_plan_w5_longdesc_20261106.md section 7: S4L 5.5M input
S4LONG_POOLS = {"s4at50": ("S4@50", "s4pool"), "s0top10": ("S0", "s0pool")}


def s4long_prompts(rows, qids, first, descriptions, top) -> dict:
    """{(query id, doc): S4 prompt} on the top `top` of `first`, as DescriptionScore builds it."""
    t = RERANK_PROMPT.read_text()
    by_id = {r["query_id"]: r for r in rows}
    out = {}
    for q in qids:
        for d, _ in first[q][:top]:
            desc = descriptions.get(d, "") or "(no description)"
            out[(q, d)] = fill(t, by_id[q]["query"], DESC_EVIDENCE.replace("{description}", desc))
    return out


def stage_s4long(S, ctx, sets, args) -> None:
    """W5: S4 (rerank_v1, DESC evidence, gpt-6-sol@low) on the long descriptions of --desc-file,
    on the top 10 of --frame-pool (s4at50: S4@50, s0top10: S0) for every query of the sets.
    Only the description text differs from S4, and the text cache key is the whole prompt, so
    long and short calls cannot collide. Writes <run-dir>/<set>/S4L_<pool><run-suffix>.jsonl
    and cost_S4L_<pool><run-suffix>.json (run-dir default runs/); ledger
    <out>/s4long/spend.json under its own cap."""
    if "descriptions_long" not in ctx:
        sys.exit("--stage s4long needs --desc-file (the long descriptions file)")
    long_ = ctx["descriptions_long"]
    # Videos whose long description fell back to the index description (marked "fallback" in the
    # --desc-file rows): their S4L prompt equals the S4 one by design, so it reuses S4's cached score.
    fallback = {
        f"{f.parent.name}/{r['video_id']}"
        for f in Path("data/mc").glob(f"*/{S.desc_file}")
        for r in map(json.loads, f.read_text().splitlines())
        if r.get("fallback")
    }
    print(f"[s4long] fallback descriptions: {sorted(fallback)}")
    pool, tag = S4LONG_POOLS[args.frame_pool]
    name = f"S4L_{tag}{args.run_suffix}"
    run_dir = Path(args.run_dir) if args.run_dir else S.runs
    stage = S4LONG_STAGE
    CAP_OF[stage] = stage
    STAGES_OF_CAP[stage] = [stage]
    S.caps = {**S.caps, stage: S4LONG_CAP}
    ledger = Ledger(S, stage)
    qids = {n: [r["query_id"] for r in rs] for n, rs in sets.items()}
    if args.query_ids:
        keep = json.loads(Path(args.query_ids).read_text())
        qids = {n: [q for q in v if q in set(keep.get(n, []))] for n, v in qids.items()}
    if args.limit_queries is not None:
        qids = {n: v[: args.limit_queries] for n, v in qids.items()}
    first = {n: read_ranking(S.runs / n / f"{pool}.jsonl") for n in sets}
    m = text_models(VERIFIER_SPEC, S)
    tokens, calls, pairs, missing, short_new = 0.0, 0, 0, set(), 0
    for n, rs in sets.items():
        long_p = s4long_prompts(rs, qids[n], first[n], long_, S.top)
        short_p = s4long_prompts(rs, qids[n], first[n], ctx["descriptions"], S.top)
        for key, p in long_p.items():
            pairs += 1
            if not long_.get(key[1]):
                missing.add(key[1])
            if p == short_p[key] and key[1] not in fallback:
                sys.exit(f"[{stage}] long and short prompts coincide for {key}")
            if not m.verifier_cached.is_cached(p):
                tokens += P2.text_tokens(p)
                calls += 1
        short_new += sum(not m.verifier_cached.is_cached(p) for p in short_p.values())
    print(
        f"[{stage}] {name}: pool {pool} top {S.top}, {sum(map(len, qids.values()))} queries, "
        f"{pairs} pairs, {calls} calls to pay; candidates without a long description: "
        f"{len(missing)}; S4 short on the same pairs: {short_new} uncached"
    )
    ledger.check(stage, tokens)
    if args.project_only:
        return
    log: list = []
    scorer = DescriptionScore(Watch(m.verifier, s4_parses, log), RERANK_PROMPT.read_text(), long_)
    reranker = HolisticReranker("S4L", scorer, args.inner)
    ra = rargs(S, args, run_dir)

    def run():
        for n, rs in sets.items():
            P2.rerank_set(
                m,
                n,
                rs,
                {name: (reranker, qids[n], S.top)},
                ra,
                stage,
                merge=True,
                first=first[n],
                cost_file=f"cost_{name}.json",
            )

    left = passes(run, m.errors, args.passes, args.wait, stage)
    paid, n_paid = m.paid_cost()
    ledger.add(
        stage,
        paid,
        n_paid,
        {
            "errors": left,
            "unparsed": len(log),
            "pool": pool,
            "run": name,
            "run_dir": str(run_dir),
            "desc_file": S.desc_file,
            "queries": {k: len(v) for k, v in qids.items()},
            "projected": round(tokens),
            "missing_long": len(missing),
        },
    )
    manifest(
        S,
        stage,
        {"verifier": VERIFIER_SPEC},
        paid,
        [run_dir / n_ / f"{name}.jsonl" for n_ in sets],
        {"pool": pool, "top": S.top, "prompts": [RERANK_PROMPT.name], "desc_file": S.desc_file},
    )


def stage_frame_verifier(S, ctx, sets, args, sub) -> None:
    """S5 with a second (Gemini) frame verifier on the frame subsets: same prompt, frames,
    labels and pool; outputs under results/full/<frame-stage>/runs, never results/full/runs."""
    spec, media = args.frame_verifier, media_level(args.frame_media_resolution)
    slug = frame_slug(spec, media)
    stage = args.frame_stage or f"frame_verifier_{slug}"
    if stage in CAP_OF or stage == "runs":
        sys.exit(f"--frame-stage {stage!r} would share a main stage's folder")
    CAP_OF[stage] = stage  # its own cap
    STAGES_OF_CAP[stage] = [stage]
    S.caps = {**S.caps, stage: args.frame_cap}
    rate = args.rate_ref or GEMINI_IMAGE_TOKENS.get(media) or GEMINI_IMAGE_TOKENS_DEFAULT[media]
    verifier = {
        "spec": spec,
        "media_resolution": media,
        "rate": rate,
        "rate_file": S.out / stage / f"frame_rate_{slug}.json",
        "label": f"S5_{slug}",
    }
    run_frames(
        S,
        ctx,
        sets,
        args,
        stage,
        gemini_vision_models(S, spec, media),
        sub["frames"],
        ("S5",),
        f"_{slug}",
        S.out / stage / "runs",
        verifier,
    )


def stage_repeats(S, ctx, sets, args) -> None:
    """Decision 5: S2@50 / S4@50 on the second-verifier subset and S5 on 100 twin-both queries,
    with {"repeat": seed} in the cache key (fresh calls), same splits and frames."""
    sub = read_subsets(S)
    seed = args.repeat_seed
    m = text_models(VERIFIER_SPEC, S, reseed=seed)
    run_text_subset(
        S,
        ctx,
        sets,
        args,
        "repeats",
        m,
        sub["repeats"]["desc"],
        f"_r{seed}",
        S.s2,
        S.s4,
        {"verifier": VERIFIER_SPEC, "repeat_seed": seed},
    )
    run_frames(
        S,
        ctx,
        sets,
        args,
        "repeats",
        vision_models(S, reseed=seed),
        sub["repeats"]["S5"],
        ("S5",),
        f"_r{seed}",
        S.out / "repeats",
    )


# ---- stage: eval --------------------------------------------------------------------------------


def group_rows(rows, key):
    if key == "length_bin":
        return [r for r in rows if r["dataset"] in VRIPT]
    return rows


GROUP_KEYS = (None, "k", "dataset", "length_bin")
METRICS = ("ndcg@10", "recall@1", "recall@5", "mrr")


def means(sc: dict, rows: list[dict]) -> dict:
    return {str(key): PV.mean_table(sc, group_rows(rows, key), key) for key in GROUP_KEYS}


def paired_cells(sa, sb, rows, n_boot) -> dict:
    return {
        m: {
            str(key): PV.paired(sa, sb, group_rows(rows, key), m, key, n_boot) for key in GROUP_KEYS
        }
        for m in ("ndcg@10", "recall@1")
    }


def twin_above(rank: dict, rows: list[dict], n_boot: int) -> dict:
    def f(r):
        if r["query_id"] not in rank:
            return None
        pos = {d: i for i, (d, _) in enumerate(rank[r["query_id"]])}
        big = len(pos) + 1
        return float(pos.get(r["twin"], big) < pos.get(r["source"], big))

    return P2.by_k(rows, f, n_boot)


def source_meta(S) -> dict:
    """Per source: duration, scenes, caption words (Vript) from constraints.jsonl / captions."""
    meta = {}
    caps = {d: load_captions(Path(S.cfg["data_root"]) / d) for d in VRIPT}
    for c in read_jsonl(Path(S.cfg["out_dir"]) / "constraints.jsonl"):
        cap = caps.get(c["dataset"], {}).get(c["video_id"])
        meta[c["source"]] = {
            "duration": float(c["duration_s"]),
            "scenes": c.get("n_scenes") or 1,
            "words": len(cap.caption.split()) if cap else None,
            "dataset": c["dataset"],
        }
    return meta


def ols_boot(y, X, clusters, n_boot: int, seed: int = 0) -> dict:
    """OLS with cluster-bootstrap SEs / 95% CIs (resampling clusters, normal equations summed per
    cluster) and VIFs of the non-intercept columns."""
    y, X = np.asarray(y, float), np.asarray(X, float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    keys = sorted(set(clusters))
    pos = {c: i for i, c in enumerate(keys)}
    idx = np.array([pos[c] for c in clusters])
    p, C = X.shape[1], len(keys)
    xtx = np.zeros((C, p, p))
    xty = np.zeros((C, p))
    np.add.at(xtx, idx, X[:, :, None] * X[:, None, :])
    np.add.at(xty, idx, X * y[:, None])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, C, size=(n_boot, C))
    W = np.zeros((n_boot, C))
    np.add.at(W, (np.repeat(np.arange(n_boot), C), picks.ravel()), 1.0)
    A = np.einsum("bc,cij->bij", W, xtx)
    b = W @ xty
    betas = (np.linalg.pinv(A) @ b[:, :, None])[:, :, 0]
    lo, hi = np.percentile(betas, [2.5, 97.5], axis=0)
    vif = []
    for j in range(1, p):
        others = np.delete(X, j, axis=1)
        fit = others @ np.linalg.lstsq(others, X[:, j], rcond=None)[0]
        ss = ((X[:, j] - X[:, j].mean()) ** 2).sum()
        r2 = 1 - ((X[:, j] - fit) ** 2).sum() / ss if ss > 0 else 1.0
        vif.append(float(1 / (1 - r2)) if r2 < 1 else math.inf)
    return {
        "n": len(y),
        "clusters": C,
        "beta": beta.tolist(),
        "se": betas.std(0).tolist(),
        "ci_low": lo.tolist(),
        "ci_high": hi.tolist(),
        "vif": [None, *vif],
    }


# Plan section 3 also lists caption words per minute. It is left out: Vript writes about the same
# number of words per scene (median 126-128, p10-p90 116-132), so words per minute is scenes per
# minute times a constant (r = .99 over the 200 Vript sources, VIF about 100 with both in).
# Scenes per minute is kept: it is a property of the video (how often the content changes), while
# words per minute adds only the annotators' per-scene caption length. rate_collinearity() reports
# the diagnosis in summary.json.
COVARIATES = (
    "const",
    "log2_duration",
    "k",
    "twin",
    "vript_long",
    "scenes_per_min",
)


def covariates(r: dict, meta: dict) -> list[float] | None:
    m = meta.get(r["source"])
    if m is None or m["words"] is None:  # same sample as before: Vript sources with a caption
        return None
    minutes = m["duration"] / 60
    return [
        1.0,
        math.log2(m["duration"]),
        float(r["k"]),
        float(r["set"] == "twin"),
        float(r["dataset"] == "vript_long"),
        m["scenes"] / minutes,
    ]


def rate_collinearity(meta: dict, keep: set | None = None) -> dict:
    """Why caption words per minute is not a covariate: its correlation with scenes per minute
    over the Vript sources (with keep, S.keep_sources, only the drawn ones), overall and per
    dataset, the VIF of each when both are in (1 / (1 - r^2)), and the spread of caption words
    per scene."""
    out = {}
    groups = {
        "all": [
            m
            for s, m in meta.items()
            if m["dataset"] in VRIPT and m["words"] and (keep is None or s in keep)
        ]
    }
    groups |= {d: [m for m in groups["all"] if m["dataset"] == d] for d in VRIPT}
    for g, ms in groups.items():
        if len(ms) < 3:
            continue
        wpm = np.array([m["words"] / (m["duration"] / 60) for m in ms])
        spm = np.array([m["scenes"] / (m["duration"] / 60) for m in ms])
        wps = np.array([m["words"] / m["scenes"] for m in ms])
        r = float(np.corrcoef(wpm, spm)[0, 1])
        out[g] = {
            "sources": len(ms),
            "corr_words_scenes_per_min": r,
            "vif_both_in": float(1.0 / (1.0 - r * r)) if abs(r) < 1 else None,
            "words_per_scene_p10_p50_p90": np.percentile(wps, [10, 50, 90]).round(1).tolist(),
        }
    return {
        "dropped": "caption_words_per_min",
        "kept": "scenes_per_min",
        "restricted_to_kept_sources": keep is not None,
        **out,
    }


def regressions(S, ctx, sets, ranks, scores, subsets, base, lines, n_boot) -> dict:
    """Plan section 3: gain on log2 duration and covariates within Vript, both sets pooled;
    secondary model with the mean log2 duration of the 10 pool candidates; mechanism model on
    the per-query description drop."""
    meta = source_meta(S)
    videos = ctx["corpus"].videos
    frame_ids = {n: set(v) for n, v in subsets["frames"].items()}
    specs = [(f"{a} - S0", a, "S0", None) for a in (S.S2, S.S4, "S2@10", "S4@10")]
    specs += [
        ("S3 - " + base["S3"], "S3", base["S3"], "frames"),
        ("S5 - " + base["S5"], "S5", base["S5"], "frames"),
    ]
    # Tier 1: S5 and S3g on every query (on = "all": no frame-subset restriction)
    for a in ("S5", "S3g"):
        if all(a in ranks[n] and covers(ranks[n][a], sets[n]) for n in sets):
            specs.append((f"{a} - {base['S5']} (all queries)", a, base["S5"], "all"))
    out = {}
    for label, a, b, on in specs:
        Y, X, X2, cl = [], [], [], []
        for name, rs in sets.items():
            sa, sb = scores[name].get(a), scores[name].get(b)
            if sa is None or sb is None:
                continue
            for r in rs:
                q = r["query_id"]
                if r["dataset"] not in VRIPT or q not in sa or q not in sb:
                    continue
                if on == "frames" and q not in frame_ids.get(name, set()):
                    continue
                x = covariates({**r, "set": name}, meta)
                if x is None:
                    continue
                pool = ranks[name][
                    "S0" if on is None or subsets["pool"] == "S0" else subsets["pool"]
                ][q][:10]
                cand = np.mean(
                    [math.log2(max(videos[d].duration_s, 0.1)) for d, _ in pool if d in videos]
                )
                Y.append(sa[q]["ndcg@10"] - sb[q]["ndcg@10"])
                X.append(x)
                X2.append([*x, float(cand)])
                cl.append(r["source"])
        if len(Y) > len(COVARIATES) + 2:
            out[label] = {
                "main": {**ols_boot(Y, X, cl, n_boot), "terms": list(COVARIATES)},
                "with_candidate_duration": {
                    **ols_boot(Y, X2, cl, n_boot),
                    "terms": [*COVARIATES, "mean_log2_candidate_duration"],
                },
            }
    drop = description_drop(S, sets, lines)
    Y, X, cl = [], [], []
    for name, rs in sets.items():
        for r in rs:
            d = drop.get((name, r["query_id"]))
            x = covariates({**r, "set": name}, meta) if r["dataset"] in VRIPT else None
            if d is not None and x is not None:
                Y.append(1 - d["yes"])
                X.append(x)
                cl.append(r["source"])
    if len(Y) > len(COVARIATES) + 2:
        out["mechanism: description drop"] = {
            "main": {**ols_boot(Y, X, cl, n_boot), "terms": list(COVARIATES)}
        }
    return out


def description_drop(S, sets, lines) -> dict:
    """{(set, qid): yes / no / unclear shares of the source's split constraints in its own S2
    verdicts} for every query whose source S2 checked, i.e. whose source is in S0's top n (n = the
    S2 depth, 50 on the full run); other queries have no verdict for the source and are left out.
    The constraints are the splitter's (S1-S3) constraints of the query, not the mined facts, and
    the verdict is the verifier's answer on the source's own description. The description drop is
    1 - yes: "unclear" and "no" both count as not confirmed."""
    out = {}
    for name in sets:
        for q, line in lines[name].get(S.S2, {}).items():
            v = line["trace"]["verdicts"].get(line["source"])
            if v:
                out[(name, q)] = {
                    a: sum(x == a for x in v) / len(v) for a in ("yes", "no", "unclear")
                } | {"constraints": len(v)}
    return out


def drop_groups(r: dict) -> list[str]:
    keys = ["all", f"k={r['k']}"]
    if r["dataset"] in VRIPT:
        keys.append(f"Vript {r['length_bin']}")
        keys.append(f"Vript {r['length_bin']} k={r['k']}")
    if r["k"] == 1:
        keys.append(f"k=1 type={r['facts'][0]['type']}")
    return keys


def drop_tables(S, sets, lines, n_boot) -> dict:
    """Per set and group: mean over queries of the per-query description drop (1 - yes share of
    the source's split constraints), cluster-bootstrap CI over sources. Queries whose source is
    outside S0's top 50 are not counted (see description_drop)."""
    drop = description_drop(S, sets, lines)
    out = {}
    for name, rs in sets.items():
        groups: dict[str, list] = {}
        for r in rs:
            d = drop.get((name, r["query_id"]))
            if d is None:
                continue
            for g in drop_groups(r):
                groups.setdefault(g, []).append((1 - d["yes"], r["source"]))
        out[name] = {
            g: P2.share_ci([v for v, _ in vs], [c for _, c in vs], n_boot)
            for g, vs in sorted(groups.items())
        }
    return out


def drop_verdicts(S, sets, lines) -> dict:
    """Per set and group: queries in the group, queries whose source S2 checked (source in S0's
    top 50), split constraints checked, and the pooled yes / no / unclear shares over those
    constraints (the smoke run's §4.4 figure is the pooled yes share, over the top 20)."""
    drop = description_drop(S, sets, lines)
    out = {}
    for name, rs in sets.items():
        groups: dict[str, dict] = {}
        for r in rs:
            d = drop.get((name, r["query_id"]))
            for g in drop_groups(r):
                c = groups.setdefault(
                    g,
                    {"queries": 0, "checked": 0, "constraints": 0, "yes": 0, "no": 0, "unclear": 0},
                )
                c["queries"] += 1
                if d is None:
                    continue
                c["checked"] += 1
                c["constraints"] += d["constraints"]
                for a in ("yes", "no", "unclear"):
                    c[a] += round(d[a] * d["constraints"])
        for c in groups.values():
            n = c["constraints"] or 1
            for a in ("yes", "no", "unclear"):
                c[a] = c[a] / n
        out[name] = dict(sorted(groups.items()))
    return out


def visibility_table(S, n_boot) -> dict:
    """Index visibility: share of confirmed grounded facts Gemini sees in the source's own 0.5
    fps index frames (build_v2 index check), per Vript bin, dataset and fact type."""
    path = Path(S.cfg["out_dir"]) / "constraints.jsonl"
    groups: dict[str, list] = {}
    for c in read_jsonl(path):
        if S.keep_sources is not None and c["source"] not in S.keep_sources:
            continue
        for g in c.get("grounding", []):
            if not g.get("confirmed") or g.get("index") is None:
                continue
            keys = ["all", f"dataset={c['dataset']}", f"type={g.get('type')}"]
            if c["dataset"] in VRIPT:
                keys.append(f"Vript {c['length_bin']}")
            for k in keys:
                groups.setdefault(k, []).append(
                    (0.0 if g.get("index_dropped") else 1.0, c["source"])
                )
    return {
        g: P2.share_ci([v for v, _ in vs], [c for _, c in vs], n_boot)
        for g, vs in sorted(groups.items())
    }


def s6_rows(ranks, lines, qrels, ids_by_set, s4: str, s5: str = "S5") -> list[dict]:
    """Rows for trigger_sim's sweep: final = S5 when triggered, else S4 (s4 = S4@50)."""
    rows = []
    for name, ids in ids_by_set.items():
        for q in ids:
            if q not in ranks[name].get(s4, {}) or q not in ranks[name].get(s5, {}):
                continue
            a4, a5, g = ranks[name][s4][q], ranks[name][s5][q], qrels[name][q]
            m4, m5 = PV.query_metrics(a4, g), PV.query_metrics(a5, g)
            c4 = float(lines[name][s4][q]["cost"]["tokens_in"])
            rows.append(
                {
                    "query_id": q,
                    "set": name,
                    "source": lines[name][s4][q]["source"],
                    "q4": m4["ndcg@10"],
                    "q5": m5["ndcg@10"],
                    "r4": m4["recall@1"],
                    "r5": m5["recall@1"],
                    "s4_margin": TS.margin(a4),
                    "c4": c4,
                    "c2": 0.0,
                    "c_base": c4,
                    "c5": float(lines[name][s5][q]["cost"]["tokens_in"]),
                }
            )
    return rows


def dev_s6_rows() -> list[dict]:
    """Dev = the smoke outputs: S4@50 (deep pool) and S5 (pilot) on the 120 twin-both queries,
    expanded labels (data/mc/smoke_v2/judge)."""
    ids = json.loads((DEV / "pilot/frame_subset_twin_both.json").read_text())
    rows = {r["query_id"]: r for r in read_jsonl(DEV_DATA / "queries_twin.jsonl")}
    keep = [rows[q] for q in ids["twin"]]
    qrels = {"twin": PV.load_qrels(DEV_DATA / "judge/qrels_expanded.jsonl", keep)}
    lines = {
        "twin": {
            "S4@50": {ln["query_id"]: ln for ln in read_jsonl(DEV / "deep_pool/twin/S4@50.jsonl")},
            "S5": {ln["query_id"]: ln for ln in read_jsonl(DEV / "pilot/twin/S5.jsonl")},
        }
    }
    ranks = {
        "twin": {
            s: {q: [(d, x) for d, x in ln["ranking"]] for q, ln in v.items()}
            for s, v in lines["twin"].items()
        }
    }
    return s6_rows(ranks, lines, qrels, ids, "S4@50")


def s6_eval(ranks, lines, qrels, subsets, s4, extra: dict | None = None) -> dict:
    """extra: {label: {set: ids}} further test groups (Tier 1: every query once S5 covers a
    set); "all queries" pools the "all nested" and "all near-miss" groups."""
    feat, direction = "s4_margin", "le"
    dev = dev_s6_rows()
    dev_pts = TS.sweep(dev, feat, direction)
    thresholds = {str(f): (TS.operating_point(dev_pts, f) or {}).get("t") for f in TS.FRACS}
    tests = {
        "twin-both": s6_rows(ranks, lines, qrels, subsets["twin_both"], s4),
        "nested sample": s6_rows(ranks, lines, qrels, subsets["nested_sample"], s4),
    }
    tests["pooled"] = tests["twin-both"] + tests["nested sample"]
    for label, ids in (extra or {}).items():
        tests[label] = s6_rows(ranks, lines, qrels, ids, s4)
    if "all nested" in tests and "all near-miss" in tests:
        tests["all queries"] = tests["all nested"] + tests["all near-miss"]
    out = {
        "dev": {"n": len(dev), "thresholds": thresholds, "auc": TS.auc_summary(dev_pts)},
        "test": {},
    }
    for name, rows in tests.items():
        if len(rows) < 5:
            continue
        cell = {
            "n": len(rows),
            "auc": TS.auc_summary(TS.sweep(rows, feat, direction)),
            "S4": float(np.mean([r["q4"] for r in rows])),
            "S5": float(np.mean([r["q5"] for r in rows])),
            "dev_threshold": {},
            "crossfit": {},
        }
        for f, t in thresholds.items():
            if t is not None:
                cell["dev_threshold"][f] = {
                    "t": t,
                    **TS.apply_t(rows, feat, t),
                    **TS.point_cis(rows, feat, direction, t),
                }
            cell["crossfit"][f] = TS.crossfit(rows, feat, direction, float(f))
        out["test"][name] = cell
    return out


def compare_runs(a: dict, b: dict, ids, qrels, n_boot) -> dict:
    """A run and its repeat on the same queries: nDCG@10 means, paired difference, top-1 agreement."""
    ids = [q for q in ids if q in a and q in b]
    rows = [{"query_id": q, "source": q.split("#")[0], "k": 0} for q in ids]
    sa = PV.score_system({q: a[q] for q in ids}, qrels)
    sb = PV.score_system({q: b[q] for q in ids}, qrels)
    if not ids:
        return {"n": 0}
    return {
        "n": len(ids),
        "run": float(np.mean([sa[q]["ndcg@10"] for q in ids])),
        "repeat": float(np.mean([sb[q]["ndcg@10"] for q in ids])),
        "diff": PV.paired(sb, sa, rows, "ndcg@10", None, n_boot).get("all"),
        "top1_same": float(np.mean([a[q][0][0] == b[q][0][0] for q in ids])),
    }


def covers(rank: dict, rows: list[dict]) -> bool:
    """Every query of rows has a ranking."""
    return bool(rows) and all(r["query_id"] in rank for r in rows)


def test_gate(S, ctx, sets, ranks) -> dict:
    """The retrieve stage's gate recomputed on the kept queries (--sources-file)."""
    s0s = {n: ranks[n]["S0"] for n in sets}
    counts = {
        n: {str(k): sum(r["k"] == k for r in rs) for k in (1, 2, 3, 4)} for n, rs in sets.items()
    }
    return {
        "corpus": len(ctx["corpus"].doc_ids),
        "depth": max(len(v) for v in s0s["twin"].values()),
        "sources": len(S.keep_sources),
        "sources_file": S.sources_file,
        "queries": {n: len(rs) for n, rs in sets.items()},
        "queries_by_k": counts,
        "checks": None,
        "twin_both_s0_top10": len(both_in_top(s0s["twin"], sets["twin"], S.top)),
        "recall": {n: recall_table(rs, s0s[n], n_boot=S.n_boot) for n, rs in sets.items()},
    }


def stage_eval(S, ctx, sets, args) -> dict:
    sub = read_subsets(S)
    pool = sub["pool"]
    ranks, lines = {}, {}
    for name in sets:
        ranks[name], lines[name] = load_runs(S.runs / name, S.rescore)
        if S.keep:  # the runs may cover more queries than the test set
            ranks[name] = {
                s: {q: v for q, v in r.items() if q in S.keep[name]} for s, r in ranks[name].items()
            }
            lines[name] = {
                s: {q: v for q, v in r.items() if q in S.keep[name]} for s, r in lines[name].items()
            }
    base = {"S3": "S2@10", "S5": "S4@10"} if pool == "S0" else {"S3": pool, "S5": pool}
    base["S3g"] = base["S5"]  # Tier 1 graded frame verifier: same pool as S5
    graded = [f"S2g@{S.s2}", *(f"S2g@{r}" for r in S.rescore)]  # Tier 1, present when run
    all_sys = ["S0", "S1", S.S2, S.S4, *(f"S{x}@{r}" for r in S.rescore for x in (2, 4)), *graded]
    frame_sys = ["S0", "S2@10", "S4@10", S.S4, "S3", "S5", "S3g"]
    all_pairs = [
        ("S1", "S0"),
        (S.S2, "S0"),
        (S.S4, "S0"),
        (S.S4, S.S2),
        ("S2@10", "S0"),
        ("S4@10", "S0"),
        (S.S4, "S4@10"),
        ("S4@10", "S2@10"),
        (S.S4, graded[0]),
        (graded[0], S.S2),
        (graded[0], "S0"),
        ("S4@10", "S2g@10"),
    ]
    frame_pairs = list(
        dict.fromkeys(
            [
                ("S3", base["S3"]),
                ("S5", base["S5"]),
                ("S3", "S5"),
                ("S5", S.S4),
                ("S3", "S0"),
                ("S5", "S0"),
                ("S3g", "S5"),
                ("S3g", base["S5"]),
                ("S3g", "S3"),
                ("S3g", "S0"),
            ]
        )
    )
    subsets = {
        "twin-both": ("twin", sub["twin_both"]["twin"]),
        "nested sample": ("natural", sub["nested_sample"]["natural"]),
    }
    # Tier 1: the frame systems on the whole set once S5 covers every query of it
    whole = {"natural": "all nested", "twin": "all near-miss"}
    for name, label in whole.items():
        if "S5" in ranks[name] and covers(ranks[name]["S5"], sets[name]):
            subsets[label] = (name, [r["query_id"] for r in sets[name]])
    qrels_path = S.judge_dir / "qrels_expanded.jsonl"
    kinds = {"strict": None, **({"expanded": qrels_path} if qrels_path.exists() else {})}
    primary = "expanded" if "expanded" in kinds else "strict"
    res: dict = {
        "pool": pool,
        "frame_base": base,
        "primary_labels": primary,
        "sources_file": getattr(S, "sources_file", None),
        "sources": len(S.keep_sources) if S.keep_sources else None,
        "labels": {},
    }
    qrels_by = {}
    scores_by = {}
    for kind, path in kinds.items():
        qrels = {n: PV.load_qrels(path, rs) for n, rs in sets.items()}
        qrels_by[kind] = qrels
        scores = {n: {s: PV.score_system(r, qrels[n]) for s, r in ranks[n].items()} for n in sets}
        scores_by[kind] = scores
        cell: dict = {"means": {}, "paired": {}, "subset_means": {}, "subset_paired": {}}
        for name, rs in sets.items():
            sc = scores[name]
            cell["means"][name] = {s: means(sc[s], rs) for s in all_sys if s in sc}
            cell["paired"][name] = {
                f"{a} - {b}": paired_cells(sc[a], sc[b], rs, S.n_boot)
                for a, b in all_pairs
                if a in sc and b in sc
            }
        for label, (name, ids) in subsets.items():
            sc, rs = scores[name], [r for r in sets[name] if r["query_id"] in set(ids)]
            cell["subset_means"][label] = {s: means(sc[s], rs) for s in frame_sys if s in sc}
            cell["subset_paired"][label] = {
                f"{a} - {b}": paired_cells(sc[a], sc[b], rs, S.n_boot)
                for a, b in frame_pairs
                if a in sc and b in sc
            }
        res["labels"][kind] = cell
    twin_rows = sets["twin"]
    both = [r for r in twin_rows if r["query_id"] in set(sub["twin_both"]["twin"])]
    res["twin_above_source"] = {
        "twin-both": {
            s: twin_above(ranks["twin"][s], both, S.n_boot) for s in frame_sys if s in ranks["twin"]
        },
        "all twin": {
            s: twin_above(ranks["twin"][s], twin_rows, S.n_boot)
            for s in dict.fromkeys([*all_sys, *frame_sys])
            if s in ranks["twin"]
        },
    }
    gate_path = S.out / "retrieve" / "gate.json"
    if S.keep:
        res["gate"] = test_gate(S, ctx, sets, ranks)
    else:
        res["gate"] = json.loads(gate_path.read_text()) if gate_path.exists() else None
    res["costs"] = {n: cost_table(S.runs / n, S.keep[n] if S.keep else None) for n in sets}
    if "S5" in ranks["twin"] and S.S4 in ranks["twin"]:
        extra = {
            label: {name: ids} for label, (name, ids) in subsets.items() if label in whole.values()
        }
        res["s6"] = s6_eval(ranks, lines, qrels_by[primary], sub, S.S4, extra)
    res["regression"] = {
        kind: regressions(S, ctx, sets, ranks, scores_by[kind], sub, base, lines, S.n_boot)
        for kind in kinds
    }
    res["description_drop"] = drop_tables(S, sets, lines, S.n_boot)
    res["description_drop_verdicts"] = drop_verdicts(S, sets, lines)
    res["regression_rate_collinearity"] = rate_collinearity(source_meta(S), S.keep_sources)
    res["index_visibility"] = visibility_table(S, S.n_boot)
    res["second"] = second_eval(S, sets, sub, lines, qrels_by[primary])
    res["repeats"] = repeats_eval(S, sets, sub, ranks, qrels_by[primary])
    stats = S.judge_dir / "judge_stats.json"
    res["judge"] = json.loads(stats.read_text()) if stats.exists() else None
    res["spend"] = {  # this script's ledgers; index and build spend are in their own reports
        p.parent.name: data["entries"]
        for p in sorted(S.out.glob("*/spend.json"))
        if "entries" in (data := json.loads(p.read_text()))
    }
    (S.out / "eval").mkdir(parents=True, exist_ok=True)
    (S.out / "eval" / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (S.out / "report.md").write_text(report(S, res, sets, all_sys, frame_sys))
    write_manifest(
        S.out / "eval", models={}, extra={"stage": "eval", "smoke": S.smoke, "n_boot": S.n_boot}
    )
    print(f"wrote {S.out}/report.md and {S.out}/eval/summary.json")
    if S.smoke:
        smoke_check(S, res)
    return res


def second_eval(S, sets, sub, lines, qrels) -> dict:
    out = {}
    name_ = S.cfg["second_verifier"].split(":", 1)[1].split("@")[0]
    for name in sets:
        folder = S.out / "second" / name
        if not folder.exists():
            continue
        r2, l2 = load_runs(folder)
        ids = sub["second"][name]
        rows = [r for r in sets[name] if r["query_id"] in set(ids)]
        a, b = f"{S.S4[:2]}@{S.second}_{name_}", f"S2@{S.second}_{name_}"
        if a not in r2 or b not in r2:
            continue
        keep = {r["query_id"] for r in rows}  # the subset's queries (test set: its part of them)
        r2 = {s: {q: v for q, v in r.items() if q in keep} for s, r in r2.items()}
        sa, sb = PV.score_system(r2[a], qrels[name]), PV.score_system(r2[b], qrels[name])
        sol2 = lines[name].get(f"S2@{S.second}") or lines[name].get(S.S2, {})
        out[name] = {
            "S4 - S2": paired_cells(sa, sb, rows, S.n_boot),
            "verdicts": SV.verdict_agreement(
                SV.verdict_items({q: ln["trace"] for q, ln in sol2.items() if q in set(ids)}),
                SV.verdict_items({q: ln["trace"] for q, ln in l2[b].items()}),
            ),
        }
    return out


def repeats_eval(S, sets, sub, ranks, qrels) -> dict:
    out = {}
    for name in sets:
        folder = S.out / "repeats" / name
        if not folder.exists():
            continue
        rr, _ = load_runs(folder)
        for s, r in rr.items():
            orig = s.rsplit("_r", 1)[0]
            ids = sorted(q for q in r if S.keep is None or q in S.keep[name])
            if orig in ranks[name]:
                out[f"{name} {s}"] = compare_runs(ranks[name][orig], r, ids, qrels[name], S.n_boot)
    return out


# ---- report -------------------------------------------------------------------------------------


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def ci(d: dict | None) -> str:
    return "-" if not d else f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def grid(means_: dict, systems, key: str, groups, metric="ndcg@10") -> list[str]:
    lines = ["| system | " + " | ".join(map(str, groups)) + " |", "|---" * (len(groups) + 1) + "|"]
    for s in systems:
        if s not in means_:
            continue
        cells = [
            f3(means_[s][key].get(g if key != "None" else "all", {}).get(metric)) for g in groups
        ]
        lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def k_grid(means_: dict, systems, metric="ndcg@10") -> list[str]:
    lines = ["| system | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
    for s in systems:
        if s in means_:
            cells = [f3(means_[s]["None"].get("all", {}).get(metric))]
            cells += [f3(means_[s]["k"].get(k, {}).get(metric)) for k in (1, 2, 3, 4)]
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def paired_lines(table: dict, on: str) -> list[str]:
    out = []
    for pair, by_m in table.items():
        for key in ("None", "k", "dataset", "length_bin"):
            for g, d in by_m["ndcg@10"][key].items():
                label = g if key != "length_bin" else f"Vript {g}"
                out.append(
                    f"| {pair} | {on} | {label} | {d['n']} | {ci(d)} | {ci(by_m['recall@1'][key].get(g))} |"
                )
    return out


def report(S, res, sets, all_sys, frame_sys) -> str:
    L = ["# Full run: results (generated by pipeline/run_systems.py --stage eval)", ""]
    L += [
        (
            f"- Mode: {'SMOKE reproduction (results/full_smoke)' if S.smoke else 'full corpus'}. "
            f"S2 / S4 on S0's top {S.s2} / {S.s4} (verifier {VERIFIER_SPEC}, splitter {SPLIT_SPEC}); "
            f"S2@10 / S4@10 rescored from the same calls. S3 / S5 on {res['pool']}'s top {S.top} "
            f"(own 0.5 fps index frames, labelled, no cap); frame comparisons against {res['frame_base']}."
        ),
        (
            f"- Labels: strict (source only) and expanded (two-model frame judge, {S.judge_dir}); "
            f"primary {res['primary_labels']} (decision 2). Twin-above-source uses ranks only."
        ),
        f"- CIs: 95%, paired cluster bootstrap over source videos, {S.n_boot:,} resamples.",
        *(
            [
                (
                    f"- Test set: {res['sources']} sources from {res['sources_file']}; every "
                    "subset, cost table and repeat comparison is restricted to their queries. "
                    "S2g@50 / S2g@10 (graded per-constraint verifier on descriptions, mean grade) "
                    "and S3g (on frames) are the Tier 1 systems; 'all nested' / 'all near-miss' "
                    "are the frame systems on every query of the set."
                )
            ]
            if res.get("sources_file")
            else []
        ),
        "",
    ]
    if res.get("gate"):
        g = res["gate"]
        L += [
            "## Full-corpus gate (S0 source recall)",
            "",
            (
                f"Corpus {g['corpus']}, queries {g['queries']}, by k {g['queries_by_k']}; checks {g['checks']}; "
                f"twin-both on S0's top 10: {g['twin_both_s0_top10']}"
                + (f"; sources {g['sources']}." if g.get("sources") else ".")
            ),
            "",
            *gate_report(g),
            "",
        ]
    for kind, cell in res["labels"].items():
        L += [f"## {kind} labels", ""]
        for name in sets:
            for metric in ("ndcg@10", "recall@1"):
                L += [
                    f"### {name}: {metric} by k (all queries)",
                    "",
                    *k_grid(cell["means"][name], all_sys, metric),
                ]
            L += [
                f"### {name}: nDCG@10 by Vript bin",
                "",
                *grid(cell["means"][name], all_sys, "length_bin", BINS),
            ]
            L += [
                f"### {name}: nDCG@10 by dataset",
                "",
                *grid(
                    cell["means"][name],
                    all_sys,
                    "dataset",
                    sorted(cell["means"][name]["S0"]["dataset"]),
                ),
            ]
        for label, m in cell["subset_means"].items():
            for metric in ("ndcg@10", "recall@1"):
                L += [f"### {label}: {metric} by k", "", *k_grid(m, frame_sys, metric)]
            L += [f"### {label}: nDCG@10 by Vript bin", "", *grid(m, frame_sys, "length_bin", BINS)]
        L += [
            f"### Paired differences ({kind})",
            "",
            "| pair | on | group | n | nDCG@10 | R@1 |",
            "|---|---|---|---|---|---|",
        ]
        for name in sets:
            L += paired_lines(cell["paired"][name], name)
        for label, t in cell["subset_paired"].items():
            L += paired_lines(t, label)
        L.append("")
    L += [
        "## Twin ranked above the source (rate [95% CI])",
        "",
        "| on | system | all | k=1 | k=2 | k=3 | k=4 |",
        "|---|---|---|---|---|---|---|",
    ]
    for on, table in res["twin_above_source"].items():
        for s, t in table.items():
            L.append(
                f"| {on} | {s} | "
                + " | ".join(P2.rate_cell(t.get(g, {})) for g in ("all", "1", "2", "3", "4"))
                + " |"
            )
    L.append("")
    if res.get("s6"):
        s6 = res["s6"]
        L += [
            "## S6 trigger (S4 top-1 minus top-2 margin; offline)",
            "",
            (
                f"Dev (smoke twin-both, S4@50 + S5, expanded labels): n {s6['dev']['n']}, thresholds "
                f"{s6['dev']['thresholds']}, AUC {s6['dev']['auc']}."
            ),
            "",
            "| test | n | S4 | S5 | AUC (share) | target | t (dev) | share | tokens/query | nDCG@10 | kept | cross-fit share | cross-fit kept |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
        ]
        for name, c in s6["test"].items():
            for f in map(str, TS.FRACS):
                d, cf = c["dev_threshold"].get(f, {}), c["crossfit"].get(f, {})
                L.append(
                    f"| {name} | {c['n']} | {c['S4']:.3f} | {c['S5']:.3f} | {f3(c['auc']['auc_share'])} | {f} | "
                    f"{d.get('t', '-')} | {f3(d.get('share'))} | {d.get('cost', 0):,.0f} | {f3(d.get('ndcg10'))} | "
                    f"{f3(d.get('kept_ndcg'))} | {f3(cf.get('share', [None])[0])} | {f3(cf.get('kept', [None])[0])} |"
                )
        L.append("")
    L += ["## Length analysis within Vript (plan section 3)", ""]
    col = res.get("regression_rate_collinearity") or {}
    if col.get("all"):
        a = col["all"]
        L += [
            "Covariates: log2 source duration, k, twin, Vript long and scenes per minute. Caption "
            "words per minute (plan section 3) is left out: over the "
            f"{a['sources']} Vript sources its correlation with scenes per minute is "
            f"{a['corr_words_scenes_per_min']:.3f}"
            + "".join(f", {d} {col[d]['corr_words_scenes_per_min']:.3f}" for d in VRIPT if d in col)
            + ", because Vript captions run a near-constant length per scene (words per scene "
            f"p10 / median / p90 {' / '.join(f'{x:.0f}' for x in a['words_per_scene_p10_p50_p90'])}"
            "). With both in, each had a VIF of about 100. Scenes per minute is kept as the "
            "video's content-change rate.",
            "",
        ]
    for kind, regs in res["regression"].items():
        L += [
            f"### Regressions, {kind} labels (coefficient [95% CI], cluster-bootstrap SE, VIF)",
            "",
        ]
        for label, models_ in regs.items():
            for which, r in models_.items():
                names = r.get("terms") or [*COVARIATES, "mean_log2_candidate_duration"]
                L += [
                    f"{label} ({which}; n {r['n']}, sources {r['clusters']})",
                    "",
                    "| term | b | 95% CI | SE | VIF |",
                    "|---|---|---|---|---|",
                ]
                for i, t in enumerate(names):
                    vif = r["vif"][i]
                    L.append(
                        f"| {t} | {r['beta'][i]:+.4f} | [{r['ci_low'][i]:+.4f}, {r['ci_high'][i]:+.4f}] | "
                        f"{r['se'][i]:.4f} | {'-' if vif is None else f'{vif:.2f}'} |"
                    )
                L.append("")
    L += [
        "### Index visibility (confirmed facts visible in the source's 0.5 fps index frames)",
        "",
        "| group | n | share |",
        "|---|---|---|",
    ]
    L += [f"| {g} | {c['n']} | {P2.rate_cell(c)} |" for g, c in res["index_visibility"].items()]
    L += [
        "",
        "### Description drop (share of the source's split constraints its own description does not confirm, S2 traces)",
        "",
        (
            f"Definition: for each query whose source is in S0's top {S.s2} (so S2 checked it), the "
            "S2 verifier's verdicts on the source's own description for each of the query's split "
            "constraints (splitter output, not the mined facts); drop = 1 - (share answered yes), so "
            "'unclear' and 'no' both count as not confirmed. 'drop' is the mean of the per-query "
            "drop with a cluster-bootstrap CI over sources; 'checked / queries' is how many of the "
            "group's queries had their source in the pool (the others have no verdict, and those "
            "sources are likely the ones their descriptions describe worst, so the drop is a lower "
            "bound); yes / no / unclear are pooled over the checked constraints. The smoke run's "
            "'11-14%' (results_smoke §4.4) is the pooled yes share on the 16-frame index (top 20), "
            "the complement of this drop. Twin queries are built differently: k-1 facts shared with "
            "the twin plus one distinct fact, mined by comparing the two captions (twin_facts_v1), "
            "mostly objects, scenes and on-screen text; nested facts are mined as distinctive fine "
            "details (mostly attributes and objects). Twin constraints are therefore confirmed more "
            "often at every k, including k = 1 (only the distinct fact)."
        ),
        "",
        "| set | group | checked / queries | constraints | drop | yes | no | unclear |",
        "|---|---|---|---|---|---|---|---|",
    ]
    ver = res.get("description_drop_verdicts") or {}
    for name, t in res["description_drop"].items():
        for g, c in t.items():
            v = ver.get(name, {}).get(g)
            extra = (
                f"{v['checked']} / {v['queries']} | {v['constraints']} | {P2.rate_cell(c)} | "
                f"{v['yes']:.3f} | {v['no']:.3f} | {v['unclear']:.3f}"
                if v
                else f"{c['n']} | - | {P2.rate_cell(c)} | - | - | -"
            )
            L.append(f"| {name} | {g} | {extra} |")
    L += [
        "",
        "## Cost per query (recorded cost of every call, cached or not)",
        "",
        "| set | system | queries | calls | tokens in | tokens out | images |",
        "|---|---|---|---|---|---|---|",
    ]
    for name, table in res["costs"].items():
        for s, c in table.items():
            L.append(
                f"| {name} | {s} | {c['queries']} | {c['calls']:.1f} | {c['tokens_in']:,.0f} | "
                f"{c['tokens_out']:,.0f} | {c['images']:.1f} |"
            )
    if res["second"]:
        L += ["", "## Second verifier (" + S.cfg["second_verifier"] + ")", ""]
        for name, c in res["second"].items():
            L.append(
                f"- {name}: S4 - S2 nDCG@10 {ci(c['S4 - S2']['ndcg@10']['None'].get('all'))}; "
                f"S2 verdict agreement with {VERIFIER_SPEC}: kappa {f3(c['verdicts']['kappa'])} (n {c['verdicts']['n']})"
            )
    if res["repeats"]:
        L += [
            "",
            "## Run-to-run repeats (decision 5; fresh cache keys)",
            "",
            "| run | n | nDCG@10 run | repeat | repeat - run | top-1 same |",
            "|---|---|---|---|---|---|",
        ]
        for k, c in res["repeats"].items():
            if c.get("n"):
                L.append(
                    f"| {k} | {c['n']} | {c['run']:.3f} | {c['repeat']:.3f} | {ci(c['diff'])} | {c['top1_same']:.3f} |"
                )
    if res.get("judge"):
        j = res["judge"]
        L += [
            "",
            "## Label quality",
            "",
            (
                f"Sanity (source acceptance): {j.get('sanity')}. "
                f"Fact agreement: {j.get('agreement', {}).get('facts', {}).get('all')}."
            ),
        ]
    L += [
        "",
        "## Spend (new input tokens per stage)",
        "",
        "| stage | entry | calls | tokens in | tokens out |",
        "|---|---|---|---|---|",
    ]
    for stage, entries in res["spend"].items():
        L += [
            f"| {stage} | {e['stage']} | {e['calls']:,} | {e['tokens_in']:,} | {e['tokens_out']:,} |"
            for e in entries
        ]
    return "\n".join(L) + "\n"


# ---- the smoke reproduction check ---------------------------------------------------------------


def smoke_check(S, res) -> None:
    """Our means against the pilot (summary.json), deep pool and twin-both summaries."""
    pilot = json.loads((DEV / "pilot/summary.json").read_text())
    deep = json.loads((DEV / "deep_pool/summary.json").read_text())
    both = json.loads((DEV / "pilot/twin_both_summary.json").read_text())
    rows = []

    def ours(kind, name, s, metric, subset=None):
        cell = res["labels"].get(kind, {})
        t = cell["subset_means"][subset] if subset else cell["means"][name]
        return t.get(s, {}).get("None", {}).get("all", {}).get(metric)

    names = {
        "S0": "S0",
        "S1": "S1",
        "S2": "S2@20",
        "S4": "S4@20",
        "S2@10": "S2@10",
        "S4@10": "S4@10",
    }
    for kind in ("strict", "expanded"):
        if kind not in res["labels"] or kind not in pilot["labels"]:
            continue
        for name in SETS:
            for theirs, mine in names.items():
                for m in ("ndcg@10", "recall@1"):
                    ref = (
                        pilot["labels"][kind]["means"][name]
                        .get(theirs, {})
                        .get("overall", {})
                        .get("all", {})
                        .get(m)
                    )
                    rows.append(
                        (f"pilot {kind} {name} {theirs}", mine, m, ref, ours(kind, name, mine, m))
                    )
            for s in ("S0", "S2@10", "S4@10", "S3", "S5"):
                if name == "natural":
                    ref = (
                        pilot["labels"][kind]["subset_means"][name]
                        .get(s, {})
                        .get("overall", {})
                        .get("all", {})
                        .get("ndcg@10")
                    )
                    rows.append(
                        (
                            f"pilot {kind} natural frame subset {s}",
                            s,
                            "ndcg@10",
                            ref,
                            ours(kind, name, s, "ndcg@10", "nested sample"),
                        )
                    )
        for s in ("S0", "S2@10", "S4@10", "S3", "S5"):
            ref = (
                both["labels"]
                .get(kind, {})
                .get("means", {})
                .get(s, {})
                .get("all", {})
                .get("ndcg@10")
            )
            rows.append(
                (
                    f"twin-both {kind} {s}",
                    s,
                    "ndcg@10",
                    ref,
                    ours(kind, "twin", s, "ndcg@10", "twin-both"),
                )
            )
    for name in SETS:
        ref = deep["sets"][name]["labels"]["strict"]["means"]["S4@50"]["all"]["ndcg@10"]
        rows.append(
            (
                f"deep pool strict {name} S4@50",
                "S4@50",
                "ndcg@10",
                ref,
                ours("strict", name, "S4@50", "ndcg@10"),
            )
        )
    for g in ("all", "1", "2", "3", "4"):
        ref = both.get("twin_above_source", {}).get("S5", {}).get(g, {}).get("rate")
        mine = res["twin_above_source"]["twin-both"].get("S5", {}).get(g, {}).get("rate")
        rows.append((f"twin-both twin above source S5 k={g}", "S5", "rate", ref, mine))
    bad = [r for r in rows if r[3] is None or r[4] is None or abs(r[3] - r[4]) > 1e-9]
    lines = [
        "# Smoke reproduction: pipeline/run_systems.py --smoke vs the smoke reports",
        "",
        f"{len(rows) - len(bad)} of {len(rows)} values identical (tolerance 1e-9).",
        "",
        "| reference | ours | metric | reference value | ours | match |",
        "|---|---|---|---|---|---|",
    ]
    for ref_name, mine, m, a, b in rows:
        ok = a is not None and b is not None and abs(a - b) <= 1e-9
        lines.append(f"| {ref_name} | {mine} | {m} | {f3(a)} | {f3(b)} | {'yes' if ok else 'NO'} |")
    spent = {k: sum(e["tokens_in"] for e in v) for k, v in res["spend"].items()}
    lines += ["", f"New input tokens spent by this reproduction, per stage: {spent}", ""]
    (S.out / "smoke_check.md").write_text("\n".join(lines))
    print("\n".join(lines[:3]), f"spend {spent}")


# ---- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=("retrieve", "select", "desc", "second", "frames", "repeats", "eval", "s4long"),
    )
    ap.add_argument("--smoke", action="store_true", help="reproduce the smoke run from caches")
    ap.add_argument("--smoke-cap", type=float, default=2e5)
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--s0-only", action="store_true", help="retrieve: S0 and the gate only (free)")
    ap.add_argument("--project-only", action="store_true")
    ap.add_argument("--force", action="store_true", help="select: redraw frozen subsets")
    ap.add_argument("--workers", type=int, default=32, help="queries in flight")
    ap.add_argument("--inner", type=int, default=4, help="calls per query in flight")
    ap.add_argument("--canary", type=int, default=10)
    ap.add_argument("--canary-frames", type=int, default=2)
    ap.add_argument("--rate-tol", type=float, default=0.02, help="frames canary: tokens/image")
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--canary-always", action="store_true", help="run canaries on cache hits too")
    ap.add_argument("--passes", type=int, default=4, help="retry passes for failed calls")
    ap.add_argument("--wait", type=float, default=60.0, help="seconds before a retry pass (x pass)")
    ap.add_argument("--repeat-seed", type=int, default=1)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--frame-verifier", help="frames: S5 only, with this (gemini) verifier spec")
    ap.add_argument("--frame-media-resolution", help="frame verifier: low/medium/high (unset)")
    ap.add_argument("--frame-stage", help="frame verifier: output folder under the out dir")
    ap.add_argument("--frame-cap", type=float, default=40e6, help="frame verifier: token cap")
    ap.add_argument("--rate-ref", type=float, help="frame verifier: tokens per image (projection)")
    ap.add_argument("--canary-only", action="store_true", help="frame verifier: stop after canary")
    ap.add_argument(
        "--frame-pool",
        choices=("s4at50", "s0top10"),
        default="s4at50",
        help="frames: s4at50 = the frozen subsets.json pool; s0top10 = S5 on S0's top 10 (W4)",
    )
    ap.add_argument("--run-suffix", default="", help="frame pool s0top10: run file S5<suffix>")
    ap.add_argument("--query-ids", help="frame pool s0top10: only these {set: [query ids]}")
    ap.add_argument("--limit-queries", type=int, help="frame pool s0top10: first N per set")
    ap.add_argument("--desc-file", help="s4long: long descriptions file in each dataset dir")
    ap.add_argument("--desc-prompt", default="prompts/describe_v4d.txt", help="s4long: its prompt")
    ap.add_argument("--run-dir", help="s4long: write the run files here instead of runs/")
    ap.add_argument(
        "--sources-file",
        help="keep only the queries of these sources ({'sources': [...]}); subsets intersected",
    )
    args = ap.parse_args(argv)
    S = settings(args)
    ctx = load_context(S, need_index=args.stage in ("retrieve",))
    sets = load_sets(S, ctx)
    if args.sources_file:
        sets = restrict_sets(S, sets, args.sources_file)
    {
        "retrieve": stage_retrieve,
        "select": stage_select,
        "desc": stage_desc,
        "second": stage_second,
        "frames": stage_frames,
        "repeats": stage_repeats,
        "eval": stage_eval,
        "s4long": stage_s4long,
    }[args.stage](S, ctx, sets, args)


if __name__ == "__main__":
    main()
