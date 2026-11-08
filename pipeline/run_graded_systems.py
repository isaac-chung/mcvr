"""Tier 1 of the methodology revision (methodology_plan_20261031.md): graded
per-constraint verifiers on the whole-query 0-10 scale, S5 on every test query, the trigger over
the full query distribution, and the label fill. Thin layer over pipeline/run_systems.py (FS),
whose settings, ledgers, caps, canaries, retry passes, models and eval helpers it reuses; nothing
already run is recomputed (S0, splits, S2, S4, the 230 + 150 S5 runs, labels).

Paid stages (each prints its projection of new input tokens and stops when the stage's cap in
configs/mc_tier1.yaml `caps`, or configs/mc_full.yaml `total_cap`, would be passed; every call
goes through the shared LLM cache, so a rerun pays only for calls that failed):

    uv run --env-file .env python pipeline/run_graded_systems.py --stage desc-graded --project-only
    uv run --env-file .env python pipeline/run_graded_systems.py --stage desc-graded     # A1: S2g@50, all
    uv run --env-file .env python pipeline/run_graded_systems.py --stage second-graded   # A3: gpt-5.5
    uv run --env-file .env python pipeline/run_graded_systems.py --stage frames-graded   # A2: S3g
    uv run --env-file .env python pipeline/run_graded_systems.py --stage frames-all      # B: S5, rest
    uv run --env-file .env python pipeline/judge_labels.py --stage run \\
        --fill-queries results/full/tier1/fill_ids.json                        # D (see fill-plan)

The test set is the 100-source draw of configs/mc_tier1.yaml `sources_file` (results/full/select/
sources100.json, seed 11); every stage filters the query sets to it unless --all-sources is given.

Free stages:

    uv run python pipeline/run_graded_systems.py --stage fill-plan   # D: unjudged top-10 pairs, ids file
    uv run python pipeline/run_graded_systems.py --stage eval        # A-C: results/full/tier1/report.md

`--dump-prompts FILE` on desc-graded / second-graded writes the uncached prompts as JSONL
(one {"prompt": ...} per line) and stops; once they are in the cache, the stage runs as cache hits.

Outputs: results/full/runs/<set>/{S2g@50, S3g}.jsonl, S5.jsonl extended to every query (merge),
results/full/second/<set>/S2g@50_<model>.jsonl, results/full/<stage>/{spend, manifest}.json and
results/full/tier1/{summary.json, report.md, fill_ids.json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from src.datasets.mc import read_jsonl
from src.grounding import frame_labels
from src.systems.base import merge_reranked
from src.systems.single import cluster_bootstrap_ci
from src.systems.verify import (
    VERIFIER_SPEC,
    DescriptionCheck,
    FrameCheck,
    GradedVerifier,
    aggregate_grades,
    numbered,
    parse_grades,
    rank_by_score,
)

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FS = _load(ROOT / "pipeline/run_systems.py", "_tier1_systems")
P2, PV, TS = FS.P2, FS.PV, FS.TS

TIER1_CONFIG = Path("configs/mc_tier1.yaml")
STAGE_DIRS = {
    "desc-graded": "tier1_desc",
    "second-graded": "tier1_second",
    "frames-graded": "tier1_frames",
    "frames-all": "tier1_frames_all",
}
for _d in STAGE_DIRS.values():  # each Tier 1 stage has its own cap
    FS.CAP_OF[_d] = _d
    FS.STAGES_OF_CAP[_d] = [_d]
KS = (1, 2, 3, 4)
SET_LABEL = {"natural": "nested", "twin": "near-miss"}


# ---- settings -----------------------------------------------------------------------------------


def settings(args) -> SimpleNamespace:
    """FS.settings (configs/mc_full.yaml paths, depths, caps) plus the Tier 1 config as S.tier,
    its caps added, and S.tier1 = results/full/tier1."""
    base = argparse.Namespace(
        smoke=False, smoke_cap=2e5, config=args.config, out=args.out, n_boot=args.n_boot
    )
    S = FS.settings(base)
    S.tier = yaml.safe_load(Path(args.tier1_config or TIER1_CONFIG).read_text())
    S.caps = {**S.caps, **S.tier["caps"]}
    S.tier1 = S.out / "tier1"
    S.depth_g = int(S.tier["graded_depth"])
    S.S2G = f"S2g@{S.depth_g}"
    S.draw = None
    if S.tier.get("sources_file") and not getattr(args, "all_sources", False):
        path = Path(S.tier["sources_file"])
        if not path.exists():
            sys.exit(f"{path} missing: draw it with pipeline/draw_test_set.py --out {path}")
        S.draw = json.loads(path.read_text())
    return S


def draw_sets(S, sets: dict) -> dict:
    """The query sets restricted to the drawn sources (all of a drawn source's queries)."""
    if S.draw is None:
        return sets
    keep = set(S.draw["sources"])
    out = {n: [r for r in rs if r["source"] in keep] for n, rs in sets.items()}
    print(
        f"[tier1] test set: {len(keep)} sources (seed {S.draw.get('seed')}), queries "
        f"{ {n: len(v) for n, v in out.items()} }"
    )
    return out


def second_ids(S, sets: dict, sub: dict) -> dict[str, list[str]]:
    """Second-verifier subset: the frozen one on the full set; inside the draw, 150 per set
    balanced over k from the same seeded shuffle (PV.subset_balanced, seed 0)."""
    if S.draw is None:
        return sub["second"]
    n = int(S.tier.get("second_verifier_subset", S.cfg["second_verifier_subset"]))
    return {name: PV.subset_balanced(rs, n, 0) for name, rs in sets.items()}


def graded_parses(text: str) -> bool:
    data = FS.parse_json(text)
    return isinstance(data, dict) and isinstance(
        data.get("constraints", data.get("grades")), (list, dict)
    )


class WatchVision:
    """As FS.Watch for a vision model: logs answers that do not parse."""

    def __init__(self, inner, ok, log: list):
        self.inner, self.ok, self.log = inner, ok, log

    def describe(self, prompt: str, images: list[bytes], labels=None):
        text, cost = self.inner.describe(prompt, images, labels=labels)
        if text and not self.ok(text):
            self.log.append(text[:200])
        return text, cost


def splits_of(S, name: str) -> dict[str, list[str]]:
    return json.loads((S.runs / name / "splits.json").read_text())


# ---- A1 / A3: S2g on descriptions -------------------------------------------------------------


def desc_prompts(S, sets, ids, first, descriptions, template: str, depth: int) -> dict:
    """{(set, query_id, doc): prompt} as DescriptionCheck builds them, S0's top `depth`."""
    out = {}
    for name, rs in sets.items():
        splits = splits_of(S, name)
        keep = set(ids.get(name, []))
        for r in rs:
            q = r["query_id"]
            if q not in keep:
                continue
            for d, _ in first[name][q][:depth]:
                desc = descriptions.get(d, "") or "(no description)"
                out[(name, q, d)] = (
                    template.replace("{query}", r["query"])
                    .replace("{constraints}", numbered(splits[q]))
                    .replace("{description}", desc)
                )
    return out


def run_desc_graded(S, ctx, sets, args, stage, spec, ids, out_dir, suffix, models_spec) -> None:
    ledger = FS.Ledger(S, stage)
    m = FS.text_models(spec, S)
    sub = FS.subset_sets(sets, ids)
    first = {n: FS.read_ranking(S.runs / n / "S0.jsonl") for n in sets}
    template = Path(S.tier["graded_desc_prompt"]).read_text()
    depth = S.depth_g
    prompts = desc_prompts(S, sets, ids, first, ctx["descriptions"], template, depth)
    new = [p for p in dict.fromkeys(prompts.values()) if not m.verifier_cached.is_cached(p)]
    proj = sum(P2.text_tokens(p) for p in new)
    print(
        f"[{stage}] {spec}: {sum(map(len, sub.values()))} queries x top {depth} = "
        f"{len(prompts)} calls, {len(new)} to pay"
    )
    if args.dump_prompts:
        path = Path(args.dump_prompts)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w") as handle:
            for p in new:
                handle.write(json.dumps({"prompt": p}) + "\n")
        print(f"[{stage}] wrote {len(new)} uncached prompts to {path}")
        return
    ledger.check(stage, proj)
    if args.project_only:
        return
    splitters = FS.fixed_splitter(S, sets, ids)
    log: list = []
    out = FS.rargs(S, args, out_dir)
    system = f"{S.S2G}{suffix}"
    paths = []
    for name in sets:
        if not sub[name]:
            continue
        verifier = FS.Watch(m.verifier, graded_parses, log)
        reranker = GradedVerifier(
            "S2g",
            splitters[name],
            DescriptionCheck(verifier, template, ctx["descriptions"], parse_grades),
            args.inner,
            S.tier["aggregate"],
        )
        if (new or args.canary_always) and not args.no_canary and name == FS.SETS[0]:
            can = FS.canary_ids({name: sub[name]}, args.canary)[name]
            P2.rerank_set(
                m,
                name,
                sub[name],
                {system: (reranker, can, depth)},
                FS.rargs(S, args, S.out / stage / "canary"),
                "canary",
                first=first[name],
            )
            if log or m.errors:
                sys.exit(f"[{stage}] canary failed: unparsed {log[:2]}, errors {m.errors[:2]}")
            print(f"[{stage}] canary ok ({len(can)} queries)")
            if args.canary_only:
                paid, n = m.paid_cost()
                ledger.add(stage, paid, n, {"canary_only": True, "queries": len(can)})
                print(
                    f"[{stage}] --canary-only: stopping after the canary; its calls are cached "
                    f"and written to {S.out / stage / 'canary' / name}"
                )
                return
        plan = {system: (reranker, [r["query_id"] for r in sub[name]], depth)}
        FS.passes(
            lambda name=name, plan=plan: P2.rerank_set(
                m, name, sub[name], plan, out, stage, first=first[name]
            ),
            m.errors,
            args.passes,
            args.wait,
            f"{stage} {name}",
        )
        paths.append(Path(out.out) / name / f"{system}.jsonl")
    paid, n = m.paid_cost()
    ledger.add(stage, paid, n, {"errors": len(m.errors), "unparsed": len(log)})
    FS.manifest(
        S,
        stage,
        models_spec,
        paid,
        paths,
        {
            "depth": depth,
            "prompt": Path(S.tier["graded_desc_prompt"]).name,
            "aggregate": S.tier["aggregate"],
            "queries": {k: len(v) for k, v in ids.items()},
        },
    )


def stage_desc_graded(S, ctx, sets, args) -> None:
    ids = {n: [r["query_id"] for r in rs] for n, rs in sets.items()}
    run_desc_graded(
        S,
        ctx,
        sets,
        args,
        STAGE_DIRS["desc-graded"],
        VERIFIER_SPEC,
        ids,
        None,
        "",
        {"verifier": VERIFIER_SPEC, "splitter": FS.SPLIT_SPEC},
    )


def stage_second_graded(S, ctx, sets, args) -> None:
    sub = FS.read_subsets(S)
    spec = S.cfg["second_verifier"]
    name = spec.split(":", 1)[1].split("@")[0]
    run_desc_graded(
        S,
        ctx,
        sets,
        args,
        STAGE_DIRS["second-graded"],
        spec,
        second_ids(S, sets, sub),
        S.out / "second",
        f"_{name}",
        {"second_verifier": spec, "splitter": FS.SPLIT_SPEC},
    )


# ---- A2: S3g on frames ---------------------------------------------------------------------------


def frame_projection(m, S, sets, qids, first, frames, template, per_image) -> tuple:
    """New input tokens, calls and images of S3g (prompts as FrameCheck builds them)."""
    cache: dict = {}

    def load(doc):
        if doc not in cache:
            cache[doc] = frames(doc)
        return cache[doc]

    tokens, calls, images = 0.0, 0, 0
    for name, rs in sets.items():
        splits = splits_of(S, name)
        by_id = {r["query_id"]: r for r in rs}
        for q in qids.get(name, []):
            for d, _ in first[name][q][: S.top]:
                imgs, times = load(d)
                p = (
                    template.replace("{n}", str(len(imgs)))
                    .replace("{query}", by_id[q]["query"])
                    .replace("{constraints}", numbered(splits[q]))
                )
                if not m.vlm_cached.is_cached(p, imgs, frame_labels(times)):
                    tokens += P2.text_tokens(p) + 12 * len(imgs) + per_image * len(imgs)
                    calls += 1
                    images += len(imgs)
    return tokens, calls, images


def stage_frames_graded(S, ctx, sets, args) -> None:
    stage = STAGE_DIRS["frames-graded"]
    ledger = FS.Ledger(S, stage)
    sub = FS.read_subsets(S)
    pool = sub["pool"]
    # every query of the test set (S3g covers the test set like S5 does after frames-all); on the
    # full set, the frozen frame subsets as in the original plan
    qids = (
        {n: [r["query_id"] for r in rs] for n, rs in sets.items()}
        if S.draw is not None
        else sub["frames"]
    )
    first = {n: FS.read_ranking(S.runs / n / f"{pool}.jsonl") for n in sets}
    frames = P2.frame_source(ctx)
    m = FS.vision_models(S)
    template = Path(S.tier["graded_frames_prompt"]).read_text()
    rate_file = S.out / "frames" / "frame_rate.json"
    per_image = (
        json.loads(rate_file.read_text())["tokens_per_image"] if rate_file.exists() else FS.RATE_REF
    )
    proj, calls, images = frame_projection(m, S, sets, qids, first, frames, template, per_image)
    print(
        f"[{stage}] S3g on {pool} top {S.top}: {sum(map(len, qids.values()))} queries, {calls} "
        f"calls, {images:,} images to pay at {per_image:.1f} tokens per image"
    )
    ledger.check(stage, proj)
    if args.project_only:
        return
    splitters = FS.fixed_splitter(S, sets, qids)
    log: list = []
    vlm = WatchVision(m.vlm, graded_parses, log)
    out = FS.rargs(S, args)

    def plan(name):
        return {
            "S3g": (
                GradedVerifier(
                    "S3g",
                    splitters[name],
                    FrameCheck(vlm, template, frames, parse_grades),
                    args.inner,
                    S.tier["aggregate"],
                ),
                qids.get(name, []),
                S.top,
            )
        }

    def run(ids, ra):
        for name in sets:  # merged file: rows of every saved query for the writer
            p = {s: (rr, ids.get(name, []), n) for s, (rr, _, n) in plan(name).items()}
            P2.rerank_set(m, name, S.all_sets[name], p, ra, stage, merge=True, first=first[name])

    if (calls or args.canary_always) and not args.no_canary:
        can = FS.canary_ids(FS.subset_sets(sets, qids), args.canary_frames)
        run(can, FS.rargs(S, args, S.out / stage / "canary"))
        if log or m.errors:
            sys.exit(f"[{stage}] canary failed: unparsed {log[:2]}, errors {m.errors[:2]}")
        print(f"[{stage}] canary ok ({sum(map(len, can.values()))} queries)")
    left = FS.passes(lambda: run(qids, out), m.errors, args.passes, args.wait, stage)
    paid, n = m.paid_cost()
    ledger.add(
        stage,
        paid,
        n,
        {
            "errors": left,
            "unparsed": len(log),
            "pool": pool,
            "queries": {k: len(v) for k, v in qids.items()},
        },
    )
    FS.manifest(
        S,
        stage,
        {"verifier": VERIFIER_SPEC},
        paid,
        [S.runs / n_ / "S3g.jsonl" for n_ in sets],
        {
            "pool": pool,
            "top": S.top,
            "prompt": Path(S.tier["graded_frames_prompt"]).name,
            "aggregate": S.tier["aggregate"],
            "tost_margin": S.tier["tost_margin"],
        },
    )


# ---- B: S5 on every remaining query -------------------------------------------------------------


def remaining_ids(S, sets) -> dict[str, list[str]]:
    """Query ids without an S5 ranking yet (the frozen frame subsets are already run)."""
    out = {}
    for name, rs in sets.items():
        done = PV.saved_query_ids(S.runs / name / "S5.jsonl")
        out[name] = [r["query_id"] for r in rs if r["query_id"] not in done]
    return out


def stage_frames_all(S, ctx, sets, args) -> None:
    ids = remaining_ids(S, sets)
    print(f"[frames-all] S5 still to run: { {n: len(v) for n, v in ids.items()} }")
    if not any(ids.values()):
        print("[frames-all] nothing to do: S5 covers every query")
        return
    # S5.jsonl is merged, so the writer needs the rows of every saved query, not only the test set's
    FS.run_frames(
        S, ctx, S.all_sets, args, STAGE_DIRS["frames-all"], FS.vision_models(S), ids, ("S5",)
    )


# ---- D: label fill plan (free) -------------------------------------------------------------------


def rankings(S) -> dict[str, dict[str, list[str]]]:
    out: dict = {}
    for path in sorted(S.runs.glob("*/S*.jsonl")):
        for line in read_jsonl(path):
            out.setdefault(path.stem, {})[line["query_id"]] = [d for d, _ in line["ranking"]]
    return out


def judged_pairs(S) -> dict[str, set[str]]:
    out: dict = {}
    path = S.judge_dir / "judged.jsonl"
    if path.exists():
        for line in read_jsonl(path):
            out.setdefault(line["source"], set()).add(line["candidate"])
    return out


def fill_plan(rows: list[dict], rank: dict, judged: dict, systems: list[str], top: int) -> dict:
    """Unjudged (source, candidate) pairs in the top `top` of any run (what judge.py
    --fill-queries will judge), with the systems that brought each in and the query ids."""
    pairs: dict[tuple, set] = {}
    ids = set()
    for r in rows:
        q, source = r["query_id"], r["source"]
        for system, by_q in rank.items():
            for d in by_q.get(q, [])[:top]:
                if d in judged.get(source, set()) or d == r.get("twin"):
                    continue
                pairs.setdefault((source, d), set()).add(system)
                ids.add(q)
    by_system = {s: sum(s in v for v in pairs.values()) for s in rank}
    only_new = sum(1 for v in pairs.values() if v <= set(systems))
    return {
        "pairs": len(pairs),
        "queries": len(ids),
        "by_system": by_system,
        "only_new_systems": only_new,
        "query_ids": sorted(ids),
    }


def stage_fill_plan(S, ctx, sets, args) -> None:
    rows = [r for rs in sets.values() for r in rs]
    rank = rankings(S)
    plan = fill_plan(rows, rank, judged_pairs(S), S.tier["fill"]["systems"], S.top)
    by_set = {
        n: [r["query_id"] for r in rs if r["query_id"] in set(plan["query_ids"])]
        for n, rs in sets.items()
    }
    S.tier1.mkdir(parents=True, exist_ok=True)
    ids_file = Path(S.tier["fill"]["ids_file"])
    ids_file.write_text(json.dumps(by_set, indent=1))
    present = [s for s in S.tier["fill"]["systems"] if s in rank]
    missing = [s for s in S.tier["fill"]["systems"] if s not in rank]
    # judge pair: both models, ~5.9k input tokens over the two calls (methodology plan)
    tokens = plan["pairs"] * 5.9e3
    print(
        f"[fill-plan] runs present: {sorted(rank)}; new systems present {present}, not yet run "
        f"{missing}\n[fill-plan] unjudged top-{S.top} pairs: {plan['pairs']} over "
        f"{plan['queries']} queries (by system {plan['by_system']}; reached only by the new "
        f"systems: {plan['only_new_systems']}); about {tokens / 1e6:.1f}M judge input tokens\n"
        f"[fill-plan] wrote {ids_file} ({ {n: len(v) for n, v in by_set.items()} }); then:\n"
        f"  uv run --env-file .env python pipeline/judge_labels.py --stage run --project-only "
        f"--fill-queries {ids_file}\n  uv run --env-file .env python pipeline/judge_labels.py "
        f"--stage run --fill-queries {ids_file}\n  uv run python pipeline/judge_labels.py --stage stats"
    )
    (S.tier1 / "fill_plan.json").write_text(
        json.dumps({k: v for k, v in plan.items() if k != "query_ids"}, indent=1)
    )


# ---- eval (free) ---------------------------------------------------------------------------------


def rescore_graded(line: dict, first: list, top: int, how: str) -> list:
    """The graded system on the first `top` candidates, rescored from the saved grades."""
    head = line["trace"]["first_stage"][:top]
    scores = [aggregate_grades(line["trace"]["grades"][d], how) for d in head]
    return merge_reranked(first, rank_by_score([(d, 0.0) for d in head], scores))


def load_runs(S, sets, sub) -> tuple[dict, dict]:
    """FS.load_runs per set (S2@10 / S4@10 rescored) plus S2g@10 and the min-aggregated S2g / S3g
    rescored from the saved grades, and the second verifier's runs under their own names."""
    ranks, lines = {}, {}
    pool, how2 = sub["pool"], S.tier["secondary_aggregate"]
    for name in sets:
        ranks[name], lines[name] = FS.load_runs(S.runs / name, S.rescore)
        if S.S2G in lines[name]:
            s0 = ranks[name]["S0"]
            for r in S.rescore:
                ranks[name][f"S2g@{r}"] = {
                    q: rescore_graded(ln, s0[q], r, S.tier["aggregate"])
                    for q, ln in lines[name][S.S2G].items()
                }
            ranks[name][f"{S.S2G}-{how2}"] = {
                q: rescore_graded(ln, s0[q], S.depth_g, how2)
                for q, ln in lines[name][S.S2G].items()
            }
        if "S3g" in lines[name] and pool in ranks[name]:
            ranks[name][f"S3g-{how2}"] = {
                q: rescore_graded(ln, ranks[name][pool][q], S.top, how2)
                for q, ln in lines[name]["S3g"].items()
            }
        second = S.out / "second" / name
        if second.exists():
            r2, l2 = FS.load_runs(second, ())
            ranks[name].update(r2)
            lines[name].update(l2)
    return ranks, lines


def boot(diffs, clusters, n_boot: int, seed: int = 0) -> dict:
    """Mean of diffs with a 95% and a 90% (TOST) cluster-bootstrap CI and a two-sided
    bootstrap p-value (share of resampled means on the other side of zero, doubled)."""
    groups: dict[str, list[float]] = {}
    for v, c in zip(diffs, clusters, strict=True):
        groups.setdefault(c, []).append(v)
    sums = np.array([sum(g) for g in groups.values()])
    counts = np.array([len(g) for g in groups.values()])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    lo90, hi90 = np.percentile(means, [5, 95])
    mean = float(np.mean(diffs))
    p = 2 * min(float((means <= 0).mean()), float((means >= 0).mean()))
    return {
        "n": len(diffs),
        "clusters": len(sums),
        "mean": mean,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "ci90_low": float(lo90),
        "ci90_high": float(hi90),
        "p": min(1.0, p),
    }


def paired(sa: dict, sb: dict, rows: list[dict], n_boot: int, metric="ndcg@10") -> dict | None:
    rs = [r for r in rows if r["query_id"] in sa and r["query_id"] in sb]
    if len(rs) < 5:
        return None
    return boot(
        [sa[r["query_id"]][metric] - sb[r["query_id"]][metric] for r in rs],
        [r["source"] for r in rs],
        n_boot,
    )


def by_group(sa, sb, rows, n_boot) -> dict:
    out = {"all": paired(sa, sb, rows, n_boot)}
    for k in KS:
        out[f"k={k}"] = paired(sa, sb, [r for r in rows if r["k"] == k], n_boot)
    return out


def mean_of(sc: dict, rows: list[dict], metric="ndcg@10") -> dict:
    ids = [r["query_id"] for r in rows if r["query_id"] in sc]
    out = {"n": len(ids), "all": float(np.mean([sc[q][metric] for q in ids])) if ids else None}
    for k in KS:
        ks = [
            q
            for q, r in zip(ids, [r for r in rows if r["query_id"] in sc], strict=True)
            if r["k"] == k
        ]
        out[f"k={k}"] = float(np.mean([sc[q][metric] for q in ks])) if ks else None
    return out


def eval_graded(S, sets, ranks, scores, sub, n_boot) -> dict:
    """A1 / A3: S2g against S4 (same scale) and S2 (same framing) on all queries; the second
    verifier on its subset; min aggregation as the secondary."""
    g, how2 = S.S2G, S.tier["secondary_aggregate"]
    systems = ["S0", "S2@50", "S2@10", g, "S2g@10", f"{g}-{how2}", "S4@50", "S4@10"]
    pairs = [
        ("S4@50", g),
        ("S4@10", "S2g@10"),
        (g, "S2@50"),
        (g, "S0"),
        ("S4@50", "S0"),
        (f"{g}-{how2}", g),
        ("S4@50", f"{g}-{how2}"),
    ]
    out: dict = {"systems": systems, "sets": {}, "second": {}}
    for name, rs in sets.items():
        sc = scores[name]
        out["sets"][name] = {
            "means": {s: mean_of(sc[s], rs) for s in systems if s in sc},
            "paired": {
                f"{a} - {b}": by_group(sc[a], sc[b], rs, n_boot)
                for a, b in pairs
                if a in sc and b in sc
            },
        }
        model = S.cfg["second_verifier"].split(":", 1)[1].split("@")[0]
        a, b, c = f"S4@50_{model}", f"{g}_{model}", f"S2@50_{model}"
        rows = [r for r in rs if r["query_id"] in set(second_ids(S, sets, sub)[name])]
        if a in sc and b in sc:
            out["second"][name] = {
                "model": model,
                "means": {s: mean_of(sc[s], rows) for s in (a, b, c) if s in sc},
                "paired": {
                    f"{a} - {b}": by_group(sc[a], sc[b], rows, n_boot),
                    **({f"{b} - {c}": by_group(sc[b], sc[c], rows, n_boot)} if c in sc else {}),
                },
            }
    return out


def eval_frames_graded(S, sets, scores, sub, n_boot) -> dict:
    """A2: S3g against S5 on the frame subsets, with the TOST at the pre-set margin."""
    pool, how2, margin = sub["pool"], S.tier["secondary_aggregate"], float(S.tier["tost_margin"])
    systems = [pool, "S3", "S5", "S3g", f"S3g-{how2}"]
    pairs = [("S3g", "S5"), ("S3g", "S3"), ("S3g", pool), ("S5", pool), (f"S3g-{how2}", "S5")]
    out: dict = {"tost_margin": margin, "systems": systems, "subsets": {}}
    groups = {
        "contested": [r for r in sets["twin"] if r["query_id"] in set(sub["twin_both"]["twin"])],
        "nested sample": [
            r for r in sets["natural"] if r["query_id"] in set(sub["nested_sample"]["natural"])
        ],
    }
    if S.draw is not None:  # S3g and S5 cover the whole test set
        groups = {"all near-miss": sets["twin"], "all nested": sets["natural"], **groups}
    for label, rows in groups.items():
        name = "twin" if label in ("contested", "all near-miss") else "natural"
        sc = scores[name]
        cell = {
            "n": len(rows),
            "means": {s: mean_of(sc[s], rows) for s in systems if s in sc},
            "paired": {
                f"{a} - {b}": by_group(sc[a], sc[b], rows, n_boot)
                for a, b in pairs
                if a in sc and b in sc
            },
        }
        d = cell["paired"].get("S3g - S5", {}).get("all")
        if d:
            cell["tost"] = {
                "equivalent": -margin < d["ci90_low"] and d["ci90_high"] < margin,
                "ci90": [d["ci90_low"], d["ci90_high"]],
                "margin": margin,
            }
        out["subsets"][label] = cell
    return out


def eval_frames_all(S, sets, ranks, scores, sub, n_boot) -> dict:
    """B: S5 against S4@50 on every query with an S5 ranking (today the 380 of the frame subsets,
    after --stage frames-all all 1,989), by set, k, and contested / unselected."""
    pool = sub["pool"]
    contested = set(sub["twin_both"]["twin"]) | set(sub["nested_sample"]["natural"])
    out: dict = {"pool": pool, "coverage": {}, "sets": {}, "pooled": {}}
    pooled = []
    for name, rs in sets.items():
        sc = scores[name]
        if "S5" not in sc:
            continue
        rows = [r for r in rs if r["query_id"] in sc["S5"]]
        out["coverage"][name] = {"with_S5": len(rows), "queries": len(rs)}
        in_pool = [
            r
            for r in rows
            if r["source"] in {d for d, _ in ranks[name][pool][r["query_id"]][: S.top]}
        ]
        groups = {
            "all": rows,
            "selected (frame subsets)": [r for r in rows if r["query_id"] in contested],
            "unselected": [r for r in rows if r["query_id"] not in contested],
            f"source in {pool} top {S.top}": in_pool,
            f"source outside {pool} top {S.top}": [r for r in rows if r not in in_pool],
        }
        out["sets"][name] = {
            g: {
                "n": len(gr),
                "S4": mean_of(sc[pool], gr),
                "S5": mean_of(sc["S5"], gr),
                "S5 - S4": by_group(sc["S5"], sc[pool], gr, n_boot),
                "S5 - S0": by_group(sc["S5"], sc["S0"], gr, n_boot) if "S0" in sc else None,
            }
            for g, gr in groups.items()
            if gr
        }
        pooled += [{**r, "_set": name} for r in rows]
    if pooled:
        sc_all = {
            s: {q: v for name in sets for q, v in scores[name].get(s, {}).items()}
            for s in (pool, "S5")
        }
        out["pooled"] = {
            "n": len(pooled),
            "S5 - S4": by_group(sc_all["S5"], sc_all[pool], pooled, n_boot),
        }
    return out


def eval_length(S, sets, scores, sub, n_boot) -> dict:
    """B / W8: S5 - S4@50 nDCG@10 on log2 duration and the covariates of FS.regressions over
    every Vript query with an S5 ranking; the slope's CI and the minimal detectable slope
    (80% power, two-sided 5%: 2.8 x bootstrap SE)."""
    meta = FS.source_meta(S)
    pool = sub["pool"]
    Y, X, cl = [], [], []
    for name, rs in sets.items():
        sc = scores[name]
        if "S5" not in sc:
            continue
        for r in rs:
            q = r["query_id"]
            if r["dataset"] not in FS.VRIPT or q not in sc["S5"] or q not in sc[pool]:
                continue
            x = FS.covariates({**r, "set": name}, meta)
            if x is None:
                continue
            Y.append(sc["S5"][q]["ndcg@10"] - sc[pool][q]["ndcg@10"])
            X.append(x)
            cl.append(r["source"])
    if len(Y) <= len(FS.COVARIATES) + 2:
        return {"n": len(Y)}
    fit = FS.ols_boot(Y, X, cl, n_boot)
    j = FS.COVARIATES.index("log2_duration")
    return {
        "n": len(Y),
        "clusters": fit["clusters"],
        "terms": list(FS.COVARIATES),
        "beta": fit["beta"],
        "ci_low": fit["ci_low"],
        "ci_high": fit["ci_high"],
        "slope_log2_duration": {
            "beta": fit["beta"][j],
            "ci": [fit["ci_low"][j], fit["ci_high"][j]],
            "se": fit["se"][j],
            "mde_80_power": 2.8 * fit["se"][j],
        },
    }


def trigger_rows(S, sets, ranks, lines, qrels, sub) -> list[dict]:
    """FS.s6_rows plus k, the S4 top score, the contested flag and a seeded uniform draw."""
    s4 = sub["pool"] if sub["pool"] != "S0" else "S4@10"
    contested = set(sub["twin_both"]["twin"]) | set(sub["nested_sample"]["natural"])
    ids = {n: [r["query_id"] for r in rs] for n, rs in sets.items()}
    rows = FS.s6_rows(ranks, lines, qrels, ids, s4)
    by_id = {r["query_id"]: r for rs in sets.values() for r in rs}
    for r in rows:
        r["k"] = by_id[r["query_id"]]["k"]
        r["s4_top"] = float(ranks[r["set"]][s4][r["query_id"]][0][1])
        r["contested"] = r["query_id"] in contested
        r["u"] = random.Random(f"trigger|{r['query_id']}").random()
        a4, a5 = ranks[r["set"]][s4][r["query_id"]], ranks[r["set"]]["S5"][r["query_id"]]
        r["top1_changed"] = a4[0][0] != a5[0][0]
    return rows


def apply_rule(rows: list[dict], fire, n_boot: int) -> dict:
    """Share sent, nDCG@10 of 'S5 when the rule fires, else S4', gain kept, top-1 changes caught."""
    n = len(rows)
    trig = [bool(fire(r)) for r in rows]
    src = [r["source"] for r in rows]
    q = [r["q5"] if f else r["q4"] for r, f in zip(rows, trig, strict=True)]
    q4, q5 = float(np.mean([r["q4"] for r in rows])), float(np.mean([r["q5"] for r in rows]))
    d = [qq - r["q4"] for qq, r in zip(q, rows, strict=True)]
    d5 = [r["q5"] - r["q4"] for r in rows]
    changed = [r["top1_changed"] for r in rows]
    cost = [r["c_base"] + (r["c5"] if f else 0.0) for r, f in zip(rows, trig, strict=True)]
    return {
        "n": n,
        "share": sum(trig) / n,
        "share_ci": cluster_bootstrap_ci([float(f) for f in trig], src, n_boot),
        "S4": q4,
        "S5": q5,
        "ndcg10": float(np.mean(q)),
        "gain": float(np.mean(d)),
        "gain_ci": cluster_bootstrap_ci(d, src, n_boot),
        "kept": TS.kept(float(np.mean(q)), q4, q5),
        "kept_ci": TS.ratio_ci(d, d5, src, n_boot),
        "tokens_per_query": float(np.mean(cost)),
        "fired": sum(trig),
        "top1_changes": sum(changed),
        "top1_changes_caught": sum(c for c, f in zip(changed, trig, strict=True) if f),
        "top1_changes_missed": sum(c for c, f in zip(changed, trig, strict=True) if not f),
    }


def eval_trigger(S, sets, ranks, lines, qrels, sub, n_boot) -> dict:
    """C: the margin trigger at the dev threshold and the free baselines over every query with
    an S5 ranking, by set and pooled, contested as a subgroup; cross-fit operating points."""
    T = S.tier["trigger"]
    t_m, t_k, t_s = (
        float(T["margin_threshold"]),
        int(T["k_threshold"]),
        float(T["top_score_threshold"]),
    )
    rows = trigger_rows(S, sets, ranks, lines, qrels, sub)
    if not rows:
        return {"n": 0}
    margin_share = sum(r["s4_margin"] <= t_m for r in rows) / len(rows)
    rules = {
        f"margin <= {t_m:g}": lambda r: r["s4_margin"] <= t_m,
        f"k >= {t_k}": lambda r: r["k"] >= t_k,
        f"S4 top score < {t_s:g}": lambda r: r["s4_top"] < t_s,
        f"random, share {margin_share:.2f}": lambda r: r["u"] < margin_share,
        "always (S5)": lambda r: True,
    }
    groups = {"pooled": rows}
    groups |= {SET_LABEL[n]: [r for r in rows if r["set"] == n] for n in sets}
    groups["contested (near-miss, frame subset)"] = [
        r for r in rows if r["set"] == "twin" and r["contested"]
    ]
    groups["nested sample"] = [r for r in rows if r["set"] == "natural" and r["contested"]]
    groups["unselected"] = [r for r in rows if not r["contested"]]
    out: dict = {"n": len(rows), "thresholds": {"margin": t_m, "k": t_k, "top": t_s}, "groups": {}}
    for g, rs in groups.items():
        if len(rs) < 5:
            continue
        pts = TS.sweep(rs, "s4_margin", "le")
        cell = {
            "n": len(rs),
            "rules": {name: apply_rule(rs, fire, n_boot) for name, fire in rules.items()},
            "auc_margin": TS.auc_summary(pts),
            "crossfit_margin": {
                str(f): TS.crossfit(rs, "s4_margin", "le", float(f)) for f in T["fracs"]
            },
        }
        out["groups"][g] = cell
    # send rates of the rules over every query (S5 not needed): the full distribution
    s4 = sub["pool"] if sub["pool"] != "S0" else "S4@10"
    send = {}
    for name, rs in sets.items():
        rk = ranks[name][s4]
        m = np.array([TS.margin(rk[r["query_id"]]) for r in rs])
        top = np.array([float(rk[r["query_id"]][0][1]) for r in rs])
        k = np.array([r["k"] for r in rs])
        send[SET_LABEL[name]] = {
            "n": len(rs),
            f"margin <= {t_m:g}": float((m <= t_m).mean()),
            f"k >= {t_k}": float((k >= t_k).mean()),
            f"S4 top score < {t_s:g}": float((top < t_s).mean()),
        }
    out["send_rates_all_queries"] = send
    return out


def holm(ps: list[float]) -> list[float]:
    """Holm step-down adjusted p-values (as paper/robustness_checks.holm)."""
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for j, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - j) * ps[i]))
        adj[i] = run
    return adj


def holm_table(res: dict) -> list[dict]:
    """Holm over the primary contrasts (all-queries cells): the graded gaps per set and verifier,
    S3g vs S5 per frame subset, S5 vs S4 pooled."""
    items = []
    for name, cell in res["graded"]["sets"].items():
        for pair in (
            f"S4@50 - {res['graded']['systems'][3]}",
            f"{res['graded']['systems'][3]} - S2@50",
        ):
            d = cell["paired"].get(pair, {}).get("all")
            if d:
                items.append({"contrast": pair, "on": SET_LABEL[name], **d})
    for name, cell in res["graded"]["second"].items():
        for pair, d in cell["paired"].items():
            if d.get("all"):
                items.append(
                    {"contrast": pair, "on": f"{SET_LABEL[name]}, second subset", **d["all"]}
                )
    for label, cell in res["frames_graded"]["subsets"].items():
        d = cell["paired"].get("S3g - S5", {}).get("all")
        if d:
            items.append({"contrast": "S3g - S5", "on": label, **d})
    d = res["frames_all"].get("pooled", {}).get("S5 - S4", {}).get("all")
    if d:
        items.append({"contrast": "S5 - S4@50", "on": "all queries with S5", **d})
    adj = holm([it["p"] for it in items]) if items else []
    for it, a in zip(items, adj, strict=True):
        it["p_holm"] = a
    return items


def stage_eval(S, ctx, sets, args) -> dict:
    sub = FS.read_subsets(S)
    n_boot = S.n_boot
    ranks, lines = load_runs(S, sets, sub)
    for name, rs in sets.items():  # the test set's queries only (runs may cover more)
        keep = {r["query_id"] for r in rs}
        ranks[name] = {s: {q: v for q, v in r.items() if q in keep} for s, r in ranks[name].items()}
        lines[name] = {s: {q: v for q, v in r.items() if q in keep} for s, r in lines[name].items()}
    qrels_path = S.judge_dir / "qrels_expanded.jsonl"
    kinds = {"expanded": qrels_path, "strict": None} if qrels_path.exists() else {"strict": None}
    res: dict = {"pool": sub["pool"], "n_boot": n_boot, "labels": {}}
    for kind, path in kinds.items():
        qrels = {n: PV.load_qrels(path, rs) for n, rs in sets.items()}
        scores = {n: {s: PV.score_system(r, qrels[n]) for s, r in ranks[n].items()} for n in sets}
        cell = {
            "graded": eval_graded(S, sets, ranks, scores, sub, n_boot),
            "frames_graded": eval_frames_graded(S, sets, scores, sub, n_boot),
            "frames_all": eval_frames_all(S, sets, ranks, scores, sub, n_boot),
            "length": eval_length(S, sets, scores, sub, n_boot),
        }
        if kind == next(iter(kinds)):
            cell["trigger"] = eval_trigger(S, sets, ranks, lines, qrels, sub, n_boot)
        cell["holm"] = holm_table(cell)
        res["labels"][kind] = cell
    res["runs_present"] = {n: sorted(ranks[n]) for n in sets}
    S.tier1.mkdir(parents=True, exist_ok=True)
    (S.tier1 / "summary.json").write_text(json.dumps(res, indent=1, default=_json))
    (S.tier1 / "report.md").write_text(report(S, res, sets))
    print(f"wrote {S.tier1}/report.md and {S.tier1}/summary.json")
    return res


def _json(o):
    if isinstance(o, float) and math.isnan(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, set):
        return sorted(o)
    return str(o)


# ---- report -------------------------------------------------------------------------------------


def f3(v) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v:.3f}"


def ci(d: dict | None) -> str:
    if not d:
        return "-"
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}] (n {d['n']})"


def pct(v) -> str:
    return "-" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{100 * v:.1f}%"


def means_grid(means: dict, systems: list[str]) -> list[str]:
    L = ["| system | n | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|---|"]
    for s in systems:
        if s in means:
            m = means[s]
            L.append(
                f"| {s} | {m['n']} | "
                + " | ".join(f3(m[g]) for g in ("all", *[f"k={k}" for k in KS]))
                + " |"
            )
    return [*L, ""]


def paired_grid(paired: dict) -> list[str]:
    L = ["| contrast | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
    for pair, by in paired.items():
        L.append(
            f"| {pair} | "
            + " | ".join(ci(by.get(g)) for g in ("all", *[f"k={k}" for k in KS]))
            + " |"
        )
    return [*L, ""]


def report(S, res, sets) -> str:
    L = [
        "# Tier 1: graded verifiers, S5 on all queries, trigger over the full distribution",
        "",
        "Generated by pipeline/run_graded_systems.py --stage eval. Systems present: "
        + "; ".join(f"{SET_LABEL[n]} {v}" for n, v in res["runs_present"].items())
        + f". Pool for the frame systems: {res['pool']}'s top {S.top}. CIs: 95% paired cluster "
        f"bootstrap over source videos, {res['n_boot']:,} resamples; TOST uses the 90% CI.",
        "",
        (
            "Cells that need a run not made yet are empty or missing; the eval runs on whatever is in "
            "results/full/runs."
        ),
        "",
    ]
    for kind, cell in res["labels"].items():
        L += [f"# {kind} labels", ""]
        g = cell["graded"]
        L += [
            f"## A1: graded per-constraint verifier {S.S2G} vs S4 (same 0-10 scale) and S2 (same framing)",
            "",
        ]
        for name, c in g["sets"].items():
            L += [f"### {SET_LABEL[name]}: nDCG@10 by k", "", *means_grid(c["means"], g["systems"])]
            L += [f"### {SET_LABEL[name]}: paired differences", "", *paired_grid(c["paired"])]
        if g["second"]:
            L += ["## A3: second verifier on its 150-query subsets", ""]
            for name, c in g["second"].items():
                L += [
                    f"### {SET_LABEL[name]} ({c['model']})",
                    "",
                    *means_grid(c["means"], list(c["means"])),
                    *paired_grid(c["paired"]),
                ]
        else:
            L += ["## A3: second verifier: not run yet", ""]
        fg = cell["frames_graded"]
        L += [f"## A2: S3g vs S5 on the frame subsets (TOST margin +-{fg['tost_margin']})", ""]
        for label, c in fg["subsets"].items():
            L += [
                f"### {label} (n {c['n']})",
                "",
                *means_grid(c["means"], fg["systems"]),
                *paired_grid(c["paired"]),
            ]
            if c.get("tost"):
                t = c["tost"]
                L += [
                    (
                        f"TOST S3g - S5: 90% CI [{t['ci90'][0]:+.3f}, {t['ci90'][1]:+.3f}] within "
                        f"+-{t['margin']}: {'equivalent' if t['equivalent'] else 'not shown equivalent'}."
                    ),
                    "",
                ]
        fa = cell["frames_all"]
        L += [
            f"## B: S5 vs {fa['pool']} on every query with an S5 ranking (coverage {fa['coverage']})",
            "",
        ]
        for name, groups in fa["sets"].items():
            L += [
                f"### {SET_LABEL[name]}",
                "",
                "| group | n | S4 | S5 | S5 - S4 (all) | k=1 | k=2 | k=3 | k=4 |",
                "|---|---|---|---|---|---|---|---|---|",
            ]
            for gname, c in groups.items():
                d = c["S5 - S4"]
                L.append(
                    f"| {gname} | {c['n']} | {f3(c['S4']['all'])} | {f3(c['S5']['all'])} | {ci(d['all'])} | "
                    + " | ".join(ci(d.get(f"k={k}")) for k in KS)
                    + " |"
                )
            L.append("")
        if fa.get("pooled"):
            L += [f"Pooled over sets: S5 - S4 {ci(fa['pooled']['S5 - S4']['all'])}.", ""]
        ln = cell["length"]
        L += [
            "## B / W8: length regression of S5 - S4 on log2 duration (Vript, every query with S5)",
            "",
        ]
        if ln.get("slope_log2_duration"):
            s = ln["slope_log2_duration"]
            L += [
                (
                    f"n {ln['n']} queries ({ln['clusters']} sources); slope per doubling of duration "
                    f"{s['beta']:+.4f} [{s['ci'][0]:+.4f}, {s['ci'][1]:+.4f}]; bootstrap SE {s['se']:.4f}, "
                    f"minimal detectable slope at 80% power {s['mde_80_power']:.4f}. Covariates: {ln['terms']}."
                ),
                "",
            ]
        else:
            L += [f"too few queries (n {ln.get('n')})", ""]
        if "trigger" in cell:
            tr = cell["trigger"]
            L += [
                f"## C: trigger at the dev threshold and free baselines ({kind} labels; n {tr['n']} queries with S5)",
                "",
            ]
            L += [
                "Send rates over every query (no S5 needed):",
                "",
                "| set | n | "
                + " | ".join(k for k in list(next(iter(tr["send_rates_all_queries"].values())))[1:])
                + " |",
                "|---|---|---|---|---|",
            ]
            for name, c in tr["send_rates_all_queries"].items():
                L.append(
                    f"| {name} | {c['n']} | "
                    + " | ".join(pct(v) for k, v in c.items() if k != "n")
                    + " |"
                )
            L.append("")
            for gname, c in tr["groups"].items():
                L += [
                    f"### {gname} (n {c['n']}; AUC of the margin sweep {f3(c['auc_margin'].get('auc_share'))})",
                    "",
                    "| rule | sent | nDCG@10 | gain over S4 | kept | top-1 changes caught / missed | tokens/query |",
                    "|---|---|---|---|---|---|---|",
                ]
                for rname, r in c["rules"].items():
                    L.append(
                        f"| {rname} | {pct(r['share'])} [{pct(r['share_ci'][0])}, {pct(r['share_ci'][1])}] | "
                        f"{f3(r['ndcg10'])} | {r['gain']:+.3f} [{r['gain_ci'][0]:+.3f}, {r['gain_ci'][1]:+.3f}] | "
                        f"{pct(r['kept'])} [{pct(r['kept_ci'][0])}, {pct(r['kept_ci'][1])}] | "
                        f"{r['top1_changes_caught']} / {r['top1_changes_missed']} | {r['tokens_per_query']:,.0f} |"
                    )
                L.append("")
                L += [
                    "Cross-fit margin threshold (5 folds over sources, 20 repetitions; mean [2.5%, 97.5%]):",
                    "",
                ]
                for f, cf in c["crossfit_margin"].items():
                    L.append(
                        f"- keep >= {f}: sent {pct(cf['share'][0])} [{pct(cf['share'][1])}, {pct(cf['share'][2])}], kept {pct(cf['kept'][0])} [{pct(cf['kept'][1])}, {pct(cf['kept'][2])}]"
                    )
                L.append("")
        if cell["holm"]:
            L += [
                "## Holm over the primary contrasts",
                "",
                "| contrast | on | n | mean [95% CI] | p | p (Holm) |",
                "|---|---|---|---|---|---|",
            ]
            for it in cell["holm"]:
                L.append(
                    f"| {it['contrast']} | {it['on']} | {it['n']} | {it['mean']:+.3f} [{it['ci_low']:+.3f}, {it['ci_high']:+.3f}] | {it['p']:.4f} | {it['p_holm']:.4f} |"
                )
            L.append("")
    return "\n".join(L)


# ---- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--stage",
        required=True,
        choices=(
            "desc-graded",
            "second-graded",
            "frames-graded",
            "frames-all",
            "fill-plan",
            "eval",
        ),
    )
    ap.add_argument("--config", help="full-run config (default configs/mc_full.yaml)")
    ap.add_argument("--tier1-config", help="default configs/mc_tier1.yaml")
    ap.add_argument("--out")
    ap.add_argument("--project-only", action="store_true")
    ap.add_argument(
        "--dump-prompts", help="desc stages: write the uncached prompts (JSONL) and stop"
    )
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--canary", type=int, default=10)
    ap.add_argument("--canary-frames", type=int, default=2)
    ap.add_argument("--rate-tol", type=float, default=0.02)
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--canary-always", action="store_true")
    ap.add_argument("--canary-only", action="store_true", help="desc stages: stop after the canary")
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--wait", type=float, default=60.0)
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--all-sources", action="store_true", help="ignore sources_file: every source")
    args = ap.parse_args(argv)
    S = settings(args)
    ctx = FS.load_context(S, need_index=False)
    S.all_sets = FS.load_sets(S, ctx)  # every query: merge-writes need rows for saved queries too
    sets = draw_sets(S, S.all_sets)
    {
        "desc-graded": stage_desc_graded,
        "second-graded": stage_second_graded,
        "frames-graded": stage_frames_graded,
        "frames-all": stage_frames_all,
        "fill-plan": stage_fill_plan,
        "eval": stage_eval,
    }[args.stage](S, ctx, sets, args)


if __name__ == "__main__":
    main()
