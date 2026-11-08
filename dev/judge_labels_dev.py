"""Pooled frame judge over the v2 queries (docs/plan.md 4.5). PAID: two vision models.

Reads the natural and twin queries of the config's out_dir (queries.jsonl, queries_twin.jsonl)
and the system rankings (one jsonl per system, lines {"query_id", "ranking": [[doc, score],
...]}; by default every *.jsonl in judge.rankings_dir, or --rankings S0=path S4=path ...). Per
source, the pool is the union of every system's top 10 over all its queries, plus the twin and
the source; one call per (source, candidate) and judge model (Gemini, gpt-6-astra) covers all of
the source's facts, on the candidate's index frames (at most 16, uniform, 512 px;
`src.judge_frames`). Extras reached only at k >= 3 are judged on a seeded 25% sample.

Writes to judge.out_dir (data/mc/smoke_v2/judge/): judged.jsonl (per source and candidate, per
fact, both models' answers), qrels_strict.jsonl, qrels_expanded.jsonl, judge_stats.json
(agreement per fact type and per k, verdict pairs, false-negative estimate for the unjudged
extras, sanity checks, pool sizes), judge_report.md and manifest.json.

    uv run --env-file .env python dev/judge_labels_dev.py --estimate
    uv run --env-file .env python dev/judge_labels_dev.py --max-input-tokens 30000000

Fill mode (`--fill-queries FILE`, JSON {set: [query ids]} or a list): judges only the candidates
in those queries' top 10 (of the given rankings) that have no line in judged.jsonl yet, whatever
the extras sample said, appends them (marked "fill") and rewrites qrels_strict / qrels_expanded
from the whole judged.jsonl; judge_stats.json and judge_report.md are left as they were, and the
fill's counts and cost go to fill_<file stem>.json.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from concurrent.futures import ThreadPoolExecutor
from functools import cache
from pathlib import Path

import yaml

from src import judge_frames as jf
from src.datasets.mc import Video, load_videos, read_jsonl, split_global_id
from src.frames import FramePolicy
from src.grounding_v2 import FrameStore, Grounder
from src.types import Cost


def load_build_v2():
    path = Path(__file__).resolve().parent / "build_queries_dev.py"
    spec = importlib.util.spec_from_file_location("build_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@cache
def videos_of(root: str, dataset: str) -> dict[str, Video]:
    return {v.video_id: v for v in load_videos(Path(root) / dataset)}


def video(root: str, gid: str) -> tuple[str, Video]:
    dataset, video_id = split_global_id(gid)
    return dataset, videos_of(root, dataset)[video_id]


def read_rankings(paths: dict[str, Path]) -> dict[str, dict[str, list[str]]]:
    return {
        system: {line["query_id"]: [d for d, _ in line["ranking"]] for line in read_jsonl(path)}
        for system, path in paths.items()
    }


def ranking_paths(cfg: dict, given: list[str] | None) -> dict[str, Path]:
    if given:
        return {s: Path(p) for s, p in (item.split("=", 1) for item in given)}
    folder = Path(cfg.get("judge", {}).get("rankings_dir", "results/smoke_v2/runs"))
    return {p.stem: p for p in sorted(folder.glob("*.jsonl"))}


def read_queries(out_dir: Path) -> list[dict]:
    rows = []
    for name in ("queries.jsonl", "queries_twin.jsonl"):
        if (out_dir / name).exists():
            rows += read_jsonl(out_dir / name)
    return rows


def judge_tasks(rows, rankings, jcfg):
    pools = jf.build_pools(rows, rankings, jcfg.get("top", jf.TOP))
    plan = jf.plan_judging(pools, jcfg.get("extra_share", jf.EXTRA_SHARE), jcfg.get("full_k", 2))
    tasks = [(s, d) for s, cands in plan.items() for d, ok in cands.items() if ok]
    return pools, plan, tasks


def read_judged(path: Path) -> dict[str, dict[str, dict]]:
    judged: dict[str, dict[str, dict]] = {}
    if path.exists():
        for line in read_jsonl(path):
            judged.setdefault(line["source"], {})[line["candidate"]] = line["answers"]
    return judged


def fill_query_ids(path: Path) -> list[str]:
    given = json.loads(Path(path).read_text())
    ids = given if isinstance(given, list) else [q for v in given.values() for q in v]
    return sorted(set(ids))


def fill_tasks(rows, rankings, ids, judged, top=jf.TOP) -> tuple[list[tuple[str, str]], dict]:
    """(source, candidate) pairs in the top `top` of any ranking for the given query ids that
    are not judged yet (each once, in query order), and per-query counts {query_id: n new}."""
    by_id = {r["query_id"]: r for r in rows}
    tasks, per_query, seen = [], {}, set()
    for q in ids:
        row = by_id[q]
        source = row["source"]
        new = [
            d
            for d in jf.judged_in_top(row, rankings, top)
            if d not in judged.get(source, {}) and d != row.get("twin")
        ]
        per_query[q] = len(new)
        for d in new:
            if (source, d) not in seen:
                seen.add((source, d))
                tasks.append((source, d))
    return tasks, per_query


def report(st: dict) -> str:
    lines = ["# Pooled frame judge (two models)", ""]
    lines += [f"Pools: {st['pools']}", f"Sanity: {st['sanity']}", ""]
    lines += ["## Agreement per fact type", ""]
    lines += ["| type | n | raw agreement | kappa | kappa (yes vs not) |", "|---|---|---|---|---|"]
    for t, s in st["agreement"]["facts"].items():
        kap = "-" if s["kappa"] is None else f"{s['kappa']:.2f}"
        kyes = "-" if s["kappa_yes"] is None else f"{s['kappa_yes']:.2f}"
        lines.append(f"| {t} | {s['n']} | {s['raw_agreement']:.2f} | {kap} | {kyes} |")
    lines += ["", "## Relevance decisions per k", ""]
    lines += ["| k | n | raw agreement | kappa | relevant A | relevant B | both |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for k, s in st["agreement"]["relevance"].items():
        kap = "-" if s["kappa"] is None else f"{s['kappa']:.2f}"
        lines.append(
            f"| {k} | {s['n']} | {s['raw_agreement']:.2f} | {kap} | {s['relevant_a']} | "
            f"{s['relevant_b']} | {s['relevant_both']} |"
        )
    lines += ["", f"Verdict pairs (A|B): {st['agreement']['verdicts']}", ""]
    lines += ["## Unjudged extras (k >= 3), estimated false negatives", ""]
    lines += ["| k | queries | sampled | relevant | rate | unjudged | est. missed / query |"]
    lines += ["|---|---|---|---|---|---|---|"]
    for k, c in st["false_negatives"].items():
        rate = "-" if c["rate"] is None else f"{c['rate']:.3f}"
        per = "-" if c["est_missed_per_query"] is None else f"{c['est_missed_per_query']:.3f}"
        lines.append(
            f"| {k} | {c['queries']} | {c['judged']} | {c['relevant']} | {rate} | "
            f"{c['unjudged']} | {per} |"
        )
    rule = (
        "Disagreement rule: a candidate is relevant only if both models say yes to every fact; "
        "any disagreement counts as not relevant."
    )
    lines += ["", rule, ""]
    return "\n".join(lines)


def merge_model_results(tasks, specs, results: dict) -> list[tuple]:
    """Per task, the answers and costs of each model's separate call merged into one
    (source, doc, answers, cost, n_frames) tuple, as the single-pool path returns it.
    results: {(task index, spec): (answers, cost, n_frames)}."""
    done = []
    for i, (source, doc) in enumerate(tasks):
        answers: dict[str, dict] = {}
        cost: dict[str, Cost] = {}
        n_frames = 0
        for spec in specs:
            a, c, n_frames = results[(i, spec)]
            for key, by_model in a.items():
                answers.setdefault(key, {}).update(by_model)
            cost.update(c)
        done.append((source, doc, answers, cost, n_frames))
    return done


def run_split(tasks, specs, store, grounder, facts_of, root, cap, side, args) -> list[tuple]:
    """Each (task, model) call in its model's own pool: --gemini-workers for Gemini,
    --openai-workers for OpenAI (default --workers), so a high OpenAI rate limit is not held
    back by Gemini's. Same calls and cache keys as the single-pool path."""
    frames_cache: dict[str, tuple] = {}

    def frames_of(doc):
        if doc not in frames_cache:
            dataset, v = video(root, doc)
            frames, times = store.index_frames(dataset, v)
            frames_cache[doc] = jf.judge_frames(frames, times, cap, side)
        return frames_cache[doc]

    def call(item):
        i, spec = item
        source, doc = tasks[i]
        frames, times = frames_of(doc)
        answers, cost = grounder.on_frames(frames, times, facts_of[source], specs=(spec,))
        return (i, spec), (answers, cost, len(frames))

    pools = {
        spec: ThreadPoolExecutor(
            (args.gemini_workers if spec.startswith("gemini") else args.openai_workers)
            or args.workers
        )
        for spec in specs
    }
    for _, doc in tasks:  # load frames once, before the pools race for them
        frames_of(doc)
    futures = [pools[spec].submit(call, (i, spec)) for i in range(len(tasks)) for spec in specs]
    results = dict(f.result() for f in futures)
    for pool in pools.values():
        pool.shutdown()
    return merge_model_results(tasks, specs, results)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_smoke_v2.yaml")
    parser.add_argument("--rankings", nargs="*", help="SYSTEM=path.jsonl ...")
    parser.add_argument("--limit", type=int, default=None, help="first N sources only")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--gemini-workers", type=int, default=0, help="separate pool for Gemini calls"
    )
    parser.add_argument(
        "--openai-workers", type=int, default=0, help="separate pool for OpenAI calls"
    )
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--max-input-tokens", type=float, default=None)
    parser.add_argument(
        "--fill-queries", help="JSON of query ids: judge only their unjudged top-10 candidates"
    )
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    jcfg = cfg.get("judge", {})
    root = cfg["data_root"]
    b2 = load_build_v2()

    rows = read_queries(Path(cfg["out_dir"]))
    sources = list(dict.fromkeys(r["source"] for r in rows))[: args.limit]
    rows = [r for r in rows if r["source"] in set(sources)]
    rankings = read_rankings(ranking_paths(cfg, args.rankings))
    print(f"{len(rows)} queries from {len(sources)} sources; systems {sorted(rankings)}")
    pools, plan, tasks = judge_tasks(rows, rankings, jcfg)
    out = Path(jcfg.get("out_dir", Path(cfg["out_dir"]) / "judge"))
    if args.fill_queries:
        old = read_judged(out / "judged.jsonl")
        ids = fill_query_ids(Path(args.fill_queries))
        tasks, per_query = fill_tasks(rows, rankings, ids, old, jcfg.get("top", jf.TOP))
        print(
            f"fill: {len(tasks)} unjudged (source, candidate) pairs over "
            f"{sum(1 for n in per_query.values() if n)} of {len(ids)} queries"
        )
    facts_of, of_query = {}, {}
    for s in sources:
        facts_of[s], oq = jf.source_facts([r for r in rows if r["source"] == s])
        of_query.update(oq)

    specs = tuple(jcfg.get("models", jf.JUDGES))
    store = FrameStore(
        root, index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8))
    )
    cap, side = jcfg.get("max_frames", jf.MAX_FRAMES), jcfg.get("max_side", jf.MAX_SIDE)
    template = jf.JUDGE_PROMPT.read_text()
    media = jcfg.get("media_resolution", cfg.get("grounding_media_resolution"))
    projected = 0.0
    for s, d in tasks:
        dataset, v = video(root, d)
        n = min(cap, len(store.index_times(dataset, v)))
        for spec in specs:
            projected += b2.text_tokens(template) + 15 * len(facts_of[s])
            projected += b2.image_tokens(spec, n, media) if spec.startswith("gemini") else n * 260
    print(
        f"{len(tasks)} (source, candidate) judgments x {len(specs)} models; projected input "
        f"{projected / 1e6:.2f}M tokens if none is cached"
    )
    if args.estimate:
        return
    max_tokens = args.max_input_tokens or jcfg.get("max_input_tokens")
    if max_tokens is not None and projected > max_tokens:
        raise SystemExit(f"projected {projected:,.0f} > --max-input-tokens {max_tokens:,.0f}")
    budget = b2.Budget(max_tokens)
    models = {s: b2.GuardedVision(b2.make_grounder_model(s, media), budget, s) for s in specs}
    grounder = Grounder(models, template, store)

    def one(task):
        source, doc = task
        dataset, v = video(root, doc)
        frames, times = store.index_frames(dataset, v)
        frames, times = jf.judge_frames(frames, times, cap, side)
        answers, cost = grounder.on_frames(frames, times, facts_of[source])
        return source, doc, answers, cost, len(frames)

    if args.gemini_workers or args.openai_workers:
        done = run_split(tasks, specs, store, grounder, facts_of, root, cap, side, args)
    else:
        with ThreadPoolExecutor(args.workers) as ex:
            done = list(ex.map(one, tasks))
    judged: dict[str, dict[str, dict]] = (
        read_judged(out / "judged.jsonl") if args.fill_queries else {}
    )
    total: dict[str, Cost] = {}
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "judged.jsonl", "a" if args.fill_queries else "w") as handle:
        for source, doc, answers, cost, n_frames in done:
            judged.setdefault(source, {})[doc] = answers
            for spec, c in cost.items():
                total[spec] = total.get(spec, Cost()) + c
            entry = pools[source][doc]
            handle.write(
                json.dumps(
                    {
                        "source": source,
                        "candidate": doc,
                        **entry,
                        **({"fill": Path(args.fill_queries).stem} if args.fill_queries else {}),
                        "n_frames": n_frames,
                        "facts": facts_of[source],
                        "answers": answers,
                        "cost": {s: c.to_dict() for s, c in cost.items()},
                    }
                )
                + "\n"
            )
    strict, expanded = jf.qrels(rows, of_query, judged, specs)
    for name, table in (("qrels_strict", strict), ("qrels_expanded", expanded)):
        with open(out / f"{name}.jsonl", "w") as handle:
            handle.writelines(
                json.dumps({"query_id": q, "relevant": rel}) + "\n" for q, rel in table.items()
            )
    if args.fill_queries:
        new_pos = {
            q: [d for d in expanded[q] if d in {doc for s, doc in tasks if s == r["source"]}]
            for q, r in ((r["query_id"], r) for r in rows)
        }
        fill = {
            "queries": len(per_query),
            "queries_with_new": sum(1 for n in per_query.values() if n),
            "judged": len(tasks),
            "new_positive_pairs": sum(len(v) for v in new_pos.values()),
            "new_positive_pairs_in_fill_queries": sum(len(new_pos[q]) for q in per_query),
            "rankings": sorted(rankings),
            "cost": {s: c.to_dict() for s, c in total.items()},
        }
        (out / f"fill_{Path(args.fill_queries).stem}.json").write_text(json.dumps(fill, indent=1))
        print(json.dumps(fill, indent=1))
        return
    st = {
        "pools": {
            "sources": len(pools),
            "candidates": sum(len(p) for p in pools.values()),
            "judged": len(tasks),
            "systems": sorted(rankings),
        },
        "agreement": jf.agreement_stats(rows, facts_of, of_query, judged, specs[:2]),
        "false_negatives": jf.false_negative_estimate(
            rows,
            of_query,
            rankings,
            pools,
            plan,
            judged,
            specs,
            jcfg.get("full_k", 2),
            jcfg.get("top", jf.TOP),
        ),
        "sanity": jf.sanity(rows, of_query, judged, specs),
        "extra_positives": sum(len(v) - 1 for v in expanded.values()),
        "cost": {s: c.to_dict() for s, c in total.items()},
    }
    (out / "judge_stats.json").write_text(json.dumps(st, indent=1))
    (out / "judge_report.md").write_text(report(st))
    from src.manifest import write_manifest

    write_manifest(
        out,
        models={"judge": list(specs), "judge_media_resolution": media},
        cost=sum(total.values(), Cost()),
        extra={"config": args.config, "rankings": sorted(rankings), "prompt": jf.JUDGE_PROMPT.name},
    )
    print(report(st))


if __name__ == "__main__":
    main()
