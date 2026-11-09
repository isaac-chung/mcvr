"""Full run G5: pooled frame judge, and the judge prompt-fix test (decision 6). PAID.

Thin layer over dev/judge_labels_dev.py and src.judge_frames (reused unchanged): per
query source, the pool is the union of every system's top 10 over all its queries
(results/full/runs/<set>/S*.jsonl: S0, S1, S2@50, S4@50, S3, S5, ...) plus the twin and the
source; one call per (source, candidate) and judge model (Gemini LOW + gpt-6-astra, both must
say yes to every fact of a query) on the candidate's index frames (at most 16, 512 px); extras
reached only at k >= 3 are judged on a seeded 25% sample. New here: config paths, resume
(pairs already in judged.jsonl are skipped), a spend ledger with the `judge` cap of
configs/mc_full.yaml, a 10-pair canary, retry passes for failed calls (rate limits), separate
Gemini / OpenAI pools, --prompt, and the prompt-fix test.

    uv run --env-file .env python pipeline/judge_labels.py --stage run --project-only
    uv run --env-file .env python pipeline/judge_labels.py --stage run
    uv run --env-file .env python pipeline/judge_labels.py --stage run \
        --fill-queries results/full/select/frames_subset.json      # all top-10 pairs, no sample
    uv run --env-file .env python pipeline/judge_labels.py --stage run \
        --fill-queries <ids.json> --extra-runs QW@50=<run.jsonl> ...  # fill for runs outside runs/
    uv run python pipeline/judge_labels.py --stage stats                # qrels + stats, no calls
    uv run --env-file .env python pipeline/judge_labels.py --stage prompt-test \
        --prompt prompts/judge_frames_v2.txt                         # decision 6, smoke pairs

Outputs (judge.out_dir, data/mc/full/judge/): judged.jsonl, qrels_strict.jsonl,
qrels_expanded.jsonl, judge_stats.json, judge_report.md, manifest.json; spend in
results/full/judge/spend.json. --smoke reads results/full_smoke/runs and the smoke labels
(data/mc/smoke_v2/judge + deep_pool judged.jsonl count as judged) and writes to
results/full_smoke/judge/.

prompt-test: the 100 smoke sources' (source, source) pairs, their (source, twin) pairs and a
seeded sample of 100 other judged pool pairs, judged by both models with --prompt; compared
with the judge_frames_v1 answers on the same pairs (data/mc/smoke_v2/judge/judged.jsonl) and the
run-to-run band (base and repeat R of results/smoke_v2/judge_noise). Adopt only if source
acceptance passes the band's top and fact kappa does not fall. Writes
results/full/judge_prompt_test/{report.md, summary.json, answers.jsonl, spend.json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path

from src import judge_frames as jf
from src.datasets.mc import read_jsonl
from src.frames import FramePolicy
from src.grounding_v2 import FrameStore, Grounder, agreement_table, fill
from src.manifest import write_manifest
from src.types import Cost

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FS = _load(ROOT / "pipeline/run_systems.py", "_full_systems")
JP = _load(ROOT / "dev/judge_labels_dev.py", "_full_judge_pool")
B2 = JP.load_build_v2()
FS.CAP_OF.update({"judge": "judge", "judge_prompt_test": "judge"})
FS.STAGES_OF_CAP["judge"] = ["judge", "judge_prompt_test"]

SMOKE_DATA = Path("data/mc/smoke_v2")
SMOKE_JUDGED = (SMOKE_DATA / "judge/judged.jsonl", SMOKE_DATA / "deep_pool/judged.jsonl")
NOISE = Path("results/smoke_v2/judge_noise/summary.json")


# ---- inputs ------------------------------------------------------------------------------------


def setup(S):
    cfg = S.cfg
    jcfg = cfg["judge"]
    rows = JP.read_queries(Path(cfg["out_dir"]))
    facts_of, of_query = {}, {}
    by_source: dict[str, list] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    for s, rs in by_source.items():
        facts_of[s], oq = jf.source_facts(rs)
        of_query.update(oq)
    out = S.out / "judge" if S.smoke else Path(jcfg["out_dir"])
    return jcfg, rows, facts_of, of_query, out


def rankings(S) -> dict[str, dict[str, list[str]]]:
    """{system: {query_id: doc ids}} over every runs/<set>/S*.jsonl (sets merged)."""
    out: dict[str, dict[str, list[str]]] = {}
    for path in sorted(S.runs.glob("*/S*.jsonl")):
        for line in read_jsonl(path):
            out.setdefault(path.stem, {})[line["query_id"]] = [d for d, _ in line["ranking"]]
    return out


def judged_union(paths) -> dict[str, dict[str, dict]]:
    judged: dict[str, dict[str, dict]] = {}
    for p in paths:
        for source, cands in JP.read_judged(Path(p)).items():
            for d, answers in cands.items():
                judged.setdefault(source, {}).setdefault(d, answers)
    return judged


class Judge:
    """Both judge models (cached, guarded by a hard per-call budget) over the index frames."""

    def __init__(self, cfg, jcfg, template: str, budget_tokens: float, paid: list):
        self.specs = tuple(jcfg["models"])
        self.media = jcfg.get("media_resolution", cfg.get("grounding_media_resolution"))
        self.cap, self.side = (
            jcfg.get("max_frames", jf.MAX_FRAMES),
            jcfg.get("max_side", jf.MAX_SIDE),
        )
        self.root, self.template = cfg["data_root"], template
        self.store = FrameStore(
            self.root,
            index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8)),
        )
        self.budget = B2.Budget(budget_tokens)
        self.models = {}
        for spec in self.specs:
            cached = B2.make_grounder_model(spec, self.media)
            if spec.startswith("openai"):
                cached.inner.client = cached.inner.client.with_options(max_retries=10, timeout=300)
            cached.inner = FS.PV.Metered(cached.inner, paid)
            self.models[spec] = B2.GuardedVision(cached, self.budget, spec)
        self.grounder = Grounder(self.models, template, self.store)
        self.frames = lru_cache(maxsize=256)(self._frames)

    def _frames(self, doc: str):
        dataset, v = JP.video(self.root, doc)
        frames, times = self.store.index_frames(dataset, v)
        return jf.judge_frames(frames, times, self.cap, self.side)

    def projection(self, tasks, facts_of, check_cache: bool = False) -> tuple[float, int]:
        """Projected new input tokens (and calls) of the tasks; with check_cache, calls already
        in the vlm cache count as free (loads every candidate's frames)."""
        tokens, calls = 0.0, 0
        for s, d in tasks:
            dataset, v = JP.video(self.root, d)
            n = min(self.cap, len(self.store.index_times(dataset, v)))
            for spec in self.specs:
                if check_cache:
                    frames, times = self.frames(d)
                    from src.grounding import frame_labels

                    prompt = fill(self.template, len(frames), facts_of[s])
                    inner = self.models[spec].inner
                    if inner.is_cached(prompt, frames, frame_labels(list(times))):
                        continue
                tokens += B2.text_tokens(self.template) + 15 * len(facts_of[s])
                tokens += (
                    B2.image_tokens(spec, n, self.media) if spec.startswith("gemini") else n * 260
                )
                calls += 1
        return tokens, calls

    def run(self, tasks, facts_of, workers: dict, errors: list) -> dict:
        """{(source, doc): (answers, cost, n_frames)} for the tasks whose calls all succeeded."""

        def call(item):
            i, spec = item
            source, doc = tasks[i]
            try:
                frames, times = self.frames(doc)
                answers, cost = self.grounder.on_frames(
                    frames, times, facts_of[source], specs=(spec,)
                )
            except Exception as error:  # noqa: BLE001 (logged; the pair is retried next pass)
                errors.append(f"{source} {doc} {spec}: {error!r}"[:300])
                return (i, spec), None
            return (i, spec), (answers, cost, len(frames))

        pools = {
            s: ThreadPoolExecutor(workers["gemini" if s.startswith("gemini") else "openai"])
            for s in self.specs
        }
        futures = [pools[s].submit(call, (i, s)) for i in range(len(tasks)) for s in self.specs]
        results = dict(f.result() for f in futures)
        for pool in pools.values():
            pool.shutdown()
        done = {}
        for i, task in enumerate(tasks):
            parts = [results[(i, s)] for s in self.specs]
            if any(p is None for p in parts):
                continue
            answers: dict = {}
            cost: dict = {}
            for a, c, _ in parts:
                for key, by_model in a.items():
                    answers.setdefault(key, {}).update(by_model)
                cost.update(c)
            done[task] = (answers, cost, parts[0][2])
        return done


def run_with_passes(judge: Judge, tasks, facts_of, args, label: str):
    done, errors = {}, []
    todo = list(tasks)
    for i in range(args.passes):
        errors.clear()
        done.update(
            judge.run(
                todo,
                facts_of,
                {"gemini": args.gemini_workers, "openai": args.openai_workers},
                errors,
            )
        )
        todo = [t for t in todo if t not in done]
        if not todo:
            break
        print(f"[{label}] pass {i + 1}: {len(todo)} pairs failed, e.g. {errors[:1]}")
        if i + 1 < args.passes:
            time.sleep(args.wait * (i + 1))
    return done, todo, errors


# ---- stage: run / fill --------------------------------------------------------------------------


def write_stats(S, jcfg, rows, facts_of, of_query, out: Path, rank, prompt: Path, specs) -> dict:
    judged = judged_union([out / "judged.jsonl", *(SMOKE_JUDGED if S.smoke else ())])
    pools, plan, _ = JP.judge_tasks(rows, rank, jcfg)
    strict, expanded = jf.qrels(rows, of_query, judged, specs)
    for name, table in (("qrels_strict", strict), ("qrels_expanded", expanded)):
        with open(out / f"{name}.jsonl", "w") as handle:
            handle.writelines(
                json.dumps({"query_id": q, "relevant": rel}) + "\n" for q, rel in table.items()
            )
    judged_rows = [r for r in rows if r["source"] in judged]
    st = {
        "prompt": prompt.name,
        "pools": {
            "sources": len(pools),
            "candidates": sum(len(p) for p in pools.values()),
            "planned": sum(sum(v.values()) for v in plan.values()),
            "systems": sorted(rank),
            "judged_pairs": sum(len(v) for v in judged.values()),
        },
        "agreement": jf.agreement_stats(judged_rows, facts_of, of_query, judged, specs[:2]),
        "false_negatives": jf.false_negative_estimate(
            judged_rows,
            of_query,
            rank,
            pools,
            plan,
            judged,
            specs,
            jcfg.get("full_k", 2),
            jcfg.get("top", jf.TOP),
        ),
        "sanity": jf.sanity(judged_rows, of_query, judged, specs),
        "extra_positives": sum(len(v) - 1 for v in expanded.values()),
    }
    (out / "judge_stats.json").write_text(json.dumps(st, indent=1))
    (out / "judge_report.md").write_text(JP.report(st))
    return st


def stage_run(S, args) -> None:
    jcfg, rows, facts_of, of_query, out = setup(S)
    out.mkdir(parents=True, exist_ok=True)
    prompt = Path(args.prompt or jcfg.get("prompt", jf.JUDGE_PROMPT))
    rank = rankings(S)
    if not rank:
        sys.exit(f"no rankings in {S.runs}: run pipeline/run_systems.py first")
    for spec in args.extra_runs:  # NAME=PATH, e.g. the local rerankers of cross_encoders.py
        name, path = spec.split("=", 1)
        for line in read_jsonl(Path(path)):
            rank.setdefault(name, {})[line["query_id"]] = [d for d, _ in line["ranking"]]
    judged = judged_union([out / "judged.jsonl", *(SMOKE_JUDGED if S.smoke else ())])
    pools, _plan, planned = JP.judge_tasks(rows, rank, jcfg)
    if args.fill_queries:
        tasks, _per_query = JP.fill_tasks(
            rows, rank, JP.fill_query_ids(Path(args.fill_queries)), judged, jcfg.get("top", jf.TOP)
        )
        why = "fill:" + Path(args.fill_queries).stem
    else:
        tasks = [t for t in planned if t[1] not in judged.get(t[0], {})]
        why = "pool"
    print(
        f"{len(rows)} queries; systems {sorted(rank)}; planned pairs {len(planned)}, "
        f"to judge {len(tasks)} ({why}); prompt {prompt.name}"
    )
    ledger = FS.Ledger(S, "judge")
    paid: list = []
    judge = Judge(S.cfg, jcfg, prompt.read_text(), max(0.0, ledger.cap - ledger.total), paid)
    proj, calls = judge.projection(tasks, facts_of, args.check_cache)
    print(f"{calls} calls to pay")
    ledger.check("judge", proj)
    if args.project_only or not tasks:
        if not args.project_only:
            write_stats(S, jcfg, rows, facts_of, of_query, out, rank, prompt, judge.specs)
        return
    order = list(tasks)
    if calls and not args.no_canary:  # canary: 10 pairs, every fact answered by both models
        can = order[: args.canary]
        done, left, errors = run_with_passes(
            judge, can, facts_of, argparse.Namespace(**{**vars(args), "passes": 1}), "canary"
        )
        missing = sum(
            1
            for a, _, _ in done.values()
            for k in a.values()
            for s in judge.specs
            if k.get(s, {}).get("model_answer") in (None, "none")
        )
        spent = sum(paid, Cost()).tokens_in
        print(f"[judge] canary: {len(done)} pairs, {missing} unanswered facts, {spent:,} tokens in")
        if left or missing > len(can):
            sys.exit(
                f"[judge] canary failed: {len(left)} pairs failed ({errors[:1]}), {missing} unanswered"
            )
        append(out, done, facts_of, pools, prompt, why)
        order = order[args.canary :]
    done, left, errors = run_with_passes(judge, order, facts_of, args, "judge")
    append(out, done, facts_of, pools, prompt, why)
    (S.out / "judge").mkdir(parents=True, exist_ok=True)
    (S.out / "judge" / "errors.json").write_text(
        json.dumps({"left": left, "errors": errors[:200]}, indent=1)
    )
    cost = sum(paid, Cost())
    ledger.add(
        "judge",
        cost,
        len(paid),
        {"pairs": len(tasks) - len(left), "failed": len(left), "why": why, "prompt": prompt.name},
    )
    write_stats(S, jcfg, rows, facts_of, of_query, out, rank, prompt, judge.specs)
    write_manifest(
        out,
        models={"judge": list(judge.specs), "judge_media_resolution": judge.media},
        cost=cost,
        extra={"prompt": prompt.name, "rankings": sorted(rank), "smoke": S.smoke},
    )
    print(
        f"judged {len(tasks) - len(left)} pairs, {len(left)} failed (rerun to retry); wrote {out}"
    )


def append(out: Path, done: dict, facts_of, pools, prompt: Path, why: str) -> None:
    with open(out / "judged.jsonl", "a") as handle:
        for (source, doc), (answers, cost, n_frames) in done.items():
            entry = pools.get(source, {}).get(doc, {})
            handle.write(
                json.dumps(
                    {
                        "source": source,
                        "candidate": doc,
                        **entry,
                        "why_run": why,
                        "prompt": prompt.name,
                        "n_frames": n_frames,
                        "facts": facts_of[source],
                        "answers": answers,
                        "cost": {s: c.to_dict() for s, c in cost.items()},
                    }
                )
                + "\n"
            )


def stage_stats(S, args) -> None:
    jcfg, rows, facts_of, of_query, out = setup(S)
    prompt = Path(args.prompt or jcfg.get("prompt", jf.JUDGE_PROMPT))
    st = write_stats(
        S, jcfg, rows, facts_of, of_query, out, rankings(S), prompt, tuple(jcfg["models"])
    )
    print(JP.report(st))


# ---- stage: prompt-test (decision 6) ------------------------------------------------------------


def acceptance(rows, of_query, judged, specs) -> dict:
    """Source acceptance per query (all, by set, per model alone)."""
    out = {}
    for name, sp in (("both", specs), *((s.split(":")[0], (s,)) for s in specs)):
        vals = [
            jf.relevant(judged[r["source"]][r["source"]], of_query[r["query_id"]], sp)
            for r in rows
            if r["source"] in judged.get(r["source"], {})
        ]
        out[name] = {"n": len(vals), "rate": sum(vals) / len(vals) if vals else None}
        for set_ in ("natural", "twin"):
            v = [
                jf.relevant(judged[r["source"]][r["source"]], of_query[r["query_id"]], sp)
                for r in rows
                if r.get("set", "natural") == set_ and r["source"] in judged.get(r["source"], {})
            ]
            out[f"{name} {set_}"] = {"n": len(v), "rate": sum(v) / len(v) if v else None}
    return out


def fact_items(pairs, judged, facts_of) -> list[dict]:
    items = []
    for s, d in pairs:
        types = {f["key"]: f["type"] for f in facts_of[s]}
        for key, by_model in judged.get(s, {}).get(d, {}).items():
            items.append(
                {
                    "answers": by_model,
                    "type": types.get(key),
                    "pair": "source" if s == d else "other",
                }
            )
    return items


def relevance_pairs(rows, of_query, judged, pairs, specs) -> int:
    """(query, candidate) pairs judged relevant over the non-source test pairs."""
    want = {(s, d) for s, d in pairs if s != d}
    return sum(
        jf.relevant(judged[r["source"]][d], of_query[r["query_id"]], specs)
        for r in rows
        for d in judged.get(r["source"], {})
        if (r["source"], d) in want and d != r.get("twin")
    )


def stage_prompt_test(S, args) -> None:
    cfg = S.cfg_smoke
    jcfg = cfg["judge"]
    rows = JP.read_queries(SMOKE_DATA)
    facts_of, of_query = {}, {}
    for s in dict.fromkeys(r["source"] for r in rows):
        facts_of[s], oq = jf.source_facts([r for r in rows if r["source"] == s])
        of_query.update(oq)
    base = JP.read_judged(SMOKE_JUDGED[0])
    specs = tuple(jcfg["models"])
    sources = sorted(facts_of)
    twins = sorted({(r["source"], r["twin"]) for r in rows if r.get("twin")})
    others = sorted(
        (s, d) for s, c in base.items() for d in c if d != s and (s, d) not in set(twins)
    )
    random.Random("prompt-test|0").shuffle(others)
    pairs = [(s, s) for s in sources] + twins + sorted(others[: args.sample])
    pairs = [p for p in pairs if p[1] in base.get(p[0], {})]
    prompt = Path(args.prompt or "prompts/judge_frames_v2.txt")
    out = S.out / "judge_prompt_test"
    out.mkdir(parents=True, exist_ok=True)
    ledger = FS.Ledger(S, "judge_prompt_test")
    paid: list = []
    left_tokens = max(0.0, args.max_tokens - ledger.total)  # --max-tokens covers every run
    # the per-call guard projects astra images at build_v2's rate (about 2x the measured one)
    judge = Judge(cfg, jcfg, prompt.read_text(), 2 * left_tokens, paid)
    proj, calls = judge.projection(pairs, facts_of, check_cache=True)
    print(
        f"prompt-test: {len(pairs)} pairs ({len(sources)} source, {len(twins)} twin, "
        f"{len(pairs) - len(sources) - len(twins)} other); {calls} calls, {proj:,.0f} tokens projected"
    )
    if proj > left_tokens:
        sys.exit(f"projection above what --max-tokens leaves ({left_tokens:,.0f})")
    ledger.check("judge_prompt_test", proj)
    if args.project_only:
        return
    done, left, _errors = run_with_passes(judge, pairs, facts_of, args, "prompt-test")
    new = {}
    with open(out / "answers.jsonl", "w") as handle:
        for (s, d), (answers, cost, n) in done.items():
            new.setdefault(s, {})[d] = answers
            handle.write(
                json.dumps(
                    {
                        "source": s,
                        "candidate": d,
                        "prompt": prompt.name,
                        "n_frames": n,
                        "answers": answers,
                        "cost": {k: c.to_dict() for k, c in cost.items()},
                    }
                )
                + "\n"
            )
    cost = sum(paid, Cost())
    ledger.add(
        "judge_prompt_test",
        cost,
        len(paid),
        {"pairs": len(done), "failed": len(left), "prompt": prompt.name},
    )
    tested = [p for p in pairs if p[1] in new.get(p[0], {})]
    old = {s: {d: base[s][d] for d in base[s] if (s, d) in set(tested)} for s in base}
    twin_rows = [r for r in rows if r.get("twin")]
    noise = json.loads(NOISE.read_text())["conditions"] if NOISE.exists() else {}
    band = [
        noise.get(c, {}).get("queries", {}).get("all=all", {}).get("rate") for c in ("base", "R")
    ]
    res = {
        "prompt": prompt.name,
        "pairs": len(tested),
        "failed": len(left),
        "cost": cost.to_dict(),
        "band": band,
    }
    for name, j in (("v1", old), ("candidate", new)):
        items = fact_items(tested, j, facts_of)
        res[name] = {
            "source_acceptance": acceptance(rows, of_query, j, specs),
            "fact_agreement": agreement_table(items, *specs[:2], by="pair"),
            "twin_relevant": sum(
                jf.relevant(j[r["source"]][r["twin"]], of_query[r["query_id"]], specs)
                for r in twin_rows
                if r["twin"] in j.get(r["source"], {})
            ),
            "other_relevant_query_pairs": relevance_pairs(rows, of_query, j, tested, specs),
        }
    acc = res["candidate"]["source_acceptance"]["both"]["rate"]
    # kappa on the non-source pairs (twin + sampled pool pairs): on (source, source) pairs both
    # models say yes to almost every fact, so kappa there is degenerate and inflates "all"
    k_new = res["candidate"]["fact_agreement"]["other"]["kappa"]
    k_old = res["v1"]["fact_agreement"]["other"]["kappa"]
    top = max(b for b in band if b is not None) if any(b is not None for b in band) else 0.85
    res["decision"] = {
        "rule": (
            "adopt if source acceptance > top of the run-to-run band and fact kappa on the "
            "non-source pairs >= v1 on the same items"
        ),
        "band_top": top,
        "acceptance": acc,
        "kappa_candidate": k_new,
        "kappa_v1": k_old,
        "adopt": bool(
            acc is not None
            and acc > max(top, 0.85)
            and k_new is not None
            and k_old is not None
            and k_new >= k_old
        ),
    }
    (out / "summary.json").write_text(json.dumps(res, indent=1))
    (out / "report.md").write_text(prompt_report(res))
    write_manifest(
        out,
        models={"judge": list(specs), "judge_media_resolution": judge.media},
        cost=cost,
        extra={"prompt": prompt.name, "pairs": len(tested)},
    )
    print(prompt_report(res))


def prompt_report(res: dict) -> str:
    f = lambda v: "-" if v is None else f"{v:.3f}"
    L = [
        f"# Judge prompt-fix test: {res['prompt']} vs judge_frames_v1 (smoke pairs)",
        "",
        (
            f"{res['pairs']} pairs judged by both models ({res['failed']} failed). Input tokens: "
            f"{res['cost']['tokens_in']:,} in / {res['cost']['tokens_out']:,} out. Run-to-run band of v1 "
            f"source acceptance (base, repeat R): {res['band']}."
        ),
        "",
        "| measure | v1 | candidate |",
        "|---|---|---|",
    ]
    for key in ("both", "both natural", "both twin", "gemini", "openai"):
        a, b = (res[x]["source_acceptance"].get(key, {}) for x in ("v1", "candidate"))
        L.append(
            f"| source acceptance, {key} (n {a.get('n')}) | {f(a.get('rate'))} | {f(b.get('rate'))} |"
        )
    for grp in ("all", "source", "other"):
        a, b = (res[x]["fact_agreement"].get(grp, {}) for x in ("v1", "candidate"))
        L.append(
            f"| fact kappa, {grp} pairs (n {a.get('n')}) | {f(a.get('kappa'))} | {f(b.get('kappa'))} |"
        )
        L.append(
            f"| yes share Gemini / astra, {grp} | {f(a.get('yes_a'))} / {f(a.get('yes_b'))} | "
            f"{f(b.get('yes_a'))} / {f(b.get('yes_b'))} |"
        )
    for key in ("twin_relevant", "other_relevant_query_pairs"):
        L.append(f"| {key} | {res['v1'][key]} | {res['candidate'][key]} |")
    d = res["decision"]
    L += [
        "",
        (
            f"Decision rule: {d['rule']}. Band top {f(d['band_top'])}; acceptance {f(d['acceptance'])}; "
            f"kappa {f(d['kappa_candidate'])} vs {f(d['kappa_v1'])}. **Adopt: {d['adopt']}**."
        ),
        "",
    ]
    return "\n".join(L)


# ---- main ---------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("run", "stats", "prompt-test"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-cap", type=float, default=2e5)
    ap.add_argument("--config")
    ap.add_argument("--out")
    ap.add_argument("--prompt", help="judge prompt (default judge.prompt or judge_frames_v1)")
    ap.add_argument("--fill-queries", help="JSON {set: [query ids]}: judge all their top-10 pairs")
    ap.add_argument(
        "--extra-runs",
        nargs="*",
        default=[],
        help="NAME=PATH run files (jsonl) added to the rankings, e.g. for a label fill",
    )
    ap.add_argument("--project-only", action="store_true")
    ap.add_argument("--check-cache", action="store_true", help="projection skips cached calls")
    ap.add_argument("--no-canary", action="store_true")
    ap.add_argument("--canary", type=int, default=10)
    ap.add_argument("--gemini-workers", type=int, default=16)
    ap.add_argument("--openai-workers", type=int, default=128)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--wait", type=float, default=60.0)
    ap.add_argument("--sample", type=int, default=100, help="prompt-test: other pool pairs")
    ap.add_argument("--max-tokens", type=float, default=2.5e6, help="prompt-test hard limit")
    args = ap.parse_args(argv)
    fs_args = argparse.Namespace(
        smoke=args.smoke and args.stage != "prompt-test",
        smoke_cap=args.smoke_cap,
        config=args.config,
        out=args.out,
        n_boot=FS.N_BOOT,
    )
    S = FS.settings(fs_args)
    if args.stage == "prompt-test":
        import yaml

        S.cfg_smoke = {**S.cfg, **yaml.safe_load(FS.SMOKE_CONFIG.read_text())}
    {"run": stage_run, "stats": stage_stats, "prompt-test": stage_prompt_test}[args.stage](S, args)


if __name__ == "__main__":
    main()
