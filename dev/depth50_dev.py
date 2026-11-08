"""Deeper rerank pool: does S4 on S0's top 50 recover first-stage misses? PAID (gpt-6-sol text
calls for ranks 21-50, two vision judges for new top-10 candidates).

S4 (holistic 0-10 rerank, `prompts/rerank_v1.txt`, gpt-6-sol@low over the 0.5 fps descriptions)
is run on S0's top 50 for every natural (nested) and twin query of the v2 pilot
(results/smoke_v2/pilot/<set>/S0.jsonl). The prompts and cache keys are those of
dev/run_systems_dev.py (its `Models`, read-only), so the top-20 calls are cache hits and only
ranks 21-50 are paid. S4@20 is rescored from the same saved scores (S0's top 20 only), so the two
differ only in pool depth.

The judge then labels the (source, candidate) pairs in the top 10 of S4@50, S4@20 and S0 that
data/mc/smoke_v2/judge/judged.jsonl has no line for (Gemini LOW + gpt-6-astra, both must say yes
to every fact; same prompt, frames and cache keys as dev/judge_labels_dev.py). New judgments
go to data/mc/smoke_v2/deep_pool/judged.jsonl only; expanded qrels are rebuilt from the union.

    uv run --env-file .env python dev/depth50_dev.py --stage project
    uv run --env-file .env python dev/depth50_dev.py --stage rerank
    uv run --env-file .env python dev/depth50_dev.py --stage judge
    uv run python dev/depth50_dev.py --stage eval

Every paid stage projects its new input tokens and stops when the running total in
results/smoke_v2/deep_pool/spend.json plus the projection would pass --cap (12M).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from src import judge_frames as jf
from src.datasets.mc import McCorpus, corpus_descriptions, index_options, read_jsonl
from src.frames import FramePolicy
from src.grounding_v2 import FrameStore, Grounder
from src.systems.base import merge_reranked
from src.systems.rerank import (
    DESC_EVIDENCE,
    RERANK_PROMPT,
    DescriptionScore,
    HolisticReranker,
    fill,
)
from src.systems.verify import rank_by_score
from src.types import Cost, Query

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """A sibling script as a module (read-only use of its functions)."""
    spec = importlib.util.spec_from_file_location(
        f"_dp_{name}", ROOT / "dev" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


P2 = load_script("run_systems_dev")
PV = P2.PV
JP = load_script("judge_labels_dev")

CONFIG = Path("configs/mc_smoke_v2.yaml")
PILOT = Path("results/smoke_v2/pilot")
OUT = Path("results/smoke_v2/deep_pool")
DATA_OUT = Path("data/mc/smoke_v2/deep_pool")
JUDGE_DIR = Path("data/mc/smoke_v2/judge")
CAP = 12e6
DEPTH = 50
SHALLOW = 20
METRICS = ("ndcg@10", "recall@1", "mrr")
GROUPS = ("all", 1, 2, 3, 4)


# ---- pure helpers -----------------------------------------------------------------------------


def rescore(first_stage, scores: dict[str, float], n: int):
    """S4 on the first n of first_stage from saved per-candidate scores (ties in first-stage
    order), the rest of first_stage after, as pilot_verify.read_rankings derives S4@10."""
    head = first_stage[:n]
    ranked = rank_by_score([(d, 0.0) for d, _ in head], [scores[d] for d, _ in head])
    return merge_reranked(first_stage, ranked)


def rank_of(ranking, doc: str) -> int | None:
    """1-based rank of doc, None when absent."""
    for i, (d, _) in enumerate(ranking, 1):
        if d == doc:
            return i
    return None


def share_in_top(rankings: dict, rows: list[dict], qrels: dict, n: int) -> dict:
    """Per k and overall: share of queries with a relevant video (per qrels) in the top n."""
    groups: dict = {}
    for r in rows:
        q = r["query_id"]
        if q not in rankings:
            continue
        rel = {d for d, g in qrels[q].items() if g > 0}
        hit = any(d in rel for d, _ in rankings[q][:n])
        for g in ("all", r["k"]):
            groups.setdefault(g, []).append(hit)
    return {str(g): {"n": len(v), "share": float(np.mean(v))} for g, v in groups.items()}


def rescued(s0: dict, deep: dict, rows: list[dict], lo: int, hi: int, top: int) -> dict:
    """Per k and overall: queries whose source S0 ranks in [lo, hi], and how many of them the
    deep rerank puts in its top `top`."""
    out: dict = {}
    for r in rows:
        q = r["query_id"]
        if q not in s0 or q not in deep:
            continue
        rank = rank_of(s0[q], r["source"])
        if rank is None or not lo <= rank <= hi:
            continue
        new = rank_of(deep[q], r["source"])
        for g in ("all", str(r["k"])):
            cell = out.setdefault(g, {"n": 0, "into_top": 0, "to_rank1": 0})
            cell["n"] += 1
            cell["into_top"] += bool(new is not None and new <= top)
            cell["to_rank1"] += new == 1
    return out


def qrels_from_lists(table: dict[str, list[str]]) -> dict[str, dict[str, int]]:
    return {q: {d: 1 for d in docs} for q, docs in table.items()}


# ---- data -------------------------------------------------------------------------------------


def load_context(cfg: dict):
    """(descriptions, sets): the 0.5 fps descriptions and the pilot's query rows (sources with
    an empty description left out, as pilot_v2.load_index / load_sets do)."""
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    options = index_options(cfg)
    corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], sample["videos"])
    descriptions = corpus_descriptions(
        corpus, Path(cfg["describe_prompt"]).name, options["filename"]
    )
    missing = sorted(d for d in corpus.doc_ids if not descriptions.get(d))
    return descriptions, P2.load_sets(Path(cfg["out_dir"]), missing)


def read_ranking_file(path: Path) -> dict:
    return {line["query_id"]: [(d, s) for d, s in line["ranking"]] for line in read_jsonl(path)}


def all_rankings(sets: dict) -> dict[str, dict[str, dict]]:
    """{set: {system: {query_id: ranking}}} for S0, S4@20 (rescored from the deep scores),
    S4@50 and the pilot's saved S4 (top 20)."""
    out = {}
    for name in sets:
        s0 = read_ranking_file(PILOT / name / "S0.jsonl")
        deep = {line["query_id"]: line for line in read_jsonl(OUT / name / "S4@50.jsonl")}
        out[name] = {
            "S0": s0,
            "S4@20": {
                q: rescore(s0[q], line["trace"]["scores"], SHALLOW) for q, line in deep.items()
            },
            "S4@50": {q: [(d, s) for d, s in line["ranking"]] for q, line in deep.items()},
            "S4 (pilot)": read_ranking_file(PILOT / name / "S4.jsonl"),
        }
    return out


# ---- stages -----------------------------------------------------------------------------------


def rerank_prompts(sets: dict, descriptions: dict) -> list[str]:
    template = RERANK_PROMPT.read_text()
    prompts = []
    for name, rows in sets.items():
        s0 = read_ranking_file(PILOT / name / "S0.jsonl")
        for r in rows:
            for d, _ in s0[r["query_id"]][:DEPTH]:
                desc = descriptions.get(d, "") or "(no description)"
                prompts.append(
                    fill(template, r["query"], DESC_EVIDENCE.replace("{description}", desc))
                )
    return prompts


def stage_rerank(cfg, sets, descriptions, args, ledger) -> None:
    models = P2.Models(cfg)
    prompts = rerank_prompts(sets, descriptions)
    new = [p for p in prompts if not models.verifier_cached.is_cached(p)]
    proj = sum(P2.text_tokens(p) for p in new)
    print(f"S4@50: {len(prompts)} calls, {len(new)} not cached")
    ledger.check("rerank", proj)
    if args.project_only:
        return
    template = RERANK_PROMPT.read_text()
    reranker = HolisticReranker(
        "S4@50", DescriptionScore(models.verifier, template, descriptions), args.inner
    )
    for name, rows in sets.items():
        s0 = read_ranking_file(PILOT / name / "S0.jsonl")
        by_id = {r["query_id"]: r for r in rows}
        queries = [Query(r["query_id"], r["query"], r["source"]) for r in rows]
        start = time.perf_counter()
        records = PV.rerank_all(reranker, queries, s0, DEPTH, args.workers, f"{name} S4@50")
        PV.write_rankings(OUT / name / "S4@50.jsonl", "S4@50", records, by_id)
        print(f"{name}: {len(records)} queries in {time.perf_counter() - start:.0f} s")
    paid, n = models.paid_cost()
    failed = sum(
        1
        for name in sets
        for line in read_jsonl(OUT / name / "S4@50.jsonl")
        for s in line["trace"]["scores"].values()
        if s < 0
    )
    ledger.add("rerank", paid, n, {"errors": len(models.errors), "unscored": failed})
    (OUT / "rerank_errors.json").write_text(json.dumps(models.errors[:200], indent=1))


def judge_setup(cfg):
    jcfg = cfg["judge"]
    rows = JP.read_queries(Path(cfg["out_dir"]))
    facts_of, of_query = {}, {}
    for s in dict.fromkeys(r["source"] for r in rows):
        facts_of[s], oq = jf.source_facts([r for r in rows if r["source"] == s])
        of_query.update(oq)
    return jcfg, rows, facts_of, of_query


def judged_union() -> dict[str, dict[str, dict]]:
    judged = JP.read_judged(JUDGE_DIR / "judged.jsonl")
    for source, cands in JP.read_judged(DATA_OUT / "judged.jsonl").items():
        for d, answers in cands.items():
            judged.setdefault(source, {}).setdefault(d, answers)
    return judged


def judge_tasks(sets, rows, judged, systems: tuple[str, ...]) -> tuple[list, dict]:
    """Unjudged (source, candidate) pairs in the top 10 of the given systems, over all pilot
    queries, and per system the pairs it alone would need."""
    ranks = all_rankings(sets)
    merged = {
        s: {q: [d for d, _ in r] for name in sets for q, r in ranks[name][s].items()}
        for s in systems
    }
    ids = [r["query_id"] for rs in sets.values() for r in rs]
    tasks, _ = JP.fill_tasks(rows, merged, ids, judged, jf.TOP)
    per = {s: JP.fill_tasks(rows, {s: merged[s]}, ids, judged, jf.TOP)[0] for s in systems}
    return tasks, per


def stage_judge(cfg, sets, args, ledger) -> None:
    jcfg, rows, facts_of, _of_query = judge_setup(cfg)
    judged = judged_union()
    systems = tuple(args.judge_systems.split(","))
    tasks, per = judge_tasks(sets, rows, judged, systems)
    print(f"unjudged pairs: {len(tasks)} (" + ", ".join(f"{s} {len(t)}" for s, t in per.items()))
    b2 = JP.load_build_v2()
    root = cfg["data_root"]
    specs = tuple(jcfg["models"])
    store = FrameStore(
        root, index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8))
    )
    cap, side = jcfg.get("max_frames", jf.MAX_FRAMES), jcfg.get("max_side", jf.MAX_SIDE)
    template = jf.JUDGE_PROMPT.read_text()
    media = jcfg.get("media_resolution", cfg.get("grounding_media_resolution"))
    proj = 0.0
    for s, d in tasks:
        dataset, v = JP.video(root, d)
        n = min(cap, len(store.index_times(dataset, v)))
        for spec in specs:
            proj += b2.text_tokens(template) + 15 * len(facts_of[s])
            proj += b2.image_tokens(spec, n, media) if spec.startswith("gemini") else n * 260
    ledger.check("judge", proj)
    if args.project_only or not tasks:
        return
    paid: list[Cost] = []
    models, budget = {}, b2.Budget(args.cap - ledger.total)  # hard stop per call
    for spec in specs:
        cached = b2.make_grounder_model(spec, media)
        cached.inner = PV.Metered(cached.inner, paid)
        if spec.startswith("openai"):
            cached.inner.inner.client = cached.inner.inner.client.with_options(
                max_retries=10, timeout=300
            )
        models[spec] = b2.GuardedVision(cached, budget, spec)
    grounder = Grounder(models, template, store)
    frames_cache: dict[str, tuple] = {}
    for _, d in tasks:
        if d not in frames_cache:
            dataset, v = JP.video(root, d)
            frames, times = store.index_frames(dataset, v)
            frames_cache[d] = jf.judge_frames(frames, times, cap, side)
    errors: list[str] = []

    def call(item):
        i, spec = item
        source, doc = tasks[i]
        frames, times = frames_cache[doc]
        try:
            answers, cost = grounder.on_frames(frames, times, facts_of[source], specs=(spec,))
        except Exception as error:  # noqa: BLE001
            errors.append(f"{source} {doc} {spec}: {error!r}"[:300])
            return (i, spec), None
        return (i, spec), (answers, cost, len(frames))

    pools = {
        spec: ThreadPoolExecutor(args.gemini_workers if spec.startswith("gemini") else 128)
        for spec in specs
    }
    futures = [pools[spec].submit(call, (i, spec)) for i in range(len(tasks)) for spec in specs]
    results = dict(f.result() for f in futures)
    for pool in pools.values():
        pool.shutdown()
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    written = 0
    with open(DATA_OUT / "judged.jsonl", "a") as handle:
        for i, (source, doc) in enumerate(tasks):
            parts = [results[(i, spec)] for spec in specs]
            if any(p is None for p in parts):
                continue
            answers: dict[str, dict] = {}
            cost: dict[str, Cost] = {}
            for a, c, _ in parts:
                for key, by_model in a.items():
                    answers.setdefault(key, {}).update(by_model)
                cost.update(c)
            line = {
                "source": source,
                "candidate": doc,
                "why": [s for s, t in per.items() if (source, doc) in set(t)],
                "n_frames": parts[0][2],
                "facts": facts_of[source],
                "answers": answers,
                "cost": {s: c.to_dict() for s, c in cost.items()},
            }
            handle.write(json.dumps(line) + "\n")
            written += 1
    (OUT / "judge_errors.json").write_text(json.dumps(errors, indent=1))
    ledger.add("judge", sum(paid, Cost()), len(paid), {"pairs": written, "errors": len(errors)})


# ---- evaluation -------------------------------------------------------------------------------


def stage_eval(cfg, sets, args, ledger) -> None:
    _, rows_all, _, of_query = judge_setup(cfg)
    judged = judged_union()
    new_judged = JP.read_judged(DATA_OUT / "judged.jsonl")
    _, expanded_old = jf.qrels(rows_all, of_query, JP.read_judged(JUDGE_DIR / "judged.jsonl"))
    _, expanded = jf.qrels(rows_all, of_query, judged)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    with open(DATA_OUT / "qrels_expanded.jsonl", "w") as handle:
        handle.writelines(
            json.dumps({"query_id": q, "relevant": rel}) + "\n" for q, rel in expanded.items()
        )
    new_lines = list(read_jsonl(DATA_OUT / "judged.jsonl")) if new_judged else []
    eval_ids = {r["query_id"] for rs in sets.values() for r in rs}
    new_pos = sum(
        len(set(expanded[q]) - set(expanded_old.get(q, []))) for q in expanded if q in eval_ids
    )
    ranks = all_rankings(sets)
    result: dict = {
        "judge": {
            "new_pairs": sum(len(v) for v in new_judged.values()),
            "new_pairs_by_system": {
                s: sum(1 for line in new_lines if s in line.get("why", []))
                for s in ("S4@50", "S4@20", "S0")
            },
            "new_positive_query_pairs": new_pos,
            "new_positive_source_pairs": sum(
                1
                for line in new_lines
                if any(
                    line["candidate"] in expanded[q]
                    for q in expanded
                    if q.startswith(line["source"] + "#")
                )
            ),
        },
        "sets": {},
    }
    systems = ("S0", "S4@20", "S4@50")
    for name, rows in sets.items():
        sys_r = ranks[name]
        consistency = sum(
            [d for d, _ in sys_r["S4@20"][q][:SHALLOW]] != [d for d, _ in r[:SHALLOW]]
            for q, r in sys_r["S4 (pilot)"].items()
            if q in sys_r["S4@20"]
        )
        res = {"n": len(rows), "s4_20_vs_pilot_mismatch": consistency, "labels": {}}
        for kind in ("strict", "expanded"):
            qrels = (
                PV.load_qrels(None, rows)
                if kind == "strict"
                else qrels_from_lists({r["query_id"]: expanded[r["query_id"]] for r in rows})
            )
            sc = {s: PV.score_system(sys_r[s], qrels) for s in systems}
            res["labels"][kind] = {
                "recall_top20": share_in_top(sys_r["S0"], rows, qrels, SHALLOW),
                "recall_top50": share_in_top(sys_r["S0"], rows, qrels, DEPTH),
                "means": {s: PV.mean_table(sc[s], rows, "k") | PV.mean_table(sc[s], rows, None)
                          for s in systems},
                "paired": {
                    f"{a} - {b}": {m: PV.paired(sc[a], sc[b], rows, m) | PV.paired(
                        sc[a], sc[b], rows, m, "k") for m in METRICS}
                    for a, b in (("S4@50", "S4@20"), ("S4@50", "S0"), ("S4@20", "S0"))
                },
            }  # fmt: skip
        res["rescued_21_50"] = rescued(sys_r["S0"], sys_r["S4@50"], rows, 21, 50, 10)
        res["lost_1_20"] = {  # sources S4@20 had in its top 10 that S4@50 pushes out
            "n": sum(
                1
                for r in rows
                if (a := rank_of(sys_r["S4@20"][r["query_id"]], r["source"])) and a <= 10
                and not ((b := rank_of(sys_r["S4@50"][r["query_id"]], r["source"])) and b <= 10)
            )
        }  # fmt: skip
        if name == "twin":
            in50 = {
                r["query_id"]
                for r in rows
                if (x := rank_of(sys_r["S0"][r["query_id"]], r["source"])) and x <= DEPTH
            }
            res["twin_win"] = {s: P2.twin_win_rate(sys_r[s], rows) for s in systems}
            res["twin_win_source_in_top50"] = {
                s: P2.twin_win_rate(sys_r[s], rows, in50) for s in systems
            }
        result["sets"][name] = res
    result["spend"] = {"total_tokens_in": ledger.total, "entries": ledger.entries}
    (OUT / "summary.json").write_text(json.dumps(result, indent=1, default=str))
    (OUT / "report.md").write_text(report(result))
    print(report(result))


def f3(v: float) -> str:
    return f"{v:.3f}"


def ci(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def report(res: dict) -> str:
    j = res["judge"]
    lines = [
        "# Deeper rerank pool: S4 on S0's top 50 vs top 20 (smoke v2, 0.5 fps index)",
        "",
        (
            "S4 = holistic 0-10 rerank (`prompts/rerank_v1.txt`, gpt-6-sol@low) over the 0.5 fps "
            "descriptions. S4@20 is rescored from the same calls on S0's top 20 only, so S4@50 and "
            "S4@20 differ only in pool depth. Strict labels: the source only. Expanded: the source "
            "plus candidates both frame judges (Gemini LOW + gpt-6-astra) accept on every fact, from "
            "`data/mc/smoke_v2/judge/judged.jsonl` plus the new judgments in "
            "`data/mc/smoke_v2/deep_pool/judged.jsonl`. CIs: 95% bootstrap over source videos. "
            "Script `dev/depth50_dev.py`."
        ),
        "",
        (
            f"Judge fill: {j['new_pairs']} new (source, candidate) pairs, unjudged and in the top 10 "
            f"of S4@50, S4@20 or S0 (by system: {j['new_pairs_by_system']}); "
            f"{j['new_positive_source_pairs']} of them positive for at least one query, "
            f"{j['new_positive_query_pairs']} new positive (query, video) labels over the evaluated "
            "queries."
        ),
        "",
        f"Spend: {res['spend']['total_tokens_in']:,.0f} new input tokens "
        + "("
        + ", ".join(f"{e['stage']} {e['tokens_in']:,}" for e in res["spend"]["entries"])
        + ").",
        "",
    ]
    for name, s in res["sets"].items():
        label = "nested (natural)" if name == "natural" else "twin"
        lines += [f"## {label}: {s['n']} queries", ""]
        lines += [
            (
                f"S4@20 rescored vs the pilot's saved S4 (top-20 order differs): "
                f"{s['s4_20_vs_pilot_mismatch']} queries."
            ),
            "",
        ]
        for kind, lab in s["labels"].items():
            lines += [f"### {kind} labels", ""]
            lines += ["Recall headroom: share of queries with a relevant video in S0's top n.", ""]
            lines += ["| n | all | k=1 | k=2 | k=3 | k=4 |", "|---|---|---|---|---|---|"]
            for n, key in ((20, "recall_top20"), (50, "recall_top50")):
                lines.append(
                    f"| {n} | "
                    + " | ".join(
                        f3(lab[key][str(g)]["share"]) for g in GROUPS if str(g) in lab[key]
                    )
                    + " |"
                )
            if kind == "expanded":
                lines += ["", ("(expanded at 50 is a lower bound: S0 ranks 11-50 are judged only "
                          "where some system put them in its top 10)")]  # fmt: skip
            lines += ["", "| system | metric | all | k=1 | k=2 | k=3 | k=4 |"]
            lines += ["|---|---|---|---|---|---|---|"]
            for sysname, table in lab["means"].items():
                for m in METRICS:
                    lines.append(
                        f"| {sysname} | {m} | "
                        + " | ".join(f3(table[g][m]) for g in GROUPS if g in table)
                        + " |"
                    )
            lines += ["", "| pair | metric | all | k=1 | k=2 | k=3 | k=4 |"]
            lines += ["|---|---|---|---|---|---|---|"]
            for pair, by_m in lab["paired"].items():
                for m, t in by_m.items():
                    lines.append(
                        f"| {pair} | {m} | " + " | ".join(ci(t[g]) for g in GROUPS if g in t) + " |"
                    )
            lines.append("")
        r = s["rescued_21_50"]
        lines += [
            "### Sources S0 ranked 21-50",
            "",
            "| | all | k=1 | k=2 | k=3 | k=4 |",
            "|---|---|---|---|---|---|",
            "| queries | " + " | ".join(str(r.get(str(g), {}).get("n", 0)) for g in GROUPS) + " |",
            "| S4@50 top 10 | "
            + " | ".join(str(r.get(str(g), {}).get("into_top", 0)) for g in GROUPS)
            + " |",
            "| S4@50 rank 1 | "
            + " | ".join(str(r.get(str(g), {}).get("to_rank1", 0)) for g in GROUPS)
            + " |",
            "",
            f"Sources in S4@20's top 10 that S4@50 pushes out of its top 10: {s['lost_1_20']['n']}.",
            "",
        ]
        if "twin_win" in s:
            lines += ["### Twin ranked above the source", ""]
            lines += ["| system | on | all | k=1 | k=2 | k=3 | k=4 |"]
            lines += ["|---|---|---|---|---|---|---|"]
            for key, on in (("twin_win", "all"), ("twin_win_source_in_top50", "source in S0@50")):
                for sysname, t in s[key].items():
                    lines.append(
                        f"| {sysname} | {on} (n={t['all']['n']}) | "
                        + " | ".join(f3(t[str(g)]["rate"]) for g in GROUPS if str(g) in t)
                        + " |"
                    )
            lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", required=True, choices=("project", "rerank", "judge", "eval"))
    ap.add_argument("--cap", type=float, default=CAP)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--inner", type=int, default=4, help="calls per query in flight")
    ap.add_argument("--gemini-workers", type=int, default=16)
    ap.add_argument("--judge-systems", default="S4@50,S4@20,S0")
    ap.add_argument("--project-only", action="store_true")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    OUT.mkdir(parents=True, exist_ok=True)
    ledger = P2.Ledger(OUT / "spend.json", args.cap)
    descriptions, sets = load_context(cfg)
    if args.stage == "project":
        args.project_only = True
        stage_rerank(cfg, sets, descriptions, args, ledger)
    elif args.stage == "rerank":
        stage_rerank(cfg, sets, descriptions, args, ledger)
    elif args.stage == "judge":
        stage_judge(cfg, sets, args, ledger)
    else:
        stage_eval(cfg, sets, args, ledger)


if __name__ == "__main__":
    main()
