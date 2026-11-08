"""A describe-prompt variant of the smoke index, and S0 / S2 / S4 on it. PAID (Gemini, OpenAI).

Tests whether longer descriptions (prompts/describe_v4b.txt, "3 to 5 sentences" instead of
describe_v4's "1 to 3") change the smoke results. Same 500 smoke videos, same 16 saved frames
(frames16/), same describer (configs/mc_smoke.yaml) and embedder; the variant's descriptions go
to data/mc/<dataset>/descriptions_<prompt stem>.jsonl (`mc.descriptions_file`), so the
describe_v4 ones in descriptions.jsonl are never touched, and its index to
results/_indexes/<hash>/ (`index_settings` hashes the prompt).

Stages (all calls through the shared caches, so a rerun is free):

- index: describe the sample with the variant prompt and build its index (--measure N: describe
  N videos only and project the cost of the sample).
- coverage: for each query source, one judge call per (source, prompt) (prompts/judge_v1.txt,
  openai:gpt-6-sol@low, text only) on the source's own description and all facts of its
  queries; per query, the share of its k facts answered "yes".
- systems: S0 (gate numbers, as the dev S0 gate) and S2 / S4 (top 20, as
  dev/verifiers_dev.py) on the variant index, rankings to results/smoke/variant_<tag>/.
- report: description word counts, fact coverage, gate, S0 / S2 / S4 (and S2@10 / S4@10)
  against the describe_v4 pilot (results/smoke/pilot/), strict and expanded labels, paired
  differences with bootstrap CIs over source videos. Writes results/smoke/variant_<tag>.md and
  variant_<tag>/summary.json (tag: the prompt stem without "describe_", e.g. v4b).

    uv run --env-file .env python dev/description_length_dev.py --stage index --measure 5
    uv run --env-file .env python dev/description_length_dev.py --stage all

verifiers_dev.py is imported read-only (another agent owns it).
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

from src.cache import shared_cache
from src.datasets.mc import (
    McCorpus,
    RateLimit,
    RateLimitedVision,
    corpus_descriptions,
    describe_corpus,
    description_index,
    descriptions_file,
    index_settings,
    index_store,
    read_jsonl,
)
from src.embedders import make_embedder
from src.judge import JUDGE_PROMPT, JUDGE_SPEC, judge_prompt, parse_judgment
from src.llm import make_llm, make_vision
from src.manifest import write_manifest
from src.systems.single import cluster_bootstrap_ci, gate, gate_row, search_all
from src.types import Cost, Query

ROOT = Path(__file__).resolve().parents[1]
PILOT_DIR = Path("results/smoke/pilot")
GATE_JSON = Path("results/smoke/gate.json")
# assumed Gemini flash-lite price per 1M tokens (as in docs/artifacts.md); the API reports none
PRICE_IN, PRICE_OUT = 0.10, 0.40
SYSTEMS = ("S0", "S2", "S4", "S2@10", "S4@10")
METRICS = ("recall@1", "recall@5", "ndcg@10", "mrr")
KEYS = (None, "k", "dataset", "length_bin")


def pilot_module():
    """dev/verifiers_dev.py, imported read-only."""
    path = ROOT / "dev" / "verifiers_dev.py"
    spec = importlib.util.spec_from_file_location("pilot_verify", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- pure helpers (tested offline) -------------------------------------------------------------


def word_count(text: str) -> int:
    return len(text.split())


def word_stats(counts: list[int]) -> dict:
    if not counts:
        return {"n": 0}
    p10, med, p90 = np.percentile(counts, [10, 50, 90])
    return {"n": len(counts), "median": float(med), "p10": float(p10), "p90": float(p90)}


def description_words(corpus: McCorpus, descriptions: dict[str, str]) -> dict:
    """Word stats of the non-empty descriptions overall, by dataset and by length bin."""
    groups: dict[str, dict[str, list[int]]] = {"overall": {}, "dataset": {}, "length_bin": {}}
    for doc, text in descriptions.items():
        if not text:
            continue
        v, n = corpus.videos[doc], word_count(text)
        groups["overall"].setdefault("all", []).append(n)
        groups["dataset"].setdefault(v.dataset, []).append(n)
        groups["length_bin"].setdefault(v.length_bin, []).append(n)
    return {key: {g: word_stats(c) for g, c in sorted(t.items())} for key, t in groups.items()}


def source_facts(rows: list[dict]) -> dict[str, list[str]]:
    """Per source, the fact texts of all its queries in first-seen order (the k facts of a query
    extend those of k - 1, so this is the largest query's list)."""
    out: dict[str, list[str]] = {}
    for r in sorted(rows, key=lambda r: (r["source"], r["k"])):
        facts = out.setdefault(r["source"], [])
        for f in r["facts"]:
            if f["text"] not in facts:
                facts.append(f["text"])
    return out


def judge_description(llm, template: str, description: str, facts: list[str]) -> dict:
    """One call: does the description state each fact? {"answers", "cost", "raw"}."""
    prompt = judge_prompt(template, description.strip(), [{"text": t} for t in facts])
    raw, cost = llm.complete(prompt)
    answers = parse_judgment(raw, len(facts))
    return {"answers": [a["answer"] for a in answers], "cost": cost.to_dict(), "raw": raw}


def coverage_rows(rows: list[dict], facts: dict[str, list[str]], judged: dict) -> list[dict]:
    """Per query: share of its k facts the source's description states ("yes") and contradicts
    ("no"), from the per-source judgments (fact text -> answer)."""
    out = []
    for r in rows:
        if r["source"] not in judged:
            continue
        by_text = dict(zip(facts[r["source"]], judged[r["source"]]["answers"], strict=True))
        answers = [by_text[f["text"]] for f in r["facts"]]
        out.append(
            {
                "query_id": r["query_id"],
                "source": r["source"],
                "k": r["k"],
                "dataset": r["dataset"],
                "length_bin": r["length_bin"],
                "yes": sum(a == "yes" for a in answers) / len(answers),
                "no": sum(a == "no" for a in answers) / len(answers),
            }
        )
    return out


def grouped_means(rows: list[dict], value: str, key: str | None) -> dict:
    groups: dict = {}
    for r in rows:
        groups.setdefault("all" if key is None else r[key], []).append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        vals = [r[value] for r in rs]
        lo, hi = cluster_bootstrap_ci(vals, [r["source"] for r in rs])
        out[g] = {"n": len(rs), "mean": float(np.mean(vals)), "ci_low": lo, "ci_high": hi}
    return out


def paired_rows(a: list[dict], b: list[dict], value: str) -> list[dict]:
    """Rows of a with value replaced by a - b, for the query ids in both."""
    other = {r["query_id"]: r[value] for r in b}
    return [{**r, value: r[value] - other[r["query_id"]]} for r in a if r["query_id"] in other]


def describe_projection(cost: Cost, n_done: int, n_total: int) -> dict:
    scale = n_total / max(n_done, 1)
    usd = (cost.tokens_in * PRICE_IN + cost.tokens_out * PRICE_OUT) / 1e6
    return {
        "videos": n_done,
        "calls": cost.llm_calls,
        "images": cost.images,
        "tokens_in": cost.tokens_in,
        "tokens_out": cost.tokens_out,
        "usd_measured": round(usd, 4),
        "usd_projected": round(usd * scale, 2),
        "projected_for": n_total,
        "price_per_1m": [PRICE_IN, PRICE_OUT],
    }


# ---- stages -----------------------------------------------------------------------------------


class Ctx:
    def __init__(self, args):
        self.args = args
        self.cfg = yaml.safe_load(Path(args.config).read_text())
        self.prompt = Path(args.prompt)
        self.stem = self.prompt.stem
        self.file = descriptions_file(self.prompt)
        smoke = Path(self.cfg["smoke_dir"])
        self.sample = json.loads((smoke / "sample.json").read_text())
        self.queries_file = Path(args.queries or smoke / "queries.jsonl")
        tag = self.stem.removeprefix("describe_")  # describe_v4b -> v4b
        self.out = Path(args.out or f"results/smoke/variant_{tag}")
        self.report = Path(args.report or f"results/smoke/variant_{tag}.md")
        self.full = McCorpus.pool(
            self.cfg["data_root"], self.cfg["datasets"], self.sample["videos"]
        )

    def descriptions(self, variant: bool) -> dict[str, str]:
        if variant:
            return corpus_descriptions(self.full, self.prompt.name, self.file)
        return corpus_descriptions(self.full, Path(self.cfg["describe_prompt"]).name)

    def kept_corpus(self, descriptions: dict[str, str]) -> tuple[McCorpus, list[str]]:
        """The smoke videos with a non-empty description under both the variant and the main
        prompt (so both indexes hold the same videos), and the ids left out."""
        main = self.descriptions(False)
        missing = [d for d in self.full.doc_ids if not descriptions.get(d) or not main.get(d)]
        kept = [d for d in self.full.doc_ids if d not in set(missing)]
        return McCorpus.pool(self.cfg["data_root"], self.cfg["datasets"], kept), missing

    def index(self, corpus: McCorpus, descriptions: dict[str, str]):
        cfg = self.cfg
        settings = index_settings(
            corpus.doc_ids, cfg["embedder"], cfg["describer"], self.prompt, cfg["n_frames"]
        )
        store = index_store(cfg["results_dir"], settings)
        embedder = make_embedder(cfg["embedder"], cache=shared_cache("embed", cfg["results_dir"]))
        return description_index(corpus, descriptions, embedder, store), settings, store, embedder


def measure_ids(doc_ids: list[str], n: int) -> list[str]:
    """n videos spread evenly over the corpus parts (global ids are "<dataset>/<video id>"), drawn
    by a fixed seed, so every prompt is measured on the same videos."""
    import random

    by_part: dict[str, list[str]] = {}
    for d in doc_ids:
        by_part.setdefault(d.split("/")[0], []).append(d)
    rng, picked = random.Random(0), []
    parts = sorted(by_part)
    for p in parts:
        rng.shuffle(by_part[p])
    i = 0
    while len(picked) < min(n, len(doc_ids)):
        part = by_part[parts[i % len(parts)]]
        if part:
            picked.append(part.pop())
        i += 1
    return picked


def limit_rows(rows: list[dict], n: int) -> list[dict]:
    """n queries spread evenly over k, drawn by a fixed seed from the file order, so a later run
    on all queries contains these and reuses their cached calls."""
    import random

    by_k: dict[int, list[dict]] = {}
    for r in rows:
        by_k.setdefault(r["k"], []).append(r)
    rng, picked = random.Random(0), []
    ks = sorted(by_k)
    for k in ks:
        rng.shuffle(by_k[k])
    i = 0
    while len(picked) < min(n, len(rows)):
        group = by_k[ks[i % len(ks)]]
        if group:
            picked.append(group.pop())
        i += 1
    keep = {r["query_id"] for r in picked}
    return [r for r in rows if r["query_id"] in keep]


def stage_index(ctx: Ctx) -> None:
    cfg, args = ctx.cfg, ctx.args
    ids = measure_ids(ctx.full.doc_ids, args.measure) if args.measure else ctx.full.doc_ids
    vlm = make_vision(cfg["describer"], shared_cache("vlm", cfg["results_dir"]))
    limited = RateLimitedVision(vlm.inner, RateLimit(cfg["describe_per_minute"]))
    vlm.inner = limited
    start = time.perf_counter()
    records = describe_corpus(
        ctx.full, vlm, ctx.prompt, cfg["describer"], cfg["n_frames"], cfg["workers"], ids, ctx.file
    )
    seconds = time.perf_counter() - start
    recorded = sum((Cost(**r["cost"]) for r in records.values()), Cost())
    proj = describe_projection(recorded, len(records), len(ctx.full.doc_ids))
    print(json.dumps({"paid_this_run": limited.cost.to_dict(), "recorded": proj}, indent=1))
    if args.measure:
        for d in ids:
            print(f"{d}: {word_count(records[d]['description'])} words")
        return
    descriptions = ctx.descriptions(True)
    corpus, missing = ctx.kept_corpus(descriptions)
    index, settings, store, embedder = ctx.index(corpus, descriptions)
    build = {
        "settings": settings,
        "describe": {
            "videos": len(records),
            "left_out_empty_description": missing,
            "wall_seconds": round(seconds, 1),
            "recorded": proj,
            "paid_this_run": limited.cost.to_dict(),
        },
        "embed": {
            "docs": len(index.doc_ids),
            "paid_calls": embedder.cost.llm_calls,
            "paid_tokens": embedder.cost.tokens_in,
        },
    }
    (store / "build.json").write_text(json.dumps(build, indent=1) + "\n")
    write_manifest(
        store,
        models={"describer": cfg["describer"], "embedder": cfg["embedder"]},
        seeds={"sample_seed": ctx.sample["seed"]},
        cost=recorded + embedder.cost,
        extra={"step": f"smoke variant index ({ctx.prompt.name})", "settings": settings},
    )
    print(f"index {store}: {len(index.doc_ids)} docs; empty: {missing}")


def stage_coverage(ctx: Ctx) -> None:
    pv = pilot_module()
    cfg = ctx.cfg
    rows = read_jsonl(ctx.queries_file)
    facts = source_facts(rows)
    errors: list[str] = []
    llm = pv.SafeCalls(
        pv.more_retries(make_llm(JUDGE_SPEC, shared_cache("llm", cfg["results_dir"]))), errors
    )
    template = JUDGE_PROMPT.read_text()
    result = {}
    for name, variant in (("v4", False), (ctx.stem, True)):
        desc = ctx.descriptions(variant)
        todo = [s for s in facts if desc.get(s)]

        def one(s, desc=desc):
            return s, judge_description(llm, template, desc[s], facts[s])

        with ThreadPoolExecutor(ctx.args.workers) as pool:
            result[name] = dict(pool.map(one, todo))
        cost = sum((Cost(**j["cost"]) for j in result[name].values()), Cost())
        print(f"coverage {name}: {len(todo)} sources, {cost.llm_calls} calls, {cost.tokens_in} in")
    ctx.out.mkdir(parents=True, exist_ok=True)
    (ctx.out / "coverage.json").write_text(
        json.dumps({"facts": facts, "judged": result, "errors": errors}, indent=1)
    )


def stage_systems(ctx: Ctx) -> None:
    pv = pilot_module()
    cfg, args = ctx.cfg, ctx.args
    descriptions = ctx.descriptions(True)
    corpus, missing = ctx.kept_corpus(descriptions)
    index, settings, store, embedder = ctx.index(corpus, descriptions)
    wanted = [s for s in args.systems.split(",") if s]
    rows = [r for r in read_jsonl(ctx.queries_file) if r["source"] not in set(missing)]
    if args.limit:
        rows = limit_rows(rows, args.limit)
    rows_by_id = {r["query_id"]: r for r in rows}
    queries = [Query(r["query_id"], r["query"], r["source"]) for r in rows]
    if args.measure:
        queries = queries[:: max(1, len(queries) // args.measure)][: args.measure]
    size, at = len(corpus.doc_ids), cfg["gate_at"]
    s0, _ = search_all(index, queries, size)
    gate_rows = [
        gate_row(rows_by_id[q.id], s0[q.id], size, cfg["full_corpus"], at) for q in queries
    ]
    ctx.out.mkdir(parents=True, exist_ok=True)
    out = ctx.out / "measure" if args.measure else ctx.out
    if args.limit:
        out = ctx.out / f"limit{args.limit}"
    out.mkdir(parents=True, exist_ok=True)
    gate_result = gate(gate_rows, cfg["gate_threshold"], at)
    meta = {"corpus_size": size, "left_out_empty_description": missing, "store": str(store)}
    (out / "gate.json").write_text(json.dumps({**gate_result, "meta": meta, "rows": gate_rows}))

    errors: list[str] = []
    llm_cache = shared_cache("llm", cfg["results_dir"])
    splitter = pv.QuerySplitter(
        pv.SafeCalls(pv.more_retries(make_llm(pv.SPLIT_SPEC, llm_cache)), errors),
        pv.SPLIT_PROMPT.read_text(),
    )
    verifier = pv.SafeCalls(pv.more_retries(make_llm(pv.VERIFIER_SPEC, llm_cache)), errors)
    plan = {
        "S2": pv.ConstraintVerifier(
            "S2",
            splitter,
            pv.DescriptionCheck(verifier, pv.DESC_PROMPT.read_text(), descriptions),
            args.inner,
        ),
        "S4": pv.HolisticReranker(
            "S4",
            pv.DescriptionScore(verifier, pv.RERANK_PROMPT.read_text(), descriptions),
            args.inner,
        ),
    }
    records = {
        "S0": {q.id: {"ranking": s0[q.id], "cost": Cost(searches=1).to_dict()} for q in queries}
    }
    if "S1" in wanted:
        with ThreadPoolExecutor(args.workers * args.inner) as pool:
            splits = dict(zip([q.id for q in queries], pool.map(lambda q: splitter(q.text), queries)))
        subs = [
            Query(f"{q.id}#c{i}", c, q.cluster)
            for q in queries
            for i, c in enumerate(splits[q.id][0], 1)
        ]
        sub_rank, _ = search_all(index, subs, size)
        records["S1"] = {}
        for q in queries:
            cons, cost = splits[q.id]
            lists = [sub_rank[f"{q.id}#c{i}"] for i in range(1, len(cons) + 1)]
            records["S1"][q.id] = {
                "ranking": pv.fuse(lists),
                "cost": (cost + Cost(searches=len(cons))).to_dict(),
                "trace": {"constraints": cons},
            }
    for s, reranker in plan.items():
        if s not in wanted:
            continue
        print(f"{s}: {len(queries)} queries x top-{args.n}", flush=True)
        records[s] = pv.rerank_all(reranker, queries, s0, args.n, args.workers, s)
        print(f"  {s} cost: {pv.per_call(records[s])}", flush=True)
    for s, rec in records.items():
        pv.write_rankings(out / f"{s}.jsonl", s, rec, rows_by_id)
    costs = {s: pv.per_call(r) for s, r in records.items() if s not in ("S0", "S1")}
    costs["n_errors"], costs["errors"] = len(errors), errors[:50]
    (out / "cost.json").write_text(json.dumps(costs, indent=1))
    total = sum((Cost(**r["cost"]) for rec in records.values() for r in rec.values()), Cost())
    write_manifest(
        out,
        models={
            "verifier": pv.VERIFIER_SPEC,
            "splitter": pv.SPLIT_SPEC,
            "embedder": cfg["embedder"],
            "describer": cfg["describer"],
        },
        seeds={"sample_seed": ctx.sample["seed"]},
        cost=total + embedder.cost,
        extra={"step": f"smoke variant S0/S2/S4 ({ctx.prompt.name})", "index": settings},
    )
    print(json.dumps(costs, indent=1))


# ---- report -----------------------------------------------------------------------------------


def stat_cell(s: dict) -> str:
    return f"{s['median']:.0f} [{s['p10']:.0f}, {s['p90']:.0f}]" if s["n"] else ""


def fmt_ci(d: dict, signed: bool = True) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(d['mean'])} [{f.format(d['ci_low'])}, {f.format(d['ci_high'])}]"


def stage_report(ctx: Ctx) -> None:
    pv = pilot_module()
    stem = ctx.stem
    rows = read_jsonl(ctx.queries_file)
    names = ("v4", stem)
    desc = {"v4": ctx.descriptions(False), stem: ctx.descriptions(True)}
    words = {n: description_words(ctx.full, desc[n]) for n in names}

    cov = json.loads((ctx.out / "coverage.json").read_text())
    shared = set(cov["judged"]["v4"]) & set(cov["judged"][stem])
    cov_rows = {
        n: coverage_rows(
            rows, cov["facts"], {s: j for s, j in cov["judged"][n].items() if s in shared}
        )
        for n in names
    }

    gates = {
        "v4": json.loads(GATE_JSON.read_text()),
        stem: json.loads((ctx.out / "gate.json").read_text()),
    }
    systems = {
        "v4": {s: r for s, r in pv.read_rankings(PILOT_DIR).items() if s in SYSTEMS},
        stem: {s: r for s, r in pv.read_rankings(ctx.out).items() if s in SYSTEMS},
    }
    labels = {
        "strict": pv.load_qrels(None, rows),
        "expanded": pv.load_qrels(Path(ctx.cfg["smoke_dir"]) / "qrels_expanded.jsonl", rows),
    }
    summary: dict = {"words": words, "coverage": {}, "gate": {}, "labels": {}}
    L = [
        f"# Describe-prompt variant: {stem} vs describe_v4 on the smoke set",
        "",
        (
            f'`prompts/{ctx.prompt.name}` differs from `describe_v4.txt` only in "3 to 5 '
            'sentences" (v4: "1 to 3"). Same 500 smoke videos, same 16 saved frames, same '
            f"describer ({ctx.cfg['describer']}) and embedder ({ctx.cfg['embedder']}). "
            f"Descriptions: `data/mc/<dataset>/{ctx.file}`; index: "
            f"`{gates[stem]['meta']['store']}` ({gates[stem]['meta']['corpus_size']} videos; "
            f"empty: {gates[stem]['meta']['left_out_empty_description']}). v4 numbers are the "
            f"existing gate (`{GATE_JSON}`) and pilot (`{PILOT_DIR}`). Script "
            "`dev/description_length_dev.py`."
        ),
        "",
        "## Description length (words)",
        "",
        "| group | n | v4 median [p10, p90] | " + stem + " median [p10, p90] |",
        "|---|---|---|---|",
    ]
    for key in ("overall", "dataset", "length_bin"):
        for g, a in words["v4"][key].items():
            b = words[stem][key].get(g, {"n": 0})
            L.append(f"| {g} | {a['n']}/{b['n']} | {stat_cell(a)} | {stat_cell(b)} |")
    L += [
        "",
        "## Fact coverage of the source's own description",
        "",
        (
            f"Per query: share of its k mined facts that the source's description states "
            f"(judge `{JUDGE_SPEC}`, `{JUDGE_PROMPT.name}` with the description as the text, one "
            f"call per (source, prompt), {len(shared)} sources with both descriptions). Mean "
            '[95% CI, bootstrap over sources]; "no" = share contradicted.'
        ),
        "",
        f"| group | n | v4 yes | {stem} yes | {stem} - v4 | v4 no | {stem} no |",
        "|---|---|---|---|---|---|---|",
    ]
    for key in (None, "k", "dataset", "length_bin"):
        a = grouped_means(cov_rows["v4"], "yes", key)
        b = grouped_means(cov_rows[stem], "yes", key)
        d = grouped_means(paired_rows(cov_rows[stem], cov_rows["v4"], "yes"), "yes", key)
        an = grouped_means(cov_rows["v4"], "no", key)
        bn = grouped_means(cov_rows[stem], "no", key)
        summary["coverage"][key or "overall"] = {"v4": a, stem: b, "diff": d}
        for g in a:
            L.append(
                f"| {g} | {a[g]['n']} | {fmt_ci(a[g], False)} | {fmt_ci(b[g], False)} | "
                f"{fmt_ci(d[g])} | {an[g]['mean']:.3f} | {bn[g]['mean']:.3f} |"
            )
    L += ["", "## S0 gate (predicted recall@20 of the 8,603-video corpus)", ""]
    for n in names:
        g = gates[n]
        summary["gate"][n] = {k: g[k] for k in ("passed", "failing_k", "overall", "per_k")}
        L.append(f"- {n}: **{'PASS' if g['passed'] else 'FAIL'}** (failing k: {g['failing_k']})")
    L += [
        "",
        "| group | n v4/"
        + stem
        + " | pred. R@20 v4 | pred. R@20 "
        + stem
        + " | R@1 v4 | R@1 "
        + stem
        + " | R@20 v4 | R@20 "
        + stem
        + " | median rank v4 / "
        + stem
        + " |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key in ("overall", "per_k", "per_dataset", "per_length_bin"):
        ga = {"all": gates["v4"]["overall"]} if key == "overall" else gates["v4"][key]
        gb = {"all": gates[stem]["overall"]} if key == "overall" else gates[stem][key]
        for grp, a in ga.items():
            b = gb.get(grp)
            if not b:
                continue
            ca = f"{a['predicted_recall@20']:.3f} [{a['ci_low']:.3f}, {a['ci_high']:.3f}]"
            cb = f"{b['predicted_recall@20']:.3f} [{b['ci_low']:.3f}, {b['ci_high']:.3f}]"
            label = f"k={grp}" if key == "per_k" else grp
            L.append(
                f"| {label} | {a['n']}/{b['n']} | {ca} | {cb} | {a['recall@1']:.3f} | "
                f"{b['recall@1']:.3f} | {a['recall@20']:.3f} | {b['recall@20']:.3f} | "
                f"{a['median_rank']:.0f} / {b['median_rank']:.0f} |"
            )
    for label, qrels in labels.items():
        scores = {n: {s: pv.score_system(r, qrels) for s, r in systems[n].items()} for n in names}
        sec: dict = {"means": {}, "paired_variant_vs_v4": {}, "within_variant": {}}
        L += ["", f"## Systems, {label} labels", ""]
        if label == "expanded":
            L += [
                (
                    "Expanded labels were judged on v4's S0 top 20 only, so a true match that "
                    f"only {stem} surfaces stays unlabelled (biased toward v4)."
                ),
                "",
            ]
        L += ["### nDCG@10 and R@1 (means)", ""]
        L += [
            "| system | group | n | nDCG@10 v4 | nDCG@10 " + stem + " | R@1 v4 | R@1 " + stem + " |"
        ]
        L.append("|---|---|---|---|---|---|---|")
        for s in SYSTEMS:
            for key in KEYS:
                ta = pv.mean_table(scores["v4"][s], rows, key)
                tb = pv.mean_table(scores[stem][s], rows, key)
                sec["means"].setdefault(s, {})[key or "overall"] = {"v4": ta, stem: tb}
                for g, a in ta.items():
                    b = tb.get(g, {})
                    L.append(
                        f"| {s} | {g} | {a['n']}/{b.get('n', 0)} | {a['ndcg@10']:.3f} | "
                        f"{b.get('ndcg@10', float('nan')):.3f} | {a['recall@1']:.3f} | "
                        f"{b.get('recall@1', float('nan')):.3f} |"
                    )
        L += ["", f"### Paired {stem} - v4, same system, same queries (95% CI)", ""]
        L += ["| system | group | n | nDCG@10 | R@1 |", "|---|---|---|---|---|"]
        for s in SYSTEMS:
            for key in KEYS:
                pn = pv.paired(scores[stem][s], scores["v4"][s], rows, "ndcg@10", key)
                pr = pv.paired(scores[stem][s], scores["v4"][s], rows, "recall@1", key)
                sec["paired_variant_vs_v4"].setdefault(s, {})[key or "overall"] = {
                    "ndcg@10": pn,
                    "recall@1": pr,
                }
                for g, d in pn.items():
                    L.append(f"| {s} | {g} | {d['n']} | {fmt_ci(d)} | {fmt_ci(pr[g])} |")
        L += ["", f"### Within {stem}: reranker - S0 (95% CI)", ""]
        L += ["| pair | group | n | nDCG@10 | R@1 |", "|---|---|---|---|---|"]
        for a_s in ("S4", "S2", "S4@10", "S2@10"):
            for key in KEYS:
                sa, s0 = scores[stem][a_s], scores[stem]["S0"]
                pn = pv.paired(sa, s0, rows, "ndcg@10", key)
                pr = pv.paired(sa, s0, rows, "recall@1", key)
                sec["within_variant"].setdefault(f"{a_s}-S0", {})[key or "overall"] = {
                    "ndcg@10": pn,
                    "recall@1": pr,
                }
                for g, d in pn.items():
                    L.append(f"| {a_s} - S0 | {g} | {d['n']} | {fmt_ci(d)} | {fmt_ci(pr[g])} |")
        summary["labels"][label] = sec
    cost_file = ctx.out / "cost.json"
    if cost_file.exists():
        c = json.loads(cost_file.read_text())
        L += ["", "## Cost of the variant systems (API usage)", ""]
        L += ["| system | calls | tokens in | tokens out |", "|---|---|---|---|"]
        for s in ("S2", "S4"):
            if s in c:
                L.append(f"| {s} | {c[s]['calls']} | {c[s]['tokens_in']} | {c[s]['tokens_out']} |")
        L.append(f"\nAPI errors: {c.get('n_errors', 0)}.")
    ctx.report.write_text("\n".join(L) + "\n")
    (ctx.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {ctx.report} and {ctx.out}/summary.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mc_smoke.yaml")
    ap.add_argument("--prompt", default="prompts/describe_v4b.txt")
    ap.add_argument(
        "--stage", default="all", choices=["index", "coverage", "systems", "report", "all"]
    )
    ap.add_argument("--queries", help="default <smoke_dir>/queries.jsonl")
    ap.add_argument("--out", help="default results/smoke/variant_<tag>/ (tag: v4b)")
    ap.add_argument("--report", help="default results/smoke/variant_<tag>.md")
    ap.add_argument("--measure", type=int, default=0, help="index/systems: only this many")
    ap.add_argument("--n", type=int, default=20, help="candidates for S2 and S4")
    ap.add_argument("--systems", default="S2,S4", help="systems run in the systems stage (+S1)")
    ap.add_argument("--limit", type=int, default=0, help="systems: only this many queries, by k")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--inner", type=int, default=4)
    args = ap.parse_args()
    ctx = Ctx(args)
    stages = ["index", "coverage", "systems", "report"] if args.stage == "all" else [args.stage]
    for stage in stages:
        {"index": stage_index, "coverage": stage_coverage, "systems": stage_systems,
         "report": stage_report}[stage](ctx)  # fmt: skip


if __name__ == "__main__":
    main()
