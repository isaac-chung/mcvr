"""Fact-filter judge pass: does the short indexed description state the facts the grounding
filter KEPT as often as the facts it DROPPED? (docs/open_questions.md, "Fact-filter check" and
"Judge-pass design".) The judge call is PAID only in the `judge` stage; build, dry-run and
analyze are offline.

One text-only call per source (prompts/judge_v1.txt, openai:gpt-6-sol@low), made by the same
code as the coverage stage of dev/description_length_dev.py (its `judge_description`, `pilot_module`
helpers imported, the same `make_llm`, `shared_cache("llm")`, SafeCalls and more_retries), on the
source's describe_v4 description and ONE shuffled, unlabelled, numbered list that mixes

- kept facts: natural candidates both VLM grounders confirmed (grounding[].confirmed), and
- dropped facts: natural candidates not confirmed where one grounder said "yes" and the other
  "no" (346 facts; `--dropped-rule any_yes` also takes "yes"+"unsure", 465 facts). Facts both
  grounders rejected are left out (they include false caption facts).

Natural grounding records are role == "natural" in constraints.jsonl; their types are attribute,
object, count, on_screen_text ("text") and scene. "kept" is the grounder-confirmed set (1,499),
not only the facts that ended up in a query (selection).

Stages (--set test: the 280 test sources, descriptions_fps05.jsonl as indexed in the full run;
--set dev: the 98 development sources, the control, descriptions.jsonl as in the coverage stage):

- build: results/full/fact_filter_judge[_dev]/requests.jsonl (source, description, shuffled fact
  texts, assembled prompt; nothing about groups) and key.jsonl (hidden: per fact position, id,
  group, type, grounder verdicts, index_visible, shuffle seed, and for dev the existing kept-only
  answer).
- dry-run (or --dry-run with any stage): counts, calls, projected input tokens (words x
  TOKENS_PER_WORD of dev/build_queries_dev.py), one redacted prompt with --show. No API.
- judge: the calls through the shared cache (--sources N: a fixed seeded subset, --only ids).
  Results to judged.json (answers per prompt position).
- analyze: stated / contradicted / unclear by group and type, per-source kept - dropped, 95%
  cluster bootstrap over sources (10,000 draws, seed 0, src.stats.bootstrap_ci). For --set dev:
  the control, mixed call vs the existing kept-only coverage answer of the same facts.

    env -u OPENAI_API_KEY -u GEMINI_API_KEY uv run python pipeline/fact_judge.py \
        --stage build --dry-run --show
    uv run --env-file .env python pipeline/fact_judge.py --set dev --stage judge
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.stats import bootstrap_ci

ROOT = Path(__file__).resolve().parents[1]
SEED = 0  # fixed shuffle seed (per source: Random(f"{SEED}:{source}")); bootstrap seed too
DRAWS = 10_000
NATURAL_TYPES = ("attribute", "object", "count", "on_screen_text", "scene")
SMALL_FACTS, SMALL_SOURCES = 30, 15  # a cell with fewer facts or sources is flagged small
MAIN_PROMPT = "describe_v4.txt"
DATASETS = ("shot2story", "carebench", "vript_short", "vript_long")

SETS = {
    "test": {
        "constraints": "data/mc/full/constraints.jsonl",
        "descriptions": "descriptions_fps05.jsonl",
        "out": "results/full/fact_filter_judge",
    },
    "dev": {
        "constraints": "data/mc/smoke_v2/constraints.jsonl",
        "descriptions": "descriptions.jsonl",  # what description_length_dev.py's coverage "v4" used
        "out": "results/full/fact_filter_judge_dev",
        "reference": "results/smoke/variant_v4c/coverage.json",  # judged["v4"]: kept-only calls
        "dev_sources": "results/smoke/variant_v4c/coverage.json",  # facts: the 98 sources
    },
}


def load_module(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / rel)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---- pure helpers -----------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def verdicts(g: dict) -> dict[str, str]:
    return {spec: a["answer"] for spec, a in g["answers"].items()}


def group_of(g: dict, dropped_rule: str) -> str | None:
    """"kept" (both grounders yes), "dropped" (selected for the comparison) or None."""
    if g["confirmed"]:
        return "kept"
    said = sorted(verdicts(g).values())
    if dropped_rule == "one_yes_one_no":
        return "dropped" if said == ["no", "yes"] else None
    return "dropped" if "yes" in said else None


def natural_facts(record: dict, dropped_rule: str) -> list[dict]:
    out = []
    for g in record["grounding"]:
        if g.get("role") != "natural":
            continue
        group = group_of(g, dropped_rule)
        if group:
            out.append(
                {
                    "key": g["key"],
                    "fact_id": int(g["key"].lstrip("n")),
                    "text": g["text"],
                    "group": group,
                    "type": g.get("type"),
                    "verdicts": verdicts(g),
                    "index_visible": (g.get("index") or {}).get("visible"),
                }
            )
    return out


def shuffled(facts: list[dict], source: str) -> list[dict]:
    order = list(facts)
    random.Random(f"{SEED}:{source}").shuffle(order)
    return order


def load_descriptions(root: Path, filename: str) -> dict[str, str]:
    """Global id -> description written with describe_v4 (as corpus_descriptions)."""
    out = {}
    for ds in DATASETS:
        path = root / "data" / "mc" / ds / filename
        if path.exists():
            for r in read_jsonl(path):
                if r.get("prompt") == MAIN_PROMPT:
                    out[f"{ds}/{r['video_id']}"] = r["description"]
    return out


def assemble(template: str, description: str, texts: list[str]) -> str:
    """The prompt exactly as variant_index.judge_description builds it."""
    from src.judge import judge_prompt

    return judge_prompt(template, description.strip(), [{"text": t} for t in texts])


# ---- build ------------------------------------------------------------------------------------


def build(args) -> tuple[list[dict], list[dict], dict]:
    from src.judge import JUDGE_PROMPT

    cfg = SETS[args.set]
    template = (ROOT / JUDGE_PROMPT).read_text()
    records = read_jsonl(ROOT / cfg["constraints"])
    desc = load_descriptions(ROOT, cfg["descriptions"])
    ref, dev_sources = {}, None
    if args.set == "dev":
        cov = json.loads((ROOT / cfg["reference"]).read_text())
        dev_sources = set(cov["facts"])
        for s, texts in cov["facts"].items():
            judged = cov["judged"]["v4"].get(s)
            if judged:
                ref[s] = dict(zip(texts, judged["answers"], strict=True))
        records = [r for r in records if r["source"] in dev_sources]
    requests, key, notes = [], [], {"sources_without_description": [], "sources_one_group": []}
    totals = Counter()
    for r in sorted(records, key=lambda r: r["source"]):
        s = r["source"]
        facts = natural_facts(r, args.dropped_rule)
        totals["kept_all"] += sum(f["group"] == "kept" for f in facts)
        totals["dropped_all"] += sum(f["group"] == "dropped" for f in facts)
        if not facts:
            continue
        if not desc.get(s):
            notes["sources_without_description"].append(s)
            continue
        order = shuffled(facts, s)
        texts = [f["text"] for f in order]
        groups = {f["group"] for f in order}
        if len(groups) < 2:
            notes["sources_one_group"].append(s)
        requests.append(
            {
                "source": s,
                "dataset": r["dataset"],
                "description": desc[s],
                "facts": texts,
                "prompt": assemble(template, desc[s], texts),
            }
        )
        for pos, f in enumerate(order, 1):
            row = {"source": s, "pos": pos, **f, "seed": f"{SEED}:{s}"}
            if args.set == "dev":
                row["ref_answer"] = ref.get(s, {}).get(f["text"])  # kept-only existing call
            key.append(row)
    notes["totals_before_skips"] = dict(totals)
    return requests, key, notes


def write_build(args, requests, key, notes) -> Path:
    out = ROOT / SETS[args.set]["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in requests))
    (out / "key.jsonl").write_text("".join(json.dumps(k) + "\n" for k in key))
    (out / "build_notes.json").write_text(json.dumps({"dropped_rule": args.dropped_rule, **notes}, indent=1))
    return out


# ---- dry run ----------------------------------------------------------------------------------


def dry_run(args, requests, key, notes) -> None:
    b2 = load_module("build_v2", "dev/build_queries_dev.py")
    toks = [b2.text_tokens(r["prompt"]) for r in requests]
    print(f"set {args.set}; dropped rule {args.dropped_rule}; shuffle seed {SEED}")
    print(f"before skips: {notes['totals_before_skips']}")
    print(f"sources without a description (skipped): {notes['sources_without_description']}")
    print(f"sources with one group only: {len(notes['sources_one_group'])}")
    by_group = Counter(k["group"] for k in key)
    print(f"calls: {len(requests)}; facts in calls: {dict(by_group)}")
    print("per type (kept / dropped):")
    for t in sorted({k["type"] for k in key}):
        print(
            f"  {t:15s} {sum(k['group'] == 'kept' and k['type'] == t for k in key):5d} "
            f"{sum(k['group'] == 'dropped' and k['type'] == t for k in key):5d}"
        )
    print(f"facts per call: median {np.median([len(r['facts']) for r in requests]):.0f}, max "
          f"{max(len(r['facts']) for r in requests)}")
    print(
        f"projected input tokens (words x {b2.TOKENS_PER_WORD}, as dev/build_queries_dev.py): "
        f"per call median {np.median(toks):.0f}, min {min(toks):.0f}, max {max(toks):.0f}; "
        f"total {sum(toks):,.0f}"
    )
    pos = [(k["pos"] - 1) / max(1, n_of[k["source"]] - 1) for k in key if k["group"] == "dropped"
           for n_of in [Counter(x["source"] for x in key)] if n_of[k["source"]] > 1]
    print(f"shuffle check: mean relative position of dropped facts {np.mean(pos):.3f} (0.5 = ungrouped)")
    if args.show:
        mixed_sources = [i for i, r in enumerate(requests)
                         if {k["group"] for k in key if k["source"] == r["source"]} == {"kept", "dropped"}]
        r = requests[mixed_sources[min(args.show_index, len(mixed_sources) - 1)]]
        print(f"\n=== assembled prompt, source {r['source']} (no API key in it; hidden key not shown) ===")
        print(r["prompt"])
        mine = [k for k in key if k["source"] == r["source"]]
        print("=== end; group sequence by position (hidden, for the shuffle check only):")
        print(" ".join(k["group"][0].upper() for k in mine))


# ---- judge ------------------------------------------------------------------------------------


def judge(args, requests) -> None:
    import yaml

    from src.cache import shared_cache
    from src.judge import JUDGE_PROMPT, JUDGE_SPEC
    from src.llm import make_llm
    from src.types import Cost

    vi = load_module("variant_index", "dev/description_length_dev.py")
    pv = vi.pilot_module()
    cfg = yaml.safe_load((ROOT / "configs/mc_full.yaml").read_text())
    if args.only:
        wanted = set(args.only.split(","))
        requests = [r for r in requests if r["source"] in wanted]
    if args.sources:
        pick = random.Random(SEED).sample(sorted(r["source"] for r in requests), min(args.sources, len(requests)))
        requests = [r for r in requests if r["source"] in set(pick)]
    errors: list[str] = []
    llm = pv.SafeCalls(
        pv.more_retries(make_llm(JUDGE_SPEC, shared_cache("llm", ROOT / cfg["results_dir"]))), errors
    )
    template = (ROOT / JUDGE_PROMPT).read_text()

    def one(r):
        j = vi.judge_description(llm, template, r["description"], r["facts"])
        return r["source"], j

    with ThreadPoolExecutor(args.workers) as pool:
        judged = dict(pool.map(one, requests))
    cost = sum((Cost(**j["cost"]) for j in judged.values()), Cost())
    print(f"judge: {len(judged)} calls, {cost.llm_calls} paid (rest cached), {cost.tokens_in} in, {len(errors)} errors")
    out = ROOT / SETS[args.set]["out"]
    path = out / "judged.json"
    merged = json.loads(path.read_text())["judged"] if path.exists() else {}
    merged.update(judged)
    path.write_text(json.dumps({"judge": JUDGE_SPEC, "judged": merged, "errors": errors}, indent=1))


# ---- analyze ----------------------------------------------------------------------------------


def rate_ci(values: list[float], clusters: list[str]) -> tuple[float, float, float]:
    v = np.array(values, dtype=float)
    lo, hi = bootstrap_ci(v, clusters, n=DRAWS, seed=SEED)
    return float(v.mean()), lo, hi


def cell(rows: list[dict], label: str) -> dict:
    n_src = len({r["source"] for r in rows})
    out = {"cell": label, "facts": len(rows), "sources": n_src,
           "small": len(rows) < SMALL_FACTS or n_src < SMALL_SOURCES}
    for name, ans in (("stated", "yes"), ("contradicted", "no"), ("unclear", "unclear")):
        m, lo, hi = rate_ci([r["answer"] == ans for r in rows], [r["source"] for r in rows])
        out[name] = {"rate": m, "ci": [lo, hi]}
    return out


def per_source_diff(rows: list[dict]) -> dict:
    """Mean over sources with both groups of (kept stated rate - dropped stated rate)."""
    by = defaultdict(lambda: {"kept": [], "dropped": []})
    for r in rows:
        by[r["source"]][r["group"]].append(r["answer"] == "yes")
    diffs, src = [], []
    for s, g in by.items():
        if g["kept"] and g["dropped"]:
            diffs.append(np.mean(g["kept"]) - np.mean(g["dropped"]))
            src.append(s)
    if not diffs:
        return {"sources": 0}
    m, lo, hi = rate_ci(diffs, src)
    return {"sources": len(diffs), "mean_diff": m, "ci": [lo, hi]}


def joined(args) -> tuple[list[dict], dict]:
    out = ROOT / SETS[args.set]["out"]
    key = read_jsonl(out / "key.jsonl")
    if getattr(args, "sources_file", None):
        keep = set(json.loads(Path(args.sources_file).read_text())["sources"])
        key = [k for k in key if k["source"] in keep]
    data = json.loads((out / "judged.json").read_text())
    judged, bad = data["judged"], []
    if getattr(args, "sources_file", None):
        judged = {s: j for s, j in judged.items() if s in keep}
    rows = []
    for k in key:
        j = judged.get(k["source"])
        if j is None:
            continue
        if not j.get("raw"):
            bad.append(k["source"])
            continue
        rows.append({**k, "answer": j["answers"][k["pos"] - 1]})
    return rows, {"sources_judged": len(judged), "sources_with_error": len(set(bad)), "errors": data["errors"][:5]}


def analyze(args) -> None:
    rows, meta = joined(args)
    res = {"meta": meta, "cells": [], "diff": {}}
    for g in ("kept", "dropped"):
        res["cells"].append(cell([r for r in rows if r["group"] == g], g))
    # Dropped facts (one grounder yes, one no) split by which grounder said yes.
    for label, yes_prefix in (("dropped/gemini_yes", "gemini:"), ("dropped/astra_yes", "openai:")):
        res["cells"].append(cell(
            [r for r in rows if r["group"] == "dropped"
             and any(v == "yes" and k.startswith(yes_prefix) for k, v in r["verdicts"].items())],
            label))
    for t in NATURAL_TYPES:
        for g in ("kept", "dropped"):
            sub = [r for r in rows if r["group"] == g and r["type"] == t]
            if sub:
                res["cells"].append(cell(sub, f"{g}/{t}"))
    res["diff"]["overall"] = per_source_diff(rows)
    for t in NATURAL_TYPES:
        sub = [r for r in rows if r["type"] == t]
        if sub:
            res["diff"][t] = per_source_diff(sub)
    print(json.dumps(meta))
    print(f"{'cell':26s} {'facts':>5s} {'src':>4s}  stated [95% CI]            contradicted   unclear  flag")
    for c in res["cells"]:
        s = c["stated"]
        print(f"{c['cell']:26s} {c['facts']:5d} {c['sources']:4d}  {s['rate']:.3f} [{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
              f"   {c['contradicted']['rate']:.3f}        {c['unclear']['rate']:.3f}   {'SMALL' if c['small'] else ''}")
    print("per-source stated difference, kept - dropped (sources with both groups):")
    for k, d in res["diff"].items():
        if d.get("sources"):
            small = " SMALL" if d["sources"] < SMALL_SOURCES else ""
            print(f"  {k:15s} n_sources={d['sources']:3d} {d['mean_diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}]{small}")
    if args.set == "dev":
        res["control"] = control(rows)
    if getattr(args, "sources_file", None):
        res["meta"]["sources_file"] = args.sources_file
    out = Path(args.out) if getattr(args, "out", None) else ROOT / SETS[args.set]["out"]
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1))
    if getattr(args, "tables_dir", None):
        path = Path(args.tables_dir) / "fact_judge.tex"
        path.write_text(fact_judge_table(res, out / "analysis.json"))
        print(f"wrote {path}")


def fact_judge_table(res: dict, source: Path) -> str:
    """tables/fact_judge.tex: the kept / dropped cells and the per-source difference."""
    cells = {c["cell"]: c for c in res["cells"]}
    d = res["diff"]["overall"]

    def f(x: float) -> str:
        return f"{x:.3f}".lstrip("0") if x >= 0 else f"{x:.3f}".replace("-0.", "-.")

    def row(label: str, c: dict) -> str:
        s, k = c["stated"], c["contradicted"]
        return (
            f"{label} & {c['facts']:,} & {c['sources']} & {f(s['rate'])} [{f(s['ci'][0])}, "
            f"{f(s['ci'][1])}] & {f(k['rate'])} [{f(k['ci'][0])}, {f(k['ci'][1])}] & "
            f"{f(c['unclear']['rate'])} \\\\"
        )

    n_src = cells["kept"]["sources"]
    lines = [
        f"% Generated from {source} (pipeline/fact_judge.py, --stage analyze).",
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\setlength{\\tabcolsep}{3pt}",
        "\\resizebox{\\columnwidth}{!}{%",
        "\\begin{tabular}{@{}lrrccr@{}}",
        "\\toprule",
        "Group & Facts & Sources & Stated [95\\% CI] & Contradicted [95\\% CI] & Unclear \\\\",
        "\\midrule",
        row("Kept (both grounders yes)", cells["kept"]),
        row("Dropped (one grounder yes)", cells["dropped"]),
        "\\addlinespace[2pt]",
        row("\\quad of which Gemini yes, \\texttt{gpt-6-astra} no", cells["dropped/gemini_yes"]),
        row("\\quad of which Gemini no, \\texttt{gpt-6-astra} yes", cells["dropped/astra_yes"]),
        "\\bottomrule",
        "\\end{tabular}}",
        "\\caption{Does the indexed description state the facts the grounding filter kept? A "
        "text-only judge (\\texttt{gpt-6-sol}, low effort) read each source's short indexed "
        "description (about 44 words), never the video, and marked every fact of that source as "
        "stated, contradicted or unclear (not mentioned). Kept and dropped facts of a source were "
        f"shuffled together, unlabelled, in one call per source; {n_src} test sources. "
        "\\emph{Dropped}: exactly one of the two grounders (Gemini~3.5 Flash-Lite, "
        "\\texttt{gpt-6-astra}) said yes; facts both rejected are left out. Intervals are "
        f"{DRAWS:,}-draw cluster bootstraps over sources. Per-source stated rate, kept minus "
        f"dropped: {d['mean_diff']:+.3f} [{d['ci'][0]:+.3f}, {d['ci'][1]:+.3f}] over "
        f"{d['sources']} sources with both groups.}}".replace("+0.", "+.").replace("-0.", "-."),
        "\\label{tab:fact-judge}",
        "\\end{table}",
    ]
    return "\n".join(lines) + "\n"


def control_cell(kept: list[dict]) -> dict:
    mixed = np.array([r["answer"] == "yes" for r in kept], dtype=float)
    ref = np.array([r["ref_answer"] == "yes" for r in kept], dtype=float)
    diff = mixed - ref
    lo, hi = bootstrap_ci(diff, [r["source"] for r in kept], n=DRAWS, seed=SEED)
    table = Counter((r["ref_answer"], r["answer"]) for r in kept)
    return {
        "facts": len(kept),
        "ref_stated": float(ref.mean()),
        "mixed_stated": float(mixed.mean()),
        "diff": float(diff.mean()),
        "diff_ci": [lo, hi],
        "answer_agreement": float(np.mean([r["answer"] == r["ref_answer"] for r in kept])),
        "ref_to_mixed": {f"{a}->{b}": n for (a, b), n in sorted(table.items())},
    }


def control(rows: list[dict]) -> dict:
    """Mixed call vs the existing kept-only coverage call (results/smoke/variant_v4c/coverage.json,
    judged["v4"]) on the same facts and description. Those 366 facts are the old single-grounder
    query facts; only those that are kept or dropped under the two-grounder rule are in the mixed
    call. Reported for the kept ones (the comparison that matters), all, and the dropped ones."""
    have = [r for r in rows if r.get("ref_answer")]
    out = {
        "kept": control_cell([r for r in have if r["group"] == "kept"]),
        "dropped": control_cell([r for r in have if r["group"] == "dropped"]),
        "all": control_cell(have),
    }
    print("\nCONTROL (dev): mixed call vs existing kept-only call, same facts (stated = yes):")
    for g, c in out.items():
        print(
            f"  {g:8s} n={c['facts']:3d} kept-only {c['ref_stated']:.3f} -> mixed "
            f"{c['mixed_stated']:.3f}; diff {c['diff']:+.3f} [{c['diff_ci'][0]:+.3f}, "
            f"{c['diff_ci'][1]:+.3f}]; same answer {c['answer_agreement']:.2f}; {c['ref_to_mixed']}"
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="test", choices=list(SETS))
    ap.add_argument("--stage", default="dry-run", choices=["build", "dry-run", "judge", "analyze"])
    ap.add_argument("--dry-run", action="store_true", help="build in memory, print projections, no API")
    ap.add_argument("--dropped-rule", default="one_yes_one_no", choices=["one_yes_one_no", "any_yes"])
    ap.add_argument("--show", action="store_true", help="dry-run: print one assembled prompt")
    ap.add_argument("--show-index", type=int, default=0)
    ap.add_argument("--sources", type=int, default=0, help="judge: a fixed seeded subset of N sources")
    ap.add_argument("--only", help="judge: comma-separated source ids")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--sources-file", help="analyze: keep only these sources ({'sources': [...]})")
    ap.add_argument("--out", help="analyze: folder for analysis.json (default the set's out)")
    ap.add_argument("--tables-dir", help="analyze: also write fact_judge.tex there")
    args = ap.parse_args()
    if args.stage in ("build", "dry-run") or args.dry_run:
        requests, key, notes = build(args)
        if args.stage == "build":
            print(f"wrote {write_build(args, requests, key, notes)} ({len(requests)} requests, {len(key)} facts)")
        if args.dry_run or args.stage == "dry-run":
            dry_run(args, requests, key, notes)
        return
    if args.stage == "judge":
        out = ROOT / SETS[args.set]["out"]
        judge(args, read_jsonl(out / "requests.jsonl"))
    elif args.stage == "analyze":
        analyze(args)


if __name__ == "__main__":
    main()
