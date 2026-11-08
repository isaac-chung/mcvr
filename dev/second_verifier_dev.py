"""Second-verifier check of RQ1 on the v2 smoke pilot: does "S4 beats S2 on descriptions" hold
with other verifier models? PAID (text only).

Reuses, read-only, everything of dev/run_systems_dev.py's description stage: the same S0 top
20 (results/smoke_v2/pilot/<set>/S0.jsonl), the same luna constraints (<set>/splits.json, no new
split calls), the same 0.5 fps describe_v4 descriptions, and the same prompts
(prompts/verify_desc_v1.txt for S2, prompts/rerank_v1.txt for S4) through the unchanged
src.systems.verify / rerank classes. Only the verifier model changes:

    gpt-6-sol   openai:gpt-6-sol@low   (existing pilot rankings; no calls)
    gpt-5.5     openai:gpt-5.5@low     (accepts reasoning_effort; no temperature is sent)
    gemini      gemini:gemini-3.5-flash-lite   (text only; the model that wrote the descriptions)

Queries: a seeded subset of 150 per set (natural nested and twin), balanced over k
(pilot_verify.subset_balanced, seed 0). The paid stage projects new input tokens first and stops
if the projection plus spend.json would pass --cap (default 8M).

    uv run --env-file .env python dev/second_verifier_dev.py --stage run --project-only
    uv run --env-file .env python dev/second_verifier_dev.py --stage run
    uv run python dev/second_verifier_dev.py --stage eval

Outputs: results/smoke_v2/second_verifier/ (<set>/subset.json, <set>/S{0,2,4}_<verifier>.jsonl,
spend.json, cost.json, summary.json, report.md).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import spearmanr

from src.cache import shared_cache
from src.datasets.mc import McCorpus, corpus_descriptions, index_options, read_jsonl
from src.grounding_v2 import cohen_kappa
from src.llm import make_llm
from src.systems.rerank import (
    DESC_EVIDENCE,
    RERANK_PROMPT,
    DescriptionScore,
    HolisticReranker,
    fill,
)
from src.systems.verify import DESC_PROMPT, ConstraintVerifier, DescriptionCheck, numbered
from src.types import Cost, Query

ROOT = Path(__file__).resolve().parents[1]


def load_script(name: str):
    """A sibling smoke script as a module (read-only reuse of its functions)."""
    spec = importlib.util.spec_from_file_location(
        f"_sv_{name}", ROOT / "dev" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PV = load_script("verifiers_dev")
P2 = load_script("run_systems_dev")

CONFIG = Path("configs/mc_smoke_v2.yaml")
PILOT = Path("results/smoke_v2/pilot")
OUT = Path("results/smoke_v2/second_verifier")
JUDGE_DIR = Path("data/mc/smoke_v2/judge")
SETS = ("natural", "twin")
REFERENCE = "gpt-6-sol"
VERIFIERS = {
    "gpt-6-sol": "openai:gpt-6-sol@low",  # existing pilot rankings, not rerun
    "gpt-5.5": "openai:gpt-5.5@low",
    "gemini": "gemini:gemini-3.5-flash-lite",
}
NEW = ("gpt-5.5", "gemini")
TOTAL, SEED, N = 150, 0, 20
CAP = 8e6
METRICS = ("ndcg@10", "recall@1")
PAIRS = (("S4", "S2"), ("S4", "S0"), ("S2", "S0"))


# ---- helpers (offline-tested) -----------------------------------------------------------------


class FixedSplitter:
    """Stands in for QuerySplitter: returns the saved luna constraints of a query text, at no
    cost, so S2 sees exactly the pilot's constraints."""

    def __init__(self, by_text: dict[str, list[str]]):
        self.by_text = by_text

    def __call__(self, text: str) -> tuple[list[str], Cost]:
        return list(self.by_text[text]), Cost()


def constraints_by_text(rows: list[dict], splits: dict[str, list[str]]) -> dict[str, list[str]]:
    """{query text: constraints}; two queries with the same text must have the same split."""
    out: dict[str, list[str]] = {}
    for r in rows:
        cons = splits[r["query_id"]]
        if out.setdefault(r["query"], cons) != cons:
            raise ValueError(f"query text with two splits: {r['query']!r}")
    return out


def s2_prompt(template: str, query: str, constraints: list[str], desc: str) -> str:
    """The prompt DescriptionCheck builds."""
    return (
        template.replace("{query}", query)
        .replace("{constraints}", numbered(constraints))
        .replace("{description}", desc or "(no description)")
    )


def s4_prompt(template: str, query: str, desc: str) -> str:
    """The prompt DescriptionScore builds."""
    return fill(template, query, DESC_EVIDENCE.replace("{description}", desc or "(no description)"))


def project(
    llm, rows, s0, splits, descriptions, desc_t, rerank_t, n=N, chars_per_token=3.6
) -> tuple[float, int]:
    """New input tokens and calls of S2 + S4 for one model (prompts not yet in its cache)."""
    tokens, calls = 0.0, 0
    for r in rows:
        for d, _ in s0[r["query_id"]][:n]:
            desc = descriptions.get(d, "")
            for p in (
                s2_prompt(desc_t, r["query"], splits[r["query_id"]], desc),
                s4_prompt(rerank_t, r["query"], desc),
            ):
                if not llm.is_cached(p):
                    tokens += len(p) / chars_per_token
                    calls += 1
    return tokens, calls


def verdict_items(traces: dict[str, dict]) -> dict[tuple[str, str, int], str]:
    """{(query, candidate, constraint index): yes/no/unclear} from S2 traces {qid: trace}."""
    out = {}
    for q, t in traces.items():
        for d, verdicts in t["verdicts"].items():
            for i, v in enumerate(verdicts):
                out[(q, d, i)] = v
    return out


def verdict_agreement(a: dict, b: dict) -> dict:
    """Agreement of two verifiers' S2 verdicts on their shared (query, candidate, constraint)
    items: raw agreement, kappa on yes/no/unclear and on yes vs not yes, yes share of each."""
    keys = sorted(set(a) & set(b))
    va, vb = [a[k] for k in keys], [b[k] for k in keys]
    ya, yb = [str(v == "yes") for v in va], [str(v == "yes") for v in vb]
    n = len(keys)
    return {
        "n": n,
        "agree": float(np.mean([x == y for x, y in zip(va, vb, strict=True)])) if n else None,
        "kappa": cohen_kappa(va, vb),
        "kappa_yes": cohen_kappa(ya, yb),
        "share_a": {c: va.count(c) / n for c in ("yes", "no", "unclear")} if n else {},
        "share_b": {c: vb.count(c) / n for c in ("yes", "no", "unclear")} if n else {},
    }


def score_spearman(a: dict[str, dict], b: dict[str, dict]) -> dict:
    """Per query Spearman of two verifiers' S4 scores over the shared candidates ({qid: {doc:
    score}}); queries where either score list is constant are counted, not averaged."""
    rhos, constant = [], 0
    for q in sorted(set(a) & set(b)):
        docs = sorted(set(a[q]) & set(b[q]))
        x, y = [a[q][d] for d in docs], [b[q][d] for d in docs]
        if len(docs) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
            constant += 1
            continue
        rhos.append(float(spearmanr(x, y).statistic))
    return {
        "n": len(rhos),
        "constant": constant,
        "mean": float(np.mean(rhos)) if rhos else None,
        "median": float(np.median(rhos)) if rhos else None,
        "q25": float(np.percentile(rhos, 25)) if rhos else None,
    }


def top1_agreement(a: dict, b: dict, rows: list[dict]) -> float | None:
    """Share of queries whose reranked top 1 is the same under two verifiers."""
    same = [a[r["query_id"]][0][0] == b[r["query_id"]][0][0] for r in rows if r["query_id"] in a]
    return float(np.mean(same)) if same else None


def gap_difference(sc: dict, v: str, ref: str, rows: list[dict], metric: str, key=None) -> dict:
    """(S4 - S2 under v) - (S4 - S2 under ref) per query, mean with cluster-bootstrap CI."""
    a = {
        q: {metric: sc[v]["S4"][q][metric] - sc[v]["S2"][q][metric]}
        for q in sc[v]["S4"]
        if q in sc[v]["S2"]
    }
    b = {
        q: {metric: sc[ref]["S4"][q][metric] - sc[ref]["S2"][q][metric]}
        for q in sc[ref]["S4"]
        if q in sc[ref]["S2"]
    }
    return PV.paired(a, b, rows, metric, key)


# ---- data -------------------------------------------------------------------------------------


def load_data(cfg: dict) -> tuple[dict[str, str], dict[str, list[dict]]]:
    """Descriptions of the 0.5 fps index and the query rows of each set (as pilot_v2 loads
    them: sources with an empty description left out)."""
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], sample["videos"])
    descriptions = corpus_descriptions(
        corpus, Path(cfg["describe_prompt"]).name, index_options(cfg)["filename"]
    )
    missing = sorted(d for d in corpus.doc_ids if not descriptions.get(d))
    return descriptions, P2.load_sets(Path(cfg["out_dir"]), missing)


def read_lines(path: Path, ids: set[str]) -> dict[str, dict]:
    return {line["query_id"]: line for line in read_jsonl(path) if line["query_id"] in ids}


def snapshot(out: Path, sets: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Pick the subset of each set and copy the pilot's S0 / S2 / S4 lines of it (gpt-6-sol)
    into out, so later reruns of the pilot cannot change this comparison."""
    chosen = {}
    for name in SETS:
        have = {line["query_id"] for line in read_jsonl(PILOT / name / "S0.jsonl")}
        rows = [r for r in sets[name] if r["query_id"] in have]
        sub = out / name / "subset.json"
        if sub.exists():
            ids = json.loads(sub.read_text())
        else:
            ids = PV.subset_balanced(rows, TOTAL, SEED)
            sub.parent.mkdir(parents=True, exist_ok=True)
            sub.write_text(json.dumps(ids, indent=1))
        idset = set(ids)
        chosen[name] = [r for r in rows if r["query_id"] in idset]
        for system in ("S0", "S2", "S4"):
            path = out / name / f"{system}_{REFERENCE}.jsonl"
            if system == "S0":
                path = out / name / "S0.jsonl"
            if path.exists():
                continue
            lines = read_lines(PILOT / name / f"{system}.jsonl", idset)
            with path.open("w") as f:
                for q in ids:
                    f.write(json.dumps(lines[q]) + "\n")
        splits = json.loads((PILOT / name / "splits.json").read_text())
        (out / name / "splits.json").write_text(json.dumps({q: splits[q] for q in ids}, indent=1))
    return chosen


def read_s0(out: Path) -> dict:
    return {
        line["query_id"]: [(d, s) for d, s in line["ranking"]]
        for line in read_jsonl(out / "S0.jsonl")
    }


# ---- run --------------------------------------------------------------------------------------


def make_model(spec: str, cache, paid: list, errors: list):
    model = make_llm(spec, cache)
    if spec.startswith("openai:"):
        model = PV.more_retries(model)
    model.inner = PV.Metered(model.inner, paid)
    return model, PV.SafeCalls(model, errors)


def stage_run(cfg, args) -> None:
    out = Path(args.out)
    descriptions, sets = load_data(cfg)
    chosen = snapshot(out, sets)
    desc_t, rerank_t = DESC_PROMPT.read_text(), RERANK_PROMPT.read_text()
    cache = shared_cache("llm", cfg["results_dir"])
    ledger = P2.Ledger(out / "spend.json", args.cap)
    models = {}
    total, plan = 0.0, {}
    for v in args.verifiers:
        paid, errors = [], []
        cached, safe = make_model(VERIFIERS[v], cache, paid, errors)
        models[v] = (cached, safe, paid, errors)
        for name in SETS:
            s0 = read_s0(out / name)
            splits = json.loads((out / name / "splits.json").read_text())
            tok, calls = project(
                cached, chosen[name], s0, splits, descriptions, desc_t, rerank_t, args.n
            )
            plan[(v, name)] = (tok, calls)
            total += tok
            print(f"[project] {v} {name}: {calls} calls, {tok:,.0f} new input tokens")
    ledger.check("all", total)
    if args.project_only:
        return
    costs = json.loads((out / "cost.json").read_text()) if (out / "cost.json").exists() else {}
    for v in args.verifiers:
        cached, safe, paid, errors = models[v]
        for name in SETS:
            rows = chosen[name]
            by_id = {r["query_id"]: r for r in rows}
            s0 = read_s0(out / name)
            splits = json.loads((out / name / "splits.json").read_text())
            splitter = FixedSplitter(constraints_by_text(rows, splits))
            rerankers = {
                "S2": ConstraintVerifier(
                    "S2", splitter, DescriptionCheck(safe, desc_t, descriptions), args.inner
                ),
                "S4": HolisticReranker(
                    "S4", DescriptionScore(safe, rerank_t, descriptions), args.inner
                ),
            }
            queries = [Query(r["query_id"], r["query"], r["source"]) for r in rows]
            for system, reranker in rerankers.items():
                start = time.perf_counter()
                records = PV.rerank_all(
                    reranker, queries, s0, args.n, args.workers, f"{v} {name} {system}"
                )
                PV.write_rankings(out / name / f"{system}_{v}.jsonl", system, records, by_id)
                costs[f"{v} {name} {system}"] = {
                    **PV.per_call(records),
                    "wall_seconds": time.perf_counter() - start,
                }
                print(f"  {v} {name} {system}: {costs[f'{v} {name} {system}']}", flush=True)
        n_paid = len(paid)
        cost = sum(paid, Cost())
        costs[f"{v} errors"] = {"n": len(errors), "first": errors[:20]}
        ledger.add(v, cost, n_paid, {"spec": VERIFIERS[v], "errors": len(errors)})
        (out / "cost.json").write_text(json.dumps(costs, indent=1))


# ---- eval -------------------------------------------------------------------------------------


def stage_eval(cfg, args) -> None:
    out, judge = Path(args.out), Path(args.judge_dir)
    _, sets = load_data(cfg)
    chosen = snapshot(out, sets)
    result: dict = {"n": {n: len(r) for n, r in chosen.items()}, "labels": {}, "agreement": {}}
    rankings, lines = {}, {}
    for name in SETS:
        rankings[name], lines[name] = {}, {}
        s0 = read_s0(out / name)
        for v in VERIFIERS:
            rankings[name][v] = {"S0": s0}
            lines[name][v] = {}
            for system in ("S2", "S4"):
                path = out / name / f"{system}_{v}.jsonl"
                if not path.exists():
                    continue
                ls = {line["query_id"]: line for line in read_jsonl(path)}
                lines[name][v][system] = ls
                rankings[name][v][system] = {
                    q: [(d, s) for d, s in line["ranking"]] for q, line in ls.items()
                }
    have = [v for v in VERIFIERS if all(len(rankings[n][v]) == 3 for n in SETS)]
    result["verifiers"] = {v: VERIFIERS[v] for v in have}
    for kind in ("strict", "expanded"):
        res: dict = {}
        for name in SETS:
            rows = chosen[name]
            qrels = PV.load_qrels(judge / f"qrels_{kind}.jsonl", rows)
            sc = {
                v: {s: PV.score_system(r, qrels) for s, r in rankings[name][v].items()}
                for v in have
            }
            res[name] = {"means": {}, "paired": {}, "gap_vs_ref": {}}
            for v in have:
                res[name]["means"][v] = {
                    s: {
                        "overall": PV.mean_table(sc[v][s], rows, None)["all"],
                        "k": PV.mean_table(sc[v][s], rows, "k"),
                    }
                    for s in ("S0", "S2", "S4")
                }
                res[name]["paired"][v] = {
                    f"{a} - {b}": {
                        m: {
                            "overall": PV.paired(sc[v][a], sc[v][b], rows, m)["all"],
                            "k": PV.paired(sc[v][a], sc[v][b], rows, m, "k"),
                        }
                        for m in METRICS
                    }
                    for a, b in PAIRS
                }
                if v != REFERENCE and REFERENCE in have:
                    res[name]["gap_vs_ref"][v] = {
                        m: gap_difference(sc, v, REFERENCE, rows, m)["all"] for m in METRICS
                    }
        result["labels"][kind] = res
    for name in SETS:
        ag: dict = {"s2_verdicts": {}, "s4_spearman": {}, "s4_top1": {}, "s4_failed": {}}
        items = {
            v: verdict_items({q: ln["trace"] for q, ln in lines[name][v]["S2"].items()})
            for v in have
        }
        scores = {
            v: {q: ln["trace"]["scores"] for q, ln in lines[name][v]["S4"].items()} for v in have
        }
        for v in have:
            allsc = [s for d in scores[v].values() for s in d.values()]
            ag["s4_failed"][v] = sum(s < 0 for s in allsc)
        for a, b in combinations(have, 2):
            key = f"{a} vs {b}"
            ag["s2_verdicts"][key] = verdict_agreement(items[a], items[b])
            ag["s4_spearman"][key] = score_spearman(scores[a], scores[b])
            ag["s4_top1"][key] = top1_agreement(
                rankings[name][a]["S4"], rankings[name][b]["S4"], chosen[name]
            )
        result["agreement"][name] = ag
    spend = out / "spend.json"
    result["spend"] = json.loads(spend.read_text())["entries"] if spend.exists() else []
    cost = out / "cost.json"
    result["costs"] = json.loads(cost.read_text()) if cost.exists() else {}
    (out / "summary.json").write_text(json.dumps(result, indent=1, default=str))
    notes = out / "notes.md"  # hand-written findings, placed after the title
    text = report(result)
    if notes.exists():
        head, _, rest = text.partition("\n\n")
        text = f"{head}\n\n{notes.read_text().strip()}\n\n{rest}"
    (out / "report.md").write_text(text)
    print(f"wrote {out}/report.md and summary.json")


# ---- report -----------------------------------------------------------------------------------


def ci(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def report(res: dict) -> str:
    vs = list(res["verifiers"])
    L = [
        "# Second verifier check of RQ1 (S4 vs S2 on descriptions)",
        "",
        (
            f"Generated {datetime.now(UTC).isoformat(timespec='seconds')} by "
            "`dev/second_verifier_dev.py`."
        ),
        "",
        f"Setup: seeded subset of {res['n']['natural']} natural nested and {res['n']['twin']} "
        "twin queries (pilot_verify.subset_balanced, seed 0, balanced over k), the pilot's S0 "
        "top 20, the pilot's luna constraints (splits.json), the 0.5 fps describe_v4 "
        "descriptions, prompts verify_desc_v1 (S2) and rerank_v1 (S4). Verifiers: "
        + ", ".join(f"{v} = `{s}`" for v, s in res["verifiers"].items())
        + f". {REFERENCE} rows are the pilot's saved rankings on the same queries (no new calls). "
        "gpt-5.5 got reasoning_effort=low and no temperature; Gemini got the client default "
        "(temperature 0, default thinking).",
        "",
        (
            "**Caveat: Gemini (gemini-3.5-flash-lite) also wrote the descriptions**, so as a "
            "description reader it may agree more with its own phrasing and favour description-"
            "based reranking (S2 and S4 alike); its rows are not independent of the index."
        ),
        "",
        "CIs: 95%, cluster bootstrap over source videos (2,000 resamples).",
        "",
    ]
    for kind, by_set in res["labels"].items():
        L += [f"## {kind.capitalize()} labels", ""]
        for name, r in by_set.items():
            L += [f"### {name}", "", "Means (nDCG@10 / R@1):", ""]
            L += ["| verifier | S0 | S2 | S4 |", "|---|---|---|---|"]
            for v in vs:
                m = r["means"][v]
                L.append(
                    f"| {v} | "
                    + " | ".join(
                        f"{m[s]['overall']['ndcg@10']:.3f} / {m[s]['overall']['recall@1']:.3f}"
                        for s in ("S0", "S2", "S4")
                    )
                    + " |"
                )
            L += ["", "Paired differences, overall:", ""]
            L += ["| verifier | pair | nDCG@10 | R@1 |", "|---|---|---|---|"]
            for v in vs:
                for pair, d in r["paired"][v].items():
                    L.append(
                        f"| {v} | {pair} | {ci(d['ndcg@10']['overall'])} | "
                        f"{ci(d['recall@1']['overall'])} |"
                    )
            L += ["", "Paired nDCG@10 by k (n per k in brackets):", ""]
            L += [
                "| verifier | pair | k = 1 | k = 2 | k = 3 | k = 4 |",
                "|---|---|---|---|---|---|",
            ]
            for v in vs:
                for pair, d in r["paired"][v].items():
                    cells = [
                        f"{ci(d['ndcg@10']['k'][k])} ({d['ndcg@10']['k'][k]['n']})"
                        for k in sorted(d["ndcg@10"]["k"], key=str)
                    ]
                    L.append(f"| {v} | {pair} | " + " | ".join(cells) + " |")
            L += ["", "Paired R@1 by k:", ""]
            L += [
                "| verifier | pair | k = 1 | k = 2 | k = 3 | k = 4 |",
                "|---|---|---|---|---|---|",
            ]
            for v in vs:
                for pair, d in r["paired"][v].items():
                    cells = [ci(d["recall@1"]["k"][k]) for k in sorted(d["recall@1"]["k"], key=str)]
                    L.append(f"| {v} | {pair} | " + " | ".join(cells) + " |")
            if r["gap_vs_ref"]:
                L += ["", f"(S4 - S2) gap of a verifier minus that of {REFERENCE}:", ""]
                L += ["| verifier | nDCG@10 | R@1 |", "|---|---|---|"]
                for v, d in r["gap_vs_ref"].items():
                    L.append(f"| {v} | {ci(d['ndcg@10'])} | {ci(d['recall@1'])} |")
            L.append("")
    L += ["## Verifier agreement", ""]
    for name, ag in res["agreement"].items():
        L += [f"### {name}", ""]
        L += [
            (
                "S2 verdicts per (query, candidate, constraint); S4 scores: per-query Spearman over "
                "the 20 candidates (queries with a constant score list excluded), and same top 1."
            ),
            "",
            (
                "| pair | S2 items | S2 agree | S2 kappa (3-way) | S2 kappa (yes vs not) | "
                "S4 Spearman mean / median (n, constant) | S4 same top 1 |"
            ),
            "|---|---|---|---|---|---|---|",
        ]
        for key, a in ag["s2_verdicts"].items():
            s = ag["s4_spearman"][key]
            L.append(
                f"| {key} | {a['n']} | {a['agree']:.3f} | {a['kappa']:.3f} | "
                f"{a['kappa_yes']:.3f} | {s['mean']:.3f} / {s['median']:.3f} "
                f"({s['n']}, {s['constant']}) | {ag['s4_top1'][key]:.3f} |"
            )
        L += ["", "S2 verdict shares (yes / no / unclear):", ""]
        shares = {}
        for key, a in ag["s2_verdicts"].items():
            x, y = key.split(" vs ")
            shares.setdefault(x, a["share_a"])
            shares.setdefault(y, a["share_b"])
        for v, sh in shares.items():
            L.append(f"- {v}: {sh['yes']:.3f} / {sh['no']:.3f} / {sh['unclear']:.3f}")
        L.append(f"- S4 failed scores (-1): {ag['s4_failed']}")
        L.append("")
    L += ["## Spend (new paid calls, this script)", ""]
    for e in res["spend"]:
        L.append(
            f"- {e['stage']} (`{e.get('spec', '')}`): {e['calls']:,} calls, "
            f"{e['tokens_in']:,} in / {e['tokens_out']:,} out tokens, errors {e.get('errors', 0)}"
        )
    L += [
        "",
        "Per system (all calls, cached included):",
        "",
        "| run | calls | tokens in | tokens out | in per call | model version |",
        "|---|---|---|---|---|---|",
    ]
    for key, c in res["costs"].items():
        if "calls" in c:
            L.append(
                f"| {key} | {c['calls']:,} | {c['tokens_in']:,} | {c['tokens_out']:,} | "
                f"{c['in_per_call']:.0f} | {c['model_version']} |"
            )
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--stage", required=True, choices=("run", "eval"))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--judge-dir", default=str(JUDGE_DIR))
    ap.add_argument("--verifiers", default=",".join(NEW))
    ap.add_argument("--cap", type=float, default=CAP)
    ap.add_argument("--n", type=int, default=N)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--inner", type=int, default=4)
    ap.add_argument("--project-only", action="store_true")
    args = ap.parse_args(argv)
    args.verifiers = [v for v in args.verifiers.split(",") if v]
    unknown = [v for v in args.verifiers if v not in NEW]
    if unknown:
        sys.exit(f"--verifiers: unknown {unknown}; choose from {NEW}")
    cfg = yaml.safe_load(Path(args.config).read_text())
    if args.stage == "run":
        stage_run(cfg, args)
    else:
        stage_eval(cfg, args)


if __name__ == "__main__":
    main()
