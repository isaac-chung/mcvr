"""First pilot of the main systems on the smoke set. PAID (verifier and split calls).

S0 (one search over the 496-video description index, as in the dev S0 gate), S1 (split
the query, one search per constraint, RRF), S2 / S3 (per-constraint verification of S0's top-N
against descriptions / frames) and S4 / S5 (holistic 0-10 reranking against descriptions /
frames). Verifier `openai:gpt-6-sol@low`, splitter `openai:gpt-6-luna@low`; all calls go through
the shared caches, so a rerun is free.

S1, S2 and S4 run on all queries with N = 20; S3 and S5 on a seeded subset balanced over k
(--subset-per-k per k) with N = 10. Every ranking goes to results/smoke/pilot/<system>.jsonl
(full depth, with trace), and the report to results/smoke/pilot_report.md. Labels are strict
(the source video only); with --qrels (e.g. data/mc/smoke/qrels_expanded.jsonl) the saved
rankings are also scored against expanded labels.

    uv run --env-file .env python dev/verifiers_dev.py --measure 5   # cost check
    uv run --env-file .env python dev/verifiers_dev.py               # full pilot
    uv run python dev/verifiers_dev.py --eval-only [--qrels FILE]    # rescore only
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml

from src.cache import shared_cache
from src.datasets.mc import (
    McCorpus,
    corpus_descriptions,
    description_index,
    index_settings,
    index_store,
    read_jsonl,
)
from src.embedders import make_embedder
from src.llm import CachedVision, make_llm
from src.manifest import write_manifest
from src.metrics import ndcg_at_k, recall_at_k
from src.systems.base import Reranker, check_reranked, merge_reranked
from src.systems.decompose import (
    SPLIT_PROMPT,
    SPLIT_SPEC,
    GoldConstraints,
    QuerySplitter,
    fuse,
)
from src.systems.rerank import RERANK_PROMPT, DescriptionScore, FrameScore, HolisticReranker
from src.systems.single import cluster_bootstrap_ci, search_all
from src.systems.verify import (
    DESC_PROMPT,
    FRAMES_PROMPT,
    VERIFIER_SPEC,
    ConstraintVerifier,
    DescriptionCheck,
    FrameCheck,
    LabelledOpenAIVision,
    rank_by_score,
    saved_frame_source,
)
from src.types import Cost, Query, Ranking

OUT = Path("results/smoke/pilot")
REPORT = Path("results/smoke/pilot_report.md")
TOKEN_CAP = 40_000_000
METRICS = ("recall@1", "recall@5", "ndcg@10", "mrr")


class SafeCalls:
    """Wraps a cached model: an API error after the client's retries is logged and answered
    with an empty string (counts as unclear / unscored). Nothing is cached for it, so a rerun
    tries again."""

    def __init__(self, inner, log: list):
        self.inner, self.log = inner, log

    def complete(self, prompt: str):
        try:
            return self.inner.complete(prompt)
        except Exception as error:  # noqa: BLE001
            self.log.append(repr(error)[:300])
            return "", Cost()

    def describe(self, prompt: str, images: list[bytes], labels=None):
        try:
            return self.inner.describe(prompt, images, labels=labels)
        except Exception as error:  # noqa: BLE001
            self.log.append(repr(error)[:300])
            return "", Cost()


def more_retries(model, retries: int = 10):
    inner = model.inner
    inner.client = inner.client.with_options(max_retries=retries, timeout=300)
    return model


# ---- data -------------------------------------------------------------------------------------


def load_smoke(cfg: dict, queries_file: Path):
    smoke = Path(cfg["smoke_dir"])
    sample = json.loads((smoke / "sample.json").read_text())
    corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], sample["videos"])
    descriptions = corpus_descriptions(corpus, Path(cfg["describe_prompt"]).name)
    missing = {d for d in corpus.doc_ids if not descriptions.get(d)}
    if missing:
        kept = [d for d in corpus.doc_ids if d not in missing]
        corpus = McCorpus.pool(cfg["data_root"], cfg["datasets"], kept)
    rows = [r for r in read_jsonl(queries_file) if r["source"] not in missing]
    return corpus, descriptions, rows, sample


def subset_ids(rows: list[dict], per_k: int, seed: int = 0) -> list[str]:
    """per_k query ids per k, drawn with a seeded shuffle of the sorted ids."""
    out = []
    for k in sorted({r["k"] for r in rows}):
        ids = sorted(r["query_id"] for r in rows if r["k"] == k)
        random.Random(f"{seed}|k{k}").shuffle(ids)
        out += sorted(ids[:per_k])
    return out


# ---- running ----------------------------------------------------------------------------------


def rerank_all(
    reranker: Reranker,
    queries: list[Query],
    first: dict[str, Ranking],
    n: int,
    workers: int,
    label: str,
) -> dict[str, dict]:
    done, start = [0], time.perf_counter()

    def one(q: Query) -> tuple[str, dict]:
        candidates = first[q.id][:n]
        result = reranker.rerank(q, candidates)
        check_reranked(candidates, result.ranking)
        done[0] += 1
        if done[0] % 20 == 0:
            print(f"  {label}: {done[0]}/{len(queries)} ({time.perf_counter() - start:.0f} s)")
        record = {
            "ranking": merge_reranked(first[q.id], result.ranking),
            "cost": (Cost(searches=1) + result.cost).to_dict(),
            "trace": {"n": n, "first_stage": [d for d, _ in candidates], **result.trace},
        }
        return q.id, record

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, queries))


def write_rankings(path: Path, system: str, records: dict[str, dict], rows_by_id: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for qid, rec in records.items():
            row = rows_by_id[qid]
            line = {
                "query_id": qid,
                "system": system,
                "source": row["source"],
                "k": row["k"],
                "dataset": row["dataset"],
                "length_bin": row["length_bin"],
                "ranking": [[d, round(float(s), 6)] for d, s in rec["ranking"]],
                "cost": rec["cost"],
                "trace": rec.get("trace", {}),
            }
            f.write(json.dumps(line) + "\n")


def per_call(records: dict[str, dict]) -> dict:
    total = sum((Cost(**r["cost"]) for r in records.values()), Cost())
    calls = max(total.llm_calls, 1)
    return {
        "queries": len(records),
        "calls": total.llm_calls,
        "images": total.images,
        "tokens_in": total.tokens_in,
        "tokens_out": total.tokens_out,
        "in_per_call": total.tokens_in / calls,
        "out_per_call": total.tokens_out / calls,
        "model_version": total.model_version,
    }


# ---- evaluation -------------------------------------------------------------------------------


def load_qrels(path: Path | None, rows: list[dict]) -> dict[str, dict[str, int]]:
    """Strict labels (source = 1) plus, when path is given, the expanded labels in it. Lines may
    be {"query_id", "doc_id" | "video", "grade" | "relevance"}, {"query_id", "relevant": [...]}
    or {"query_id", "qrels": {doc: grade}}; the source always stays relevant."""
    qrels = {r["query_id"]: {r["source"]: 1} for r in rows}
    if path is None:
        return qrels
    for line in read_jsonl(path):
        qid = line.get("query_id")
        if qid not in qrels:
            continue
        grades: dict[str, int] = {}
        if isinstance(line.get("qrels"), dict):
            grades = {d: int(g) for d, g in line["qrels"].items()}
        elif isinstance(line.get("relevant"), list):
            grades = {d: 1 for d in line["relevant"]}
        else:
            doc = line.get("doc_id", line.get("video"))
            grade = line.get("grade", line.get("relevance", 1))
            if doc is not None:
                grades = {doc: int(bool(grade)) if isinstance(grade, bool) else int(grade)}
        for d, g in grades.items():
            qrels[qid][d] = max(qrels[qid].get(d, 0), g)
    return qrels


def query_metrics(ranking: Ranking, grades: dict[str, int]) -> dict[str, float]:
    relevant = {d for d, g in grades.items() if g > 0}
    rr = next((1.0 / i for i, (d, _) in enumerate(ranking, 1) if d in relevant), 0.0)
    return {
        # recall@1/@5: any relevant video in the top 1/5 (with one positive: plain recall)
        "recall@1": float(any(d in relevant for d, _ in ranking[:1])),
        "recall@5": float(any(d in relevant for d, _ in ranking[:5])),
        "ndcg@10": ndcg_at_k(ranking, grades, 10),
        "mrr": rr,
        "frac_recall@5": recall_at_k(ranking, grades, 5),
    }


def score_system(rankings: dict[str, Ranking], qrels) -> dict[str, dict[str, float]]:
    return {q: query_metrics(r, qrels[q]) for q, r in rankings.items()}


def mean_table(scores: dict, rows: list[dict], key: str | None) -> dict:
    groups: dict = {}
    for r in rows:
        if r["query_id"] in scores:
            groups.setdefault("all" if key is None else r[key], []).append(r["query_id"])
    return {
        g: {"n": len(ids), **{m: float(np.mean([scores[q][m] for q in ids])) for m in METRICS}}
        for g, ids in sorted(groups.items(), key=lambda x: str(x[0]))
    }


def paired(
    a: dict, b: dict, rows: list[dict], metric: str, key: str | None = None, n_boot: int = 2000
) -> dict:
    """Mean of a - b per group over shared queries, with a bootstrap CI over source videos."""
    groups: dict = {}
    for r in rows:
        q = r["query_id"]
        if q in a and q in b:
            groups.setdefault("all" if key is None else r[key], []).append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda x: str(x[0])):
        diffs = [a[r["query_id"]][metric] - b[r["query_id"]][metric] for r in rs]
        lo, hi = cluster_bootstrap_ci(diffs, [r["source"] for r in rs], n_boot)
        out[g] = {"n": len(rs), "mean": float(np.mean(diffs)), "ci_low": lo, "ci_high": hi}
    return out


def headroom(s0: dict[str, Ranking], rows: list[dict], n: int, ids: set | None = None) -> dict:
    """Per k: share of queries whose positive is in S0's top n but not at rank 1."""
    groups: dict = {}
    for r in rows:
        if (ids is not None and r["query_id"] not in ids) or r["query_id"] not in s0:
            continue
        top = [d for d, _ in s0[r["query_id"]][:n]]
        hit = r["source"] in top and top[0] != r["source"]
        groups.setdefault(r["k"], []).append(hit)
        groups.setdefault("all", []).append(hit)
    return {
        g: {"n": len(v), "share": float(np.mean(v))}
        for g, v in sorted(groups.items(), key=lambda x: (x[0] != "all", str(x[0])))
    }


def read_rankings(out: Path, top: int = 10) -> dict[str, dict[str, Ranking]]:
    """Saved rankings by system, plus S2@<top>, S2-gold@<top> and S4@<top>: S2 and S4 rescored on S0's top
    <top> only from their saved verdicts and scores (no calls), so they compare with S3 and S5
    at the same N."""
    systems = {}
    for path in sorted(out.glob("S*.jsonl")):
        lines = read_jsonl(path)
        systems[path.stem] = {
            line["query_id"]: [(d, s) for d, s in line["ranking"]] for line in lines
        }
        if path.stem in ("S2", "S4", "S2-gold"):
            derived = {}
            for line in lines:
                first = line["trace"]["first_stage"][:top]
                if path.stem.startswith("S2"):
                    v = line["trace"]["verdicts"]
                    scores = [sum(a == "yes" for a in v[d]) for d in first]
                else:
                    scores = [line["trace"]["scores"][d] for d in first]
                head = rank_by_score([(d, 0.0) for d in first], scores)
                derived[line["query_id"]] = merge_reranked(systems["S0"][line["query_id"]], head)
            systems[f"{path.stem}@{top}"] = derived
    return systems


def fmt(v) -> str:
    return f"{v:.3f}" if isinstance(v, float) else str(v)


def ci_cell(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def evaluate(rows: list[dict], systems: dict, qrels, subset: list[str]) -> dict:
    scores = {s: score_system(r, qrels) for s, r in systems.items()}
    sub = set(subset)
    sub_rows = [r for r in rows if r["query_id"] in sub]
    result: dict = {"tables": {}, "paired_vs_s0": {}, "subset": {}}
    for s, sc in scores.items():
        result["tables"][s] = {
            key or "overall": mean_table(sc, rows, key)
            for key in (None, "k", "dataset", "length_bin")
        }
        if s != "S0":
            result["paired_vs_s0"][s] = {
                m: {key or "overall": paired(sc, scores["S0"], rows, m, key) for key in (None, "k")}
                for m in ("ndcg@10", "recall@1")
            }
    pairs = [
        ("S3", "S2@10"),
        ("S5", "S4@10"),
        ("S3", "S2"),
        ("S5", "S4"),
        ("S3", "S5"),
        ("S2", "S4"),
        ("S3", "S0"),
        ("S5", "S0"),
        ("S2@10", "S0"),
    ]
    for a, b in pairs:
        if a in scores and b in scores:
            result["subset"][f"{a}-{b}"] = {
                m: {
                    key or "overall": paired(scores[a], scores[b], sub_rows, m, key)
                    for key in (None, "k")
                }
                for m in ("ndcg@10", "recall@1")
            }
    result["subset_tables"] = {
        s: {
            key or "overall": mean_table({q: v for q, v in sc.items() if q in sub}, rows, key)
            for key in (None, "k")
        }
        for s, sc in scores.items()
    }
    return result


def report_lines(title: str, ev: dict, systems: list[str]) -> list[str]:
    lines = [f"## {title}", ""]
    lines += ["### All queries (S3/S5 rows: subset only)", ""]
    for key in ("overall", "k", "dataset", "length_bin"):
        lines += [f"**{key}**", "", "| system | group | n | " + " | ".join(METRICS) + " |"]
        lines.append("|---" * (3 + len(METRICS)) + "|")
        for s in systems:
            for g, st in ev["tables"].get(s, {}).get(key, {}).items():
                vals = " | ".join(fmt(st[m]) for m in METRICS)
                lines.append(f"| {s} | {g} | {st['n']} | {vals} |")
        lines.append("")
    lines += ["### Paired difference vs S0 (95% CI, bootstrap over source videos)", ""]
    lines += ["| system | group | n | nDCG@10 | R@1 |", "|---|---|---|---|---|"]
    for s in systems:
        if s not in ev["paired_vs_s0"]:
            continue
        p = ev["paired_vs_s0"][s]
        for key in ("overall", "k"):
            for g, d in p["ndcg@10"][key].items():
                lines.append(
                    f"| {s} | {g} | {d['n']} | {ci_cell(d)} | {ci_cell(p['recall@1'][key][g])} |"
                )
    lines += ["", "### Subset (same queries): means", ""]
    lines += ["| system | group | n | " + " | ".join(METRICS) + " |"]
    lines.append("|---" * (3 + len(METRICS)) + "|")
    for s in systems:
        for key in ("overall", "k"):
            for g, st in ev["subset_tables"].get(s, {}).get(key, {}).items():
                vals = " | ".join(fmt(st[m]) for m in METRICS)
                lines.append(f"| {s} | {g} | {st['n']} | {vals} |")
    lines += ["", "### Subset: paired differences", ""]
    lines += ["| pair | group | n | nDCG@10 | R@1 |", "|---|---|---|---|---|"]
    for pair, p in ev["subset"].items():
        for key in ("overall", "k"):
            for g, d in p["ndcg@10"][key].items():
                lines.append(
                    f"| {pair} | {g} | {d['n']} | {ci_cell(d)} | {ci_cell(p['recall@1'][key][g])} |"
                )
    return [*lines, ""]


# ---- follow-up: gold constraints, larger frame subset, splitter error ------------------------

FOLLOWUP_METRICS = ("ndcg@10", "recall@1", "mrr")
MATCH_PROMPT = Path("prompts/split_match_v1.txt")
GROUP_KEYS = (None, "k", "dataset", "length_bin")
# (a, b, on the frame subset only?)
FOLLOWUP_PAIRS = (
    ("S2-gold", "S2", False),
    ("S2-gold@10", "S2@10", False),
    ("S3-gold", "S3", True),
    ("S3", "S2@10", True),
    ("S5", "S4@10", True),
    ("S3-gold", "S2-gold@10", True),
    ("S3-gold", "S5", True),
)


class Metered:
    """Wraps an uncached model (inside the cache) and sums the cost of the calls that really go
    out, so a rerun over a partly cached set reports only the new spend. Other attributes
    (params, client) pass through, so cache keys do not change."""

    def __init__(self, inner, log: list):
        self.inner, self.log = inner, log

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, prompt: str):
        text, cost = self.inner.complete(prompt)
        self.log.append(cost)
        return text, cost

    def describe(self, prompt: str, images: list[bytes], labels=None):
        if labels is None:
            text, cost = self.inner.describe(prompt, images)
        else:
            text, cost = self.inner.describe(prompt, images, labels=labels)
        self.log.append(cost)
        return text, cost


def per_k_counts(rows: list[dict], total: int) -> dict[int, int]:
    """total split as evenly as possible over k (the extra ones to the smallest k), capped by the
    queries available at each k."""
    ks = sorted({r["k"] for r in rows})
    base, extra = divmod(total, len(ks))
    have = {k: sum(r["k"] == k for r in rows) for k in ks}
    return {k: min(have[k], base + (i < extra)) for i, k in enumerate(ks)}


def subset_balanced(rows: list[dict], total: int, seed: int = 0) -> list[str]:
    """About total / (number of k) query ids per k, from the same seeded shuffle as subset_ids, so
    the subset for a larger total is a superset of a smaller one (and of subset_ids' subset)."""
    out = []
    for k, n in per_k_counts(rows, total).items():
        ids = sorted(r["query_id"] for r in rows if r["k"] == k)
        random.Random(f"{seed}|k{k}").shuffle(ids)
        out += sorted(ids[:n])
    return out


def saved_query_ids(path: Path) -> set[str]:
    return {line["query_id"] for line in read_jsonl(path)} if path.exists() else set()


def project_new(
    plan: dict[str, tuple[list[str], int]], done: dict[str, set[str]], in_per_call: dict
) -> dict:
    """Input tokens of the calls not yet made: per system, (queries, N) minus the queries whose
    rankings are already saved, times the measured input tokens per call."""
    est = {}
    for s, (ids, n) in plan.items():
        new = [q for q in ids if q not in done.get(s, set())]
        est[s] = {
            "new_queries": len(new),
            "calls": len(new) * n,
            "tokens_in": len(new) * n * in_per_call[s],
        }
    return {"systems": est, "total_in": sum(e["tokens_in"] for e in est.values())}


def fit_subset(rows: list[dict], total: int, seed: int, budget: float, estimate) -> int:
    """The largest balanced subset size <= total whose estimate(subset ids) stays under budget
    (steps of one query; the balanced draw keeps every smaller subset nested)."""
    while total > 0 and estimate(subset_balanced(rows, total, seed)) > budget:
        total -= 1
    return total


def token_overlap(a: str, b: str) -> float:
    """Token F1 of two strings (lower case, alphanumeric tokens, a few stop words dropped)."""
    stop = {"a", "an", "the", "is", "are", "of", "in", "on", "with", "and", "to", "at", "it"}

    def toks(t: str) -> list[str]:
        return [
            w
            for w in "".join(c if c.isalnum() else " " for c in t.lower()).split()
            if w not in stop
        ]

    x, y = toks(a), toks(b)
    if not x or not y:
        return 0.0
    common = sum(min(x.count(w), y.count(w)) for w in set(x))
    if not common:
        return 0.0
    p, r = common / len(y), common / len(x)
    return 2 * p * r / (p + r)


def parse_match(text: str, n_facts: int, n_cons: int) -> dict:
    """{"facts": [full|partial|none per fact], "constraints": [fact number or None per
    constraint]} from a split_match_v1 answer; missing entries count as none / None."""
    from src.constraints import as_int, parse_json

    data = parse_json(text)
    data = data if isinstance(data, dict) else {}
    facts = ["none"] * n_facts
    for pos, item in enumerate(data.get("facts") or [], 1):
        if isinstance(item, dict):
            i = as_int(item.get("fact"), pos)
            m = str(item.get("match", "none")).strip().lower()
            if 1 <= i <= n_facts:
                facts[i - 1] = m if m in ("full", "partial", "none") else "none"
    cons: list[int | None] = [None] * n_cons
    for pos, item in enumerate(data.get("constraints") or [], 1):
        if isinstance(item, dict):
            i = as_int(item.get("constraint"), pos)
            f = item.get("fact")
            f = as_int(f, 0) if f is not None else 0
            if 1 <= i <= n_cons and 1 <= f <= n_facts:
                cons[i - 1] = f
    return {"facts": facts, "constraints": cons}


def match_split(llm, template: str, facts: list[str], constraints: list[str]) -> dict:
    from src.systems.verify import numbered

    prompt = template.replace("{facts}", numbered(facts)).replace(
        "{constraints}", numbered(constraints)
    )
    text, _ = llm.complete(prompt)
    return parse_match(text, len(facts), len(constraints))


def split_stats(rows: list[dict], splits: dict[str, list[str]], matches: dict | None) -> dict:
    """Splitter error per k and overall: parsed-constraint count vs k, token-overlap match
    (each gold fact's best token F1 over the parsed constraints, matched at >= 0.5) and, when
    matches (query id -> match_split result) are given, the LLM match rates."""
    groups: dict = {}
    for r in rows:
        if r["query_id"] in splits:
            groups.setdefault(r["k"], []).append(r)
            groups.setdefault("all", []).append(r)
    out = {}
    for g, rs in sorted(groups.items(), key=lambda x: (x[0] != "all", str(x[0]))):
        counts = [len(splits[r["query_id"]]) for r in rs]
        diffs = [c - r["k"] for c, r in zip(counts, rs, strict=True)]
        best = [
            max(token_overlap(f["text"], c) for c in splits[r["query_id"]])
            for r in rs
            for f in r["facts"]
        ]
        st = {
            "n": len(rs),
            "mean_constraints": float(np.mean(counts)),
            "share_eq_k": float(np.mean([d == 0 for d in diffs])),
            "share_gt_k": float(np.mean([d > 0 for d in diffs])),
            "share_lt_k": float(np.mean([d < 0 for d in diffs])),
            "overlap_mean_best_f1": float(np.mean(best)),
            "overlap_fact_recall@0.5": float(np.mean([b >= 0.5 for b in best])),
        }
        if matches:
            ms = [matches[r["query_id"]] for r in rs if r["query_id"] in matches]
            fa = [m for x in ms for m in x["facts"]]
            ca = [c for x in ms for c in x["constraints"]]
            st.update(
                {
                    "llm_fact_full": float(np.mean([m == "full" for m in fa])),
                    "llm_fact_full_or_partial": float(np.mean([m != "none" for m in fa])),
                    "llm_constraint_precision": float(np.mean([c is not None for c in ca])),
                    "llm_exact_split": float(
                        np.mean(
                            [
                                len(x["constraints"]) == len(x["facts"])
                                and all(m == "full" for m in x["facts"])
                                and all(c is not None for c in x["constraints"])
                                for x in ms
                            ]
                        )
                    ),
                    "llm_constraints_per_matched_fact": float(
                        sum(c is not None for c in ca) / max(1, sum(m != "none" for m in fa))
                    ),
                }
            )
        out[g] = st
    return out


def evaluate_followup(rows: list[dict], systems: dict, qrels, subset: list[str]) -> dict:
    """Means and paired differences (FOLLOWUP_PAIRS) for nDCG@10, R@1 and MRR, overall and per
    k, dataset and length bin; frame-subset pairs only over the subset, all over shared
    queries."""
    scores = {s: score_system(r, qrels) for s, r in systems.items()}
    sub = set(subset)
    sub_rows = [r for r in rows if r["query_id"] in sub]
    res: dict = {"pairs": {}, "means_all": {}, "means_subset": {}}
    for a, b, on_sub in FOLLOWUP_PAIRS:
        if a not in scores or b not in scores:
            continue
        rs = sub_rows if on_sub else rows
        res["pairs"][f"{a} - {b}"] = {
            "subset": on_sub,
            **{
                m: {
                    key or "overall": paired(scores[a], scores[b], rs, m, key) for key in GROUP_KEYS
                }
                for m in FOLLOWUP_METRICS
            },
        }
    for s, sc in scores.items():
        res["means_all"][s] = {key or "overall": mean_table(sc, rows, key) for key in GROUP_KEYS}
        res["means_subset"][s] = {
            key or "overall": mean_table({q: v for q, v in sc.items() if q in sub}, rows, key)
            for key in GROUP_KEYS
        }
    return res


def followup_lines(title: str, ev: dict, all_order: list[str], sub_order: list[str]) -> list[str]:
    head = "| system | group | n | nDCG@10 | R@1 | MRR |"
    lines = [f"## {title}", "", "### Means, all queries", "", head, "|---|---|---|---|---|---|"]
    for s in all_order:
        for key in ("overall", "k", "dataset", "length_bin"):
            for g, st in ev["means_all"].get(s, {}).get(key, {}).items():
                vals = " | ".join(fmt(st[m]) for m in FOLLOWUP_METRICS)
                lines.append(f"| {s} | {g} | {st['n']} | {vals} |")
    lines += ["", "### Means, frame subset", "", head, "|---|---|---|---|---|---|"]
    for s in sub_order:
        for key in ("overall", "k", "dataset", "length_bin"):
            for g, st in ev["means_subset"].get(s, {}).get(key, {}).items():
                vals = " | ".join(fmt(st[m]) for m in FOLLOWUP_METRICS)
                lines.append(f"| {s} | {g} | {st['n']} | {vals} |")
    lines += ["", "### Paired differences (95% CI, bootstrap over source videos)", ""]
    lines += ["| pair | on | group | n | nDCG@10 | R@1 | MRR |", "|---|---|---|---|---|---|---|"]
    for pair, p in ev["pairs"].items():
        on = "subset" if p["subset"] else "all"
        for key in ("overall", "k", "dataset", "length_bin"):
            for g, d in p["ndcg@10"][key].items():
                cells = " | ".join(ci_cell(p[m][key][g]) for m in FOLLOWUP_METRICS)
                lines.append(f"| {pair} | {on} | {g} | {d['n']} | {cells} |")
    return [*lines, ""]


# ---- main -------------------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/mc_smoke.yaml")
    ap.add_argument("--queries", help="default <smoke_dir>/queries.jsonl")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--systems", default="S1,S2,S4,S3,S5")
    ap.add_argument("--n", type=int, default=20, help="candidates for S2 and S4")
    ap.add_argument("--n-frames-systems", type=int, default=10, help="candidates for S3, S5")
    ap.add_argument("--subset-per-k", type=int, default=15)
    ap.add_argument(
        "--subset-total",
        type=int,
        default=0,
        help="frame subset of this many queries, balanced over k (superset of --subset-per-k's)",
    )
    ap.add_argument(
        "--token-budget",
        type=float,
        default=25e6,
        help="cap on projected new input tokens; the frame subset is shrunk to fit",
    )
    ap.add_argument("--project-only", action="store_true", help="print the projection, stop")
    ap.add_argument("--split-stats", action="store_true", help="splitter error vs gold facts")
    ap.add_argument("--measure", type=int, default=0, help="cost check on this many queries")
    ap.add_argument("--workers", type=int, default=8, help="queries in parallel")
    ap.add_argument("--inner", type=int, default=4, help="candidates in parallel per query")
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--qrels", help="expanded qrels jsonl to score against as well")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    queries_file = Path(args.queries or Path(cfg["smoke_dir"]) / "queries.jsonl")
    corpus, descriptions, rows, sample = load_smoke(cfg, queries_file)
    rows_by_id = {r["query_id"]: r for r in rows}
    queries = [Query(r["query_id"], r["query"], r["source"]) for r in rows]
    subset = (
        subset_balanced(rows, args.subset_total, seed=sample["seed"])
        if args.subset_total
        else subset_ids(rows, args.subset_per_k, seed=sample["seed"])
    )
    out = Path(args.out)
    wanted = [s for s in args.systems.split(",") if s]

    if not args.eval_only:
        settings = index_settings(
            corpus.doc_ids,
            cfg["embedder"],
            cfg["describer"],
            cfg["describe_prompt"],
            cfg["n_frames"],
        )
        embedder = make_embedder(cfg["embedder"], cache=shared_cache("embed", cfg["results_dir"]))
        index = description_index(
            corpus, descriptions, embedder, index_store(cfg["results_dir"], settings)
        )
        depth = len(corpus.doc_ids)
        errors: list[str] = []
        llm_cache = shared_cache("llm", cfg["results_dir"])
        paid: dict[str, list] = {"luna": [], "sol_text": [], "sol_vision": []}
        split_cached = more_retries(make_llm(SPLIT_SPEC, llm_cache))
        split_cached.inner = Metered(split_cached.inner, paid["luna"])
        split_llm = SafeCalls(split_cached, errors)
        verifier_cached = more_retries(make_llm(VERIFIER_SPEC, llm_cache))
        verifier_cached.inner = Metered(verifier_cached.inner, paid["sol_text"])
        verifier = SafeCalls(verifier_cached, errors)
        vlm_inner = Metered(
            LabelledOpenAIVision(VERIFIER_SPEC.split(":", 1)[1].split("@")[0], "low"),
            paid["sol_vision"],
        )
        vlm = SafeCalls(
            more_retries(
                CachedVision(vlm_inner, shared_cache("vlm", cfg["results_dir"]), VERIFIER_SPEC)
            ),
            errors,
        )
        splitter = QuerySplitter(split_llm, SPLIT_PROMPT.read_text())
        gold = GoldConstraints.from_rows(rows)
        frames = saved_frame_source(lambda d: corpus.frames_dir(d, cfg["n_frames"]))

        if args.measure:
            ids = set(subset[:: max(1, len(subset) // args.measure)][: args.measure])
            run_queries = [q for q in queries if q.id in ids]
            sub_queries = run_queries
            out = out / "measure"
        else:
            run_queries = queries
            sub_queries = [q for q in queries if q.id in set(subset)]

        s0, _ = search_all(index, run_queries, depth)
        records = {
            "S0": {
                q.id: {"ranking": s0[q.id], "cost": Cost(searches=1).to_dict()} for q in run_queries
            }
        }

        with ThreadPoolExecutor(args.workers * args.inner) as pool:
            splits = dict(
                zip(
                    [q.id for q in run_queries],
                    pool.map(lambda q: splitter(q.text), run_queries),
                    strict=True,
                )
            )
        if "S1" in wanted:
            subs = [
                Query(f"{q.id}#c{i}", c, q.cluster)
                for q in run_queries
                for i, c in enumerate(splits[q.id][0], 1)
            ]
            sub_rank, _ = search_all(index, subs, depth)
            records["S1"] = {}
            for q in run_queries:
                cons, cost = splits[q.id]
                lists = [sub_rank[f"{q.id}#c{i}"] for i in range(1, len(cons) + 1)]
                records["S1"][q.id] = {
                    "ranking": fuse(lists),
                    "cost": (cost + Cost(searches=len(cons))).to_dict(),
                    "trace": {"constraints": cons},
                }

        desc_t, frames_t = DESC_PROMPT.read_text(), FRAMES_PROMPT.read_text()
        rerank_t = RERANK_PROMPT.read_text()
        plan = {
            "S2": (
                ConstraintVerifier(
                    "S2", splitter, DescriptionCheck(verifier, desc_t, descriptions), args.inner
                ),
                run_queries,
                args.n,
            ),
            "S4": (
                HolisticReranker(
                    "S4", DescriptionScore(verifier, rerank_t, descriptions), args.inner
                ),
                run_queries,
                args.n,
            ),
            "S3": (
                ConstraintVerifier("S3", splitter, FrameCheck(vlm, frames_t, frames), args.inner),
                sub_queries,
                args.n_frames_systems,
            ),
            "S5": (
                HolisticReranker("S5", FrameScore(vlm, rerank_t, frames), args.inner),
                sub_queries,
                args.n_frames_systems,
            ),
            "S2-gold": (
                ConstraintVerifier(
                    "S2-gold", gold, DescriptionCheck(verifier, desc_t, descriptions), args.inner
                ),
                run_queries,
                args.n,
            ),
            "S3-gold": (
                ConstraintVerifier("S3-gold", gold, FrameCheck(vlm, frames_t, frames), args.inner),
                sub_queries,
                args.n_frames_systems,
            ),
        }
        if not args.measure:
            old = (
                json.loads((out / "cost.json").read_text()) if (out / "cost.json").exists() else {}
            )
            per_call_in = {
                s: old.get(s, {}).get("in_per_call", 6000.0) for s in ("S2", "S3", "S4", "S5")
            }
            per_call_in["S2-gold"], per_call_in["S3-gold"] = per_call_in["S2"], per_call_in["S3"]
            done = {s: saved_query_ids(out / f"{s}.jsonl") for s in plan}
            frame_sys = [s for s in ("S3", "S5", "S3-gold") if s in wanted]

            def estimate(ids: list[str]) -> float:
                todo = {
                    s: ([q.id for q in plan[s][1]] if s not in frame_sys else ids, plan[s][2])
                    for s in plan
                    if s in wanted
                }
                return project_new(todo, done, per_call_in)["total_in"]

            proj = project_new(
                {s: ([q.id for q in plan[s][1]], plan[s][2]) for s in plan if s in wanted},
                done,
                per_call_in,
            )
            print("projected new input tokens:", json.dumps(proj, indent=1), flush=True)
            if proj["total_in"] > args.token_budget and frame_sys and args.subset_total:
                fitted = fit_subset(
                    rows, args.subset_total, sample["seed"], args.token_budget, estimate
                )
                print(f"over budget: frame subset shrunk {args.subset_total} -> {fitted}")
                subset = subset_balanced(rows, fitted, seed=sample["seed"])
                if estimate(subset) > args.token_budget:
                    sys.exit("projected new input tokens exceed --token-budget at any subset")
                sub_queries = [q for q in queries if q.id in set(subset)]
                for s in ("S3", "S5", "S3-gold"):
                    plan[s] = (plan[s][0], sub_queries, plan[s][2])
            elif proj["total_in"] > args.token_budget:
                sys.exit("projected new input tokens exceed --token-budget")
            if args.project_only:
                print(f"frame subset: {len(subset)}; new input {estimate(subset):,.0f} tokens")
                return
        for s in [s for s in ("S2", "S4", "S3", "S5", "S2-gold", "S3-gold") if s in wanted]:
            reranker, qs, n = plan[s]
            print(f"{s}: {len(qs)} queries x top-{n}", flush=True)
            records[s] = rerank_all(reranker, qs, s0, n, args.workers, s)
            print(f"  {s} cost: {per_call(records[s])}", flush=True)
        for s, rec in records.items():
            write_rankings(out / f"{s}.jsonl", s, rec, rows_by_id)
        split_cost = sum((c for _, c in splits.values()), Cost())
        costs = {}
        if (out / "cost.json").exists() and not args.measure:
            costs = json.loads((out / "cost.json").read_text())
        costs.update({s: per_call(r) for s, r in records.items() if s != "S0"})
        costs["split"] = {
            "calls": split_cost.llm_calls,
            "tokens_in": split_cost.tokens_in,
            "tokens_out": split_cost.tokens_out,
            "model_version": split_cost.model_version,
        }
        costs["errors"] = errors[:50]
        costs["n_errors"] = len(errors)
        if args.split_stats:
            match_llm = SafeCalls(split_cached, errors)
            template = MATCH_PROMPT.read_text()
            gold_facts = {r["query_id"]: [f["text"] for f in r["facts"]] for r in rows}
            ids = [q.id for q in run_queries]
            with ThreadPoolExecutor(args.workers * args.inner) as pool:
                found = pool.map(
                    lambda q: match_split(match_llm, template, gold_facts[q], splits[q][0]), ids
                )
                matches = dict(zip(ids, found, strict=True))
            stats = split_stats(rows, {q: splits[q][0] for q in ids}, matches)
            (out / "split_stats.json").write_text(
                json.dumps(
                    {
                        "matcher": SPLIT_SPEC,
                        "prompt": MATCH_PROMPT.name,
                        "stats": stats,
                        "matches": matches,
                    },
                    indent=1,
                )
            )
            print(json.dumps(stats, indent=1))
        costs["paid_this_run"] = {
            m: (sum(log, Cost())).to_dict() | {"calls_made": len(log)} for m, log in paid.items()
        }
        (out / "cost.json").write_text(json.dumps(costs, indent=1))
        total = sum((Cost(**r["cost"]) for s, rec in records.items() for r in rec.values()), Cost())
        write_manifest(
            out,
            models={
                "verifier": VERIFIER_SPEC,
                "splitter": SPLIT_SPEC,
                "embedder": cfg["embedder"],
                "describer": cfg["describer"],
            },
            seeds={"sample_seed": sample["seed"], "subset_seed": sample["seed"]},
            cost=total + embedder.cost,
            extra={
                "step": "smoke pilot S0-S5",
                "n": args.n,
                "n_frames_systems": args.n_frames_systems,
                "subset": subset,
                "subset_total": len(subset),
                "prompts": [
                    p.name for p in (SPLIT_PROMPT, DESC_PROMPT, FRAMES_PROMPT, RERANK_PROMPT)
                ],
            },
        )
        print(json.dumps(costs, indent=1))
        if args.measure:
            project(costs, len(queries), len(subset), args)
            return

    systems = read_rankings(out)
    order = [
        s
        for s in (
            "S0",
            "S1",
            "S2",
            "S2-gold",
            "S4",
            "S2@10",
            "S2-gold@10",
            "S4@10",
            "S3",
            "S3-gold",
            "S5",
        )
        if s in systems
    ]
    if args.subset_total == 0 and "S3" in systems and len(systems["S3"]) > len(subset):
        subset = sorted(systems["S3"])  # the saved frame subset
    ev = evaluate(rows, systems, load_qrels(None, rows), subset)
    lines = [
        "# Smoke pilot: S0-S5 over the 496-video description index",
        "",
        (
            f"Queries: {len(rows)} (`{queries_file}`); S3/S5 subset: {len(subset)} queries "
            f"({args.subset_per_k} per k, seed {sample['seed']}). Verifier `{VERIFIER_SPEC}`, "
            f"splitter `{SPLIT_SPEC}`. Rankings: `{out}/<system>.jsonl`."
        ),
        "",
        "### Rerank headroom (positive in S0 top-N but not at rank 1)",
        "",
        "| N | group | n | share |",
        "|---|---|---|---|",
    ]
    for n, ids in ((args.n, None), (args.n_frames_systems, set(subset))):
        for g, d in headroom(systems["S0"], rows, n, ids).items():
            lines.append(f"| {n}{' (subset)' if ids else ''} | {g} | {d['n']} | {d['share']:.3f} |")
    lines.append("")
    lines += report_lines("Strict labels (source only)", ev, order)
    summary = {"strict": ev}
    qrels_path = Path(args.qrels) if args.qrels else None
    if qrels_path and qrels_path.exists():
        ev_x = evaluate(rows, systems, load_qrels(qrels_path, rows), subset)
        lines += report_lines(f"Expanded labels (`{qrels_path}`)", ev_x, order)
        summary["expanded"] = ev_x
    all_order = [s for s in order if s not in ("S3", "S3-gold", "S5")]
    sub_order = [
        s for s in ("S0", "S2@10", "S2-gold@10", "S4@10", "S3", "S3-gold", "S5") if s in systems
    ]
    per_k = dict(sorted(Counter(rows_by_id[q]["k"] for q in subset).items()))
    fl = [f"Frame subset: {len(subset)} queries ({per_k} per k).", ""]
    summary["followup"] = {
        "strict": evaluate_followup(rows, systems, load_qrels(None, rows), subset)
    }
    fl += followup_lines(
        "Follow-up, strict labels", summary["followup"]["strict"], all_order, sub_order
    )
    if qrels_path and qrels_path.exists():
        summary["followup"]["expanded"] = evaluate_followup(
            rows, systems, load_qrels(qrels_path, rows), subset
        )
        fl += followup_lines(
            f"Follow-up, expanded labels (`{qrels_path}`)",
            summary["followup"]["expanded"],
            all_order,
            sub_order,
        )
    if (out / "split_stats.json").exists():
        summary["split_stats"] = json.loads((out / "split_stats.json").read_text())["stats"]
    (out / "followup_tables.md").write_text("\n".join(fl))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "report_tables.md").write_text("\n".join(lines))
    print(f"wrote {out}/report_tables.md and summary.json")


def project(costs: dict, n_queries: int, n_subset: int, args) -> None:
    """Projected input tokens of the full pilot from the measured per-call means."""
    est = {}
    for s, (qs, n) in {
        "S2": (n_queries, args.n),
        "S4": (n_queries, args.n),
        "S3": (n_subset, args.n_frames_systems),
        "S5": (n_subset, args.n_frames_systems),
    }.items():
        if s in costs:
            est[s] = {
                "calls": qs * n,
                "tokens_in": qs * n * costs[s]["in_per_call"],
                "tokens_out": qs * n * costs[s]["out_per_call"],
            }
    total_in = sum(e["tokens_in"] for e in est.values())
    print(json.dumps({"projection": est, "total_in": total_in, "cap": TOKEN_CAP}, indent=1))
    if total_in > TOKEN_CAP:
        print("projected input tokens exceed the cap: shrink --subset-per-k", file=sys.stderr)


if __name__ == "__main__":
    main()
