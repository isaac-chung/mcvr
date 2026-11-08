"""Pooled frame judge for labels, with two-model agreement (docs/plan.md 4.5, replacement).

Per query source, the pool is the union of the top TOP candidates of every given system ranking
(S0, S4, and S2/S3/S5 where run) over all of the source's queries (natural and twin), plus the
twin video B and the source itself (a sanity check: it should meet its own facts). Each pooled
candidate gets one call per judge model (`prompts/judge_frames_v1.txt`) covering all of the
source's facts at once (the union over its queries, deduplicated), on the candidate's index
frames: at most MAX_FRAMES, uniformly subsampled, longest side MAX_SIDE px. Answers are "yes",
"no" or "unsure" per fact (`grounding_v2.parse_answers`).

A candidate is relevant to a query when both models say "yes" to every fact of the query, so
one judgment labels all nested queries of the source by logic. Candidates that enter the pool
only through k >= 3 queries ("extras") are judged on a seeded EXTRA_SHARE sample (AVS style);
the rest stay unjudged, and the share of sampled extras found relevant estimates how many
positives the unjudged ones hide (reported per k).

Outputs (see `dev/judge_labels_dev.py`): strict qrels (the source only), expanded qrels
(source plus candidates both models accept), and agreement of the two models (raw agreement and
Cohen's kappa) per fact type and on relevance decisions per k.
"""

from __future__ import annotations

import io
import random
from pathlib import Path

from src.constraints import seed_for
from src.grounding_v2 import GEMINI_SPEC, OPENAI_SPEC, agreement_table, cohen_kappa
from src.twins import norm_text

JUDGES = (GEMINI_SPEC, OPENAI_SPEC)
JUDGE_PROMPT = Path("prompts/judge_frames_v1.txt")
TOP = 10
MAX_FRAMES = 16
MAX_SIDE = 512
EXTRA_SHARE = 0.25
FULL_K = 2  # candidates reached by a query with k <= FULL_K are always judged


# ---- frames ------------------------------------------------------------------------------------


def subsample(n: int, cap: int = MAX_FRAMES) -> list[int]:
    """Indices of at most cap of n frames, uniform (slice midpoints), in order."""
    if n <= cap:
        return list(range(n))
    return sorted({int((i + 0.5) * n / cap) for i in range(cap)})


def resize_jpeg(jpeg: bytes, max_side: int = MAX_SIDE) -> bytes:
    """The JPEG with its longest side at most max_side (unchanged when already small)."""
    from PIL import Image

    with Image.open(io.BytesIO(jpeg)) as image:
        if max(image.size) <= max_side:
            return jpeg
        image = image.convert("RGB")
        image.thumbnail((max_side, max_side))
        out = io.BytesIO()
        image.save(out, format="JPEG", quality=90)
        return out.getvalue()


def judge_frames(
    frames: list[bytes], times: list[float], cap: int = MAX_FRAMES, max_side: int = MAX_SIDE
) -> tuple[list[bytes], list[float]]:
    keep = subsample(len(frames), cap)
    return [resize_jpeg(frames[i], max_side) for i in keep], [times[i] for i in keep]


# ---- facts and pools ---------------------------------------------------------------------------


def source_facts(rows: list[dict]) -> tuple[list[dict], dict[str, list[str]]]:
    """(facts, {query_id: fact keys}) for the query rows of one source: every fact once
    (normalised text), keyed "f<i>" in first-seen order."""
    facts, by_text, of_query = [], {}, {}
    for row in rows:
        keys = []
        for fact in row["facts"]:
            text = norm_text(fact["text"])
            if text not in by_text:
                by_text[text] = f"f{len(facts)}"
                facts.append({"key": by_text[text], "text": fact["text"], "type": fact["type"]})
            keys.append(by_text[text])
        of_query[row["query_id"]] = keys
    return facts, of_query


def build_pools(
    rows: list[dict],
    rankings: dict[str, dict[str, list[str]]],
    top: int = TOP,
    include_source: bool = True,
) -> dict[str, dict[str, dict]]:
    """{source: {candidate: {"min_k", "systems", "why"}}}. rankings: {system: {query_id: doc ids
    best first}}. "why" lists "pool" (in some top), "twin" (the twin video of a twin query) and
    "source"; min_k is the smallest k of a query whose top reached it (None if none did)."""
    pools: dict[str, dict[str, dict]] = {}

    def add(source: str, doc: str, why: str, k: int | None = None, system: str | None = None):
        entry = pools.setdefault(source, {}).setdefault(
            doc, {"min_k": None, "systems": [], "why": []}
        )
        if why not in entry["why"]:
            entry["why"].append(why)
        if system and system not in entry["systems"]:
            entry["systems"].append(system)
        if k is not None and (entry["min_k"] is None or k < entry["min_k"]):
            entry["min_k"] = k

    for row in rows:
        source = row["source"]
        pools.setdefault(source, {})
        if include_source:
            add(source, source, "source")
        if row.get("twin"):
            add(source, row["twin"], "twin")
        for system, by_query in sorted(rankings.items()):
            for doc in by_query.get(row["query_id"], [])[:top]:
                add(source, doc, "pool", row["k"], system)
    return pools


def plan_judging(
    pools: dict[str, dict[str, dict]],
    share: float = EXTRA_SHARE,
    full_k: int = FULL_K,
    seed: str = "judge_pool",
) -> dict[str, dict[str, bool]]:
    """{source: {candidate: judged?}}. Judged: the source, the twin, and candidates reached by a
    query with k <= full_k; the other candidates (extras, reached only at k > full_k) with a
    seeded probability `share` (the same draw for a (source, candidate) on every run)."""
    plan: dict[str, dict[str, bool]] = {}
    for source, pool in pools.items():
        plan[source] = {}
        for doc, entry in pool.items():
            if (
                {"source", "twin"} & set(entry["why"])
                or entry["min_k"] is not None
                and entry["min_k"] <= full_k
            ):
                plan[source][doc] = True
            else:
                rng = random.Random(seed_for(f"{source}|{doc}", seed))
                plan[source][doc] = rng.random() < share
    return plan


# ---- labels ------------------------------------------------------------------------------------


def relevant(answers: dict[str, dict[str, dict]], keys: list[str], specs: tuple[str, ...]) -> bool:
    """Every model in specs says "yes" to every fact key."""
    return bool(keys) and all(
        answers.get(k, {}).get(s, {}).get("answer") == "yes" for k in keys for s in specs
    )


def qrels(
    rows: list[dict],
    of_query: dict[str, list[str]],
    judged: dict[str, dict[str, dict]],
    specs: tuple[str, ...] = JUDGES,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """(strict, expanded): {query_id: relevant ids, source first}. judged: {source: {candidate:
    {fact key: {spec: answer}}}}. The twin video of a twin query never enters (it lacks d by
    construction; a judge that accepts it is counted separately as a contradiction)."""
    strict, expanded = {}, {}
    for row in rows:
        qid, source = row["query_id"], row["source"]
        strict[qid] = [source]
        extra = sorted(
            doc
            for doc, answers in judged.get(source, {}).items()
            if doc not in (source, row.get("twin")) and relevant(answers, of_query[qid], specs)
        )
        expanded[qid] = [source, *extra]
    return strict, expanded


def judged_in_top(
    row: dict, rankings: dict[str, dict[str, list[str]]], top: int = TOP
) -> list[str]:
    """Candidates of a query's pool: the union of the systems' top `top` for that query."""
    seen: dict[str, None] = {}
    for by_query in rankings.values():
        for doc in by_query.get(row["query_id"], [])[:top]:
            seen.setdefault(doc)
    return list(seen)


def false_negative_estimate(
    rows: list[dict],
    of_query: dict[str, list[str]],
    rankings: dict[str, dict[str, list[str]]],
    pools: dict[str, dict[str, dict]],
    plan: dict[str, dict[str, bool]],
    judged: dict[str, dict[str, dict]],
    specs: tuple[str, ...] = JUDGES,
    full_k: int = FULL_K,
    top: int = TOP,
) -> dict[str, dict]:
    """Per k > full_k (and "all"): over the queries' pooled extras (candidates reached only at
    k > full_k, other than source and twin), how many were sampled and judged, how many of those
    are relevant to the query, the rate, how many stayed unjudged, and the estimated missed
    positives (rate x unjudged), per query on average."""
    out: dict[str, dict] = {}
    for row in rows:
        if row["k"] <= full_k:
            continue
        source = row["source"]
        n_judged = n_rel = n_unjudged = 0
        for doc in judged_in_top(row, rankings, top):
            entry = pools[source][doc]
            if doc in (source, row.get("twin")) or entry["min_k"] is None:
                continue
            if entry["min_k"] <= full_k:
                continue
            if plan[source][doc] and doc in judged.get(source, {}):
                n_judged += 1
                n_rel += relevant(judged[source][doc], of_query[row["query_id"]], specs)
            else:
                n_unjudged += 1
        for name in (str(row["k"]), "all"):
            cell = out.setdefault(name, {"queries": 0, "judged": 0, "relevant": 0, "unjudged": 0})
            cell["queries"] += 1
            cell["judged"] += n_judged
            cell["relevant"] += n_rel
            cell["unjudged"] += n_unjudged
    for cell in out.values():
        rate = cell["relevant"] / cell["judged"] if cell["judged"] else None
        cell["rate"] = rate
        cell["est_missed"] = rate * cell["unjudged"] if rate is not None else None
        cell["est_missed_per_query"] = (
            cell["est_missed"] / cell["queries"] if rate is not None and cell["queries"] else None
        )
    return out


def agreement_stats(
    rows: list[dict],
    facts_of: dict[str, list[dict]],
    of_query: dict[str, list[str]],
    judged: dict[str, dict[str, dict]],
    specs: tuple[str, str] = JUDGES,
) -> dict:
    """{"facts": per fact type (grounding_v2.agreement_table over every judged (candidate,
    fact)), "relevance": per k raw agreement and kappa of the two models' relevance decisions
    over judged (query, candidate) pairs other than the source and twin, "verdicts": counts of
    (answer a, answer b)}."""
    a, b = specs
    items, verdicts = [], {}
    for source, cands in judged.items():
        types = {f["key"]: f["type"] for f in facts_of[source]}
        for answers in cands.values():
            for key, by_model in answers.items():
                items.append({"answers": by_model, "type": types.get(key)})
                pair = f"{by_model.get(a, {}).get('answer')}|{by_model.get(b, {}).get('answer')}"
                verdicts[pair] = verdicts.get(pair, 0) + 1
    by_k: dict[str, tuple[list[str], list[str]]] = {}
    for row in rows:
        for doc, answers in judged.get(row["source"], {}).items():
            if doc in (row["source"], row.get("twin")):
                continue
            ra = relevant(answers, of_query[row["query_id"]], (a,))
            rb = relevant(answers, of_query[row["query_id"]], (b,))
            for name in (str(row["k"]), "all"):
                xs, ys = by_k.setdefault(name, ([], []))
                xs.append(str(ra))
                ys.append(str(rb))
    relevance = {
        name: {
            "n": len(xs),
            "raw_agreement": sum(x == y for x, y in zip(xs, ys, strict=True)) / len(xs),
            "kappa": cohen_kappa(xs, ys),
            "relevant_a": xs.count("True"),
            "relevant_b": ys.count("True"),
            "relevant_both": sum(x == y == "True" for x, y in zip(xs, ys, strict=True)),
        }
        for name, (xs, ys) in sorted(by_k.items(), key=lambda kv: (kv[0] != "all", kv[0]))
    }
    return {
        "facts": agreement_table(items, a, b, by="type"),
        "relevance": relevance,
        "verdicts": dict(sorted(verdicts.items())),
    }


def sanity(rows: list[dict], of_query, judged, specs: tuple[str, ...] = JUDGES) -> dict:
    """Share of queries whose own source is judged relevant (should be near 1), and the number
    of twin queries whose twin is judged relevant (should be 0: it lacks d)."""
    own = [
        relevant(judged[r["source"]][r["source"]], of_query[r["query_id"]], specs)
        for r in rows
        if r["source"] in judged.get(r["source"], {})
    ]
    twins = [
        relevant(judged[r["source"]][r["twin"]], of_query[r["query_id"]], specs)
        for r in rows
        if r.get("twin") and r["twin"] in judged.get(r["source"], {})
    ]
    return {
        "source_relevant": sum(own) / len(own) if own else None,
        "n_source_checked": len(own),
        "twin_relevant": sum(twins),
        "n_twin_checked": len(twins),
    }
