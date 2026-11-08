"""S0: one search. Embed the query, cosine search over the description index, keep the top k.

Also the smoke-set gate on S0 (docs/plan.md, "Smoke and dev set"). From the positive's rank r
among the N smoke videos, the chance that it would rank in the top 20 of the full corpus: each
of the M = full - 1 other full-corpus videos beats the positive independently with probability
p, p ~ Beta(r - 1 + 0.5, N - r + 0.5) (the Jeffreys posterior from the N - 1 smoke
distractors), so the number X that beat it is BetaBinomial(M, a, b) and
P(full rank <= 20) = P(X <= 19). Predicted recall@20 is its mean over queries, with a bootstrap
CI over source videos. The old rank-scaling rule (rank / N x full <= 20, which at N = 500 is
just smoke recall@1) is kept as `scaled_recall@20` for comparison.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
from scipy.stats import betabinom

from src.indexes.base import top_k
from src.indexes.dense_text import DenseTextIndex
from src.systems.base import Tools
from src.types import Cost, Query, Ranking, Run

FULL_CORPUS = 8603  # Shot2Story 4,023 + CaReBench 1,000 + Vript short 2,080 + Vript long 1,500


class SingleSearch:
    """S0 inside the systems framework: one search of the description index."""

    def __init__(self, name: str = "S0", index: str = "DESC"):
        self.name, self.index = name, index

    def run(self, query: Query, tools: Tools) -> Run:
        return tools.finish(self.name, tools.search(self.index))


def search_all(
    index: DenseTextIndex, queries: list[Query], k: int
) -> tuple[dict[str, Ranking], Cost]:
    """S0 for many queries at once: the query texts are embedded in one batch (one API call per
    batch instead of one per query), then each is a cosine search over the normalised vectors."""
    start = time.perf_counter()
    vectors = index.embedder.encode([q.text for q in queries], "query")
    rankings = {
        q.id: top_k(index.doc_ids, index.matrix @ v, k)
        for q, v in zip(queries, vectors, strict=True)
    }
    return rankings, Cost(searches=len(queries), seconds=time.perf_counter() - start)


# ---- the gate ----------------------------------------------------------------------------


def positive_rank(ranking: Ranking, positive: str) -> int | None:
    """1-based rank of the positive in a full ranking, None if absent."""
    for i, (doc, _) in enumerate(ranking, 1):
        if doc == positive:
            return i
    return None


def p_full_rank_within(rank: int, corpus_size: int, full_corpus: int = FULL_CORPUS, at: int = 20):
    """P(full-corpus rank <= at) for a positive at `rank` among corpus_size smoke videos:
    X ~ BetaBinomial(full_corpus - 1, rank - 1 + 0.5, corpus_size - rank + 0.5), P(X <= at - 1)."""
    if not 1 <= rank <= corpus_size:
        raise ValueError(f"rank {rank} outside 1..{corpus_size}")
    a, b = rank - 1 + 0.5, corpus_size - rank + 0.5
    return float(betabinom.cdf(at - 1, full_corpus - 1, a, b))


def gate_row(
    query: dict, ranking: Ranking, corpus_size: int, full_corpus: int = FULL_CORPUS, at: int = 20
) -> dict:
    """Per query: rank, percentile = rank / corpus size, the beta-binomial P(full rank <= at),
    the old scaled rank (percentile x full_corpus) and its hit, and actual hits at 1, 5, 20."""
    rank = positive_rank(ranking, query["source"])
    if rank is None:  # not in the list: treat as last
        rank = corpus_size
    percentile = rank / corpus_size
    scaled = percentile * full_corpus
    return {
        "query_id": query["query_id"],
        "source": query["source"],
        "dataset": query["dataset"],
        "length_bin": query["length_bin"],
        "k": int(query["k"]),
        "rank": rank,
        "percentile": percentile,
        f"p_hit@{at}": p_full_rank_within(rank, corpus_size, full_corpus, at),
        "scaled_rank": scaled,
        f"scaled_hit@{at}": scaled <= at,
        "hit@1": rank <= 1,
        "hit@5": rank <= 5,
        "hit@20": rank <= 20,
    }


def cluster_bootstrap_ci(
    values: list[float], clusters: list[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float]:
    """95% percentile CI of the mean, resampling clusters (source videos) with replacement."""
    groups: dict[str, list[float]] = {}
    for v, c in zip(values, clusters, strict=True):
        groups.setdefault(c, []).append(v)
    sums = np.array([sum(g) for g in groups.values()])
    counts = np.array([len(g) for g in groups.values()])
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, len(sums), size=(n_boot, len(sums)))
    means = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def summarise(rows: Iterable[dict], at: int = 20, n_boot: int = 2000) -> dict:
    rows = list(rows)
    if not rows:
        return {"n": 0}
    probs = [r[f"p_hit@{at}"] for r in rows]
    lo, hi = cluster_bootstrap_ci(probs, [r["source"] for r in rows], n_boot)
    return {
        "n": len(rows),
        "sources": len({r["source"] for r in rows}),
        "median_rank": float(np.median([r["rank"] for r in rows])),
        "mean_percentile": float(np.mean([r["percentile"] for r in rows])),
        f"predicted_recall@{at}": float(np.mean(probs)),
        "ci_low": lo,
        "ci_high": hi,
        f"scaled_recall@{at}": float(np.mean([r[f"scaled_hit@{at}"] for r in rows])),
        "recall@1": float(np.mean([r["hit@1"] for r in rows])),
        "recall@5": float(np.mean([r["hit@5"] for r in rows])),
        "recall@20": float(np.mean([r["hit@20"] for r in rows])),
    }


def gate(rows: list[dict], threshold: float = 0.6, at: int = 20, n_boot: int = 2000) -> dict:
    """Tables per k, dataset and length bin, and the verdict: pass if the beta-binomial
    predicted recall@`at` is at least threshold at every k."""

    def table(key: str) -> dict:
        groups: dict = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        return {g: summarise(rs, at, n_boot) for g, rs in sorted(groups.items())}

    per_k = table("k")
    failing = [k for k, s in per_k.items() if s[f"predicted_recall@{at}"] < threshold]
    return {
        "threshold": threshold,
        "at": at,
        "method": "beta-binomial, Jeffreys prior; CI: bootstrap over source videos",
        "passed": bool(per_k) and not failing,
        "failing_k": failing,
        "overall": summarise(rows, at, n_boot),
        "per_k": per_k,
        "per_dataset": table("dataset"),
        "per_length_bin": table("length_bin"),
    }
