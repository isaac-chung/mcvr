"""Retrieval and plan metrics.

nDCG uses the relevance grade as the gain (as trec_eval and ir_measures do), and recall divides
by all relevant videos, so numbers match the official MultiVENT evaluation script.
"""

from __future__ import annotations

import math

from src.types import Ranking


def _dcg(gains: list[int]) -> float:
    return sum(g / math.log2(i + 2) for i, g in enumerate(gains))


def ndcg_at_k(ranking: Ranking, grades: dict[str, int], k: int = 10) -> float:
    ideal = _dcg(sorted(grades.values(), reverse=True)[:k])
    if ideal == 0:
        return 0.0
    return _dcg([grades.get(doc, 0) for doc, _ in ranking[:k]]) / ideal


def recall_at_k(ranking: Ranking, grades: dict[str, int], k: int = 10) -> float:
    relevant = {d for d, g in grades.items() if g > 0}
    if not relevant:
        return 0.0
    return sum(1 for doc, _ in ranking[:k] if doc in relevant) / len(relevant)


def judged_at_k(ranking: Ranking, grades: dict[str, int], k: int = 10) -> float:
    """Share of the top k that has a judgment (unjudged videos count as non-relevant)."""
    return sum(1 for doc, _ in ranking[:k] if doc in grades) / k


def plan_scores(chosen: tuple[str, ...], best: tuple[str, ...]) -> dict[str, float]:
    """Precision and recall of the indexes a planner searched against the oracle subset,
    and the overhead in searches (positive means it searched more than needed)."""
    hit = len(set(chosen) & set(best))
    return {
        "plan_precision": hit / len(chosen) if chosen else 0.0,
        "plan_recall": hit / len(best),
        "plan_overhead": float(len(chosen) - len(best)),
    }
