"""Reciprocal rank fusion."""

from __future__ import annotations

from collections import defaultdict

from src.types import Ranking


def rrf(rankings: list[Ranking], k: int = 60, weights: list[float] | None = None) -> Ranking:
    """Fuse ranked lists: score(d) = sum over lists of w / (k + rank), w = 1 unless weights are
    given. Ties go to the smaller id."""
    scores: dict[str, float] = defaultdict(float)
    for ranking, w in zip(rankings, weights or [1.0] * len(rankings), strict=True):
        for rank, (doc, _) in enumerate(ranking, start=1):
            scores[doc] += w / (k + rank)
    return sorted(scores.items(), key=lambda item: (-item[1], item[0]))
