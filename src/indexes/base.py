"""What a searchable index looks like."""

from __future__ import annotations

from typing import Protocol

import numpy as np

from src.types import Cost, Ranking


class Index(Protocol):
    name: str

    def search(self, text: str, k: int) -> tuple[Ranking, Cost]: ...


def top_k(doc_ids: list[str], scores: np.ndarray, k: int) -> Ranking:
    """Best k documents by score, ties broken by position for determinism."""
    k = min(k, len(doc_ids))
    order = np.lexsort((np.arange(len(scores)), -scores))[:k]
    return [(doc_ids[i], float(scores[i])) for i in order]
