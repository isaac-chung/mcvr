"""Plain records shared by every module."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np

# query id -> doc id -> relevance grade (0 means judged non-relevant)
Qrels = dict[str, dict[str, int]]
# (doc id, score), best first
Ranking = list[tuple[str, float]]


@dataclass(frozen=True)
class Query:
    id: str
    text: str
    cluster: str  # queries sharing a cluster are resampled together in the bootstrap
    tags: tuple[str, ...] = ()


@dataclass
class Cost:
    searches: int = 0
    llm_calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    seconds: float = 0.0
    images: int = 0  # frames sent to a vision model
    # model versions the APIs returned, comma separated and sorted ("" when none reported)
    model_version: str = ""

    def __add__(self, other: Cost) -> Cost:
        versions = {v for c in (self, other) for v in c.model_version.split(",") if v}
        return Cost(
            self.searches + other.searches,
            self.llm_calls + other.llm_calls,
            self.tokens_in + other.tokens_in,
            self.tokens_out + other.tokens_out,
            self.seconds + other.seconds,
            self.images + other.images,
            ",".join(sorted(versions)),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class Run:
    """What one system did for one query."""

    query_id: str
    system: str
    ranking: Ranking
    indexes: tuple[str, ...]  # indexes it searched
    cost: Cost
    trace: dict = field(default_factory=dict)


@dataclass
class Corpus:
    """Per-index payloads for the documents we search over."""

    doc_ids: list[str]
    texts: dict[str, dict[str, str]] = field(default_factory=dict)  # index name -> doc id -> text
    vectors: dict[str, np.ndarray] = field(default_factory=dict)  # index name -> [n_docs, ...]
