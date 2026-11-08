"""Systems decide which indexes to search; the store and tools do the searching and costing.

Two kinds of system share these tools. Routing systems (the pilot) choose indexes. Candidate
rerankers (the multi-constraint study, S2-S5) get the query and S0's top-N candidates and return
a reranked list plus its Cost; `RerankSystem` runs S0, hands the candidates over and appends
the rest of S0's list below them, so every system returns a list as deep as S0's.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from src.indexes.base import Index
from src.llm import LLM
from src.types import Cost, Query, Ranking, Run


class RankingStore:
    """One ranked list per (query, index), computed once and shared by every system."""

    def __init__(
        self, indexes: dict[str, Index], depth: int, texts: dict[str, dict[str, str]] | None = None
    ):
        self.indexes, self.depth = indexes, depth
        self.texts = texts or {}  # index name -> doc id -> text, for result previews
        self._memo: dict[tuple[str, str], tuple[Ranking, Cost]] = {}

    def get(self, query: Query, index: str) -> tuple[Ranking, Cost]:
        key = (query.id, index)
        if key not in self._memo:
            self._memo[key] = self.indexes[index].search(query.text, self.depth)
        return self._memo[key]

    def search_seconds(self) -> dict[str, list[float]]:
        """Seconds of every search made so far, by index name."""
        out: dict[str, list[float]] = {name: [] for name in self.indexes}
        for (_, name), (_, cost) in self._memo.items():
            out[name].append(cost.seconds)
        return out


class Tools:
    """The only things a system may touch. Costs add up here, whether or not results were cached."""

    def __init__(self, store: RankingStore, query: Query, llm: LLM | None = None):
        self.store, self.query, self.llm = store, query, llm
        self.cost = Cost()
        self.searched: list[str] = []

    @property
    def index_names(self) -> tuple[str, ...]:
        return tuple(self.store.indexes)

    def search(self, index: str) -> Ranking:
        ranking, cost = self.store.get(self.query, index)
        self.cost += cost
        if index not in self.searched:
            self.searched.append(index)
        return ranking

    def snippet(self, index: str, doc: str, chars: int = 160) -> str:
        """Start of a video's text in this index (empty for indexes without text, like VIS)."""
        text = self.store.texts.get(index, {}).get(doc, "")
        return " ".join(text.split())[:chars]

    def complete(self, prompt: str) -> str:
        if self.llm is None:
            raise RuntimeError("this system needs an LLM; set `llm` in the config")
        text, cost = self.llm.complete(prompt)
        self.cost += cost
        return text

    def finish(self, system: str, ranking: Ranking, trace: dict | None = None) -> Run:
        return Run(self.query.id, system, ranking, tuple(self.searched), self.cost, trace or {})


class System(Protocol):
    name: str

    def run(self, query: Query, tools: Tools) -> Run: ...


# ---- candidate reranking -------------------------------------------------------------------


@dataclass(frozen=True)
class Reranked:
    """A reranker's answer: the candidates in its order, what it cost, and anything to keep."""

    ranking: Ranking
    cost: Cost = field(default_factory=Cost)
    trace: dict = field(default_factory=dict)


class Reranker(Protocol):
    """Reorders first-stage candidates. It sees the query and the candidates only (doc id and
    first-stage score, best first); any text or frames it needs come from what it was built
    with (descriptions, saved frames), never from captions."""

    name: str

    def rerank(self, query: Query, candidates: Ranking) -> Reranked: ...


def check_reranked(candidates: Ranking, reranked: Ranking) -> None:
    """A reranker may only reorder (or drop) candidates, never add a video or repeat one."""
    ids = [d for d, _ in reranked]
    allowed = {d for d, _ in candidates}
    if len(ids) != len(set(ids)):
        raise ValueError("reranked list repeats a video")
    extra = [d for d in ids if d not in allowed]
    if extra:
        raise ValueError(f"reranked list has videos that were not candidates: {extra[:3]}")


def merge_reranked(first_stage: Ranking, reranked: Ranking) -> Ranking:
    """The reranked candidates in the reranker's order, then the rest of the first-stage list
    in its order (candidates the reranker dropped come back here). Tail scores are shifted below
    the reranked block, so a score threshold never lifts a tail video above a reranked one."""
    head = list(reranked)
    seen = {d for d, _ in head}
    tail = [(d, s) for d, s in first_stage if d not in seen]
    if head and tail:
        floor = min(s for _, s in head)
        top = max(s for _, s in tail)
        tail = [(d, s - top + floor - 1.0) for d, s in tail]
    return head + tail


class RerankSystem:
    """S0's top-n over `index`, reranked by `reranker`; cost = the search plus the reranking."""

    def __init__(self, reranker: Reranker, n: int = 20, index: str = "DESC", name: str = ""):
        self.reranker, self.n, self.index = reranker, n, index
        self.name = name or reranker.name

    def run(self, query: Query, tools: Tools) -> Run:
        first = tools.search(self.index)
        candidates = first[: self.n]
        result = self.reranker.rerank(query, candidates)
        check_reranked(candidates, result.ranking)
        tools.cost += result.cost
        trace = {"n": self.n, "first_stage": [d for d, _ in candidates], **result.trace}
        return tools.finish(self.name, merge_reranked(first, result.ranking), trace)
