"""S1: decompose at retrieval time. An LLM splits the query into its constraints
(`prompts/split_v1.txt`), each constraint is searched on its own over the description index,
and the lists are fused with reciprocal rank fusion (docs/plan.md, H1).

The split is shared with the verifiers (S2, S3): systems split the query themselves and never
see the mined constraints. The splitter goes through the shared llm cache, so S1, S2 and S3
reuse one split call per query.
"""

from __future__ import annotations

from pathlib import Path

from src.constraints import parse_json
from src.fusion import rrf
from src.llm import LLM
from src.systems.base import Tools
from src.types import Cost, Query, Ranking, Run

SPLIT_SPEC = "openai:gpt-6-luna@low"
SPLIT_PROMPT = Path("prompts/split_v1.txt")
MAX_CONSTRAINTS = 8


def parse_constraints(text: str, query: str, limit: int = MAX_CONSTRAINTS) -> list[str]:
    """The constraint strings of a split answer, cleaned and de-duplicated in order. An answer
    with none (empty, garbled or refused) falls back to the whole query as one constraint."""
    data = parse_json(text)
    raw = data.get("constraints", []) if isinstance(data, dict) else []
    out: list[str] = []
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, dict):
            item = item.get("text") or item.get("constraint") or ""
        value = " ".join(str(item or "").split())
        if value and value.lower() not in {c.lower() for c in out}:
            out.append(value)
    return out[:limit] or [" ".join(query.split())]


class QuerySplitter:
    """query text -> (constraints, cost), one LLM call per query (cached by the LLM wrapper)."""

    def __init__(self, llm: LLM, template: str):
        self.llm, self.template = llm, template

    def __call__(self, query: str) -> tuple[list[str], Cost]:
        text, cost = self.llm.complete(self.template.replace("{query}", query))
        return parse_constraints(text, query), cost


def fuse(rankings: list[Ranking], k: int = 60) -> Ranking:
    """RRF of the per-constraint lists; one list is returned as is (it is already ranked)."""
    if len(rankings) == 1:
        return list(rankings[0])
    return rrf(rankings, k=k)


class Decompose:
    """S1 inside the systems framework: split, one search per constraint, RRF."""

    def __init__(self, splitter: QuerySplitter, index: str = "DESC", name: str = "S1"):
        self.splitter, self.index, self.name = splitter, index, name

    def run(self, query: Query, tools: Tools) -> Run:
        constraints, cost = self.splitter(query.text)
        tools.cost += cost
        rankings = []
        for i, text in enumerate(constraints, 1):
            sub = Query(f"{query.id}#c{i}", text, query.cluster)
            ranking, search_cost = tools.store.get(sub, self.index)
            tools.cost += search_cost
            rankings.append(ranking)
        if self.index not in tools.searched:
            tools.searched.append(self.index)
        return tools.finish(self.name, fuse(rankings), {"constraints": constraints})


class GoldConstraints:
    """A drop-in for QuerySplitter that returns the query's mined facts instead of a split (the
    S2-gold / S3-gold oracle rows): query text -> (fact texts as they are, no cost). Keyed by
    query text, which is unique on the smoke set; an unknown query raises KeyError."""

    def __init__(self, by_query: dict[str, list[str]]):
        self.by_query = {" ".join(q.split()): list(f) for q, f in by_query.items()}

    @classmethod
    def from_rows(cls, rows: list[dict]) -> GoldConstraints:
        """From queries.jsonl rows ({"query", "facts": [{"text", ...}]})."""
        return cls({r["query"]: [f["text"] for f in r["facts"]] for r in rows})

    def __call__(self, query: str) -> tuple[list[str], Cost]:
        return list(self.by_query[" ".join(query.split())]), Cost()
