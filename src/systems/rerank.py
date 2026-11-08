"""S4 and S5: holistic reranker controls (docs/plan.md, systems table).

One call per candidate scores it 0-10 against the whole query (`prompts/rerank_v1.txt`), with
no decomposition: S4 reads the candidate's description, S5 looks at its 16 saved frames with
"Frame i of 16 (t = ... s):" labels. They exist so a reranking gain of S2/S3 is not credited to
decomposition. Rerank by score, ties in S0's order.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

from src.constraints import parse_json
from src.grounding import frame_labels
from src.llm import LLM, VisionLLM
from src.systems.base import Reranked
from src.systems.verify import FrameSource, rank_by_score, run_parallel
from src.types import Cost, Query, Ranking

RERANK_PROMPT = Path("prompts/rerank_v1.txt")
DESC_EVIDENCE = (
    "You cannot see the video; you get a written description of it, made from frames sampled "
    "over the whole video.\n\nDescription of the video:\n{description}"
)
FRAMES_EVIDENCE = (
    "These are {n} frames sampled at equal intervals, in order, over the whole video; each "
    'frame is preceded by a label "Frame i of {n} (t = ... s)".'
)


def parse_score(text: str) -> float:
    """The 0-10 score of an answer, clamped; -1 when there is none (ranks below every scored
    candidate, so a failed call never lifts a video)."""
    data = parse_json(text)
    value = data.get("score") if isinstance(data, dict) else None
    if value is None:
        match = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', text or "")
        value = match.group(1) if match else None
    try:
        return min(10.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return -1.0


def fill(template: str, query: str, evidence: str) -> str:
    return template.replace("{query}", query).replace("{evidence}", evidence)


class DescriptionScore:
    """S4's call: the whole query against the candidate's description."""

    def __init__(self, llm: LLM, template: str, descriptions: dict[str, str]):
        self.llm, self.template, self.descriptions = llm, template, descriptions

    def __call__(self, query: str, doc: str) -> tuple[float, Cost]:
        desc = self.descriptions.get(doc, "") or "(no description)"
        text, cost = self.llm.complete(
            fill(self.template, query, DESC_EVIDENCE.replace("{description}", desc))
        )
        return parse_score(text), cost


class FrameScore:
    """S5's call: the whole query against the candidate's labelled frames."""

    def __init__(self, vlm: VisionLLM, template: str, frames: FrameSource):
        self.vlm, self.template, self.frames = vlm, template, frames

    def __call__(self, query: str, doc: str) -> tuple[float, Cost]:
        images, times = self.frames(doc)
        if not images:
            return -1.0, Cost()
        evidence = FRAMES_EVIDENCE.replace("{n}", str(len(images)))
        text, cost = self.vlm.describe(
            fill(self.template, query, evidence), images, labels=frame_labels(times)
        )
        return parse_score(text), cost


Scorer = Callable[[str, str], tuple[float, Cost]]


class HolisticReranker:
    """A Reranker: one 0-10 score per candidate against the whole query; ties in S0's order."""

    def __init__(self, name: str, score: Scorer, workers: int = 1):
        self.name, self.score, self.workers = name, score, workers

    def rerank(self, query: Query, candidates: Ranking) -> Reranked:
        docs = [d for d, _ in candidates]
        results = run_parallel(lambda d: self.score(query.text, d), docs, self.workers)
        cost = Cost()
        for _, c in results:
            cost += c
        scores = [s for s, _ in results]
        trace = {"scores": dict(zip(docs, scores, strict=True))}
        return Reranked(rank_by_score(candidates, scores), cost, trace)
