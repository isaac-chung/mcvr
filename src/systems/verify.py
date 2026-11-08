"""S2 and S3: per-constraint verification of S0's top-N candidates (docs/plan.md, H2 and H3).

The verifier splits the query into constraints itself (`QuerySplitter`, one call per query),
then makes one call per (query, candidate) with all constraints, answering yes, no or unclear
for each. S2 reads the candidate's generated description (`prompts/verify_desc_v1.txt`); S3
looks at the candidate's 16 saved frames, the same ones the describer saw, each labelled
"Frame i of 16 (t = ... s):" (`prompts/verify_frames_v1.txt`). Neither ever sees captions.

Rerank score: the number of constraints judged "yes"; ties keep S0's order. The verdicts are
kept in the trace, so other aggregations (fewest "no", weighted unclear) can be rescored later
without new calls.

Graded variant (S2g / S3g, the review's Tier 1): the same calls with `prompts/verify_desc_v2.txt`
/ `verify_frames_v2.txt`, which ask for one integer 0-10 per constraint (10 clearly stated, 5
not mentioned, 0 contradicted), the scale the whole-query rerankers S4 / S5 use, so per-constraint
and whole-query framing are compared on one scale. `GradedVerifier` reranks by the mean grade
(pre-registered; `aggregate="min"` is the free secondary rescoring from the saved grades), ties
in S0's order; an answer that does not parse scores -1 like a failed S4 call, a constraint the
answer leaves out scores 5.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.constraints import as_int, parse_json
from src.grounding import frame_labels, saved_frame_times
from src.llm import LLM, OpenAIVision, VisionLLM
from src.systems.base import Reranked
from src.systems.decompose import QuerySplitter
from src.types import Cost, Query, Ranking

VERIFIER_SPEC = "openai:gpt-6-sol@low"
DESC_PROMPT = Path("prompts/verify_desc_v1.txt")
FRAMES_PROMPT = Path("prompts/verify_frames_v1.txt")
GRADED_DESC_PROMPT = Path("prompts/verify_desc_v2.txt")
GRADED_FRAMES_PROMPT = Path("prompts/verify_frames_v2.txt")
ANSWERS = ("yes", "no", "unclear")
GRADE_MISSING = 5.0  # a constraint the (parsed) answer leaves out: "not mentioned"
GRADE_FAILED = -1.0  # an answer that does not parse at all (as rerank.parse_score)

# doc id -> (JPEG frames in order, their sample times in seconds)
FrameSource = Callable[[str], tuple[list[bytes], list[float | None]]]


class LabelledOpenAIVision(OpenAIVision):
    """OpenAIVision that also accepts frame labels: each label goes as a text part right before
    its image, as GeminiVision does, so the verifier can refer to frames by number."""

    def describe(
        self, prompt: str, images: list[bytes], labels: list[str] | None = None
    ) -> tuple[str, Cost]:
        if labels is None:
            return super().describe(prompt, images)
        if len(labels) != len(images):
            raise ValueError(f"{len(labels)} labels for {len(images)} images")
        start = time.perf_counter()
        parts: list[dict] = [{"type": "text", "text": prompt}]
        for label, jpeg in zip(labels, images, strict=True):
            url = "data:image/jpeg;base64," + base64.b64encode(jpeg).decode()
            parts.append({"type": "text", "text": label})
            parts.append({"type": "image_url", "image_url": {"url": url, "detail": "high"}})
        resp = self._create(parts)
        cost = Cost(
            llm_calls=1,
            tokens_in=resp.usage.prompt_tokens,
            tokens_out=resp.usage.completion_tokens,
            seconds=time.perf_counter() - start,
            images=len(images),
            model_version=resp.model,
        )
        return resp.choices[0].message.content or "", cost


def saved_frame_source(frames_dir: Callable[[str], Path]) -> FrameSource:
    """A FrameSource over src.frames.saved_frames folders (frames_dir(doc) -> folder)."""

    def load(doc: str) -> tuple[list[bytes], list[float | None]]:
        files, times = saved_frame_times(frames_dir(doc))
        return [f.read_bytes() for f in files], list(times)

    return load


def numbered(constraints: list[str]) -> str:
    return "\n".join(f"{i}. {c}" for i, c in enumerate(constraints, 1))


def _answer(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = str(value or "").strip().lower()
    for answer in ANSWERS:
        if text.startswith(answer):
            return answer
    if text.startswith("true"):
        return "yes"
    if text.startswith("false"):
        return "no"
    return "unclear"


def parse_verdicts(text: str, n: int) -> list[str]:
    """One of yes / no / unclear per constraint (1-based order). Constraints the answer leaves
    out or garbles count as unclear."""
    data = parse_json(text)
    items = data.get("constraints", data.get("verdicts")) if isinstance(data, dict) else None
    if isinstance(items, dict):  # {"1": "yes", ...}
        items = [{"constraint": k, "answer": v} for k, v in items.items()]
    found: dict[int, str] = {}
    for pos, item in enumerate(items if isinstance(items, list) else [], 1):
        if not isinstance(item, dict):
            item = {"constraint": pos, "answer": item}
        index = as_int(item.get("constraint", item.get("id")), pos)
        if 1 <= index <= n and index not in found:
            found[index] = _answer(item.get("answer"))
    return [found.get(i, "unclear") for i in range(1, n + 1)]


def _grade(value) -> float | None:
    """A 0-10 grade clamped to the scale; None when the value is not a number."""
    if isinstance(value, bool):
        return 10.0 if value else 0.0
    try:
        return min(10.0, max(0.0, float(value)))
    except (TypeError, ValueError):
        return None


def parse_grades(text: str, n: int) -> list[float]:
    """One grade per constraint (1-based order) of a graded answer. Constraints the answer
    leaves out or garbles score GRADE_MISSING; an answer with no parsable constraint list at all
    scores GRADE_FAILED on every constraint, so a failed call never lifts a video."""
    data = parse_json(text)
    items = data.get("constraints", data.get("grades")) if isinstance(data, dict) else None
    if isinstance(items, dict):  # {"1": 7, ...}
        items = [{"constraint": k, "score": v} for k, v in items.items()]
    if not isinstance(items, list):
        return [GRADE_FAILED] * n
    found: dict[int, float] = {}
    for pos, item in enumerate(items, 1):
        if not isinstance(item, dict):
            item = {"constraint": pos, "score": item}
        index = as_int(item.get("constraint", item.get("id")), pos)
        grade = _grade(item.get("score", item.get("grade", item.get("answer"))))
        if 1 <= index <= n and index not in found and grade is not None:
            found[index] = grade
    if not found:
        return [GRADE_FAILED] * n
    return [found.get(i, GRADE_MISSING) for i in range(1, n + 1)]


def aggregate_grades(grades: list[float], how: str = "mean") -> float:
    """The candidate's score from its per-constraint grades: "mean" (primary) or "min"."""
    if not grades:
        return GRADE_FAILED
    if how == "mean":
        return sum(grades) / len(grades)
    if how == "min":
        return min(grades)
    raise ValueError(f"unknown aggregation {how!r}")


# ---- what a verifier looks at -----------------------------------------------------------------


class DescriptionCheck:
    """S2's check: the candidate's description, all constraints in one text call."""

    def __init__(
        self,
        llm: LLM,
        template: str,
        descriptions: dict[str, str],
        parse: Callable = parse_verdicts,
    ):
        self.llm, self.template, self.descriptions = llm, template, descriptions
        self.parse = parse  # parse_verdicts (S2) or parse_grades (S2g)

    def __call__(self, query: str, constraints: list[str], doc: str) -> tuple[list, Cost]:
        prompt = (
            self.template.replace("{query}", query)
            .replace("{constraints}", numbered(constraints))
            .replace("{description}", self.descriptions.get(doc, "") or "(no description)")
        )
        text, cost = self.llm.complete(prompt)
        return self.parse(text, len(constraints)), cost


class FrameCheck:
    """S3's check: the candidate's saved frames with labels, all constraints in one call. The
    vision model must accept `labels` (CachedVision over LabelledOpenAIVision or Gemini)."""

    def __init__(
        self, vlm: VisionLLM, template: str, frames: FrameSource, parse: Callable = parse_verdicts
    ):
        self.vlm, self.template, self.frames = vlm, template, frames
        self.parse = parse  # parse_verdicts (S3) or parse_grades (S3g)

    def __call__(self, query: str, constraints: list[str], doc: str) -> tuple[list, Cost]:
        images, times = self.frames(doc)
        if not images:
            return self.parse("", len(constraints)), Cost()
        prompt = (
            self.template.replace("{n}", str(len(images)))
            .replace("{query}", query)
            .replace("{constraints}", numbered(constraints))
        )
        text, cost = self.vlm.describe(prompt, images, labels=frame_labels(times))
        return self.parse(text, len(constraints)), cost


Check = Callable[[str, list[str], str], tuple[list, Cost]]


def run_parallel(fn: Callable, items: list, workers: int) -> list:
    if workers <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    with ThreadPoolExecutor(min(workers, len(items))) as pool:
        return list(pool.map(fn, items))


def rank_by_score(candidates: Ranking, scores: list[float]) -> Ranking:
    """Candidates by score (higher first), ties in first-stage order; the score is returned."""
    order = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
    return [(candidates[i][0], float(scores[i])) for i in order]


class ConstraintVerifier:
    """A Reranker (src.systems.base): split the query, check every candidate against all
    constraints in one call, rerank by the number of "yes" answers (ties: first-stage order)."""

    def __init__(self, name: str, splitter: QuerySplitter, check: Check, workers: int = 1):
        self.name, self.splitter, self.check, self.workers = name, splitter, check, workers

    def rerank(self, query: Query, candidates: Ranking) -> Reranked:
        constraints, cost = self.splitter(query.text)
        docs = [d for d, _ in candidates]
        results = run_parallel(lambda d: self.check(query.text, constraints, d), docs, self.workers)
        verdicts = [v for v, _ in results]
        for _, c in results:
            cost += c
        scores = [sum(a == "yes" for a in v) for v in verdicts]
        trace = {"constraints": constraints, "verdicts": dict(zip(docs, verdicts, strict=True))}
        return Reranked(rank_by_score(candidates, scores), cost, trace)


class GradedVerifier(ConstraintVerifier):
    """S2g / S3g: as ConstraintVerifier, but the check returns one 0-10 grade per constraint
    (parse_grades) and the candidate's score is their mean (or min). The grades are kept in the
    trace as "grades", so the other aggregation is a free rescoring."""

    def __init__(
        self, name: str, splitter: QuerySplitter, check: Check, workers: int = 1, aggregate="mean"
    ):
        super().__init__(name, splitter, check, workers)
        self.aggregate = aggregate

    def rerank(self, query: Query, candidates: Ranking) -> Reranked:
        constraints, cost = self.splitter(query.text)
        docs = [d for d, _ in candidates]
        results = run_parallel(lambda d: self.check(query.text, constraints, d), docs, self.workers)
        grades = [g for g, _ in results]
        for _, c in results:
            cost += c
        scores = [aggregate_grades(g, self.aggregate) for g in grades]
        trace = {
            "constraints": constraints,
            "grades": dict(zip(docs, grades, strict=True)),
            "aggregate": self.aggregate,
        }
        return Reranked(rank_by_score(candidates, scores), cost, trace)
