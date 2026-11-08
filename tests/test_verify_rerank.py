"""S1 (decompose), S2/S3 (per-constraint verification) and S4/S5 (holistic rerank), offline."""

import json

import pytest

from src.cache import Cache
from src.embedders import HashEmbedder
from src.indexes.dense_text import DenseTextIndex
from src.llm import CachedVision, FakeLLM, FakeVision
from src.systems.base import RankingStore, RerankSystem, Tools
from src.systems.decompose import Decompose, QuerySplitter, fuse, parse_constraints
from src.systems.rerank import (
    DescriptionScore,
    FrameScore,
    HolisticReranker,
    parse_score,
)
from src.systems.verify import (
    ConstraintVerifier,
    DescriptionCheck,
    FrameCheck,
    parse_verdicts,
    rank_by_score,
    saved_frame_source,
)
from src.types import Query

DOCS = {
    "a/1": "a man in a red cap holds a sign that reads OPEN",
    "a/2": "a man in a blue cap on a street",
    "b/1": "a sign that reads OPEN on a shop door",
    "b/2": "children paint a mural on a brick wall",
}
SPLIT = json.dumps({"constraints": ["a man wears a red cap", "a sign reads OPEN"]})


class LabelVision(FakeVision):
    """FakeVision that records the labels it was given."""

    def __init__(self, answers):
        super().__init__(answers)
        self.labels, self.prompts = [], []

    def describe(self, prompt, images, labels=None):
        self.labels.append(labels)
        self.prompts.append(prompt)
        return super().describe(prompt, images)


class PromptLLM:
    """Answers by a function of the prompt (order-independent, so threads are fine)."""

    def __init__(self, fn):
        self.fn, self.prompts = fn, []

    def complete(self, prompt):
        self.prompts.append(prompt)
        from src.types import Cost

        return self.fn(prompt), Cost(llm_calls=1, tokens_in=10, tokens_out=2)


@pytest.fixture
def index(tmp_path):
    return DenseTextIndex.build("DESC", DOCS, HashEmbedder(64), tmp_path / "idx")


def test_parse_constraints_and_fallback():
    assert parse_constraints(SPLIT, "q") == ["a man wears a red cap", "a sign reads OPEN"]
    dup = '{"constraints": ["x", "X ", {"text": "y"}, ""]}'
    assert parse_constraints(dup, "q") == ["x", "y"]
    assert parse_constraints("sorry", "  the  whole query ") == ["the whole query"]


def test_parse_verdicts():
    text = (
        '{"constraints": [{"constraint": 2, "answer": "No"}, {"constraint": 1, "answer": "yes"}]}'
    )
    assert parse_verdicts(text, 3) == ["yes", "no", "unclear"]
    assert parse_verdicts('{"constraints": {"1": true, "2": "maybe"}}', 2) == ["yes", "unclear"]
    assert parse_verdicts("garbled", 2) == ["unclear", "unclear"]
    assert parse_verdicts('{"constraints": ["yes", "unclear"]}', 2) == ["yes", "unclear"]


def test_parse_score():
    assert parse_score('{"score": 7}') == 7.0
    assert parse_score("score: 12") == 10.0
    assert parse_score("no idea") == -1.0


def test_rank_by_score_ties_keep_first_stage_order():
    cands = [("a", 0.9), ("b", 0.8), ("c", 0.7)]
    assert [d for d, _ in rank_by_score(cands, [1, 2, 1])] == ["b", "a", "c"]


def test_s1_splits_searches_and_fuses(index):
    q = Query("q1", "a man in a red cap holding a sign that reads OPEN", "a/1")
    llm = FakeLLM([SPLIT])
    run = Decompose(QuerySplitter(llm, "split: {query}")).run(
        q, Tools(RankingStore({"DESC": index}, 4), q)
    )
    assert run.system == "S1" and run.trace["constraints"] == [
        "a man wears a red cap",
        "a sign reads OPEN",
    ]
    assert run.cost.searches == 2 and run.cost.llm_calls == 1 and run.indexes == ("DESC",)
    assert len(run.ranking) == 4
    one = [("x", 1.0), ("y", 0.5)]
    assert fuse([one]) == one


def test_s2_counts_yes_and_never_sees_captions(index):
    descriptions = {"a/1": "DESC-A1", "a/2": "DESC-A2", "b/1": "DESC-B1", "b/2": "DESC-B2"}

    def answer(prompt):
        if prompt.startswith("split"):
            return SPLIT
        yes = {"DESC-A1": ["yes", "yes"], "DESC-B1": ["no", "yes"]}
        for key, v in yes.items():
            if key in prompt:
                return json.dumps(
                    {"constraints": [{"constraint": i + 1, "answer": a} for i, a in enumerate(v)]}
                )
        return '{"constraints": [{"constraint": 1, "answer": "unclear"}]}'

    llm = PromptLLM(answer)
    splitter = QuerySplitter(llm, "split: {query}")
    check = DescriptionCheck(llm, "Q {query}\n{constraints}\nD {description}", descriptions)
    system = RerankSystem(ConstraintVerifier("S2", splitter, check, workers=4), n=3)
    q = Query("q1", "a man in a blue cap", "a/1")
    run = system.run(q, Tools(RankingStore({"DESC": index}, 4), q))
    head = [d for d, _ in run.ranking[:3]]
    first = run.trace["first_stage"]
    assert set(head) == set(first) and len(run.ranking) == 4
    scores = {d: s for d, s in run.ranking[:3]}
    if "a/1" in first:
        assert head[0] == "a/1" and scores["a/1"] == 2.0
    assert run.cost.llm_calls == 1 + 3  # one split + one call per candidate
    assert all("caption" not in p.lower() for p in llm.prompts)
    verdict_prompts = [p for p in llm.prompts if p.startswith("Q ")]
    assert "1. a man wears a red cap\n2. a sign reads OPEN" in verdict_prompts[0]


def write_frames(root, doc, n=3):
    folder = root / doc.replace("/", "_")
    folder.mkdir(parents=True)
    for i in range(n):
        (folder / f"{i}.jpg").write_bytes(f"jpeg{doc}{i}".encode())
    times = [1.0 + 2 * i for i in range(n)]
    (folder / "frames.json").write_text(
        json.dumps({"n": n, "times": times, "saved": list(range(n))})
    )
    return folder


def test_s3_frames_are_labelled_and_cached(tmp_path):
    for doc in ("a/1", "a/2"):
        write_frames(tmp_path, doc)
    frames = saved_frame_source(lambda d: tmp_path / d.replace("/", "_"))
    imgs, times = frames("a/1")
    assert len(imgs) == 3 and times == [1.0, 3.0, 5.0]
    vision = LabelVision(['{"constraints": [{"constraint": 1, "answer": "yes", "frames": [2]}]}'])
    cached = CachedVision(vision, Cache(tmp_path / "vlm.sqlite"), "fake")
    check = FrameCheck(cached, "{n} frames. {query}\n{constraints}", frames)
    verifier = ConstraintVerifier("S3", QuerySplitter(FakeLLM([SPLIT]), "{query}"), check)
    q = Query("q1", "red cap and OPEN sign", "a/2")
    out = verifier.rerank(q, [("a/1", 0.9), ("a/2", 0.8)])
    assert vision.labels[0] == [
        "Frame 1 of 3 (t = 1.0 s):",
        "Frame 2 of 3 (t = 3.0 s):",
        "Frame 3 of 3 (t = 5.0 s):",
    ]
    assert vision.prompts[0].startswith("3 frames.")
    assert out.trace["verdicts"]["a/1"] == ["yes", "unclear"]
    assert [d for d, _ in out.ranking] == ["a/1", "a/2"]  # tie on 1 yes: S0 order
    verifier.rerank(q, [("a/1", 0.9), ("a/2", 0.8)])
    assert vision.calls == 2  # second run came from the cache


def test_s4_s5_holistic_scores(tmp_path):
    descriptions = {"a/1": "DESC-A1", "a/2": "DESC-A2"}
    llm = PromptLLM(lambda p: '{"score": 9}' if "DESC-A2" in p else '{"score": 3}')
    s4 = HolisticReranker("S4", DescriptionScore(llm, "{evidence}\nQ {query}", descriptions), 2)
    out = s4.rerank(Query("q", "a blue cap", "a/2"), [("a/1", 0.9), ("a/2", 0.8)])
    assert [d for d, _ in out.ranking] == ["a/2", "a/1"] and out.cost.llm_calls == 2
    assert "Description of the video:\nDESC-A1" in llm.prompts[0] or "DESC-A1" in llm.prompts[1]

    write_frames(tmp_path, "a/1", 2)
    frames = saved_frame_source(lambda d: tmp_path / d.replace("/", "_"))
    vision = LabelVision(["garbled"])
    s5 = HolisticReranker("S5", FrameScore(vision, "{evidence} Q {query}", frames))
    out = s5.rerank(Query("q", "x", "a/1"), [("a/1", 0.9)])
    assert out.trace["scores"]["a/1"] == -1.0 and vision.labels[0][0].startswith("Frame 1 of 2")
    assert "2 frames sampled" in vision.prompts[0]
