"""The caption judge (src/judge.py) and the missed-positive audit summaries, offline."""

import json
from pathlib import Path

from src import judge
from src.datasets.mc import Caption
from src.llm import FakeLLM

TEMPLATE = Path("prompts/judge_v1.txt").read_text()
FACTS = [{"text": "A man wears a red shirt."}, {"text": "A sign reads OPEN."}]


def scenes(*texts: str) -> Caption:
    parts = tuple({"annotator": "GPT-4V", "text": t} for t in texts)
    return Caption("v1", " ".join(texts), parts=parts)


def answer(*items) -> str:
    return json.dumps({"facts": [{"fact": i, "answer": a, "quote": q} for i, a, q in items]})


def test_caption_uses_visual_scenes_and_skips_speech():
    cap = Caption(
        "v1",
        "whole",
        parts=(
            {"annotator": "visual", "text": "A man in red."},
            {"annotator": "asr", "text": "hello there"},
            {"annotator": "visual", "text": "A sign says OPEN."},
        ),
    )
    text, info = judge.judge_caption(cap)
    assert text == "[scene 1] A man in red.\n[scene 2] A sign says OPEN."
    assert info == {"n_scenes": 2, "words": 8, "kept_words": 8, "truncated": False}
    single, info = judge.judge_caption(Caption("v2", "One long caption.", parts=()))
    assert single == "[scene 1] One long caption." and info["n_scenes"] == 1


def test_long_caption_keeps_every_scene_and_cuts_the_longest():
    cap = scenes("short scene here", " ".join(["w"] * 100), " ".join(["x"] * 50))
    text, info = judge.judge_caption(cap, max_words=60)
    lines = text.split("\n")
    assert len(lines) == 3 and lines[0] == "[scene 1] short scene here"
    assert lines[1].endswith(judge.ELLIPSIS) and lines[2].endswith(judge.ELLIPSIS)
    assert info["truncated"] and info["kept_words"] <= 60 and info["words"] == 153
    # the two long scenes are cut to the same length
    assert lines[1].count("w") == lines[2].count("x")


def test_prompt_fills_caption_and_numbered_facts():
    prompt = judge.judge_prompt(TEMPLATE, "[scene 1] cap", FACTS)
    assert (
        "[scene 1] cap" in prompt and "1. A man wears a red shirt.\n2. A sign reads OPEN." in prompt
    )
    assert "{caption}" not in prompt and "{facts}" not in prompt


def test_parse_answers_quotes_and_gaps():
    out = judge.parse_judgment(answer((1, "Yes", "a man in red"), (2, "no", "sign says CLOSED")), 2)
    assert [a["answer"] for a in out] == ["yes", "no"] and out[0]["quote"] == "a man in red"
    # yes without a quote is unclear; a missing fact is unclear with model_answer none
    out = judge.parse_judgment(answer((1, "yes", "")), 2)
    assert out[0]["answer"] == "unclear" and out[0]["model_answer"] == "yes"
    assert out[1] == {"fact": 2, "answer": "unclear", "quote": "", "model_answer": "none"}
    # garbage, a cut-off answer, a dict form
    assert [a["answer"] for a in judge.parse_judgment("no json", 2)] == ["unclear", "unclear"]
    cut = '{"facts": [{"fact": 1, "answer": "no", "quote": "blue shirt"}, {"fact": 2, "ans'
    assert [a["answer"] for a in judge.parse_judgment(cut, 2)] == ["no", "unclear"]
    as_dict = json.dumps({"facts": {"2": {"answer": "yes", "quote": "OPEN"}}})
    assert [a["answer"] for a in judge.parse_judgment(as_dict, 2)] == ["unclear", "yes"]


def test_judge_candidate_counts_and_labels():
    llm = FakeLLM([answer((1, "yes", "man in red"), (2, "no", "says CLOSED"))])
    rec = judge.judge_candidate(llm, TEMPLATE, scenes("man in red", "sign says CLOSED"), FACTS)
    assert (rec["n_yes"], rec["n_no"], rec["n_unclear"], rec["all_yes"]) == (1, 1, 0, False)
    assert judge.is_near_miss(rec, 2) and not judge.is_near_miss({**rec, "n_unclear": 1}, 2)
    assert not judge.is_near_miss({"n_yes": 0, "n_unclear": 0}, 1)  # no near-miss at k = 1
    assert rec["cost"]["llm_calls"] == 1 and not rec["caption"]["truncated"]
    llm = FakeLLM([answer((1, "yes", "man in red"), (2, "yes", "OPEN"))])
    assert judge.judge_candidate(llm, TEMPLATE, scenes("a", "b"), FACTS)["all_yes"]


def test_quote_in_caption_ignores_case_and_punctuation():
    assert judge.quote_in_caption('A Man, in "red"', "[scene 1] a man in red.")
    assert not judge.quote_in_caption("a woman", "a man in red")
    assert not judge.quote_in_caption("", "a man")


def cand(doc: str, n_yes: int, n_unclear: int = 0, k: int = 2) -> dict:
    return {"doc_id": doc, "n_yes": n_yes, "n_unclear": n_unclear, "all_yes": n_yes == k}


def test_query_summary_strict_and_expanded():
    row = {
        "k": 2,
        "source": "s",
        "candidates": [cand("a", 2), cand("b", 1), cand("c", 1, 1), cand("d", 0)],
    }
    ranking = ["a", "b", "s", "c", "d"]
    out = judge.query_summary(row, ranking)
    assert (out["n_judged"], out["n_match"], out["n_near_miss"]) == (4, 1, 1)
    assert out["strict_recall@1"] == 0.0 and out["strict_recall@5"] == 1.0
    assert out["expanded_recall@1"] == 1.0
    assert out["expanded_ndcg@10"] > out["strict_ndcg@10"] > 0


def test_group_table_shares_and_order():
    rows = [
        {"k": 1, "g": "b", "n_judged": 19, "n_match": 0, "n_near_miss": 0},
        {"k": 2, "g": "a", "n_judged": 20, "n_match": 2, "n_near_miss": 1},
        {"k": 3, "g": "a", "n_judged": 19, "n_match": 0, "n_near_miss": 0},
    ]
    for r in rows:
        for label in ("strict", "expanded"):
            for m in ("recall@1", "recall@5", "recall@20", "ndcg@10"):
                r[f"{label}_{m}"] = 1.0 if label == "expanded" else 0.0
    table = judge.group_table(rows, "g", order=("b", "a"))
    assert list(table) == ["b", "a"]
    assert table["b"]["share_near_miss"] is None  # k = 1 only
    assert table["a"]["share_near_miss"] == 0.5 and table["a"]["mean_match"] == 1.0
    assert table["a"]["share_match"] == 0.5 and table["a"]["expanded_recall@1"] == 1.0


def test_judge_is_not_imported_by_systems_or_indexes():
    for folder in ("src/systems", "src/indexes"):
        for path in Path(folder).glob("*.py"):
            assert "src.judge" not in path.read_text() and "import judge" not in (
                path.read_text()
            ), path
