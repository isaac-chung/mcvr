"""Caption judge for labels: does a candidate video meet each of a query's facts? (judge_v1)

One text call per (query, candidate): the candidate's dataset caption (numbered visual scenes as
in mining, `constraints.scene_caption`; long captions shortened, see `judge_caption`) and the
query's numbered facts. Per fact the model answers "yes", "no" or "unclear" with a short quote
from the caption. A "yes" without a quote counts as "unclear"; facts the answer leaves out or
garbles count as "unclear" too, so neither an extra positive nor a near-miss is ever made up
from a broken answer. The judge reads captions, so it belongs to the label code: no index or
system module may import it.

Derived labels for a candidate of a k-fact query: an unlabelled match meets all k facts (k yes);
a near-miss meets exactly k - 1 and the remaining one is "no" (k - 1 yes, no unclear).
"""

from __future__ import annotations

import re
from pathlib import Path

from src.constraints import as_int, parse_json, visual_parts
from src.datasets.mc import Caption
from src.llm import LLM
from src.metrics import ndcg_at_k
from src.types import Cost

JUDGE_SPEC = "openai:gpt-6-sol@low"
JUDGE_PROMPT = Path("prompts/judge_v1.txt")
MAX_WORDS = 1500
ANSWERS = ("yes", "no", "unclear")
ELLIPSIS = "[...]"


# ---- the caption the judge reads ---------------------------------------------------------------


def _cap_per_scene(lengths: list[int], budget: int) -> int:
    """The largest per-scene word cap c with sum(min(len, c)) <= budget (water filling): short
    scenes stay whole, long ones are cut to the same length."""
    lo, hi = 0, max(lengths)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if sum(min(n, mid) for n in lengths) <= budget:
            lo = mid
        else:
            hi = mid - 1
    return lo


def judge_caption(caption: Caption, max_words: int = MAX_WORDS) -> tuple[str, dict]:
    """The caption as "[scene i] ..." lines (visual parts only; the whole caption when it has
    fewer than two) and a record {"n_scenes", "words", "kept_words", "truncated"}. Over
    max_words, every scene is kept and the longest are cut to a common length, ending in
    "[...]"."""
    texts = visual_parts(caption.parts)
    if len(texts) <= 1:
        texts = [caption.caption.strip()]
    words = [t.split() for t in texts]
    lengths = [len(w) for w in words]
    total = sum(lengths)
    cap = _cap_per_scene(lengths, max_words) if total > max_words else max(lengths, default=0)
    cap = max(cap, 1)
    scenes = []
    for w in words:
        text = " ".join(w[:cap]) + (f" {ELLIPSIS}" if len(w) > cap else "")
        scenes.append(text)
    lines = "\n".join(f"[scene {i}] {t}" for i, t in enumerate(scenes, 1))
    kept = sum(min(n, cap) for n in lengths)
    info = {"n_scenes": len(texts), "words": total, "kept_words": kept, "truncated": kept < total}
    return lines, info


def numbered(facts: list[dict]) -> str:
    return "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts, 1))


def judge_prompt(template: str, caption_text: str, facts: list[dict]) -> str:
    return template.replace("{caption}", caption_text).replace("{facts}", numbered(facts))


# ---- parsing -----------------------------------------------------------------------------------


def _said(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    word = str(value or "").strip().lower()
    for answer in ANSWERS:
        if word.startswith(answer):
            return answer
    if word.startswith(("true", "satisf")):
        return "yes"
    if word.startswith(("false", "contradict")):
        return "no"
    return "unclear"


def _partial_items(text: str) -> list[dict]:
    """The complete per-fact objects of an answer cut off before its end."""
    items = []
    for match in re.finditer(r'\{[^{}]*"fact"[^{}]*\}', text or ""):
        item = parse_json(match.group(0))
        if item:
            items.append(item)
    return items


def parse_judgment(text: str, n_facts: int) -> list[dict]:
    """One {"fact", "answer", "quote", "model_answer"} per fact (1-based), in fact order.
    "answer" is yes / no / unclear; a "yes" with no quote becomes "unclear" ("model_answer"
    keeps what the model said, "none" when it left the fact out)."""
    data = parse_json(text)
    items = data.get("facts", data.get("answers"))
    if items is None:
        items = _partial_items(text)
    if isinstance(items, dict):  # {"1": {...}, ...}
        items = [
            {"fact": k, **(v if isinstance(v, dict) else {"answer": v})} for k, v in items.items()
        ]
    found: dict[int, dict] = {}
    for pos, item in enumerate(items if isinstance(items, list) else [], 1):
        if not isinstance(item, dict):
            item = {"fact": pos, "answer": item}
        index = as_int(item.get("fact"), pos)
        if not 1 <= index <= n_facts or index in found:
            continue
        said = _said(item.get("answer"))
        quote = " ".join(str(item.get("quote") or "").split())
        answer = "unclear" if said == "yes" and not quote else said
        found[index] = {"fact": index, "answer": answer, "quote": quote, "model_answer": said}
    missing = {"answer": "unclear", "quote": "", "model_answer": "none"}
    return [found.get(i, {"fact": i, **missing}) for i in range(1, n_facts + 1)]


# ---- one judgment ------------------------------------------------------------------------------


def judge_candidate(
    llm: LLM, template: str, caption: Caption, facts: list[dict], max_words: int = MAX_WORDS
) -> dict:
    """Judge one candidate's caption against the facts: {"answers", "n_yes", "n_no",
    "n_unclear", "all_yes", "caption", "cost", "raw"}."""
    text, info = judge_caption(caption, max_words)
    raw, cost = llm.complete(judge_prompt(template, text, facts))
    answers = parse_judgment(raw, len(facts))
    return {**summarise_answers(answers), "caption": info, "cost": cost.to_dict(), "raw": raw}


def summarise_answers(answers: list[dict]) -> dict:
    counts = {a: sum(x["answer"] == a for x in answers) for a in ANSWERS}
    return {
        "answers": answers,
        "n_yes": counts["yes"],
        "n_no": counts["no"],
        "n_unclear": counts["unclear"],
        "all_yes": bool(answers) and counts["yes"] == len(answers),
    }


def is_near_miss(record: dict, k: int) -> bool:
    """Exactly k - 1 of the k facts are yes and the other is a definite no (k >= 2 only)."""
    return k >= 2 and record["n_yes"] == k - 1 and record["n_unclear"] == 0


def quote_in_caption(quote: str, caption_text: str) -> bool:
    """Is the quote (case, punctuation and spacing ignored) a passage of the caption?"""

    def norm(s: str) -> str:
        return " ".join(re.sub(r"[^\w\s]", " ", s.lower()).split())

    q = norm(quote.replace(ELLIPSIS, " "))
    return bool(q) and q in norm(caption_text)


# ---- per query summary and metrics -------------------------------------------------------------


def hit_at(ranking: list[str], relevant: set[str], k: int) -> float:
    """1 if any relevant video is in the top k (with one positive, recall@k)."""
    return float(any(d in relevant for d in ranking[:k]))


def query_summary(row: dict, ranking: list[str]) -> dict:
    """Counts and strict / expanded metrics of one judged query. row has "k", "source" and
    "candidates" (judged records with "doc_id"); ranking is S0's doc ids, best first."""
    k = row["k"]
    cands = row["candidates"]
    matches = [c["doc_id"] for c in cands if c["all_yes"]]
    near = [c["doc_id"] for c in cands if is_near_miss(c, k)]
    strict = {row["source"]}
    expanded = strict | set(matches)
    ranked = [(d, 0.0) for d in ranking]
    out = {"n_judged": len(cands), "n_match": len(matches), "n_near_miss": len(near)}
    for name, rel in (("strict", strict), ("expanded", expanded)):
        for at in (1, 5, 20):
            out[f"{name}_recall@{at}"] = hit_at(ranking, rel, at)
        out[f"{name}_ndcg@10"] = ndcg_at_k(ranked, dict.fromkeys(rel, 1), 10)
    return out


def group_table(
    summaries: list[dict], key: str, order: tuple[str, ...] | None = None
) -> dict[str, dict]:
    """Means per value of key (and "all") of the per-query summaries: share of queries with
    >= 1 near-miss / match, mean counts, strict and expanded metrics. Near-miss shares use the
    queries with k >= 2 only. order fixes the order of the groups (others follow, sorted)."""
    groups: dict[str, list[dict]] = {}
    for s in summaries:
        groups.setdefault(str(s[key]), []).append(s)
    table = {}
    names = [g for g in order or () if g in groups] + sorted(set(groups) - set(order or ()))
    for name in names:
        rows = groups[name]
        multi = [r for r in rows if r["k"] >= 2]
        stats = {
            "n": len(rows),
            "judged": sum(r["n_judged"] for r in rows),
            "share_near_miss": _mean([r["n_near_miss"] > 0 for r in multi]),
            "mean_near_miss": _mean([r["n_near_miss"] for r in multi]),
            "share_match": _mean([r["n_match"] > 0 for r in rows]),
            "mean_match": _mean([r["n_match"] for r in rows]),
        }
        for metric in ("recall@1", "recall@5", "recall@20", "ndcg@10"):
            for label in ("strict", "expanded"):
                stats[f"{label}_{metric}"] = _mean([r[f"{label}_{metric}"] for r in rows])
        table[name] = stats
    return table


def _mean(values: list) -> float | None:
    return sum(values) / len(values) if values else None


def total_cost(records: list[dict]) -> Cost:
    return sum((Cost(**r["cost"]) for r in records), Cost())
