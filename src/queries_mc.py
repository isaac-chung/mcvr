"""Query writing for the multi-constraint study (`prompts/query_mc_v2.txt`).

From a source video's grounded facts in nested order, the k-fact set is the first k facts
(k = 1..4), so each set extends the one before. The LLM writes one query per set and a map from
fact number to the phrase that carries it; an answer whose phrases do not cover every fact is
retried once with a reminder (a different prompt, so the cache keeps both).
"""

from __future__ import annotations

import re
from pathlib import Path

from src.constraints import K_MAX, parse_json
from src.llm import LLM
from src.types import Cost

LLM_SPEC = "openai:gpt-6-luna@low"
QUERY_PROMPT = Path("prompts/query_mc_v2.txt")
RETRY_NOTE = (
    "\n\nYour previous answer did not give a phrase for every fact. Make sure the query covers "
    'every fact and that "phrases" has a non-empty entry for each fact number from 1 to {k}, '
    "each taken from the query."
)
QUERY_FACT_KEYS = ("text", "type", "scene", "distinctive", "grounded_frame")


def words_for(k: int) -> int:
    return 6 + 5 * k


def nested_sets(order: list[dict], k_max: int = K_MAX) -> dict[int, list[dict]]:
    """{k: the first k facts} for every k the facts allow."""
    return {k: order[:k] for k in range(1, min(k_max, len(order)) + 1)}


def query_prompt(template: str, facts: list[dict]) -> str:
    listed = "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts, 1))
    return template.replace("{words}", str(words_for(len(facts)))).replace("{facts}", listed)


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def phrase_in_query(phrase: str, query: str, min_share: float = 0.6) -> bool:
    """The phrase is part of the query: verbatim (ignoring case and punctuation), or most of its
    words appear in the query."""
    words, in_query = _words(phrase), _words(query)
    if not words:
        return False
    if " ".join(words) in " ".join(in_query):
        return True
    return sum(w in set(in_query) for w in words) / len(words) >= min_share


def parse_query(text: str, k: int) -> tuple[str, dict[str, str]]:
    """(query, {"1": phrase, ...}) from an answer; phrases keyed "1".."k" (a list is accepted)."""
    data = parse_json(text)
    query = " ".join(str(data.get("query") or "").split())
    raw = data.get("phrases") or {}
    if isinstance(raw, list):
        raw = {str(i): p for i, p in enumerate(raw, 1)}
    phrases = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            index = re.sub(r"\D", "", str(key))
            if index and 1 <= int(index) <= k and isinstance(value, str) and value.strip():
                phrases[str(int(index))] = " ".join(value.split())
    return query, phrases


def missing_facts(query: str, phrases: dict[str, str], k: int) -> list[int]:
    """Fact numbers with no phrase, or whose phrase is not part of the query (all of them when
    the query is empty)."""
    if not query:
        return list(range(1, k + 1))
    return [
        i
        for i in range(1, k + 1)
        if str(i) not in phrases or not phrase_in_query(phrases[str(i)], query)
    ]


def write_query(llm: LLM, facts: list[dict], template: str) -> dict:
    """{"query", "phrases", "valid", "attempts", "missing", "cost"} for one fact set."""
    k = len(facts)
    prompt = query_prompt(template, facts)
    total = Cost()
    for attempt in (1, 2):
        text, cost = llm.complete(prompt if attempt == 1 else prompt + RETRY_NOTE.format(k=k))
        total = total + cost
        query, phrases = parse_query(text, k)
        missing = missing_facts(query, phrases, k)
        if not missing:
            break
    return {
        "query": query,
        "phrases": phrases,
        "valid": not missing,
        "attempts": attempt,
        "missing": missing,
        "cost": total,
    }


def query_row(source: dict, k: int, facts: list[dict], written: dict) -> dict:
    """One line of queries.jsonl. source has "source" (global id), "dataset", "length_bin"."""
    return {
        "query_id": f"{source['source']}#k{k}",
        "source": source["source"],
        "dataset": source["dataset"],
        "length_bin": source["length_bin"],
        "k": k,
        "query": written["query"],
        "facts": [{key: f.get(key) for key in QUERY_FACT_KEYS} for f in facts],
        "phrases": written["phrases"],
    }


# ---- template rendering (format check: the same fact set as a plain list) -------------------

TEMPLATE_PREFIX = "video showing: "


def template_fact(text: str) -> str:
    """A fact as a list item: spaces collapsed, final full stop dropped, first letter lower-cased
    unless the first word is an acronym or starts with a quote."""
    text = " ".join(str(text).split()).rstrip(".")
    first = text.split(" ", 1)[0] if text else ""
    if first and first[0].isupper() and not (len(first) > 1 and first[1].isupper()):
        text = text[0].lower() + text[1:]
    return text


def render_template(facts: list[dict]) -> str:
    """'video showing: X; Y; Z' over the facts in their order (no LLM)."""
    return TEMPLATE_PREFIX + "; ".join(template_fact(f["text"]) for f in facts)
