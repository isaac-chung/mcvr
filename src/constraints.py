"""Constraint mining for the multi-constraint study: caption -> atomic facts -> selected candidates.

The caption of a source video is turned into numbered scenes (visual parts only), an LLM mines
atomic facts from it (`prompts/facts_mc_v2.txt`), and the selection rule picks the candidates
that frame grounding (`src.grounding`) checks and query writing (`src.queries_mc`)
uses. Captions are read here and in the label code only, never by the index or the systems.

Every random choice is seeded from the video's global id ("<dataset>/<video_id>"), so a rerun
picks the same facts, and the LLM goes through the shared cache, so a rerun is free.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from pathlib import Path

from src.llm import LLM
from src.types import Cost

LLM_SPEC = "openai:gpt-6-luna@low"
FACTS_PROMPT = Path("prompts/facts_mc_v2.txt")
# caption parts that are not visual descriptions of a scene
SKIP_ANNOTATORS = {"narration", "asr", "asr_full", "summary", "voiceover", "spatial", "temporal"}
FACT_TYPES = {
    "action",
    "object",
    "attribute",
    "on_screen_text",
    "count",
    "scene",
    "spatial_relation",
    "temporal_order",
    "other",
}
KEEP_TYPES = {"attribute", "object", "on_screen_text", "count", "scene"}
MAX_CANDIDATES = 10  # facts sent to one grounding call
MAX_BACKUPS = 2  # non-distinctive candidates, used only when distinctive ones run out
K_MAX = 4


def seed_for(global_id: str, purpose: str) -> int:
    """A stable seed from the global id and what the randomness is for."""
    return int(hashlib.sha256(f"{global_id}|{purpose}".encode()).hexdigest()[:16], 16)


# ---- captions as numbered scenes ---------------------------------------------------------------


def visual_parts(parts) -> list[str]:
    """Texts of the caption parts that describe what is seen, in their stored order."""
    texts = []
    for part in parts or ():
        if (part.get("annotator") or "").lower() in SKIP_ANNOTATORS:
            continue
        text = (part.get("text") or "").strip()
        if text:
            texts.append(text)
    return texts


def scene_caption(caption) -> tuple[str, int]:
    """The caption as "[scene i] ..." lines and the number of scenes. A caption without at least
    two visual parts (CaReBench, single-scene Vript spans) is one scene: the whole caption."""
    texts = visual_parts(caption.parts)
    if len(texts) <= 1:
        return f"[scene 1] {caption.caption.strip()}", 1
    return "\n".join(f"[scene {i}] {t}" for i, t in enumerate(texts, 1)), len(texts)


# ---- robust JSON -------------------------------------------------------------------------------


def parse_json(text: str) -> dict:
    """The first JSON object in a model answer ({} when there is none). Tolerates code fences,
    text around the object and trailing commas."""
    if not text:
        return {}
    text = re.sub(r"```(?:json)?", "", text)
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        chunk = text[match.start() :]
        for candidate in (chunk, re.sub(r",\s*([}\]])", r"\1", chunk)):
            try:
                value, _ = decoder.raw_decode(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    return {}


def as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"true", "yes", "1"}
    return bool(value)


def as_int(value, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        match = re.search(r"\d+", str(value or ""))
        return int(match.group(0)) if match else default


# ---- mining ------------------------------------------------------------------------------------


def normalize_facts(raw: list, n_scenes: int) -> list[dict]:
    """Clean mined facts: text required, type lower-cased (unknown -> other), scene clamped to
    1..n_scenes (always 1 for a one-scene caption), flags as booleans, duplicates dropped.
    Each fact gets an "id" (its position) and "selectable" (passes the type and flag rule)."""
    facts, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        text = " ".join(str(item.get("text") or "").split())
        key = text.lower().rstrip(".")
        if not text or key in seen:
            continue
        seen.add(key)
        kind = str(item.get("type") or "other").strip().lower().replace(" ", "_")
        scene = 1 if n_scenes <= 1 else min(max(as_int(item.get("scene"), 1), 1), n_scenes)
        fact = {
            "id": len(facts),
            "text": text,
            "type": kind if kind in FACT_TYPES else "other",
            "scene": scene,
            "single_frame": as_bool(item.get("single_frame")),
            "self_contained": as_bool(item.get("self_contained")),
            "distinctive": as_bool(item.get("distinctive")),
        }
        fact["selectable"] = selectable(fact)
        facts.append(fact)
    return facts


def mine_facts(llm: LLM, caption, template: str) -> tuple[list[dict], int, Cost, str]:
    """(facts, number of scenes, cost, raw answer) for one caption."""
    text, n_scenes = scene_caption(caption)
    answer, cost = llm.complete(template.replace("{caption}", text))
    return normalize_facts(parse_json(answer).get("facts", []), n_scenes), n_scenes, cost, answer


# ---- selection ---------------------------------------------------------------------------------


def selectable(fact: dict) -> bool:
    return bool(
        fact.get("single_frame") and fact.get("self_contained") and fact.get("type") in KEEP_TYPES
    )


def round_robin(facts: list[dict], rng: random.Random) -> list[dict]:
    """Facts shuffled within each scene, then taken one scene at a time in a shuffled scene order,
    so the first facts come from different parts of the span."""
    by_scene: dict[int, list[dict]] = {}
    for fact in facts:
        by_scene.setdefault(fact["scene"], []).append(fact)
    scenes = sorted(by_scene)
    rng.shuffle(scenes)
    for scene in scenes:
        rng.shuffle(by_scene[scene])
    order, depth = [], 0
    while len(order) < len(facts):
        order += [by_scene[s][depth] for s in scenes if depth < len(by_scene[s])]
        depth += 1
    return order


def select_candidates(
    facts: list[dict],
    global_id: str,
    max_candidates: int = MAX_CANDIDATES,
    max_backups: int = MAX_BACKUPS,
) -> list[dict]:
    """The facts to ground, in preference order: distinctive selectable facts round-robin over
    scenes, then up to max_backups non-distinctive ones (a query may use at most one)."""
    rng = random.Random(seed_for(global_id, "select"))
    keep = [f for f in facts if f.get("selectable", selectable(f))]
    distinctive = round_robin([f for f in keep if f["distinctive"]], rng)
    backups = round_robin([f for f in keep if not f["distinctive"]], rng)[:max_backups]
    return distinctive[: max_candidates - len(backups)] + backups


TEXT_WORDS = re.compile(
    r"[\"\u201c\u201d]|\b(logo|logos|watermark|text|reads|read|sign|signs|caption|subtitles?|"
    r"title|banner|label|lettering|written|says)\b",
    re.IGNORECASE,
)
SUBJECT_END = {
    "is", "are", "was", "were", "has", "have", "wears", "wear", "holds", "hold", "reads",
    "appears", "appear", "shows", "show", "sits", "lies", "can", "covers",
}  # fmt: skip
HEAD_END = {"wearing", "in", "with", "on", "at", "of", "who", "that", "holding", "playing", "by"}
ARTICLES = {"a", "an", "the", "one", "two", "three", "four", "five", "six", "several", "some"}


def is_text_fact(fact: dict) -> bool:
    """On-screen text, or a fact about a text or logo (quoted words, a sign, a watermark)."""
    return fact.get("type") == "on_screen_text" or bool(TEXT_WORDS.search(fact.get("text", "")))


def entity(fact: dict) -> str:
    """A rough key for the entity a fact is about: the head noun of its subject
    ("The woman wearing a black dress has ..." -> "woman"; "Three music stands are ..." ->
    "stand")."""
    words = re.findall(r"[a-z0-9'-]+", fact.get("text", "").lower())
    subject = []
    for word in words:
        if word in SUBJECT_END:
            break
        subject.append(word)
    for i, word in enumerate(subject):
        if word in HEAD_END and i:
            subject = subject[:i]
            break
    subject = [w for w in subject if w not in ARTICLES] or words[:1]
    head = subject[-1] if subject else ""
    return head[:-1] if head.endswith("s") and not head.endswith("ss") else head


def nested_order(
    grounded: list[dict], k_max: int = K_MAX, dropped: list[dict] | None = None
) -> list[dict]:
    """Grounded candidates (in candidate order) to the nested order: the first k facts form the
    k-fact set. Distinctive facts first; at most one non-distinctive fact, only to fill up. At
    most one fact about on-screen text or a logo per query (so never two about the same text).
    Facts about an entity not yet used are preferred over more facts about the same entity.
    Grounded facts excluded by the text rule are appended to `dropped` when it is given."""
    order: list[dict] = []
    blocked: list[dict] = []

    def take(pool: list[dict], limit: int) -> None:
        for prefer_new in (True, False):
            for fact in pool:
                if len(order) >= limit or any(f is fact for f in order):
                    continue
                if is_text_fact(fact) and any(is_text_fact(f) for f in order):
                    if not any(b is fact for b in blocked):
                        blocked.append(fact)
                    continue
                if prefer_new and entity(fact) in {entity(f) for f in order}:
                    continue
                order.append(fact)

    take([f for f in grounded if f["distinctive"]], k_max)
    if len(order) < k_max:
        take([f for f in grounded if not f["distinctive"]], len(order) + 1)
    if dropped is not None:
        dropped += [b for b in blocked if not any(f is b for f in order)]
    return order
