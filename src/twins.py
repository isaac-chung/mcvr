"""Twin set: controlled (k-1)-of-k near-misses from natural video pairs (docs/plan.md 4.4).

For a query source A:
1. neighbours: the top N_NEIGHBOURS corpus videos B from other source videos (`source_key`) by
   cosine similarity of caption embeddings (text-embedding-3-small over the human captions; a
   mining-only index, never used by any system);
2. one gpt-6-luna call per (A, B) with both captions (`prompts/twin_facts_v1.txt`): facts shared
   by A and B, and facts of A that B lacks or contradicts, typed and self-contained;
3. grounding with the local-frame, two-model rule (`src.grounding_v2`): every fact on A in
   its own scene (done together with A's natural candidates, so A's frames are paid once);
   then, for the pairs that still can give a twin, one pass on B: shared facts must be "yes" on
   B's local frames of their scene_b, the distinguishing fact d must be "no" from both models on
   a whole-video sample of B (`ABSENCE_POLICY`, plus those local frames);
4. keep the best B: most shared facts (up to 3), then a valid d, then neighbour rank. Pairs are
   tried best-first by what A's grounding allows, and the search stops at the first pair that
   reaches 3 shared facts and a d (at most MAX_B_PASSES passes on B per source);
5. Query_k = (k - 1 shared facts) + d for k = 1..4; d goes at a seeded position among the k
   facts. Each twin query records the twin id and which fact is d.

Captions are read here only (mining code). Every random choice is seeded from the global id.
"""

from __future__ import annotations

import random
import re
from pathlib import Path

import numpy as np

from src.constraints import K_MAX, as_int, nested_order, parse_json, seed_for
from src.frames import FramePolicy
from src.grounding_v2 import Grounder, Window, all_say, local_times, scene_windows
from src.judge import judge_caption
from src.llm import LLM
from src.types import Cost

TWIN_SPEC = "openai:gpt-6-luna@low"
EMBED_SPEC = "openai:text-embedding-3-small"
TWIN_PROMPT = Path("prompts/twin_facts_v1.txt")
N_NEIGHBOURS = 5
MAX_SHARED, MAX_DISTINCT = 6, 4
D_PREFERENCE = ("on_screen_text", "object", "attribute", "count", "scene")
TWIN_TYPES = set(D_PREFERENCE)
CAPTION_WORDS = 1200  # per caption in the twin prompt (longest scenes cut first)
EMBED_CHARS = 8000  # caption characters embedded (well under the model's input limit)
ABSENCE_POLICY = FramePolicy(8, 1.0, 24)  # d absent in B: whole video, 1 fps, 8..24 frames
MAX_B_PASSES = 3


def norm_text(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", str(text).lower()).split())


# ---- neighbours --------------------------------------------------------------------------------


def embedding_text(caption) -> str:
    return " ".join(caption.caption.split())[:EMBED_CHARS]


def nearest_neighbours(
    anchors: list[str],
    pool: list[str],
    vectors: dict[str, np.ndarray],
    source_of: dict[str, str],
    n: int = N_NEIGHBOURS,
) -> dict[str, list[tuple[str, float]]]:
    """{anchor: [(neighbour, cosine), ...]} best first: pool videos other than the anchor and
    from another source video (source_of gives `datasets.mc.source_key`). Vectors are unit
    norm (OpenAIEmbedder normalises); ties go to the smaller id."""
    ids = [p for p in pool if p in vectors]
    matrix = np.stack([vectors[p] for p in ids]) if ids else np.zeros((0, 1))
    out = {}
    for a in anchors:
        if a not in vectors or not ids:
            out[a] = []
            continue
        sims = matrix @ vectors[a]
        order = sorted(range(len(ids)), key=lambda i: (-float(sims[i]), ids[i]))
        found = []
        for i in order:
            b = ids[i]
            if b == a or source_of.get(b) == source_of.get(a):
                continue
            found.append((b, round(float(sims[i]), 6)))
            if len(found) == n:
                break
        out[a] = found
    return out


# ---- mining one pair ---------------------------------------------------------------------------


def twin_prompt(template: str, caption_a, caption_b, max_words: int = CAPTION_WORDS) -> str:
    text_a, _ = judge_caption(caption_a, max_words)
    text_b, _ = judge_caption(caption_b, max_words)
    return template.replace("{caption_a}", text_a).replace("{caption_b}", text_b)


def _scene(value, n_scenes: int) -> int:
    return 1 if n_scenes <= 1 else min(max(as_int(value, 1), 1), n_scenes)


def parse_twin_facts(text: str, n_scenes_a: int, n_scenes_b: int) -> dict[str, list[dict]]:
    """{"shared": [...], "distinct": [...]}: text required (under 25 words), type one of
    TWIN_TYPES (others dropped), scenes clamped, duplicates dropped (a fact listed as both
    shared and distinct is dropped from both), at most MAX_SHARED / MAX_DISTINCT."""
    data = parse_json(text)
    out: dict[str, list[dict]] = {"shared": [], "distinct": []}
    seen: dict[str, set[str]] = {"shared": set(), "distinct": set()}
    limits = {"shared": MAX_SHARED, "distinct": MAX_DISTINCT}
    for kind in ("shared", "distinct"):
        raw = data.get(kind)
        for item in raw if isinstance(raw, list) else []:
            if not isinstance(item, dict):
                continue
            fact_text = " ".join(str(item.get("text") or "").split())
            fact_type = str(item.get("type") or "").strip().lower().replace(" ", "_")
            key = norm_text(fact_text)
            if not key or fact_type not in TWIN_TYPES or key in seen[kind]:
                continue
            if len(fact_text.split()) > 25 or len(out[kind]) >= limits[kind]:
                continue
            seen[kind].add(key)
            fact = {
                "text": fact_text,
                "type": fact_type,
                "scene": _scene(item.get("scene_a"), n_scenes_a),
            }
            if kind == "shared":
                fact["scene_b"] = _scene(item.get("scene_b"), n_scenes_b)
            else:
                in_b = str(item.get("in_b") or "").strip().lower()
                fact["in_b"] = in_b if in_b in ("absent", "contradicted") else "absent"
                fact["b_evidence"] = " ".join(str(item.get("b_evidence") or "").split())
            out[kind].append(fact)
    both = seen["shared"] & seen["distinct"]
    for kind in ("shared", "distinct"):
        out[kind] = [f for f in out[kind] if norm_text(f["text"]) not in both]
    return out


def mine_pair(llm: LLM, template: str, caption_a, caption_b, n_a: int, n_b: int) -> dict:
    """{"shared", "distinct", "cost", "raw"} for one (A, B) pair (one LLM call)."""
    prompt = twin_prompt(template, caption_a, caption_b)
    raw, cost = llm.complete(prompt)
    return {**parse_twin_facts(raw, n_a, n_b), "cost": cost.to_dict(), "raw": raw}


# ---- A side: one keyed fact per distinct text --------------------------------------------------


def a_side_facts(pairs: list[dict], taken: dict[str, str] | None = None) -> list[dict]:
    """Keyed facts ("tw<j>") to ground on A: every shared and distinct fact of every pair, once
    per normalised text. taken maps normalised texts already grounded on A (e.g. natural
    candidates) to their keys; those are not repeated. Sets "a_key" on each pair fact."""
    taken = dict(taken or {})
    out = []
    for pair in pairs:
        for kind in ("shared", "distinct"):
            for fact in pair.get(kind, []):
                key = norm_text(fact["text"])
                if key not in taken:
                    taken[key] = f"tw{len(out)}"
                    out.append(
                        {
                            "key": taken[key],
                            "text": fact["text"],
                            "scene": fact["scene"],
                            "type": fact["type"],
                        }
                    )
                fact["a_key"] = taken[key]
    return out


def viable(pair: dict, confirmed_on_a: set[str]) -> tuple[list[dict], list[dict]]:
    """(shared, distinct) facts of a pair confirmed on A (by a_key)."""
    shared = [f for f in pair.get("shared", []) if f.get("a_key") in confirmed_on_a]
    distinct = [f for f in pair.get("distinct", []) if f.get("a_key") in confirmed_on_a]
    return shared, distinct


def d_rank(fact: dict) -> int:
    return D_PREFERENCE.index(fact["type"]) if fact["type"] in D_PREFERENCE else len(D_PREFERENCE)


# ---- B side ------------------------------------------------------------------------------------


def b_times(
    grounder: Grounder,
    dataset: str,
    video,
    caption,
    scenes: list[int],
    absence: FramePolicy = ABSENCE_POLICY,
) -> list[float]:
    """Frame times on B: local frames of the given scenes plus a whole-video sample, all away
    from B's index frames."""
    windows = scene_windows(caption, video.duration_s)
    index_times = grounder.store.index_times(dataset, video)
    duration = video.duration_s
    times: set[float] = set()
    for scene in sorted(set(scenes)):
        window = windows.get(scene, windows[min(windows)])
        local = local_times(
            window, index_times, duration, grounder.n_local, grounder.fallback, grounder.min_offset
        )
        times |= set(local)
    whole = Window(0.0, duration, "fallback")
    times |= set(
        local_times(whole, index_times, duration, fallback=absence, min_offset=grounder.min_offset)
    )
    return sorted(times)


def b_pass(
    grounder: Grounder,
    dataset: str,
    video,
    caption,
    shared: list[dict],
    distinct: list[dict],
    absence: FramePolicy = ABSENCE_POLICY,
) -> tuple[dict, dict[str, Cost]]:
    """One call per model on B with the shared facts (must be yes) and the d candidates (must
    be no). Returns ({"frame_times", "shared": {a_key: answers}, "distinct": {...},
    "shared_ok": [a_key], "distinct_ok": [a_key]}, costs)."""
    specs = tuple(grounder.models)
    times = b_times(grounder, dataset, video, caption, [f["scene_b"] for f in shared], absence)
    frames, kept = grounder.store.at(dataset, video, times)
    keyed = [{"key": f"s:{f['a_key']}", "text": f["text"]} for f in shared]
    keyed += [{"key": f"d:{f['a_key']}", "text": f["text"]} for f in distinct]
    answers, costs = grounder.on_frames(frames, kept, keyed)
    shared_ans = {f["a_key"]: answers[f"s:{f['a_key']}"] for f in shared}
    distinct_ans = {f["a_key"]: answers[f"d:{f['a_key']}"] for f in distinct}
    record = {
        "frame_times": kept,
        "shared": shared_ans,
        "distinct": distinct_ans,
        "shared_ok": [k for k, a in shared_ans.items() if all_say(a, "yes", specs)],
        "distinct_ok": [k for k, a in distinct_ans.items() if all_say(a, "no", specs)],
    }
    return record, costs


# ---- choosing the twin and its queries ---------------------------------------------------------


def twin_order(d: dict, shared: list[dict], k_max: int = K_MAX) -> list[dict]:
    """[d, shared...]: up to k_max - 1 shared facts in nested order (entities not yet used
    first; at most one on-screen-text or logo fact among d and the shared facts)."""
    pool = [{**d, "distinctive": True}] + [{**f, "distinctive": True} for f in shared]
    order = nested_order(pool, k_max)
    if not order or order[0]["a_key"] != d["a_key"]:
        return [d]
    by_key = {f["a_key"]: f for f in [d, *shared]}
    return [by_key[f["a_key"]] for f in order]


def twin_sets(order: list[dict], global_id: str) -> dict[int, tuple[list[dict], int]]:
    """{k: (facts, position of d, 0-based)} for k = 1..len(order): the first k - 1 shared facts
    plus d at a seeded position (the same seed per source, drawn once per k)."""
    d, shared = order[0], order[1:]
    rng = random.Random(seed_for(global_id, "twin_d_position"))
    out = {}
    for k in range(1, len(order) + 1):
        pos = rng.randrange(k)
        facts = list(shared[: k - 1])
        facts.insert(pos, d)
        out[k] = (facts, pos)
    return out


def pair_score(n_shared: int, has_d: bool, rank: int, k_max: int = K_MAX) -> tuple:
    """Sort key, best first: a valid d, most shared facts (up to k_max - 1), neighbour rank."""
    return (not has_d, -min(n_shared, k_max - 1), rank)


def choose_twin(
    pairs: list[dict],
    confirmed_on_a: set[str],
    run_b_pass,
    max_passes: int = MAX_B_PASSES,
    k_max: int = K_MAX,
) -> tuple[dict | None, list[dict]]:
    """(best pair or None, log). pairs: [{"b", "rank", "shared", "distinct"}] with a_key set;
    run_b_pass(pair, shared, distinct) -> b_pass record. Pairs with a d candidate confirmed on A
    are tried best first by their A-side potential; stops at a pair with k_max - 1 shared facts
    and a valid d, or after max_passes passes on B."""
    log, best, best_key = [], None, None
    ranked = []
    for pair in pairs:
        shared, distinct = viable(pair, confirmed_on_a)
        entry = {
            "b": pair["b"],
            "rank": pair["rank"],
            "shared_on_a": len(shared),
            "distinct_on_a": len(distinct),
        }
        log.append(entry)
        if distinct:
            ranked.append((pair_score(len(shared), True, pair["rank"], k_max), pair, entry))
    ranked.sort(key=lambda x: x[0])
    for _, pair, entry in ranked[:max_passes]:
        shared, distinct = viable(pair, confirmed_on_a)
        record = run_b_pass(pair, shared, distinct)
        entry["b_pass"] = record
        ok_shared = [f for f in shared if f["a_key"] in set(record["shared_ok"])]
        ok_d = sorted(
            (f for f in distinct if f["a_key"] in set(record["distinct_ok"])),
            key=lambda f: (d_rank(f), distinct.index(f)),
        )
        key = pair_score(len(ok_shared), bool(ok_d), pair["rank"], k_max)
        entry["valid_shared"], entry["valid_d"] = len(ok_shared), len(ok_d)
        if ok_d and (best_key is None or key < best_key):
            best_key = key
            best = {**pair, "valid_shared": ok_shared, "valid_distinct": ok_d, "d": ok_d[0]}
        if ok_d and len(ok_shared) >= k_max - 1:
            break
    return best, log


def twin_row(source: dict, k: int, facts: list[dict], d_pos: int, twin: dict, written: dict):
    """One line of queries_twin.jsonl."""
    keep = ("text", "type", "scene", "a_key")
    return {
        "query_id": f"{source['source']}#twin-k{k}",
        "set": "twin",
        "source": source["source"],
        "dataset": source["dataset"],
        "length_bin": source["length_bin"],
        "k": k,
        "query": written["query"],
        "phrases": written["phrases"],
        "facts": [
            {**{x: f.get(x) for x in keep}, "role": "distinct" if i == d_pos else "shared"}
            for i, f in enumerate(facts)
        ],
        "twin": twin["b"],
        "twin_rank": twin["rank"],
        "twin_similarity": twin.get("similarity"),
        "d_index": d_pos + 1,
        "d": facts[d_pos]["text"],
        "d_type": facts[d_pos]["type"],
    }
