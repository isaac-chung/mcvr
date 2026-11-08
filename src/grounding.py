"""Frame grounding of mined facts (`prompts/ground_v2.txt`).

One vision call per source video: the saved 16 frames (the same ones the describer saw; or,
with a frame policy such as GROUND_POLICY_FPS1, denser frames no system sees), each
preceded by a label "Frame i of n (t = ... s):", and a numbered list of candidate facts. For each
fact the model answers yes or no, names the frame(s) that show it and a short evidence phrase. A
fact is kept only when the answer is "yes" with at least one valid frame number and non-empty
evidence; anything else counts as "no". The vision model goes through the shared vlm cache.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from src.constraints import as_int, parse_json
from src.frames import FramePolicy, frame_labels
from src.llm import VisionLLM
from src.types import Cost

VLM_SPEC = "gemini:gemini-3.5-flash-lite"
GROUND_PROMPT = Path("prompts/ground_v2.txt")
N_FRAMES = 16
# decoupled grounding (smoke_fps05): 1 fps, at least 16, at most 240 frames; no system sees them
GROUND_POLICY_FPS1 = FramePolicy(16, 1.0, 240)


def numbered(facts: list[dict]) -> str:
    return "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts, 1))


def ground_prompt(template: str, n_frames: int, facts: list[dict]) -> str:
    return template.replace("{n}", str(n_frames)).replace("{facts}", numbered(facts))


def saved_frame_times(frames_dir: str | Path) -> tuple[list[Path], list[float]]:
    """(files, sample times) of the frames saved by src.frames.saved_frames, in order."""
    record = json.loads((Path(frames_dir) / "frames.json").read_text())
    saved = record["saved"]
    return [Path(frames_dir) / f"{i}.jpg" for i in saved], [record["times"][i] for i in saved]


def _answer(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    return "yes" if str(value or "").strip().lower().startswith(("yes", "true")) else "no"


def _frames(item: dict) -> list[int]:
    raw = item.get("frames", item.get("frame"))
    if raw is None or raw == "":
        return []
    out = []
    for value in raw if isinstance(raw, list) else [raw]:
        number = as_int(value, 0) if value not in (None, "", "null") else 0
        if number and number not in out:
            out.append(number)
    return out


def partial_items(text: str) -> list[dict]:
    """The complete per-fact objects of an answer cut off before its end (output token limit)."""
    items = []
    for match in re.finditer(r'\{[^{}]*"fact"[^{}]*\}', text or ""):
        item = parse_json(match.group(0))
        if item:
            items.append(item)
    return items


def parse_grounding(text: str, n_facts: int, n_frames: int) -> list[dict]:
    """One {"fact", "answer", "frames", "frame", "evidence", "model_answer"} per fact (1-based),
    in fact order. "answer" is "yes" only when the model said yes, named at least one frame in
    1..n_frames and gave evidence; "model_answer" keeps what the model said. Facts the answer
    leaves out, or answers it garbles, count as "no". Frames may be given as numbers or as
    their labels ("Frame 3 of 16 ..."); an answer cut off early keeps its complete items."""
    data = parse_json(text)
    items = data.get("visible", data.get("facts"))
    if items is None:  # no complete answer object: keep the complete per-fact items
        items = partial_items(text)
    if isinstance(items, dict):  # {"1": {...}, ...}
        items = [
            {"fact": k, **(v if isinstance(v, dict) else {"answer": v})} for k, v in items.items()
        ]
    found: dict[int, dict] = {}
    for pos, item in enumerate(items if isinstance(items, list) else [], 1):
        if not isinstance(item, dict):  # a bare "yes"/"no" names no frame: counts as no
            item = {"fact": pos, "answer": item}
        index = as_int(item.get("fact"), pos)
        if not 1 <= index <= n_facts or index in found:
            continue
        said = _answer(item.get("answer", item.get("visible")))
        frames = [f for f in _frames(item) if 1 <= f <= n_frames]
        evidence = " ".join(str(item.get("evidence") or "").split())
        answer = "yes" if said == "yes" and frames and evidence else "no"
        found[index] = {
            "fact": index,
            "answer": answer,
            "frames": frames if answer == "yes" else [],
            "frame": frames[0] if answer == "yes" else None,
            "evidence": evidence,
            "model_answer": said,
        }
    empty = {"answer": "no", "frames": [], "frame": None, "evidence": "", "model_answer": "none"}
    return [found.get(i, {"fact": i, **empty}) for i in range(1, n_facts + 1)]


def ground_facts(
    vlm: VisionLLM,
    frames: list[bytes],
    facts: list[dict],
    template: str,
    times: list[float | None] | None = None,
) -> tuple[list[dict], list[dict], Cost, str]:
    """(answers, confirmed facts with "grounded_frame" and "evidence", cost, raw answer).
    Each frame is labelled with its number and time (times, one per frame, when known). The
    vision model must accept `labels` (CachedVision over Gemini). No call when there are no
    facts or no frames."""
    if not facts or not frames:
        return [], [], Cost(), ""
    labels = frame_labels(times if times is not None else [None] * len(frames))
    prompt = ground_prompt(template, len(frames), facts)
    text, cost = vlm.describe(prompt, frames, labels=labels)
    answers = parse_grounding(text, len(facts), len(frames))
    confirmed = [
        {**fact, "grounded_frame": a["frame"], "evidence": a["evidence"]}
        for fact, a in zip(facts, answers, strict=True)
        if a["answer"] == "yes"
    ]
    return answers, confirmed, cost, text
