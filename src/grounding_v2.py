"""Local-frame grounding with two-model agreement (docs/plan.md 4.3; `prompts/ground_v3.txt`).

Each fact is checked on about 6 frames taken from its own scene or shot, at times chosen away
from the index frame times (the 0.5 fps index, floor 8, in data/mc/<dataset>/frames_fps0.5/), so
grounding never reuses what the describer saw. Two vision models answer for every fact (Gemini
`gemini-3.5-flash-lite` and OpenAI `gpt-6-astra`); a fact is confirmed only when both say "yes"
with a frame number and evidence, as in ground_v2. Answers are "yes", "no" or "unsure", so the
same parser also serves absence checks (twin set: both models must say "no").

Scene windows come from the caption parts' start_s / end_s, in the same order as the numbered
scenes the miner saw (`constraints.scene_caption`):
- Shot2Story and Vript long: one window per visual part (shot or scene), "scene";
- a caption with one visual part (CaReBench, single-scene Vript): the whole video is the scene,
  "whole", still about 6 frames;
- a scene without times (Vript short): the whole video at 1 fps, capped (`FALLBACK_POLICY`),
  "fallback"; recorded per fact.
Facts that share a window share one call per model (at most MAX_FACTS_PER_CALL facts per call).

Index visibility ("did the index drop it?"): one extra call per source on all of its index
frames, by default with Gemini only (the describer's model, and the cheaper choice), over the
natural candidates and the twin facts used in queries. No reuse of the 16-frame smoke grounding:
those frames are not the 0.5 fps index frames.

Local frames are written once to data/mc/<dataset>/frames_local/<video_id>/t<ms>.jpg (new
folder; index frames and their settings are never touched). All model calls go through the
shared vlm cache, so a rerun is free.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from src.constraints import SKIP_ANNOTATORS, as_int, parse_json
from src.frames import FramePolicy, frame_labels, frame_times, saved_frames
from src.grounding import partial_items
from src.llm import VisionLLM
from src.types import Cost

GEMINI_SPEC = "gemini:gemini-3.5-flash-lite"
OPENAI_SPEC = "openai:gpt-6-astra"
GROUNDERS = (GEMINI_SPEC, OPENAI_SPEC)
INDEX_CHECK = (GEMINI_SPEC,)
GROUND_PROMPT = Path("prompts/ground_v3.txt")
N_LOCAL = 6
MIN_OFFSET_S = 0.5  # local frames at least this far from any index frame, where the scene allows
MIN_SPACING_S = 0.2  # shorter scenes get fewer local frames
INDEX_POLICY = FramePolicy(8, 0.5)  # configs/mc_smoke_fps05.yaml (do not change)
FALLBACK_POLICY = FramePolicy(8, 1.0, 30)  # scenes without times: whole video, 1 fps, 8..30
MAX_FACTS_PER_CALL = 30
LOCAL_FOLDER = "frames_local"
MAX_SIDE = 768
ANSWERS = ("yes", "no", "unsure")

# ---- scene windows -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Window:
    start: float
    end: float
    kind: str  # "scene", "whole" or "fallback"

    def to_dict(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "kind": self.kind}


def _visual_part_times(parts) -> list[tuple[float | None, float | None]]:
    """(start_s, end_s) of the parts `constraints.visual_parts` keeps, in the same order."""
    out = []
    for part in parts or ():
        if (part.get("annotator") or "").lower() in SKIP_ANNOTATORS:
            continue
        if not (part.get("text") or "").strip():
            continue
        out.append((part.get("start_s"), part.get("end_s")))
    return out


def scene_windows(caption, duration: float) -> dict[int, Window]:
    """{scene number: Window} for the numbered scenes of `constraints.scene_caption`."""
    whole = Window(0.0, duration, "whole")
    spans = _visual_part_times(caption.parts)
    if len(spans) <= 1:
        return {1: whole}
    out = {}
    for i, (start, end) in enumerate(spans, 1):
        if start is None or end is None:
            out[i] = Window(0.0, duration, "fallback")
            continue
        start, end = max(0.0, float(start)), min(duration, float(end))
        out[i] = Window(start, end, "scene") if end > start else Window(0.0, duration, "fallback")
    return out


# ---- frame times off the index -----------------------------------------------------------------


def _far_point(a: float, b: float, index_times: list[float], min_offset: float) -> float:
    """A time in [a, b]: as far as possible from the index times (up to min_offset), then as
    close as possible to the middle of [a, b]."""
    mid = (a + b) / 2
    if b <= a:
        return a
    steps = 20
    best, best_key = mid, None
    for j in range(steps + 1):
        t = a + (b - a) * j / steps
        gap = min((abs(t - x) for x in index_times), default=math.inf)
        key = (-min(gap, min_offset), abs(t - mid))
        if best_key is None or key < best_key:
            best, best_key = t, key
    return best


def local_times(
    window: Window,
    index_times: list[float],
    duration: float,
    n: int = N_LOCAL,
    fallback: FramePolicy = FALLBACK_POLICY,
    min_offset: float = MIN_OFFSET_S,
) -> list[float]:
    """Frame times for a window: n slices of the window (fallback: the policy's count over the
    whole video), one time per slice, each moved within its slice away from the index frames."""
    end_cap = max(duration - 0.05, 0.0)
    start, end = window.start, min(window.end, end_cap) if end_cap > window.start else window.end
    if window.kind == "fallback":
        count = fallback.count(duration)
    else:
        count = max(1, min(n, int((end - start) / MIN_SPACING_S)))
    width = (end - start) / count
    times = []
    for i in range(count):
        t = round(
            _far_point(start + i * width, start + (i + 1) * width, index_times, min_offset), 3
        )
        if t not in times:
            times.append(t)
    return sorted(times)


def min_gap(times: list[float], index_times: list[float]) -> float | None:
    """Smallest distance from a frame time to an index frame time (None without index times)."""
    if not times or not index_times:
        return None
    return round(min(abs(t - x) for t in times for x in index_times), 3)


# ---- frames on disk ----------------------------------------------------------------------------

Grab = Callable[[Path, float, int], bytes]


def ffmpeg_grab(path: Path, t: float, max_side: int) -> bytes:
    """One JPEG at time t, longest side at most max_side (as src.frames.saved_frames)."""
    from src.frames import _grab

    scale = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    return _grab(str(path), t, scale, ["-f", "mjpeg", "-q:v", "3"])


class FrameStore:
    """Index frames (read, or sampled with the index's own settings when missing) and local
    frames at chosen times (written once under <root>/<dataset>/frames_local/<video_id>/)."""

    def __init__(
        self,
        root: str | Path = "data/mc",
        max_side: int = MAX_SIDE,
        grab: Grab | None = None,
        index_policy: FramePolicy = INDEX_POLICY,
    ):
        self.root, self.max_side = Path(root), max_side
        self.grab = grab or ffmpeg_grab
        self.index_policy = index_policy

    def video_path(self, dataset: str, video) -> Path:
        path = Path(video.path)
        return path if path.exists() else self.root / dataset / "videos" / f"{video.video_id}.mp4"

    def index_dir(self, dataset: str, video) -> Path:
        return self.root / dataset / self.index_policy.folder() / video.video_id

    def index_times(self, dataset: str, video) -> list[float]:
        """Times of the index frames: frames.json when saved, else the policy's times computed
        from the stored duration (no ffmpeg; used only to keep local frames away from them)."""
        record = self.index_dir(dataset, video) / "frames.json"
        if record.exists():
            data = json.loads(record.read_text())
            return [data["times"][i] for i in data["saved"]]
        duration = video.duration_s
        return frame_times(duration, self.index_policy.count(duration))

    def index_frames(self, dataset: str, video) -> tuple[list[bytes], list[float]]:
        """The index frames (what the describer saw) and their times."""
        out = self.index_dir(dataset, video)
        p = self.index_policy
        files = saved_frames(self.video_path(dataset, video), p.n, out, fps=p.fps, max_n=p.max_n)
        data = json.loads((out / "frames.json").read_text())
        times = [data["times"][i] for i in data["saved"]]
        return [f.read_bytes() for f in files], times

    def at(self, dataset: str, video, times: list[float]) -> tuple[list[bytes], list[float]]:
        """JPEGs at the given times (frames that fail to decode, also slightly earlier, are
        left out) and the times kept."""
        folder = self.root / dataset / LOCAL_FOLDER / video.video_id
        folder.mkdir(parents=True, exist_ok=True)
        frames, kept = [], []
        for t in times:
            file = folder / f"t{round(t * 1000):07d}.jpg"
            if not file.exists():
                for retry in (t, t * 0.97, max(t - 1.0, 0.0)):
                    jpeg = self.grab(self.video_path(dataset, video), retry, self.max_side)
                    if jpeg:  # atomic: two sources may share a twin video across threads
                        tmp = file.with_suffix(f".{threading.get_ident()}.tmp")
                        tmp.write_bytes(jpeg)
                        os.replace(tmp, file)
                        break
            if file.exists():
                frames.append(file.read_bytes())
                kept.append(t)
        return frames, kept


# ---- prompts and answers -----------------------------------------------------------------------


def numbered(facts: list[dict]) -> str:
    return "\n".join(f"{i}. {f['text']}" for i, f in enumerate(facts, 1))


def fill(template: str, n_frames: int, facts: list[dict]) -> str:
    return template.replace("{n}", str(n_frames)).replace("{facts}", numbered(facts))


def _said(value) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    word = str(value or "").strip().lower()
    if word in ("", "none", "null", "n/a"):
        return "unsure"
    if word.startswith(("yes", "true")):
        return "yes"
    if word.startswith(("no", "false", "absent")) and not word.startswith("not sure"):
        return "no"
    return "unsure"


def _frame_numbers(item: dict) -> list[int]:
    raw = item.get("frames", item.get("frame"))
    if raw is None or raw == "":
        return []
    out = []
    for value in raw if isinstance(raw, list) else [raw]:
        number = as_int(value, 0) if value not in (None, "", "null") else 0
        if number and number not in out:
            out.append(number)
    return out


def parse_answers(text: str, n_facts: int, n_frames: int) -> list[dict]:
    """One {"fact", "answer", "frames", "evidence", "model_answer"} per fact (1-based), in fact
    order. "answer" is "yes" only when the model said yes, named a frame in 1..n_frames and gave
    evidence (else "unsure"); "no" is kept; facts left out or garbled are "unsure". Accepts the
    "visible" (ground_v3) or "facts" (judge_frames_v1) key, and an answer cut off early."""
    data = parse_json(text)
    items = data.get("visible", data.get("facts", data.get("answers")))
    if items is None:
        items = partial_items(text)
    if isinstance(items, dict):
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
        said = _said(item.get("answer", item.get("visible")))
        frames = [f for f in _frame_numbers(item) if 1 <= f <= n_frames]
        evidence = " ".join(str(item.get("evidence") or "").split())
        answer = said
        if said == "yes" and not (frames and evidence):
            answer = "unsure"
        found[index] = {
            "fact": index,
            "answer": answer,
            "frames": frames if answer == "yes" else [],
            "evidence": evidence,
            "model_answer": said,
        }
    empty = {"answer": "unsure", "frames": [], "evidence": "", "model_answer": "none"}
    return [found.get(i, {"fact": i, **empty}) for i in range(1, n_facts + 1)]


def call_with_retries(fn: Callable[[], tuple[str, Cost]], retries: int = 5, sleep=time.sleep):
    """Retry rate limits, server errors and dropped connections with backoff
    (`embedders.retryable`); other errors (e.g. a 400 for a bad request) are raised at once."""
    from src.embedders import retryable

    for attempt in range(retries + 1):
        try:
            return fn()
        except Exception as error:  # the OpenAI client raises many kinds
            if attempt == retries or not retryable(error):
                raise
            sleep(min(60.0, 2.0 ** (attempt + 1)))
    raise AssertionError("unreachable")


# ---- grounding calls ---------------------------------------------------------------------------


def add_costs(total: dict[str, Cost], new: dict[str, Cost]) -> dict[str, Cost]:
    for spec, cost in new.items():
        total[spec] = total.get(spec, Cost()) + cost
    return total


class Grounder:
    """Ground keyed facts ({"key", "text", "scene", ...}) with several vision models.

    models maps a spec to a vision model that accepts frame labels (CachedVision over Gemini or
    over `systems.verify.LabelledOpenAIVision`)."""

    def __init__(
        self,
        models: dict[str, VisionLLM],
        template: str,
        store: FrameStore,
        n_local: int = N_LOCAL,
        max_facts: int = MAX_FACTS_PER_CALL,
        fallback: FramePolicy = FALLBACK_POLICY,
        min_offset: float = MIN_OFFSET_S,
        retries: int = 5,
    ):
        self.models, self.template, self.store = models, template, store
        self.n_local, self.max_facts = n_local, max_facts
        self.fallback, self.min_offset, self.retries = fallback, min_offset, retries

    def on_frames(
        self,
        frames: list[bytes],
        times: list[float],
        facts: list[dict],
        specs: tuple[str, ...] | None = None,
    ) -> tuple[dict[str, dict[str, dict]], dict[str, Cost]]:
        """({fact key: {spec: answer}}, {spec: cost}): every model answers every fact on the
        same labelled frames (facts in chunks of max_facts). No call without frames or facts."""
        out: dict[str, dict[str, dict]] = {f["key"]: {} for f in facts}
        costs: dict[str, Cost] = {}
        if not frames or not facts:
            return out, costs
        labels = frame_labels(list(times))
        for spec in specs or tuple(self.models):
            vlm = self.models[spec]
            for i in range(0, len(facts), self.max_facts):
                chunk = facts[i : i + self.max_facts]
                prompt = fill(self.template, len(frames), chunk)
                text, cost = call_with_retries(
                    lambda vlm=vlm, prompt=prompt: vlm.describe(prompt, frames, labels=labels),
                    self.retries,
                )
                costs[spec] = costs.get(spec, Cost()) + cost
                parsed = parse_answers(text, len(chunk), len(frames))
                for fact, answer in zip(chunk, parsed, strict=True):
                    frame = answer["frames"][0] if answer["frames"] else None
                    out[fact["key"]][spec] = {
                        **{k: answer[k] for k in ("answer", "frames", "evidence", "model_answer")},
                        "time": times[frame - 1] if frame else None,
                    }
        return out, costs

    def plan_local(self, dataset: str, video, caption, facts: list[dict]) -> list[dict]:
        """The local calls a source needs, without making them: [{"window", "times", "facts"}],
        one per distinct window (facts with no or an unknown scene use scene 1)."""
        windows = scene_windows(caption, video.duration_s)
        index_times = self.store.index_times(dataset, video)
        groups: dict[Window, list[dict]] = {}
        for fact in facts:
            window = windows.get(int(fact.get("scene") or 1), windows[min(windows)])
            groups.setdefault(window, []).append(fact)
        plans = []
        for window, members in groups.items():
            times = local_times(
                window, index_times, video.duration_s, self.n_local, self.fallback,
                self.min_offset,
            )  # fmt: skip
            plans.append({"window": window, "times": times, "facts": members})
        return plans

    def local(
        self, dataset: str, video, caption, facts: list[dict]
    ) -> tuple[dict[str, dict], dict[str, Cost]]:
        """({fact key: record}, {spec: cost}) where record has "window", "frame_times",
        "min_gap_to_index", "answers" {spec: answer} and "confirmed" (every model said yes)."""
        records: dict[str, dict] = {}
        costs: dict[str, Cost] = {}
        index_times = self.store.index_times(dataset, video)
        for plan in self.plan_local(dataset, video, caption, facts):
            frames, kept = self.store.at(dataset, video, plan["times"])
            answers, cost = self.on_frames(frames, kept, plan["facts"])
            add_costs(costs, cost)
            for fact in plan["facts"]:
                records[fact["key"]] = {
                    "window": plan["window"].to_dict(),
                    "frame_times": kept,
                    "min_gap_to_index": min_gap(kept, index_times),
                    "answers": answers[fact["key"]],
                    "confirmed": all_say(answers[fact["key"]], "yes", tuple(self.models)),
                }
        return records, costs

    def index_check(
        self, dataset: str, video, facts: list[dict], specs: tuple[str, ...] = INDEX_CHECK
    ) -> tuple[dict[str, dict], dict[str, Cost]]:
        """({fact key: {"answers", "visible"}}, costs) on all of the video's index frames."""
        if not facts:
            return {}, {}
        frames, times = self.store.index_frames(dataset, video)
        answers, costs = self.on_frames(frames, times, facts, specs)
        out = {
            key: {"answers": a, "visible": all_say(a, "yes", specs), "n_frames": len(frames)}
            for key, a in answers.items()
        }
        return out, costs


def all_say(answers: dict[str, dict], word: str, specs: tuple[str, ...]) -> bool:
    """Every model in specs answered `word` (False when one of them did not answer)."""
    return bool(specs) and all(answers.get(s, {}).get("answer") == word for s in specs)


def agreement_label(answers: dict[str, dict], specs: tuple[str, ...]) -> str:
    """ "all_yes", "all_no", "all_unsure" or "disagree" over the models in specs."""
    said = {answers.get(s, {}).get("answer", "unsure") for s in specs}
    return f"all_{said.pop()}" if len(said) == 1 else "disagree"


# ---- agreement statistics ----------------------------------------------------------------------


def cohen_kappa(a: list[str], b: list[str]) -> float | None:
    """Cohen's kappa of two raters over the same items (None when undefined: no items, or both
    raters always give the same single label)."""
    if not a or len(a) != len(b):
        return None
    n = len(a)
    labels = sorted(set(a) | set(b))
    observed = sum(x == y for x, y in zip(a, b, strict=True)) / n
    expected = sum((a.count(c) / n) * (b.count(c) / n) for c in labels)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def agreement_table(items: list[dict], spec_a: str, spec_b: str, by: str = "type") -> dict:
    """Agreement of two models per value of `by` (and "all"). items: {"answers": {spec: {...}},
    by: ...}. Per group: n, raw agreement on yes/no/unsure, kappa on the three answers and on
    yes vs not yes, and the share of yes per model."""
    groups: dict[str, list[tuple[str, str]]] = {}
    for item in items:
        a = item["answers"].get(spec_a, {}).get("answer")
        b = item["answers"].get(spec_b, {}).get("answer")
        if a is None or b is None:
            continue
        for name in ("all", str(item.get(by))):
            groups.setdefault(name, []).append((a, b))
    table = {}
    for name in sorted(groups, key=lambda g: (g != "all", g)):
        pairs = groups[name]
        xs, ys = [p[0] for p in pairs], [p[1] for p in pairs]
        bx, by_ = [x == "yes" for x in xs], [y == "yes" for y in ys]
        table[name] = {
            "n": len(pairs),
            "raw_agreement": sum(x == y for x, y in pairs) / len(pairs),
            "kappa": cohen_kappa(xs, ys),
            "raw_agreement_yes": sum(x == y for x, y in zip(bx, by_, strict=True)) / len(pairs),
            "kappa_yes": cohen_kappa([str(x) for x in bx], [str(y) for y in by_]),
            "yes_a": sum(bx) / len(pairs),
            "yes_b": sum(by_) / len(pairs),
        }
    return table
