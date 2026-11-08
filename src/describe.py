"""DESC for videos without released descriptions: a vision model describes sampled frames.

The prompt asks for the visual scene and any visible text, never for audio. The released human
captions are not shown to the model: the queries are built from them.
"""

from __future__ import annotations

from pathlib import Path

from src.frames import jpeg_frames
from src.llm import VisionLLM
from src.types import Cost


def describe_video(
    vlm: VisionLLM, video: str | Path, prompt_path: str | Path, n_frames: int
) -> tuple[str, Cost]:
    return describe_frames(vlm, jpeg_frames(str(video), n_frames), prompt_path)


def describe_frames(
    vlm: VisionLLM,
    frames: list[bytes],
    prompt_path: str | Path,
    labels: list[str] | None = None,
) -> tuple[str, Cost]:
    """Describe JPEG frames already sampled (for example the saved frames of the new study).
    labels (one per frame, e.g. "Frame 3 of 40 (t = 12.5 s):") go before each image; the
    vision model must accept them (Gemini). Unlabelled calls keep their cache keys."""
    if not frames:
        return "", Cost()
    prompt = Path(prompt_path).read_text().replace("{n}", str(len(frames)))
    if labels is None:
        text, cost = vlm.describe(prompt, frames)
    else:
        text, cost = vlm.describe(prompt, frames, labels=labels)
    return text.strip(), cost
