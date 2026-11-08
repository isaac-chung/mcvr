"""Frames and audio from a local video file, through the ffmpeg and ffprobe command line tools.

Videos are untrusted downloads: paths are passed as arguments and nothing from the data
directory is executed or imported.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def frame_times(duration: float, n: int) -> list[float]:
    """Midpoints of n equal slices of the clip: uniform sampling that never touches the last frame."""
    return [(i + 0.5) * duration / n for i in range(n)]


@dataclass(frozen=True)
class FramePolicy:
    """How many frames to sample from a video: a fixed n (fps None, the default), or
    ceil(duration x fps) frames with n as the floor and max_n as an optional cap. Frames are
    always uniform midpoints over the whole video (`frame_times`)."""

    n: int = 16
    fps: float | None = None
    max_n: int | None = None

    @classmethod
    def from_config(cls, cfg: dict | None, default_n: int = 16) -> FramePolicy:
        """{"n": .., "fps": .., "max_n": ..} (all optional); None gives a fixed default_n."""
        cfg = cfg or {}
        return cls(int(cfg.get("n", default_n)), cfg.get("fps"), cfg.get("max_n"))

    @property
    def fixed(self) -> bool:
        return self.fps is None

    def count(self, duration: float) -> int:
        if self.fps is None:
            return self.n
        count = max(self.n, math.ceil(round(duration * self.fps, 6)))
        return min(count, self.max_n) if self.max_n else count

    def folder(self) -> str:
        """frames<n> for a fixed n (frames for the trial 8), frames_fps<fps> otherwise."""
        if self.fps is None:
            return "frames" if self.n == 8 else f"frames{self.n}"
        return f"frames_fps{self.fps:g}"

    def settings(self) -> dict:
        """What sets the sampling (for frames.json and index hashes); max_side is separate."""
        return {"n": self.n, "fps": self.fps, "max_n": self.max_n}


def frame_labels(times: list[float | None]) -> list[str]:
    """ "Frame i of n (t = 21.4 s):" for each frame (1-based); no time when it is unknown."""
    n = len(times)
    return [
        f"Frame {i} of {n}" + (f" (t = {t:.1f} s):" if t is not None else ":")
        for i, t in enumerate(times, 1)
    ]


def saved_times(out_dir: str | Path) -> list[float]:
    """Sample times of the frames saved in out_dir (by saved_frames), in file order."""
    record = json.loads((Path(out_dir) / "frames.json").read_text())
    return [record["times"][i] for i in record["saved"]]


def probe(path: str) -> tuple[float, bool]:
    """(duration in seconds, has an audio stream)."""
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type:format=duration",
            "-of",
            "csv=p=0",
            path,
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    duration = max((float(x) for x in out if x.replace(".", "", 1).isdigit()), default=0.0)
    return duration, "audio" in out


def _grab(path: str, t: float, filters: str, codec_args: list[str]) -> bytes:
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", path, "-frames:v", "1", "-vf", filters]
    return subprocess.run([*cmd, *codec_args, "pipe:1"], capture_output=True, check=False).stdout


def rgb_frames(path: str, n: int, size: int) -> np.ndarray:
    """[n, size, size, 3] uint8 frames, squashed to size x size (what SigLIP expects).

    A frame that cannot be decoded repeats the previous good one.
    """
    duration, _ = probe(path)
    frames, last = [], None
    for t in frame_times(duration, n):
        raw = _grab(
            path, t, f"scale={size}:{size}:flags=bicubic", ["-f", "rawvideo", "-pix_fmt", "rgb24"]
        )
        if len(raw) == size * size * 3:
            last = np.frombuffer(raw, np.uint8).reshape(size, size, 3)
        if last is None:
            last = np.zeros((size, size, 3), np.uint8)
        frames.append(last)
    return np.stack(frames)


def jpeg_frames(path: str, n: int, max_side: int = 768) -> list[bytes]:
    """n JPEG frames, longest side at most max_side, aspect ratio kept."""
    duration, _ = probe(path)
    scale = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    out = [_grab(path, t, scale, ["-f", "mjpeg", "-q:v", "3"]) for t in frame_times(duration, n)]
    return [jpeg for jpeg in out if jpeg]


def saved_frames(
    path: str | Path,
    n: int,
    out_dir: str | Path,
    max_side: int = 768,
    fps: float | None = None,
    max_n: int | None = None,
) -> list[Path]:
    """n uniform frames over the whole video as out_dir/<i>.jpg, reused when already saved.
    With fps, the count is ceil(duration x fps), at least n and at most max_n (`FramePolicy`).

    The same files feed the describer and the frame verifiers, so both see identical frames.
    Frames are spread over the whole video whatever its length (seeking is by timestamp, so a
    long video costs no more to sample). out_dir/frames.json records the settings and sample
    times; a frame that cannot be decoded is retried slightly earlier, then left out. Another
    frame count or size needs another folder (frames are never overwritten).
    """
    out_dir = Path(out_dir)
    record_file = out_dir / "frames.json"
    policy = FramePolicy(n, fps, max_n)
    settings = {"n": n, "max_side": max_side}
    if fps is not None:
        settings.update(fps=fps, max_n=max_n)
    expected = {"fps": None, "max_n": None, **settings}  # old records have no fps keys
    if record_file.exists():
        record = json.loads(record_file.read_text())
        if {k: record.get(k) for k in expected} != expected:  # never overwrite another sampling
            raise ValueError(
                f"{out_dir} holds frames sampled with other settings; use another folder"
            )
        files = [out_dir / f"{i}.jpg" for i in record["saved"]]
        if all(f.exists() for f in files):
            return files
    duration, _ = probe(str(path))
    scale = f"scale='min({max_side},iw)':'min({max_side},ih)':force_original_aspect_ratio=decrease"
    out_dir.mkdir(parents=True, exist_ok=True)
    times, saved = frame_times(duration, policy.count(duration)), []
    for i, t in enumerate(times):
        for retry in (t, t * 0.97, max(t - 1.0, 0.0)):  # the frame after a bad packet may decode
            jpeg = _grab(str(path), retry, scale, ["-f", "mjpeg", "-q:v", "3"])
            if jpeg:
                (out_dir / f"{i}.jpg").write_bytes(jpeg)
                saved.append(i)
                break
    record_file.write_text(
        json.dumps({**settings, "duration_s": duration, "times": times, "saved": saved})
    )
    return [out_dir / f"{i}.jpg" for i in saved]


def wav_audio(path: str) -> bytes:
    """Mono 16 kHz WAV of the audio track; empty bytes when there is none."""
    return subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            path,
            "-vn",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-f",
            "wav",
            "pipe:1",
        ],
        capture_output=True,
        check=False,
    ).stdout
