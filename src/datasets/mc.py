"""Datasets of the multi-constraint study, stored in data/mc/<dataset>/ (see docs/artifacts.md).

Human captions are kept away from the index by structure, not by care. `McCorpus` is built from
videos.jsonl only and holds no caption text, so the describer, the index builder and the systems,
which only ever get an `McCorpus` and the generated descriptions, cannot reach a caption.
`load_captions` reads captions.jsonl and is meant for query mining, sampling and labels only; no
indexing or system module may import it (a test checks this).

One dataset folder gives a corpus keyed by its own video ids. `McCorpus.pool` joins several
folders into one corpus keyed by global ids "<dataset>/<video_id>"; descriptions and frames
still live in each dataset's own folder.
"""

from __future__ import annotations

import hashlib
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from src.describe import describe_frames
from src.embedders import Embedder
from src.frames import FramePolicy, frame_labels, saved_frames, saved_times
from src.indexes.dense_text import DenseTextIndex
from src.llm import VisionLLM
from src.types import Corpus, Cost

LENGTH_BINS = ("lt30s", "30to60s", "1to2min", "2to4min")
MAX_DURATION_S = 240.0
DATASETS = ("shot2story", "carebench", "vript_short", "vript_long")
# keys that would mean caption text slipped into videos.jsonl
CAPTION_KEYS = {"caption", "captions", "parts", "caption_origin", "text", "sentences"}


def length_bin(duration_s: float) -> str:
    """< 30 s, 30-60 s, 1-2 min, 2-4 min. Longer videos are not part of the study: an error."""
    if duration_s < 30:
        return "lt30s"
    if duration_s <= 60:
        return "30to60s"
    if duration_s <= 120:
        return "1to2min"
    if duration_s <= MAX_DURATION_S:
        return "2to4min"
    raise ValueError(f"duration {duration_s:.1f} s is over {MAX_DURATION_S:.0f} s (no length bin)")


def global_id(dataset: str, video_id: str) -> str:
    return f"{dataset}/{video_id}"


def split_global_id(gid: str) -> tuple[str, str]:
    """'<dataset>/<video_id>' -> (dataset, video_id)."""
    dataset, sep, video_id = gid.partition("/")
    if not sep or not video_id:
        raise ValueError(f"{gid!r} is not a global id '<dataset>/<video_id>'")
    return dataset, video_id


def read_jsonl(path: str | Path) -> list[dict]:
    with open(path) as handle:
        return [json.loads(line) for line in handle if line.strip()]


# ---- the corpus: videos.jsonl only --------------------------------------------------------


@dataclass(frozen=True)
class Video:
    video_id: str
    path: str  # relative to the repo root, as written in videos.jsonl
    duration_s: float
    width: int = 0
    height: int = 0
    has_audio: bool = False
    source: str = ""
    split: str = ""
    length_bin: str = ""
    source_video_id: str = ""  # Vript long: the YouTube video the span was cut from
    dataset: str = ""  # the folder name, set on load


def load_videos(data_dir: str | Path) -> list[Video]:
    """The rows of videos.jsonl. Only the known metadata fields are kept; a row carrying caption
    text is an error, not something to drop silently. The length bin is always recomputed from
    the duration (the stored labels predate the 2to4min bin and disagree at the edges)."""
    known = {f.name for f in fields(Video)} - {"dataset", "length_bin"}
    dataset = Path(data_dir).name
    videos = []
    for row in read_jsonl(Path(data_dir) / "videos.jsonl"):
        leaked = CAPTION_KEYS & row.keys()
        if leaked:
            raise ValueError(f"videos.jsonl row {row.get('video_id')} has caption keys {leaked}")
        video = Video(**{k: v for k, v in row.items() if k in known}, dataset=dataset)
        videos.append(Video(**{**asdict(video), "length_bin": length_bin(video.duration_s)}))
    return videos


def source_key(video: Video) -> str:
    """The source video a corpus unit was cut from, so a sample can take one unit per source.

    Shot2Story ids are '<youtube id>.<shot>', CaReBench ids 'v_<fineaction id>_<clip>', Vript
    long spans carry source_video_id; a Vript short is its own source. YouTube ids are shared
    across datasets (Shot2Story and Vript long overlap)."""
    if video.dataset == "shot2story":
        return "yt:" + video.video_id.rsplit(".", 1)[0]
    if video.dataset == "carebench":
        return "fineaction:" + video.video_id.rsplit("_", 1)[0]
    if video.source_video_id:
        return "yt:" + video.source_video_id
    return f"{video.dataset}:{video.video_id}"


class McCorpus:
    """The videos to describe, index and search. Has no access to captions by construction.

    `doc_ids` are the dataset's video ids for one folder, or global ids for a pooled corpus."""

    def __init__(self, data_dir: str | Path, videos: list[Video]):
        self.dir = Path(data_dir)
        self.name = self.dir.name
        self.videos = {v.video_id: v for v in videos}
        self.doc_ids = [v.video_id for v in videos]
        # doc id -> (dataset folder, video id inside it)
        self._where = {v.video_id: (self.dir, v.video_id) for v in videos}

    @classmethod
    def load(cls, data_dir: str | Path) -> McCorpus:
        return cls(data_dir, load_videos(data_dir))

    @classmethod
    def pool(
        cls,
        root: str | Path = "data/mc",
        datasets: tuple[str, ...] | list[str] = DATASETS,
        ids: list[str] | None = None,
    ) -> McCorpus:
        """One corpus over several data/mc/<dataset>/ folders, keyed by '<dataset>/<video_id>'.
        ids (global ids) keeps only those videos, in that order."""
        root = Path(root)
        videos, where = {}, {}
        for ds in datasets:
            for v in load_videos(root / ds):
                gid = global_id(ds, v.video_id)
                videos[gid], where[gid] = v, (root / ds, v.video_id)
        if ids is not None:
            missing = [i for i in ids if i not in videos]
            if missing:
                raise KeyError(f"{len(missing)} ids not in the corpus, e.g. {missing[:3]}")
            videos = {i: videos[i] for i in ids}
        corpus = cls.__new__(cls)
        corpus.dir, corpus.name = root, "+".join(datasets)
        corpus.videos, corpus.doc_ids = videos, list(videos)
        corpus._where = {d: where[d] for d in videos}
        return corpus

    def locate(self, doc_id: str) -> tuple[Path, str]:
        """(dataset folder, video id inside it) of a doc id."""
        return self._where[doc_id]

    def video_path(self, doc_id: str) -> Path:
        folder, video_id = self._where[doc_id]
        path = Path(self.videos[doc_id].path)
        return path if path.exists() else folder / "videos" / f"{video_id}.mp4"

    def frames_dir(self, doc_id: str, n: int = 8, policy: FramePolicy | None = None) -> Path:
        """frames/<video id>/ for the trial 8 frames; frames<n>/<video id>/ otherwise (frames16);
        with an fps policy, frames_fps<fps>/<video id>/ (e.g. frames_fps0.5)."""
        folder, video_id = self._where[doc_id]
        policy = policy or FramePolicy(n)
        return folder / policy.folder() / video_id

    def frames(self, doc_id: str, n: int = 8, policy: FramePolicy | None = None) -> list[bytes]:
        """n uniform JPEG frames (longest side 768 px), sampled once and reused; policy (a
        `FramePolicy`) replaces n, e.g. 0.5 fps with a floor of 8."""
        policy = policy or FramePolicy(n)
        files = saved_frames(
            self.video_path(doc_id),
            policy.n,
            self.frames_dir(doc_id, policy=policy),
            fps=policy.fps,
            max_n=policy.max_n,
        )
        return [f.read_bytes() for f in files]

    def frame_times(
        self, doc_id: str, n: int = 8, policy: FramePolicy | None = None
    ) -> list[float]:
        """Sample times (s) of the saved frames of `frames` (call that first)."""
        return saved_times(self.frames_dir(doc_id, n, policy))

    def by_bin(self) -> dict[str, list[str]]:
        bins: dict[str, list[str]] = {b: [] for b in LENGTH_BINS}
        for doc_id, v in self.videos.items():
            bins.setdefault(v.length_bin, []).append(doc_id)
        return bins

    def to_corpus(self, descriptions: dict[str, str]) -> Corpus:
        """The `Corpus` the systems search: one DESC text per video, nothing else."""
        return Corpus(
            self.doc_ids, texts={"DESC": {d: descriptions.get(d, "") for d in self.doc_ids}}
        )


# ---- descriptions and the index ------------------------------------------------------------

DESCRIPTIONS_FILE = "descriptions.jsonl"
MAIN_PROMPT = "describe_v4.txt"  # the study's describer prompt, stored in descriptions.jsonl


def descriptions_file(prompt: str | Path) -> str:
    """Where the descriptions of a prompt live in each dataset folder: descriptions.jsonl for
    the main prompt (describe_v4), descriptions_<prompt stem>.jsonl for any variant."""
    name = Path(prompt).name
    return DESCRIPTIONS_FILE if name == MAIN_PROMPT else f"descriptions_{Path(name).stem}.jsonl"


class RateLimit:
    """At most per_minute calls per minute across threads (calls are spaced evenly)."""

    def __init__(self, per_minute: float):
        self.gap = 60.0 / per_minute if per_minute > 0 else 0.0
        self._next = 0.0
        self._lock = threading.Lock()
        self.clock, self.sleep = time.monotonic, time.sleep

    def wait(self) -> None:
        with self._lock:
            now = self.clock()
            start = max(now, self._next)
            self._next = start + self.gap
        if start > now:
            self.sleep(start - now)


class RateLimitedVision:
    """Wraps the model inside a CachedVision so only paid calls wait, and adds up what they
    cost (`cost`); keeps the inner model's sampling params, so cache keys are unchanged."""

    def __init__(self, inner: VisionLLM, limit: RateLimit):
        self.inner, self.limit = inner, limit
        self.params = getattr(inner, "params", None)
        self.cost = Cost()
        self._lock = threading.Lock()

    def describe(self, prompt: str, images: list[bytes], **kwargs) -> tuple[str, Cost]:
        self.limit.wait()
        text, cost = self.inner.describe(prompt, images, **kwargs)  # e.g. Gemini frame labels
        with self._lock:
            self.cost += cost
        return text, cost


def describe_corpus(
    corpus: McCorpus,
    vlm: VisionLLM,
    prompt_path: str | Path,
    spec: str,
    n_frames: int = 8,
    workers: int = 8,
    video_ids: list[str] | None = None,
    filename: str = DESCRIPTIONS_FILE,
    frame_policy: FramePolicy | None = None,
    labels: bool = False,
) -> dict[str, dict]:
    """Describe each video from its saved frames and write <dataset folder>/<filename>
    (descriptions.jsonl by default; keyed by the dataset's own video id, merged with what is
    there; one description per video, so a prompt variant gets its own file, see
    `descriptions_file`). frame_policy (a `FramePolicy`, e.g. 0.5 fps) replaces n_frames;
    labels puts "Frame i of n (t = ... s):" before each frame (Gemini; a new cache key).

    The vision model is expected to be cached (results/_cache/vlm.sqlite), so a rerun only pays
    for videos it has not seen. Returns doc id -> record (text, model, version, cost, seconds)."""
    prompt_text = Path(prompt_path).read_text()
    ids = video_ids if video_ids is not None else corpus.doc_ids

    def one(doc_id: str) -> dict:
        start = time.perf_counter()
        if frame_policy is None and not labels:  # the original call (fixed n, no labels)
            frames, frame_labels_ = corpus.frames(doc_id, n_frames), None
        else:
            policy = frame_policy or FramePolicy(n_frames)
            frames = corpus.frames(doc_id, policy=policy)
            times = corpus.frame_times(doc_id, policy=policy) if labels else []
            frame_labels_ = frame_labels(times) if labels else None
        text, cost = describe_frames(vlm, frames, prompt_path, frame_labels_)
        extra = {}
        if frame_policy is not None:
            extra["frame_policy"] = frame_policy.settings()
        if labels:
            extra["frame_labels"] = True
        return {
            "video_id": corpus.locate(doc_id)[1],
            "description": text,
            "model": spec,
            "model_version": cost.model_version,
            "prompt": Path(prompt_path).name,
            "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest()[:12],
            "n_frames": n_frames if frame_policy is None else len(frames),
            **extra,
            "cost": cost.to_dict(),
            "wall_seconds": round(time.perf_counter() - start, 3),
        }

    with ThreadPoolExecutor(workers) as pool:
        records = dict(zip(ids, pool.map(one, ids), strict=True))
    by_folder: dict[Path, dict[str, dict]] = {}
    for doc_id, record in records.items():
        by_folder.setdefault(corpus.locate(doc_id)[0], {})[record["video_id"]] = record
    for folder, new in by_folder.items():
        out = folder / filename
        old = {r["video_id"]: r for r in read_jsonl(out)} if out.exists() else {}
        merged = {**old, **new}
        out.write_text("".join(json.dumps(merged[d]) + "\n" for d in sorted(merged)))
    return records


def load_descriptions(data_dir: str | Path) -> dict[str, str]:
    return {
        r["video_id"]: r["description"] for r in read_jsonl(Path(data_dir) / "descriptions.jsonl")
    }


def corpus_descriptions(
    corpus: McCorpus, prompt: str | None = None, filename: str = DESCRIPTIONS_FILE
) -> dict[str, str]:
    """Descriptions of the corpus videos keyed by doc id (global ids for a pooled corpus), read
    from each dataset folder's <filename> (descriptions.jsonl by default); prompt (a file name
    such as 'describe_v4.txt') keeps only descriptions written with that prompt. Missing ones
    are left out."""
    tables: dict[Path, dict[str, dict]] = {}
    out = {}
    for doc_id in corpus.doc_ids:
        folder, video_id = corpus.locate(doc_id)
        if folder not in tables:
            path = folder / filename
            tables[folder] = {r["video_id"]: r for r in read_jsonl(path)} if path.exists() else {}
        record = tables[folder].get(video_id)
        if record and (prompt is None or record.get("prompt") == prompt):
            out[doc_id] = record["description"]
    return out


def description_cost(data_dir: str | Path) -> Cost:
    """Offline cost of the descriptions as recorded in descriptions.jsonl (cached calls included)."""
    records = read_jsonl(Path(data_dir) / "descriptions.jsonl")
    return sum((Cost(**r["cost"]) for r in records), Cost())


def index_settings(
    doc_ids: list[str],
    embedder: str,
    describer: str,
    prompt_path: str | Path,
    n_frames: int,
    frame_policy: FramePolicy | None = None,
    frame_labels: bool = False,
) -> dict:
    """What determines a description index; its hash names results/_indexes/<hash>/. The frame
    policy and labels enter only when set, so fixed-n indexes keep their hashes."""
    prompt_sha = hashlib.sha256(Path(prompt_path).read_bytes()).hexdigest()[:12]
    ids_sha = hashlib.sha256(json.dumps(doc_ids).encode()).hexdigest()[:12]
    return {
        "kind": "mc-desc",
        "embedder": embedder,
        "describer": describer,
        "prompt": Path(prompt_path).name,
        "prompt_sha256": prompt_sha,
        "n_frames": n_frames,
        "n_docs": len(doc_ids),
        "doc_ids_sha256": ids_sha,
        **({"frame_policy": frame_policy.settings()} if frame_policy is not None else {}),
        **({"frame_labels": True} if frame_labels else {}),
    }


def index_options(cfg: dict) -> dict:
    """The frame options of a smoke config for describe_corpus / index_settings:
    {"frame_policy": FramePolicy or None, "labels": bool, "filename": descriptions file}.
    A config without frame_policy, frame_labels or descriptions_file gives the 16-frame run's
    defaults (None, False, descriptions.jsonl)."""
    raw = cfg.get("frame_policy")
    return {
        "frame_policy": FramePolicy.from_config(raw, cfg.get("n_frames", 8)) if raw else None,
        "labels": bool(cfg.get("frame_labels", False)),
        "filename": cfg.get("descriptions_file", DESCRIPTIONS_FILE),
    }


def index_store(results_dir: str | Path, settings: dict) -> Path:
    blob = json.dumps(settings, sort_keys=True)
    return Path(results_dir) / "_indexes" / hashlib.sha256(blob.encode()).hexdigest()[:8]


def description_index(
    corpus: McCorpus, descriptions: dict[str, str], embedder: Embedder, store: Path
) -> DenseTextIndex:
    """The one DESC index: the generated description of every corpus video, embedded once."""
    return DenseTextIndex.build(
        "DESC", corpus.to_corpus(descriptions).texts["DESC"], embedder, store
    )


# ---- the stratified smoke sample -----------------------------------------------------------

ANY_BIN = "any"


def stratified_sample(
    videos: dict[str, list[Video]],
    eligible: dict[str, set[str]],
    strata: dict[str, dict[str, int]],
    sources_per_group: int,
    seed: int = 0,
) -> dict:
    """Draw the smoke sample: strata[dataset][bin] videos per (dataset, length bin) ('any' for
    every bin), only from eligible ids, at most one unit per source video (`source_key`), then
    sources_per_group query sources per dataset spread evenly over its strata (remainder to the
    earlier strata). Deterministic for a seed. Returns {"seed", "videos", "query_sources",
    "strata"} with global ids."""
    rng = random.Random(seed)
    used_sources: set[str] = set()
    chosen: dict[tuple[str, str], list[Video]] = {}
    for ds, bins in strata.items():
        pool = sorted(
            (v for v in videos[ds] if v.video_id in eligible.get(ds, set())),
            key=lambda v: v.video_id,
        )
        for b, n in bins.items():
            cell = [v for v in pool if b == ANY_BIN or v.length_bin == b]
            rng.shuffle(cell)
            picked = []
            for v in cell:
                if len(picked) == n:
                    break
                if source_key(v) not in used_sources:
                    used_sources.add(source_key(v))
                    picked.append(v)
            if len(picked) < n:
                raise ValueError(f"{ds}/{b}: only {len(picked)} eligible videos, need {n}")
            chosen[(ds, b)] = picked
    sources: list[str] = []
    for ds, bins in strata.items():
        cells = list(bins)
        base, extra = divmod(sources_per_group, len(cells))
        for i, b in enumerate(cells):
            n = base + (1 if i < extra else 0)
            if n > len(chosen[(ds, b)]):
                raise ValueError(
                    f"{ds}/{b}: {n} query sources needed, {len(chosen[(ds, b)])} drawn"
                )
            sources += [global_id(ds, v.video_id) for v in rng.sample(chosen[(ds, b)], n)]
    all_ids = [global_id(ds, v.video_id) for (ds, _), vs in chosen.items() for v in vs]
    source_set = set(sources)
    counts: dict[str, dict] = {}
    for (ds, b), vs in chosen.items():
        bins_found: dict[str, int] = {}
        for v in vs:
            bins_found[v.length_bin] = bins_found.get(v.length_bin, 0) + 1
        counts[f"{ds}/{b}"] = {
            "videos": len(vs),
            "query_sources": sum(global_id(ds, v.video_id) in source_set for v in vs),
            "length_bins": dict(sorted(bins_found.items())),
        }
    return {"seed": seed, "videos": all_ids, "query_sources": sources, "strata": counts}


# ---- captions: query mining, sampling and labels only ----------------------------------------


@dataclass(frozen=True)
class Caption:
    video_id: str
    caption: str
    caption_origin: str = "human"
    parts: tuple[dict, ...] = ()
    integrity: bool | None = None  # Vript's flag; None where the dataset has none


def load_captions(data_dir: str | Path) -> dict[str, Caption]:
    """captions.jsonl. For constraint mining, sampling, hard negatives and labels; never for
    indexing or for anything a system sees."""
    return {
        r["video_id"]: Caption(
            r["video_id"],
            r["caption"],
            r.get("caption_origin", "human"),
            tuple(r.get("parts") or ()),
            r.get("integrity"),
        )
        for r in read_jsonl(Path(data_dir) / "captions.jsonl")
    }


def caption_leaks(texts: list[str], captions: dict[str, Caption], min_chars: int = 30) -> list[str]:
    """Ids of videos whose caption, or a caption part of at least min_chars characters, appears
    verbatim (case-insensitive) in any of texts. An audit for index inputs and system prompts."""
    haystack = "\n".join(texts).lower()
    leaked = []
    for video_id, cap in captions.items():
        pieces = [cap.caption, *(p.get("text", "") for p in cap.parts)]
        if any(len(p) >= min_chars and p.lower() in haystack for p in pieces):
            leaked.append(video_id)
    return leaked
