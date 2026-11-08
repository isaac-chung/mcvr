"""Download and prepare the Shot2Story20K test set for the multi-constraint study.

Sources (only these two HF repos are used):
  - videos:      dataset `mteb/Shot2Story20K_test`, data/test-000NN-of-00015.parquet (15 shards, ~7.3 GB)
  - annotations: dataset `mhan/shot2story`, 20k_test.json (CC BY-NC-SA 4.0)

Outputs (see the shared data contract):
  data/mc/shot2story/videos/<video_id>.mp4   bytes copied as is from the parquet `video.bytes`
  data/mc/shot2story/videos.jsonl            one row per video, no caption text
  data/mc/shot2story/captions.jsonl          visual shot captions joined in shot order (+ parts)
  data/mc/shot2story/README.md
  data/mc/shot2story/_raw/                   parquet shards (deleted only if every video extracted
                                             and probed OK) and annotations/20k_test.json (kept)

Resumable: downloads skip files already present; extraction skips mp4s already on disk with the
right size. Run from the repo root:  uv run python -I download/shot2story.py
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import statistics
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download, list_repo_files

VIDEO_REPO = "mteb/Shot2Story20K_test"
ANN_REPO = "mhan/shot2story"
ANN_FILE = "20k_test.json"
DATASET = "shot2story"
SOURCE = "hd-vila"  # YouTube clips from HD-VILA-100M
SPLIT = "test"
# Shot2Story paper (Han et al., arXiv 2312.10300): single-shot visual captions were drafted by
# MiniGPT-4 and then corrected by annotators ("over 80% ... manually corrected"); narration
# captions were written from scratch by humans; summaries were drafted by GPT-4 and corrected
# ("over 40% ... manually corrected").
CAPTION_ORIGIN = "model+human-corrected"


def length_bin(d: float) -> str:
    if d < 30:
        return "lt30s"
    if d <= 60:
        return "30to60s"
    if d <= 120:
        return "1to2min"
    return "gt2min"


def download(raw: Path) -> tuple[list[Path], Path]:
    shards = sorted(
        f for f in list_repo_files(VIDEO_REPO, repo_type="dataset")
        if f.startswith("data/") and f.endswith(".parquet")
    )
    paths = []
    for f in shards:
        p = raw / f
        if not p.exists():
            print(f"downloading {f}", flush=True)
            hf_hub_download(VIDEO_REPO, f, repo_type="dataset", local_dir=raw)
        paths.append(p)
    ann_dir = raw / "annotations"
    ann = ann_dir / ANN_FILE
    if not ann.exists():
        print(f"downloading {ANN_REPO}/{ANN_FILE}", flush=True)
        hf_hub_download(ANN_REPO, ANN_FILE, repo_type="dataset", local_dir=ann_dir)
    return paths, ann


def extract(shards: list[Path], vdir: Path) -> tuple[dict[str, str], list[str]]:
    """Write each mp4 as is. Returns {video_id: shard name} and a list of extraction errors."""
    vdir.mkdir(parents=True, exist_ok=True)
    origin: dict[str, str] = {}
    errors: list[str] = []
    for sp in shards:
        pf = pq.ParquetFile(sp)
        n_written = 0
        for rg in range(pf.num_row_groups):
            tbl = pf.read_row_group(rg, columns=["video"])
            for v in tbl.column("video").to_pylist():
                name, data = v.get("path"), v.get("bytes")
                if not name or not data:
                    errors.append(f"{sp.name}: empty video entry ({name})")
                    continue
                vid = Path(name).stem
                if vid in origin:
                    errors.append(f"{sp.name}: duplicate {vid}")
                    continue
                origin[vid] = sp.name
                out = vdir / f"{vid}.mp4"
                if out.exists() and out.stat().st_size == len(data):
                    continue
                tmp = out.with_suffix(".mp4.part")
                tmp.write_bytes(data)
                tmp.replace(out)
                n_written += 1
        print(f"{sp.name}: {n_written} written, total {len(origin)}", flush=True)
    return origin, errors


def probe(path: Path) -> dict | None:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams",
           str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        info = json.loads(r.stdout)
        streams = info.get("streams", [])
        vs = [s for s in streams if s.get("codec_type") == "video"]
        if r.returncode != 0 or not vs:
            return None
        dur = float(info["format"]["duration"])
        num, den = (vs[0].get("avg_frame_rate") or "0/1").split("/")
        fps = float(num) / float(den) if float(den) else None
        return {
            "duration_s": round(dur, 3),
            "width": int(vs[0]["width"]),
            "height": int(vs[0]["height"]),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "fps": fps,
        }
    except Exception:
        return None


def shot_bounds(shot_name: str, fps: float | None) -> tuple[float | None, float | None]:
    """'<vid>_<f0>_<f1>.mp4' -> (f0/fps, (f1+1)/fps) seconds, or (None, None)."""
    try:
        f0, f1 = Path(shot_name).stem.rsplit("_", 2)[1:]
        if not fps:
            return None, None
        return round(int(f0) / fps, 3), round((int(f1) + 1) / fps, 3)
    except ValueError:
        return None, None


def clean(t) -> str:
    return " ".join(str(t or "").split())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="repo root")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    root = Path(args.root).resolve()
    out = root / "data" / "mc" / DATASET
    raw, vdir = out / "_raw", out / "videos"
    raw.mkdir(parents=True, exist_ok=True)

    shards, ann_path = download(raw)
    origin, errors = extract(shards, vdir)
    ids = sorted(origin)

    with ThreadPoolExecutor(args.workers) as ex:
        probes = dict(zip(ids, ex.map(lambda v: probe(vdir / f"{v}.mp4"), ids)))
    failed = [v for v in ids if probes[v] is None]

    ann = {Path(e["video"]).stem: e for e in json.loads(ann_path.read_text())}
    not_hosted = sorted(set(ann) - set(origin))
    no_caption = sorted(set(origin) - set(ann))

    n_cap = 0
    with open(out / "videos.jsonl", "w") as fv, open(out / "captions.jsonl", "w") as fc:
        for v in ids:
            p = probes[v]
            if p is not None:
                row = {"video_id": v, "path": f"data/mc/{DATASET}/videos/{v}.mp4",
                       "duration_s": p["duration_s"], "width": p["width"], "height": p["height"],
                       "has_audio": p["has_audio"], "source": SOURCE, "split": SPLIT,
                       "length_bin": length_bin(p["duration_s"])}
                fv.write(json.dumps(row) + "\n")
            e = ann.get(v)
            if e is None:
                continue
            fps = p["fps"] if p else None
            bounds = [shot_bounds(s, fps) for s in e["video_names"]]
            vis = [clean(c) for c in e["captions"]]
            parts = []
            for kind, texts in (("visual", e["captions"]), ("narration", e["audio_captions"]),
                                ("asr", e["ASR"])):
                for (s0, s1), t in zip(bounds, texts):
                    if clean(t):
                        parts.append({"text": clean(t), "start_s": s0, "end_s": s1,
                                      "annotator": kind})
            for kind, key in (("summary", "whole_caption"), ("asr_full", "whole_ASR")):
                if clean(e.get(key)):
                    parts.append({"text": clean(e[key]), "start_s": None, "end_s": None,
                                  "annotator": kind})
            fc.write(json.dumps({"video_id": v, "caption": " ".join(t for t in vis if t),
                                 "caption_origin": CAPTION_ORIGIN, "parts": parts}) + "\n")
            n_cap += 1

    ok = [probes[v] for v in ids if probes[v] is not None]
    durs = [p["duration_s"] for p in ok]
    bins = {b: sum(length_bin(d) == b for d in durs) for b in ("lt30s", "30to60s", "1to2min",
                                                               "gt2min")}
    total_bytes = sum((vdir / f"{v}.mp4").stat().st_size for v in ids)
    stats = {
        "videos": len(ids), "probed_ok": len(ok), "failed_probes": failed,
        "extract_errors": errors, "captioned": n_cap, "no_audio": sum(not p["has_audio"] for p in ok),
        "not_hosted": not_hosted, "hosted_without_caption": no_caption,
        "total_gb": round(total_bytes / 1e9, 3),
        "duration_s": ({"min": min(durs), "median": statistics.median(durs),
                        "mean": round(statistics.mean(durs), 2), "max": max(durs),
                        "total_h": round(sum(durs) / 3600, 2)} if durs else {}),
        "length_bins": bins,
        "resolutions": _count(f"{p['width']}x{p['height']}" for p in ok),
    }
    write_readme(out, stats)

    all_ok = not failed and not errors and len(ids) == len(ok)
    if all_ok:
        for sp in shards:
            sp.unlink(missing_ok=True)
        shutil.rmtree(raw / "data", ignore_errors=True)
        shutil.rmtree(raw / ".cache", ignore_errors=True)
        print("all videos extracted and probed OK; parquet shards deleted")
    else:
        print("some extraction/probe failures; parquet shards kept in", raw)
    print(json.dumps(stats, indent=1))


def _count(it) -> dict[str, int]:
    c: dict[str, int] = {}
    for x in it:
        c[x] = c.get(x, 0) + 1
    return dict(sorted(c.items(), key=lambda kv: -kv[1]))


def write_readme(out: Path, s: dict) -> None:
    d = s["duration_s"]
    res = ", ".join(f"{k}: {v}" for k, v in list(s["resolutions"].items())[:5])
    text = f"""# Shot2Story20K test (multi-constraint study)

Prepared by `download/shot2story.py` on {dt.date.today().isoformat()}.

## Sources
- Videos: HF dataset `{VIDEO_REPO}`, `data/test-000NN-of-00015.parquet` (15 shards, ~7.27 GB).
  Each row has `video` (mp4 bytes + `path`), `audio` (16 kHz wav, not used) and `caption`
  (= `whole_caption`). mp4 bytes are written as is (no re-encode); `video_id` = file stem
  (`<youtube_id>.<clip_no>`).
- Annotations: HF dataset `{ANN_REPO}`, `{ANN_FILE}` (4,025 entries), joined on `video`
  == parquet `video.path`. A copy is kept in `_raw/annotations/{ANN_FILE}`.

## Counts
- Videos extracted: {s['videos']}; probed OK: {s['probed_ok']}; failed probes: {len(s['failed_probes'])};
  without audio stream: {s['no_audio']}.
- Captioned: {s['captioned']}. Annotated but not hosted by mteb: {len(s['not_hosted'])}
  ({', '.join(s['not_hosted'])}). Hosted without annotation: {len(s['hosted_without_caption'])}.
- Total video size: {s['total_gb']} GB; total duration {d.get('total_h')} h.
- Duration (s): min {d.get('min')}, median {d.get('median')}, mean {d.get('mean')}, max {d.get('max')}.
- Length bins: {', '.join(f'{k} {v}' for k, v in s['length_bins'].items())}.
- Resolutions (top): {res}.

## Captions (`captions.jsonl`)
- `caption` = the per-shot **visual** captions (`captions` field) joined with a space in shot
  order. Nothing from narration, ASR or the summary is in `caption`.
- `caption_origin` = `{CAPTION_ORIGIN}`: per the Shot2Story paper (arXiv 2312.10300), visual shot
  captions were drafted by MiniGPT-4 and then corrected by annotators (over 80% corrected).
- `parts` (annotator labels):
  - `visual`: per-shot visual caption (MiniGPT-4 draft, human-corrected).
  - `narration`: per-shot narration/audio caption, written by humans from scratch (empty shots skipped).
  - `asr`: per-shot ASR transcript (automatic).
  - `summary`: `whole_caption`, video summary drafted by GPT-4 from shot captions + ASR, then
    human-reviewed (over 40% corrected per the paper).
  - `asr_full`: `whole_ASR`, whole-video ASR (automatic).
  - Shot `start_s`/`end_s` come from the frame ranges in `video_names` (`<vid>_<f0>_<f1>.mp4`)
    divided by the clip's ffprobe average frame rate; they are approximate.

## Licence
- Annotations: CC BY-NC-SA 4.0 (`mhan/shot2story`). Research, non-commercial use only.
- `{VIDEO_REPO}` has no licence field; the clips are YouTube material from HD-VILA-100M and
  inherit those terms. Do not redistribute.

## Caveats
- 2 annotated test clips are not in the mteb copy (listed above).
- Clips are low-res (mostly 640x360) and short (see duration stats and length bins above).
- Shot boundaries in the mp4 are the original multi-shot clip; no shot files are written.
- `source` is `{SOURCE}` for every video.
"""
    (out / "README.md").write_text(text)


if __name__ == "__main__":
    main()
