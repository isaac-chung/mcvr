"""Fetch the full CaReBench test set (1,000 videos) into data/mc/carebench/.

Source: HF dataset `myang333/CaReBench` (8 parquet shards, ~3.93 GB), a re-upload of
`MCG-NJU/CaReBench` (MIT). Each row holds the mp4 bytes plus the human captions.

Steps: download shards (resumable, skips complete files) -> extract each mp4 to
videos/<video_id>.mp4 (bytes as-is, no re-encode) -> ffprobe -> write videos.jsonl,
captions.jsonl, README.md. Shards are deleted only if every video extracted and probed OK.

Run:  uv run python -I download/carebench.py [--keep-shards]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import statistics
import subprocess
import sys
from pathlib import Path

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
HF_ID = "myang333/CaReBench"
ORIG_ID = "MCG-NJU/CaReBench"
N_SHARDS = 8
SHARD_NAMES = [f"data/test-{i:05d}-of-{N_SHARDS:05d}.parquet" for i in range(N_SHARDS)]
BASE_URL = f"https://huggingface.co/datasets/{HF_ID}/resolve/main/"
OUT = REPO_ROOT / "data" / "mc" / "carebench"
REL_OUT = "data/mc/carebench"
SHARD_DIR = OUT / "_shards"
VID_DIR = OUT / "videos"
EXPECTED_ROWS = 1000


def remote_size(name: str) -> int | None:
    import urllib.error
    import urllib.request

    url = f"https://huggingface.co/api/datasets/{HF_ID}/tree/main/data"
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            for e in json.load(r):
                if e.get("path") == name:
                    return int(e.get("size") or 0) or None
    except urllib.error.URLError:
        return None
    return None


def download(name: str) -> Path:
    dest = SHARD_DIR / Path(name).name
    size = remote_size(name)
    if dest.exists() and (size is None or dest.stat().st_size == size):
        print(f"[skip] {dest.name} exists ({dest.stat().st_size:,} B)", flush=True)
        return dest
    part = dest.with_suffix(dest.suffix + ".part")
    print(f"[get ] {name} ({size or '?'} B)", flush=True)
    for attempt in range(5):
        rc = subprocess.call(["curl", "-sSL", "--fail", "-C", "-", "--retry", "5",
                              "-o", str(part), BASE_URL + name])
        if rc == 0 and (size is None or part.stat().st_size == size):
            part.rename(dest)
            return dest
        if rc == 33 or (part.exists() and size and part.stat().st_size > size):
            part.unlink(missing_ok=True)  # bad resume; restart
        print(f"  retry {attempt + 1} (curl rc={rc})", flush=True)
    raise RuntimeError(f"failed to download {name}")


def ffprobe(path: Path) -> dict:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
        capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(out.stderr.strip()[:300])
    info = json.loads(out.stdout)
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    if v is None:
        raise RuntimeError("no video stream")
    dur = info.get("format", {}).get("duration") or v.get("duration")
    return {"duration_s": round(float(dur), 3), "width": int(v["width"]), "height": int(v["height"]),
            "has_audio": any(s.get("codec_type") == "audio" for s in streams)}


def length_bin(d: float) -> str:
    if d < 30:
        return "lt30s"
    if d <= 60:
        return "30to60s"
    if d <= 120:
        return "1to2min"
    return "gt2min"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep-shards", action="store_true")
    args = ap.parse_args()
    SHARD_DIR.mkdir(parents=True, exist_ok=True)
    VID_DIR.mkdir(parents=True, exist_ok=True)

    shards = [download(n) for n in SHARD_NAMES]

    videos, captions, errors = [], [], []
    seen: set[str] = set()
    for sp in shards:
        pf = pq.ParquetFile(sp)
        for rg in range(pf.num_row_groups):
            for row in pf.read_row_group(rg).to_pylist():
                vid_name = Path(row["video"]["path"]).name
                vid = Path(vid_name).stem
                if vid in seen:
                    errors.append(f"{vid}: duplicate")
                    continue
                seen.add(vid)
                dst = VID_DIR / f"{vid}.mp4"
                data = row["video"]["bytes"]
                if not dst.exists() or dst.stat().st_size != len(data):
                    tmp = dst.with_suffix(".mp4.tmp")
                    tmp.write_bytes(data)
                    tmp.rename(dst)
                try:
                    p = ffprobe(dst)
                except Exception as e:
                    errors.append(f"{vid}: {e}")
                    continue
                videos.append({"video_id": vid, "path": f"{REL_OUT}/videos/{vid}.mp4", **p,
                               "source": "fineaction", "split": "test",
                               "length_bin": length_bin(p["duration_s"]),
                               "category": row.get("category"), "subcategory": row.get("subcategory")})
                captions.append({"video_id": vid, "caption": (row.get("caption") or "").strip(),
                                 "caption_origin": "human",
                                 "parts": [
                                     {"text": (row.get("spatial_caption") or "").strip(), "start_s": None,
                                      "end_s": None, "annotator": "spatial"},
                                     {"text": (row.get("temporal_caption") or "").strip(), "start_s": None,
                                      "end_s": None, "annotator": "temporal"}]})
            print(f"[done] {sp.name} rg{rg}: {len(videos)} ok, {len(errors)} errors", flush=True)

    videos.sort(key=lambda r: r["video_id"])
    captions.sort(key=lambda r: r["video_id"])
    with open(OUT / "videos.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in videos)
    with open(OUT / "captions.jsonl", "w") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in captions)

    durs = [r["duration_s"] for r in videos]
    bins = {b: sum(r["length_bin"] == b for r in videos) for b in ("lt30s", "30to60s", "1to2min", "gt2min")}
    total_bytes = sum((VID_DIR / f"{r['video_id']}.mp4").stat().st_size for r in videos)
    n_audio = sum(r["has_audio"] for r in videos)
    empty_caps = sum(not c["caption"] for c in captions)
    stats = {"n_videos": len(videos), "n_errors": len(errors), "total_bytes": total_bytes,
             "n_has_audio": n_audio, "n_empty_captions": empty_caps,
             "duration_min": min(durs) if durs else None, "duration_median": statistics.median(durs) if durs else None,
             "duration_mean": round(statistics.mean(durs), 2) if durs else None,
             "duration_max": max(durs) if durs else None, "duration_total_h": round(sum(durs) / 3600, 2),
             "length_bins": bins}
    print(json.dumps(stats, indent=2))
    for e in errors[:20]:
        print("ERROR", e)

    readme = f"""# CaReBench (full test set)

- Source: HF dataset `{HF_ID}` (third-party re-upload, ungated, no licence field), files
  `{SHARD_NAMES[0]}` .. `{SHARD_NAMES[-1]}` ({N_SHARDS} parquet shards, about 3.93 GB).
  Original release: `{ORIG_ID}` (HF card licence MIT; `videos/videos.zip` + `json/metadata.json`).
- Downloaded: {dt.date.today().isoformat()} by `download/carebench.py`.
- Counts: {len(videos)} videos extracted and probed ({len(errors)} errors; expected {EXPECTED_ROWS}).
  {n_audio} with an audio stream. Total size of videos/: {total_bytes / 1e9:.3f} GB.
- Duration (ffprobe): min {stats['duration_min']} s, median {stats['duration_median']} s,
  mean {stats['duration_mean']} s, max {stats['duration_max']} s, total {stats['duration_total_h']} h.
- Length bins: {json.dumps(bins)}.

## Files
- `videos/<video_id>.mp4`: mp4 bytes exactly as stored in the parquet (no re-encode).
  `video_id` = stem of the dataset's own file name (e.g. `v_00008320_0`).
- `videos.jsonl`: per-video metadata per the data contract, plus extra fields `category`, `subcategory`.
- `captions.jsonl`: `caption` = the full human caption; `parts` = the spatial and temporal captions
  (annotator labels "spatial"/"temporal", no timestamps). `caption_origin` = "human": the
  `{ORIG_ID}` card describes "detailed human-annotated captions, including manually separated
  spatial and temporal descriptions".

## Caveats
- `source` = "fineaction" is inferred (FineAction-style `v_XXXXXXXX` ids; the source was given in the
  project brief) and is not stated on either HF card. Underlying video terms are FineAction's.
- `{HF_ID}` is a mirror; an earlier spot check found captions identical to `{ORIG_ID}` metadata.json
  for one shard (125/125) and all 1,000 names present in it.
- Resolutions and aspect ratios vary (portrait clips present). No transcript is shipped.
- Separate from the pilot folder `data/carebench/`, which is untouched.
"""
    (OUT / "README.md").write_text(readme)

    ok = not errors and len(videos) == EXPECTED_ROWS
    if ok and not args.keep_shards:
        for sp in shards:
            sp.unlink()
        try:
            SHARD_DIR.rmdir()
        except OSError:
            pass
        print("shards deleted")
    elif not ok:
        print("NOT deleting shards: errors or count mismatch")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
