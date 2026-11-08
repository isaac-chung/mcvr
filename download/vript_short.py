"""Download and prepare Vript's short-form part (YouTube Shorts + TikTok).

Source: HF dataset `Mutonix/Vript` (ungated). Licence: academic use only, no
redistribution. Do not back up or re-host data/mc/vript_short/videos/.

Downloads (resumable, skips existing):
  vript_short_videos/short_video_{1..42}_of_42.zip   (untrimmed videos, ~9.4 GB)
  vript_captions/vript_short_videos_captions.jsonl   (scene-level captions)

Writes data/mc/vript_short/{videos/<id>.mp4, videos.jsonl, captions.jsonl, README.md}
per docs/data_contract (see data/mc/<dataset>/ layout). Zips are deleted only if
every zip extracted and every mp4 probed OK.

Run from the repo root:  uv run python -I download/vript_short.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import statistics
import subprocess
import sys
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ID = "Mutonix/Vript"
N_ZIPS = 42
ZIP_TMPL = "vript_short_videos/short_video_{i}_of_42.zip"
CAPTIONS_FILE = "vript_captions/vript_short_videos_captions.jsonl"
DATASET = "vript_short"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mc" / DATASET
DL = OUT / "_downloads"
VID_DIR = OUT / "videos"
TIKTOK_RE = re.compile(r"^\d{15,}$")
SCENE_RE = re.compile(r"-Scene-(\d+)$")


def length_bin(d: float) -> str:
    if d < 30:
        return "lt30s"
    if d <= 60:
        return "30to60s"
    if d <= 120:
        return "1to2min"
    return "gt2min"


def source_of(video_id: str) -> str:
    return "tiktok" if TIKTOK_RE.match(video_id) else "youtube"


def download(filename: str, workers_note: str = "") -> Path:
    from huggingface_hub import hf_hub_download

    dest = DL / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    p = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset", filename=filename,
        local_dir=str(DL), token=False,
    )
    print(f"downloaded {filename}{workers_note}", flush=True)
    return Path(p)


def extract(zip_path: Path) -> tuple[list[str], list[str]]:
    """Extract *.mp4 members (by basename only) to VID_DIR. Returns (ids, errors)."""
    ids, errs = [], []
    try:
        with zipfile.ZipFile(zip_path) as zf:
            for info in zf.infolist():
                name = os.path.basename(info.filename)
                if info.is_dir() or not name.endswith(".mp4") or name.startswith("."):
                    continue
                vid = name[:-4]
                if not re.fullmatch(r"[A-Za-z0-9_-]+", vid):
                    errs.append(f"{zip_path.name}: skipped odd name {info.filename!r}")
                    continue
                dest = VID_DIR / name
                if not (dest.exists() and dest.stat().st_size == info.file_size):
                    tmp = dest.with_suffix(".mp4.part")
                    with zf.open(info) as src, open(tmp, "wb") as dst:
                        shutil.copyfileobj(src, dst, 1 << 20)
                    tmp.replace(dest)
                ids.append(vid)
    except Exception as e:
        errs.append(f"{zip_path.name}: {e!r}")
    return ids, errs


def probe(vid: str) -> dict:
    path = VID_DIR / f"{vid}.mp4"
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        info = json.loads(r.stdout or "{}")
        streams = info.get("streams", [])
        v = next((s for s in streams if s.get("codec_type") == "video"), None)
        dur = float(info.get("format", {}).get("duration") or (v or {}).get("duration") or 0)
        if r.returncode != 0 or v is None or dur <= 0:
            return {"video_id": vid, "error": r.stderr.strip()[:300] or "no video stream/duration"}
        w, h = int(v.get("width", 0)), int(v.get("height", 0))
        rot = 0
        for sd in v.get("side_data_list", []) or []:
            if "rotation" in sd:
                rot = int(sd["rotation"])
        rot = rot or int((v.get("tags") or {}).get("rotate", 0) or 0)
        if abs(rot) % 180 == 90:
            w, h = h, w
        return {
            "video_id": vid,
            "path": f"data/mc/{DATASET}/videos/{vid}.mp4",
            "duration_s": round(dur, 3),
            "width": w,
            "height": h,
            "has_audio": any(s.get("codec_type") == "audio" for s in streams),
            "source": source_of(vid),
            "split": "all",
            "length_bin": length_bin(dur),
            "size_bytes": path.stat().st_size,
        }
    except Exception as e:
        return {"video_id": vid, "error": repr(e)}


def load_captions(path: Path) -> dict[str, dict]:
    """Group scene-level lines by video id, in scene order."""
    by_vid: dict[str, dict] = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        meta = d.get("meta", {})
        vid = meta["video_id"]
        m = SCENE_RE.search(d.get("clip_id", ""))
        scene = int(m.group(1)) if m else 10**6
        e = by_vid.setdefault(vid, {"num_clips": meta.get("num_clips"), "integrity": meta.get("integrity"),
                                    "title": meta.get("video_title"), "scenes": {}})
        e["scenes"].setdefault(d["clip_id"], (scene, d))
    return by_vid


def caption_record(vid: str, e: dict) -> dict:
    parts, texts = [], []
    for scene, d in sorted(e["scenes"].values(), key=lambda t: (t[0], t[1]["clip_id"])):
        cap = d.get("caption") or {}
        content = (cap.get("content") or "").strip()
        if content:
            texts.append(content)
            parts.append({"text": content, "start_s": None, "end_s": None, "annotator": "GPT-4V",
                          "label": "content", "clip_id": d["clip_id"], "scene": scene,
                          "scene_title": cap.get("scene_title"), "shot_type": cap.get("shot_type"),
                          "camera_movement": cap.get("camera_movement")})
        vo = (d.get("voiceover") or "").strip()
        if vo:
            parts.append({"text": vo, "start_s": None, "end_s": None, "annotator": "ASR",
                          "label": "voiceover", "clip_id": d["clip_id"], "scene": scene})
    n_cap = len(e["scenes"])
    return {"video_id": vid, "caption": " ".join(texts), "caption_origin": "model", "parts": parts,
            "n_scenes_total": e["num_clips"], "n_scenes_captioned": n_cap,
            "integrity": e["integrity"]}


def q(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--keep-zips", action="store_true")
    args = ap.parse_args()
    if shutil.which("ffprobe") is None:
        sys.exit("ffprobe not found")
    VID_DIR.mkdir(parents=True, exist_ok=True)
    DL.mkdir(parents=True, exist_ok=True)

    cap_path = download(CAPTIONS_FILE)
    zips = [ZIP_TMPL.format(i=i) for i in range(1, N_ZIPS + 1)]
    with ThreadPoolExecutor(args.workers) as ex:
        zip_paths = list(ex.map(download, zips))

    all_ids, errors, zip_of = [], [], {}
    for zp in zip_paths:
        ids, errs = extract(zp)
        errors += errs
        for v in ids:
            if v in zip_of:
                errors.append(f"duplicate video id {v} in {zp.name} and {zip_of[v]}")
            zip_of[v] = zp.name
        all_ids += ids
        print(f"extracted {zp.name}: {len(ids)} mp4", flush=True)
    ids = sorted(set(all_ids))

    with ThreadPoolExecutor(8) as ex:
        probes = list(ex.map(probe, ids))
    bad = [p for p in probes if "error" in p]
    errors += [f"probe {p['video_id']}: {p['error']}" for p in bad]
    good = [p for p in probes if "error" not in p]

    with open(OUT / "videos.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(p, ensure_ascii=False) + "\n" for p in good)

    caps = load_captions(cap_path)
    hosted = {p["video_id"] for p in good}
    cap_recs = [caption_record(v, caps[v]) for v in sorted(caps) if v in hosted]
    with open(OUT / "captions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in cap_recs)

    # ---- stats
    durs = [p["duration_s"] for p in good]
    total_bytes = sum(p["size_bytes"] for p in good)
    bins = {b: sum(p["length_bin"] == b for p in good) for b in ["lt30s", "30to60s", "1to2min", "gt2min"]}
    src = {s: sum(p["source"] == s for p in good) for s in ["youtube", "tiktok"]}
    cap_src = {s: sum(source_of(r["video_id"]) == s for r in cap_recs) for s in ["youtube", "tiktok"]}
    shares = [r["n_scenes_captioned"] / r["n_scenes_total"] for r in cap_recs if r["n_scenes_total"]]
    cap_bins = defaultdict(int)
    dur_of = {p["video_id"]: p["duration_s"] for p in good}
    for r in cap_recs:
        cap_bins[length_bin(dur_of[r["video_id"]])] += 1
    stats = {
        "videos_hosted": len(good),
        "probe_failures": len(bad),
        "videos_with_caption": len(cap_recs),
        "captioned_videos_not_in_zips": len(set(caps) - hosted),
        "captioned_with_empty_caption_text": sum(not r["caption"] for r in cap_recs),
        "scene_caption_share": {
            "mean": round(statistics.mean(shares), 3) if shares else None,
            "median": round(statistics.median(shares), 3) if shares else None,
            "share_videos_fully_captioned": round(sum(s >= 1 for s in shares) / len(shares), 3) if shares else None,
            "min": round(min(shares), 3) if shares else None,
        },
        "integrity_false_videos": sum(r["integrity"] is False for r in cap_recs),
        "videos_with_voiceover": sum(any(p["label"] == "voiceover" for p in r["parts"]) for r in cap_recs),
        "source_hosted": src,
        "source_captioned": cap_src,
        "total_bytes": total_bytes,
        "total_gb": round(total_bytes / 1e9, 2),
        "duration_s": {
            "min": round(min(durs), 2), "p25": round(q(durs, .25), 2), "median": round(statistics.median(durs), 2),
            "mean": round(statistics.mean(durs), 2), "p75": round(q(durs, .75), 2), "max": round(max(durs), 2),
            "total_h": round(sum(durs) / 3600, 2),
        } if durs else {},
        "length_bins_hosted": bins,
        "length_bins_captioned": {b: cap_bins.get(b, 0) for b in bins},
        "with_audio": sum(p["has_audio"] for p in good),
        "top_resolutions": sorted(
            ((f"{w}x{h}", n) for (w, h), n in
             __import__("collections").Counter((p["width"], p["height"]) for p in good).items()),
            key=lambda t: -t[1])[:5],
        "errors": errors[:50],
        "n_errors": len(errors),
    }
    (OUT / "_stats.json").write_text(json.dumps(stats, indent=2))
    write_readme(stats)
    print(json.dumps({k: v for k, v in stats.items() if k != "errors"}, indent=2))
    if errors:
        print(f"{len(errors)} errors, e.g.: {errors[:5]}", file=sys.stderr)

    if not errors and not args.keep_zips:
        shutil.rmtree(DL / "vript_short_videos", ignore_errors=True)
        print("all zips extracted and probed OK; zips deleted")
    else:
        print("zips kept")
    return 0 if not errors else 1


def write_readme(s: dict) -> None:
    d = s["duration_s"]
    txt = f"""# Vript short-form part (YouTube Shorts + TikTok)

- Source: Hugging Face dataset `{REPO_ID}` (not gated), files
  `vript_short_videos/short_video_{{1..42}}_of_42.zip` (untrimmed videos) and
  `{CAPTIONS_FILE}` (scene-level captions). Raw caption file kept at
  `_downloads/{CAPTIONS_FILE}`.
- Downloaded: {dt.date.today().isoformat()} by `download/vript_short.py`.
- Licence: custom (adapted from HD-VG-130M): ACADEMIC USE ONLY, NO DISTRIBUTION; the
  authors may revoke access and request deletion; YouTube/TikTok copyright policies apply.
  Do not back up or re-host `videos/`.

## Counts

| | value |
|---|---|
| Videos hosted (mp4, ffprobe OK) | {s['videos_hosted']} |
| ...YouTube / TikTok | {s['source_hosted']['youtube']} / {s['source_hosted']['tiktok']} |
| Videos with any caption | {s['videos_with_caption']} (YouTube {s['source_captioned']['youtube']}, TikTok {s['source_captioned']['tiktok']}) |
| Captioned ids with no video in the zips | {s['captioned_videos_not_in_zips']} |
| Share of scenes captioned per captioned video | mean {s['scene_caption_share']['mean']}, median {s['scene_caption_share']['median']}, fully captioned {s['scene_caption_share']['share_videos_fully_captioned']} |
| Captioned videos with `integrity: false` | {s['integrity_false_videos']} |
| Total size | {s['total_gb']} GB |
| Duration (s) | min {d.get('min')}, median {d.get('median')}, mean {d.get('mean')}, max {d.get('max')}; total {d.get('total_h')} h |
| Length bins, hosted | {s['length_bins_hosted']} |
| Length bins, captioned | {s['length_bins_captioned']} |
| With audio stream | {s['with_audio']} |

## Fields and caveats

- `source`: derived from the id format (TikTok ids are 15+ digit numbers, YouTube ids are 11 chars);
  the caption file has no platform field. `split` is `all` (Vript has no splits).
- `captions.jsonl` covers only hosted videos that have at least one captioned scene.
  `caption` = the `caption.content` field of each scene in scene order, joined by spaces.
  `shot_type`, `camera_movement`, `scene_title` are kept only inside `parts` (label `content`);
  ASR voiceover text is in `parts` with label `voiceover` and is NOT in `caption`.
- `caption_origin` = `model`: Vript scene captions were written by GPT-4V (paper); voiceover is ASR.
- Scene timestamps (`start_s`/`end_s`) are null: the caption file has none. Scene spans live in
  `<id>_cut_meta.json` inside the clips zips (`vript_short_videos_clips/`), not downloaded.
- Not every scene of a video is captioned (`n_scenes_captioned` vs `n_scenes_total` = meta `num_clips`),
  so a video caption may skip scenes. Filter by caption presence when building retrieval sets.
- Scene coverage: meta `num_clips` equals the number of captioned lines for every video, so it seems to count
  captioned scenes only. Scene numbers have gaps in 11 videos (8,776 captioned vs 8,787 by max scene number);
  missing trailing scenes can only be checked against `_cut_meta.json` in the clips zips.
- The `voiceover` field is empty in every line of the caption file (0 of 8,776), so no voiceover parts exist.
- Per the Vript paper, captions were generated with the video title and voiceover as context, so they can mention
  names or facts not visible on screen.
"""
    (OUT / "README.md").write_text(txt)


if __name__ == "__main__":
    raise SystemExit(main())
