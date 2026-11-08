"""Build Vript's long-video part as contiguous multi-scene spans (one span per source video).

Source: HF dataset `Mutonix/Vript` (ungated). Licence: academic use only, no
redistribution. Do not back up or re-host data/mc/vript_long/videos/ (derived span files).

Inputs (downloaded once into data/mc/vript_long/_downloads/, skipped if present):
  vript_meta/vript_long_videos_meta.json           (yt-dlp info, incl. `language`)
  vript_captions/vript_long_videos_captions.zip    (one json per video: scene captions, no timestamps)
  vript_long_videos_clips/clips_{1..1095}_of_1095.zip  -- NEVER downloaded whole: we read each zip's
      central directory and `<id>_cut_meta.json` (scene spans) by HTTP range requests, and then
      range-read only the scene clip members a unit needs.

Units: per length bin (30to60s: 30-60 s, 1to2min: 60-120 s, 2to4min: 120-240 s) 500 spans of whole,
consecutive, captioned scenes; each source video appears in at most one unit. Seeded selection:
the eligible pool (English, integrity true, not in vript_short, has scene spans) is shuffled; each
video in turn is offered to the bin with the fewest accepted candidates; for it we draw a target
length T ~ U(bin), a random start scene, and extend over consecutive scenes until span >= T. The
candidate is rejected if a scene in the span lacks a caption or a clip file, if the span overshoots
the bin's upper edge, or if the video ends before T. Accepted candidates beyond 500 per bin are
reserves that replace units whose joined file fails checks (duration off by > 1 s, outside the bin,
decode error).

Video: the span's clip members are range-read (one request per unit), inflated, CRC-checked, joined
with the ffmpeg concat demuxer in stream-copy mode (no re-encode), probed with ffprobe and fully
decoded once (ffmpeg -f null). Temporary clip files are deleted after joining.

HF limits anonymous `resolve` calls (3,000 per 5 min); every huggingface.co request goes through a
sliding-window limiter (default 2,000 / 5 min); each zip's signed CDN URL is cached and only
re-resolved after 40 min or on 403. HF_TOKEN from the environment or .env is sent to huggingface.co
only (never to the CDN, never printed).

Resumable and idempotent: per-zip index cache in _index/, finished units in _state/done.jsonl,
quality failures in _state/failed.jsonl; outputs are rewritten from those files on every run.

Run from the repo root:  uv run python -I download/vript_long.py [--workers 6]
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import json
import os
import random
import re
import shutil
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ID = "Mutonix/Vript"
BASE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/"
META_FILE = "vript_meta/vript_long_videos_meta.json"
CAPTIONS_FILE = "vript_captions/vript_long_videos_captions.zip"
N_CLIP_ZIPS = 1095
CLIP_ZIP_TMPL = "vript_long_videos_clips/clips_{i}_of_1095.zip"
DATASET = "vript_long"
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "mc" / DATASET
DL = OUT / "_downloads"
IDX = OUT / "_index"
STATE = OUT / "_state"
TMP = OUT / "_tmp"
VID_DIR = OUT / "videos"
SHORT_VIDEOS = ROOT / "data" / "mc" / "vript_short" / "videos.jsonl"

SEED = 0
N_PER_BIN = 500
N_RESERVE = 200
BINS = {"30to60s": (30.0, 60.0), "1to2min": (60.0, 120.0), "2to4min": (120.0, 240.0)}
MAX_DUR_DIFF = 1.0
ANNOTATOR = "gpt-4v-scene"
SCENE_RE = re.compile(r"-Scene-(\d+)$")
ID_RE = re.compile(r"[A-Za-z0-9_-]+")


# ----------------------------------------------------------------------------- HTTP

def _hf_token() -> str | None:
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            m = re.match(r"\s*(?:export\s+)?HF_TOKEN\s*=\s*(.*)$", line)
            if m and m.group(1).strip().strip("'\""):
                return m.group(1).strip().strip("'\"")
    return None


class HFLimiter:
    """Sliding window over huggingface.co calls."""

    def __init__(self, max_calls: int, window_s: float = 300.0):
        self.max_calls, self.window = max_calls, window_s
        self.calls: collections.deque[float] = collections.deque()
        self.lock = threading.Lock()
        self.total = 0

    def wait(self) -> None:
        while True:
            with self.lock:
                now = time.time()
                while self.calls and now - self.calls[0] > self.window:
                    self.calls.popleft()
                if len(self.calls) < self.max_calls:
                    self.calls.append(now)
                    self.total += 1
                    return
                sleep = self.window - (now - self.calls[0]) + 0.1
            time.sleep(max(0.1, sleep))


class _NoRedir(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


_OPENER = urllib.request.build_opener(_NoRedir)
LIMITER = HFLimiter(2000)
TOKEN = _hf_token()
_URLS: dict[str, tuple[str, float]] = {}
_URL_LOCK = threading.Lock()
URL_TTL = 40 * 60


def resolve(path: str, force: bool = False) -> str:
    """CDN URL for a repo file (one huggingface.co call, cached)."""
    with _URL_LOCK:
        hit = _URLS.get(path)
        if hit and not force and time.time() - hit[1] < URL_TTL:
            return hit[0]
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    for attempt in range(30):
        LIMITER.wait()
        try:
            _OPENER.open(urllib.request.Request(BASE + path, method="HEAD", headers=headers), timeout=60)
            url = BASE + path  # no redirect (unlikely)
            break
        except urllib.error.HTTPError as e:
            if e.code in (301, 302, 303, 307, 308):
                url = e.headers["Location"]
                if url.startswith("/"):
                    url = "https://huggingface.co" + url
                break
            if e.code == 429:
                time.sleep(int(e.headers.get("retry-after") or 30) + 1)
                continue
            if attempt >= 5:
                raise
            time.sleep(5)
        except Exception:
            if attempt >= 5:
                raise
            time.sleep(5)
    else:
        raise OSError(f"resolve failed: {path}")
    if "huggingface.co" in url.split("/")[2]:
        LIMITER.wait()  # following that URL is another huggingface.co call
    with _URL_LOCK:
        _URLS[path] = (url, time.time())
    return url


def range_get(path: str, start: int, end: int, dest=None) -> bytes | int:
    """GET bytes [start, end] of repo file `path`. If dest (file obj) given, stream into it."""
    for attempt in range(10):
        url = resolve(path, force=attempt > 0)
        req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                if r.status != 206:
                    raise OSError(f"no range support (HTTP {r.status})")
                if dest is None:
                    b = r.read()
                    if len(b) != end - start + 1:
                        raise OSError("short read")
                    return b
                dest.seek(0)
                dest.truncate()
                n = 0
                while True:
                    chunk = r.read(1 << 20)
                    if not chunk:
                        break
                    dest.write(chunk)
                    n += len(chunk)
                if n != end - start + 1:
                    raise OSError("short read")
                return n
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(int(e.headers.get("retry-after") or 30) + 1)
            elif attempt >= 9:
                raise
            else:
                time.sleep(3 * (attempt + 1))
        except Exception:
            if attempt >= 9:
                raise
            time.sleep(3 * (attempt + 1))
    raise OSError(f"range_get failed: {path}")


def download_whole(filename: str) -> Path:
    dest = DL / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    url = resolve(filename)
    with urllib.request.urlopen(url, timeout=300) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)
    print(f"downloaded {filename}", flush=True)
    return dest


# ----------------------------------------------------------------------------- zip index

class RemoteFile:
    """Seekable read-only view of a remote zip, for zipfile's central-directory parsing."""

    def __init__(self, path: str, size: int):
        self.path, self.size, self.pos = path, size, 0

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        # read ahead 256 KB to keep the central directory to a few requests
        if not hasattr(self, "_buf") or not (self._b0 <= self.pos and self.pos + n <= self._b0 + len(self._buf)):
            ln = max(n, 1 << 18)
            s = max(0, min(self.pos, self.size - ln))
            self._b0, self._buf = s, range_get(self.path, s, min(self.size, s + ln) - 1)
        o = self.pos - self._b0
        self.pos += n
        return self._buf[o:o + n]


def zip_sizes() -> dict[str, int]:
    """Sizes of the clip zips from the HF tree API (cached)."""
    fn = IDX / "tree_clips.json"
    if fn.exists():
        return json.loads(fn.read_text())
    LIMITER.wait()
    headers = {"Authorization": f"Bearer {TOKEN}"} if TOKEN else {}
    url = f"https://huggingface.co/api/datasets/{REPO_ID}/tree/main/vript_long_videos_clips"
    sizes: dict[str, int] = {}
    while url:
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as r:
            for x in json.loads(r.read()):
                if x.get("type") == "file":
                    sizes[x["path"]] = int(x["size"])
            link = r.headers.get("Link") or ""
            m = re.search(r'<([^>]+)>;\s*rel="next"', link)
            url = m.group(1) if m else None
        if url:
            LIMITER.wait()
    IDX.mkdir(parents=True, exist_ok=True)
    fn.write_text(json.dumps(sizes))
    return sizes


def index_zip(path: str, size: int, seed_cut_meta: dict | None) -> dict:
    """Central directory (offsets) + cut_meta for one clip zip; cached in _index/zips/."""
    fn = IDX / "zips" / (Path(path).stem + ".json")
    if fn.exists():
        return json.loads(fn.read_text())
    rec: dict = {"zip": path, "size": size, "members": {}, "cut_meta": {}, "empty": False}
    if size < 10000:
        rec["empty"] = True
    else:
        zf = zipfile.ZipFile(RemoteFile(path, size))
        for i in zf.infolist():
            if i.is_dir():
                continue
            base = os.path.basename(i.filename)
            rec["members"][base] = [i.filename, i.header_offset, i.compress_size, i.file_size, i.CRC, i.compress_type]
        for base, m in rec["members"].items():
            if not base.endswith("_cut_meta.json"):
                continue
            vid = base[: -len("_cut_meta.json")]
            if seed_cut_meta is not None and vid in seed_cut_meta:
                rec["cut_meta"][vid] = seed_cut_meta[vid]
            else:
                rec["cut_meta"][vid] = json.loads(zf.read(m[0]))
    fn.parent.mkdir(parents=True, exist_ok=True)
    tmp = fn.with_suffix(".part")
    tmp.write_text(json.dumps(rec))
    tmp.replace(fn)
    return rec


def load_seed_cut_meta(path: str | None) -> dict | None:
    if not path:
        return None
    seed = {}
    for r in json.load(open(path)):
        for _, d in (r.get("cut_meta") or {}).items():
            if isinstance(d, dict) and "clips" in d:
                seed[d["video_id"]] = d
    print(f"seed cut_meta for {len(seed)} videos from {path}", flush=True)
    return seed


# ----------------------------------------------------------------------------- selection

def ts(s: str) -> float:
    h, m, x = s.split(":")
    return int(h) * 3600 + int(m) * 60 + float(x)


def load_captions(path: Path) -> dict[str, dict]:
    caps = {}
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if info.is_dir() or not info.filename.endswith("_caption.json"):
                continue
            d = json.loads(zf.read(info))
            caps[d["meta"]["video_id"]] = d
    return caps


def vo_text(v) -> str:
    """Vript long `voiceover` is a list of ASR strings (sometimes a plain string)."""
    if isinstance(v, list):
        return " ".join(x.strip() for x in v if isinstance(x, str) and x.strip())
    return (v or "").strip() if isinstance(v, str) else ""


def detect_lang(text: str) -> str:
    import py3langid as langid  # offline, bundled model

    return langid.classify(text)[0] if text.strip() else "und"


def english(vid: str, meta: dict, caps: dict, lang_log: dict) -> bool:
    lang = (meta.get(vid) or {}).get("language")
    if lang:
        ok = str(lang).split("-")[0].lower() == "en"
        lang_log["meta"][("en" if ok else "other")] += 1
        return ok
    d = caps[vid]
    vo = " ".join(vo_text(c.get("voiceover")) for c in d["data"].values())
    text = ((meta.get(vid) or {}).get("title") or d["meta"].get("video_title") or "") + ". " + vo[:5000]
    lab = detect_lang(text)
    lang_log["langid"][lab] += 1
    return lab == "en"


def build_scenes(cm: dict, caps_v: dict, members: dict) -> list[dict]:
    sc = []
    for c in cm["clips"]:
        cid = c["clip_id"].removesuffix(".mp4")
        m = SCENE_RE.search(cid)
        sc.append({"clip_id": cid, "scene": int(m.group(1)) if m else None, "a": ts(c["span"][0]),
                   "b": ts(c["span"][1]), "cap": cid in caps_v, "file": (cid + ".mp4") in members})
    sc.sort(key=lambda s: s["a"])
    return sc


def select(meta, caps, zipidx, short_ids, rng_seed=SEED):
    """Deterministic candidate lists per bin (500 primaries + reserves)."""
    lang_log = {"meta": collections.Counter(), "langid": collections.Counter()}
    excl = collections.Counter()
    cm_of, zip_of = {}, {}
    for z in zipidx.values():
        for vid, cm in z["cut_meta"].items():
            cm_of[vid] = cm
            zip_of[vid] = z["zip"]
    pool = []
    for vid in sorted(caps):
        if caps[vid]["meta"].get("integrity") is not True:
            excl["integrity_not_true"] += 1
            continue
        if vid in short_ids:
            excl["in_vript_short"] += 1
            continue
        if vid not in cm_of:
            excl["no_scene_spans"] += 1
            continue
        if not ID_RE.fullmatch(vid):
            excl["odd_id"] += 1
            continue
        if not english(vid, meta, caps, lang_log):
            excl["not_english"] += 1
            continue
        pool.append(vid)
    rng = random.Random(rng_seed)
    rng.shuffle(pool)
    want = N_PER_BIN + N_RESERVE
    cands = {b: [] for b in BINS}
    rejects = {b: collections.Counter() for b in BINS}
    for vid in pool:
        open_bins = [b for b in BINS if len(cands[b]) < want]
        if not open_bins:
            break
        b = min(open_bins, key=lambda x: (len(cands[x]), list(BINS).index(x)))
        lo, hi = BINS[b]
        sc = build_scenes(cm_of[vid], caps[vid]["data"], zipidx[zip_of[vid]]["members"])
        target = rng.uniform(lo, hi)
        i = rng.randrange(len(sc))
        reason, j = None, i
        while True:
            s = sc[j]
            if not s["cap"]:
                reason = "scene_without_caption"
                break
            if not s["file"]:
                reason = "clip_file_missing"
                break
            if j > i and abs(s["a"] - sc[j - 1]["b"]) > 0.05:
                reason = "scene_gap"
                break
            dur = s["b"] - sc[i]["a"]
            if dur >= target:
                if dur > hi:
                    reason = "overshoot_bin"
                break
            if j == len(sc) - 1:
                reason = "video_ends_before_target"
                break
            j += 1
        if reason:
            rejects[b][reason] += 1
            continue
        span = sc[i: j + 1]
        cands[b].append({
            "unit_id": f"{vid}__sc{span[0]['scene']:03d}-{span[-1]['scene']:03d}",
            "source_video_id": vid, "length_bin": b, "target_s": round(target, 3),
            "start_s": round(span[0]["a"], 3), "end_s": round(span[-1]["b"], 3),
            "span_s": round(span[-1]["b"] - span[0]["a"], 3),
            "scene_ids": [s["clip_id"] for s in span],
            "scene_first": span[0]["scene"], "scene_last": span[-1]["scene"],
            "start_index": i, "n_scenes_video": len(sc), "clip_zip": zip_of[vid],
            "scene_spans": [[round(s["a"], 3), round(s["b"], 3)] for s in span],
        })
    info = {"pool_size": len(pool), "pool_excluded": dict(excl),
            "language": {"meta": dict(lang_log["meta"]), "langid_fallback": dict(lang_log["langid"])},
            "rejected_candidates": {b: dict(c) for b, c in rejects.items()},
            "accepted_candidates": {b: len(c) for b, c in cands.items()}}
    return cands, info


# ----------------------------------------------------------------------------- build one unit

def probe(path: Path) -> dict:
    cmd = ["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    info = json.loads(r.stdout or "{}")
    streams = info.get("streams", [])
    v = next((s for s in streams if s.get("codec_type") == "video"), None)
    a = next((s for s in streams if s.get("codec_type") == "audio"), None)
    dur = float(info.get("format", {}).get("duration") or 0)
    if r.returncode != 0 or v is None or dur <= 0:
        raise ValueError("probe: " + (r.stderr.strip()[:200] or "no video stream/duration"))
    w, h = int(v.get("width", 0)), int(v.get("height", 0))
    rot = 0
    for sd in v.get("side_data_list", []) or []:
        if "rotation" in sd:
            rot = int(sd["rotation"])
    if abs(rot) % 180 == 90:
        w, h = h, w
    return {"duration_s": round(dur, 3), "width": w, "height": h, "has_audio": a is not None,
            "v_dur": float(v.get("duration") or 0), "a_dur": float((a or {}).get("duration") or 0),
            "vcodec": v.get("codec_name"), "acodec": (a or {}).get("codec_name")}


def build_unit(c: dict, zipidx: dict) -> dict:
    """Returns {"ok": True, ...probe} or {"ok": False, "reason": ..., "transient": bool}."""
    uid = c["unit_id"]
    final = VID_DIR / f"{uid}.mp4"
    work = TMP / uid
    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    try:
        z = zipidx[c["clip_zip"]]
        mem = [z["members"][cid + ".mp4"] for cid in c["scene_ids"]]
        start = min(m[1] for m in mem)
        end = max(m[1] + 30 + len(m[0].encode()) + 1024 + m[2] for m in mem)  # +slack for extra field
        end = min(end, z["size"] - 1)
        raw = work / "range.bin"
        try:
            with open(raw, "w+b") as f:
                range_get(c["clip_zip"], start, end, dest=f)
        except Exception as e:
            return {"ok": False, "reason": f"download: {e!r}"[:300], "transient": True}
        clip_paths = []
        with open(raw, "rb") as f:
            for cid, (name, off, csize, usize, crc, ctype) in zip(c["scene_ids"], mem):
                f.seek(off - start)
                hdr = f.read(30)
                if hdr[:4] != b"PK\x03\x04":
                    return {"ok": False, "reason": f"bad local header {cid}", "transient": True}
                nlen = int.from_bytes(hdr[26:28], "little")
                xlen = int.from_bytes(hdr[28:30], "little")
                f.seek(off - start + 30 + nlen + xlen)
                data = f.read(csize)
                if len(data) != csize:
                    return {"ok": False, "reason": f"short member {cid}", "transient": True}
                if ctype == zipfile.ZIP_DEFLATED:
                    data = zlib.decompressobj(-15).decompress(data)
                elif ctype != zipfile.ZIP_STORED:
                    return {"ok": False, "reason": f"compression {ctype}", "transient": False}
                if len(data) != usize or (zlib.crc32(data) & 0xFFFFFFFF) != crc:
                    return {"ok": False, "reason": f"crc/size mismatch {cid}", "transient": True}
                p = work / f"{cid}.mp4"
                p.write_bytes(data)
                clip_paths.append(p)
        raw.unlink()
        lst = work / "list.txt"
        lst.write_text("".join(f"file '{p.name}'\n" for p in clip_paths))
        part = work / "joined.mp4"
        cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst),
               "-map", "0", "-c", "copy", "-movflags", "+faststart", str(part)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=1200)
        if r.returncode != 0 or not part.exists():
            return {"ok": False, "reason": "concat: " + r.stderr.strip()[-200:], "transient": False}
        try:
            pr = probe(part)
        except Exception as e:
            return {"ok": False, "reason": f"{e}"[:300], "transient": False}
        span = c["span_s"]
        if abs(pr["duration_s"] - span) > MAX_DUR_DIFF or (pr["v_dur"] and abs(pr["v_dur"] - span) > MAX_DUR_DIFF):
            return {"ok": False, "reason": f"duration_mismatch joined={pr['duration_s']} video={pr['v_dur']} span={span}",
                    "transient": False, "probe": pr}
        lo, hi = BINS[c["length_bin"]]
        if not (lo <= pr["duration_s"] <= hi):
            return {"ok": False, "reason": f"joined_duration_outside_bin {pr['duration_s']}", "transient": False}
        dec = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-threads", "2", "-i", str(part), "-map", "0",
                              "-f", "null", "-"], capture_output=True, text=True, timeout=1800)
        if dec.returncode != 0 or dec.stderr.strip():
            return {"ok": False, "reason": "decode: " + dec.stderr.strip()[:200], "transient": False}
        part.replace(final)
        pr["size_bytes"] = final.stat().st_size
        return {"ok": True, **pr}
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ----------------------------------------------------------------------------- outputs

def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(p * (len(xs) - 1))))]


def dstats(xs):
    if not xs:
        return {}
    return {"n": len(xs), "min": round(min(xs), 2), "p10": round(q(xs, .1), 2), "median": round(statistics.median(xs), 2),
            "mean": round(statistics.mean(xs), 2), "p90": round(q(xs, .9), 2), "max": round(max(xs), 2)}


def write_outputs(final_units, done, failed, caps, sel_info, t_run, args):
    vrecs, crecs = [], []
    for c in final_units:
        d = done[c["unit_id"]]
        vrecs.append({
            "video_id": c["unit_id"], "path": f"data/mc/{DATASET}/videos/{c['unit_id']}.mp4",
            "duration_s": d["duration_s"], "width": d["width"], "height": d["height"], "has_audio": d["has_audio"],
            "source": "youtube", "split": DATASET, "length_bin": c["length_bin"],
            "source_video_id": c["source_video_id"], "scene_first": c["scene_first"], "scene_last": c["scene_last"],
            "n_scenes": len(c["scene_ids"]), "source_start_s": c["start_s"], "source_end_s": c["end_s"],
            "size_bytes": d["size_bytes"],
        })
        data = caps[c["source_video_id"]]["data"]
        parts, texts = [], []
        for cid, (a, b) in zip(c["scene_ids"], c["scene_spans"]):
            e = data[cid]
            cap = e.get("caption") or {}
            content = (cap.get("content") or "").strip()
            texts.append(content)
            parts.append({"text": content, "start_s": round(a - c["start_s"], 3), "end_s": round(b - c["start_s"], 3),
                          "annotator": ANNOTATOR, "clip_id": cid, "scene": int(SCENE_RE.search(cid).group(1)),
                          "scene_title": cap.get("scene_title"), "shot_type": cap.get("shot_type"),
                          "camera_movement": cap.get("camera_movement"),
                          "voiceover_asr": vo_text(e.get("voiceover"))})
        crecs.append({"video_id": c["unit_id"], "caption": " ".join(t for t in texts if t), "caption_origin": "model",
                      "parts": parts, "source_video_id": c["source_video_id"]})
    with open(OUT / "videos.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in vrecs)
    with open(OUT / "captions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(r, ensure_ascii=False) + "\n" for r in crecs)

    fail_reasons = {b: collections.Counter() for b in BINS}
    for fr in failed.values():
        fail_reasons[fr["length_bin"]][fr["reason"].split(" ")[0].split(":")[0]] += 1
    selection = {
        "seed": SEED, "created": dt.date.today().isoformat(), "script": "download/vript_long.py",
        "rules": {
            "bins_s": {b: list(v) for b, v in BINS.items()}, "units_per_bin": N_PER_BIN, "reserves_per_bin": N_RESERVE,
            "filters": ["captions meta integrity is true", "English: yt-dlp `language` in vript_long_videos_meta.json "
                        "starts with 'en'; if missing, py3langid on title + ASR voiceover == 'en'",
                        "not in data/mc/vript_short/videos.jsonl", "has scene spans (<id>_cut_meta.json)"],
            "procedure": "shuffle eligible pool with random.Random(seed); each video in turn goes to the bin with the "
                         "fewest accepted candidates (ties: bin order); target ~ U(lo, hi); start scene ~ uniform over "
                         "the video's scenes; extend over consecutive scenes until span >= target; reject if a scene "
                         "lacks a caption or clip file, span > hi, or the video ends first. One attempt per video; "
                         "each video used at most once overall. Candidates past 500 per bin are reserves used in "
                         "order to replace units whose joined file fails (|joined - span| > 1 s, outside bin, "
                         "decode error).",
            "join": "ffmpeg concat demuxer, -c copy (no re-encode)",
        },
        "selection_stats": sel_info,
        "build_failures": {b: dict(c) for b, c in fail_reasons.items()},
        "units": [{"unit_id": c["unit_id"], "source_video_id": c["source_video_id"], "length_bin": c["length_bin"],
                   "target_s": c["target_s"], "start_s": c["start_s"], "end_s": c["end_s"], "span_s": c["span_s"],
                   "scene_ids": c["scene_ids"], "joined_duration_s": done[c["unit_id"]]["duration_s"]}
                  for c in final_units],
    }
    (OUT / "selection.json").write_text(json.dumps(selection, indent=1))

    by_bin = {b: [v for v in vrecs if v["length_bin"] == b] for b in BINS}
    words = {b: [len(c["caption"].split()) for c in crecs if c["video_id"] in {v["video_id"] for v in by_bin[b]}]
             for b in BINS}
    total = sum(v["size_bytes"] for v in vrecs)
    stats = {
        "units": len(vrecs), "units_per_bin": {b: len(x) for b, x in by_bin.items()},
        "total_bytes": total, "total_gb": round(total / 1e9, 2),
        "gb_per_bin": {b: round(sum(v["size_bytes"] for v in x) / 1e9, 2) for b, x in by_bin.items()},
        "duration_s_per_bin": {b: dstats([v["duration_s"] for v in x]) for b, x in by_bin.items()},
        "total_h": round(sum(v["duration_s"] for v in vrecs) / 3600, 2),
        "abs_joined_minus_span_s": dstats([abs(done[c["unit_id"]]["duration_s"] - c["span_s"]) for c in final_units]),
        "scenes_per_unit": {b: dstats([v["n_scenes"] for v in x]) for b, x in by_bin.items()},
        "caption_words_per_bin": {b: dstats(w) for b, w in words.items()},
        "with_audio": sum(v["has_audio"] for v in vrecs),
        "resolutions": collections.Counter(f"{v['width']}x{v['height']}" for v in vrecs).most_common(5),
        "selection": sel_info, "build_failures": {b: dict(c) for b, c in fail_reasons.items()},
        "hf_calls_this_run": LIMITER.total, "runtime_s_this_run": round(t_run, 1),
    }
    (OUT / "_stats.json").write_text(json.dumps(stats, indent=2))
    write_readme(stats)
    return stats


def write_readme(s: dict) -> None:
    sel = s["selection"]
    rows = "\n".join(
        f"| {b} | {s['units_per_bin'][b]} | {s['duration_s_per_bin'][b].get('min')} / {s['duration_s_per_bin'][b].get('median')}"
        f" / {s['duration_s_per_bin'][b].get('max')} | {s['scenes_per_unit'][b].get('median')} | "
        f"{s['caption_words_per_bin'][b].get('median')} ({s['caption_words_per_bin'][b].get('p10')}-"
        f"{s['caption_words_per_bin'][b].get('p90')}) | {s['gb_per_bin'][b]} |" for b in BINS)
    txt = f"""# Vript long-video part: multi-scene spans (YouTube)

- Source: Hugging Face dataset `{REPO_ID}` (not gated). Files: `{META_FILE}` (yt-dlp info),
  `{CAPTIONS_FILE}` (scene captions; raw copy in `_downloads/`), and scene clip members of
  `vript_long_videos_clips/clips_{{1..1095}}_of_1095.zip`, which were **range-read** (only the needed members;
  the zips were never downloaded whole). Scene spans come from `<id>_cut_meta.json` inside those zips
  (cached with each zip's central directory in `_index/zips/`).
- Built: {dt.date.today().isoformat()} by `download/vript_long.py` (seed {SEED}); per-unit choices in
  `selection.json`.
- Licence: custom (adapted from HD-VG-130M): ACADEMIC USE ONLY, NO DISTRIBUTION; the authors may revoke access;
  YouTube copyright policies apply. The span files in `videos/` are derived works: do not back up or re-host them;
  release only unit ids, source video ids and scene ranges.

## What a unit is

One contiguous span of whole scenes from one source video, joined from Vript's scene clip files with the ffmpeg
concat demuxer in stream-copy mode (no re-encode). `video_id` = `<source_video_id>__sc<first>-<last>` (Vript scene
numbers). Each source video appears in at most one unit across all bins.

## Counts

| Bin | Units | Duration s (min / median / max) | Scenes per unit (median) | Caption words, median (p10-p90) | GB |
|---|---|---|---|---|---|
{rows}

- Total: {s['units']} units, {s['total_gb']} GB, {s['total_h']} h; with audio {s['with_audio']}; resolutions {s['resolutions']}.
- |joined duration - scene span|: {s['abs_joined_minus_span_s']}.
- Eligible pool: {sel['pool_size']} videos; excluded {sel['pool_excluded']}.
- Language: meta `language` {sel['language']['meta']}; py3langid fallback (title + ASR voiceover, for videos with no
  `language`): {sel['language']['langid_fallback']}.
- Rejected selection candidates (one attempt per video): {sel['rejected_candidates']}.
- Units dropped after joining and replaced from reserves: {s['build_failures']}.

## Selection

Bins: `30to60s` = 30-60 s, `1to2min` = 60-120 s, `2to4min` = 120-240 s (note: the contract's shared bin
`gt2min` is split here; this folder has no `lt30s`/`gt4min` units). The English, `integrity: true` pool is shuffled
with `random.Random(0)`; each video is offered to the bin with the fewest accepted candidates; a target length is drawn
uniformly in the bin, a start scene uniformly over the video's scenes, and the span is extended over consecutive
scenes until it reaches the target. Candidates are rejected if any scene in the span has no caption or no clip file,
if the span passes the bin's upper edge, or if the video ends first. 500 primaries + reserves per bin; a unit is
replaced by the next reserve if its joined file differs from the span by more than 1 s, falls outside its bin, or does
not decode cleanly (full `ffmpeg -f null` decode).

## Fields and caveats

- `videos.jsonl`: contract fields plus `source_video_id`, `scene_first`, `scene_last`, `n_scenes`, `source_start_s`,
  `source_end_s` (span within the source video, from cut_meta), `size_bytes`. `source` = `youtube`,
  `split` = `vript_long`.
- `captions.jsonl`: `caption` = the scenes' `caption.content` (visual content field only) joined with spaces in scene
  order. `parts` = one entry per scene: `text` (content), `start_s`/`end_s` relative to the unit (from cut_meta),
  `annotator` = `{ANNOTATOR}`, plus `clip_id`, `scene`, `scene_title`, `shot_type`, `camera_movement` and
  `voiceover_asr` (ASR text, often non-empty; NOT part of `caption`). `caption_origin` = `model` (GPT-4V).
- Per the Vript paper, captions were written with the title and voiceover as context, so they can name things not
  visible on screen. Caption length grows with span length (more scenes); scene captions repeat setting descriptions.
- Spans start at a random scene, so they can begin or end mid-action; some spans are a single long scene.
- Hosted clips are capped at 720p (the meta `resolution` is the original YouTube format).
"""
    (OUT / "README.md").write_text(txt)


# ----------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--index-workers", type=int, default=8)
    ap.add_argument("--seed-cut-meta", default=None,
                    help="optional clips_index.json with cut_meta harvested earlier (skips those small reads)")
    ap.add_argument("--select-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="build at most N units per bin (testing)")
    ap.add_argument("--retry-failed", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    for tool in ("ffmpeg", "ffprobe"):
        if shutil.which(tool) is None:
            sys.exit(f"{tool} not found")
    for d in (VID_DIR, DL, IDX, STATE, TMP):
        d.mkdir(parents=True, exist_ok=True)
    print(f"HF token: {'yes' if TOKEN else 'no (anonymous)'}", flush=True)

    meta = json.loads(download_whole(META_FILE).read_text())
    caps = load_captions(download_whole(CAPTIONS_FILE))
    print(f"meta {len(meta)} videos, captions {len(caps)} videos", flush=True)

    sizes = zip_sizes()
    paths = [CLIP_ZIP_TMPL.format(i=i) for i in range(1, N_CLIP_ZIPS + 1)]
    missing = [p for p in paths if p not in sizes]
    if missing:
        sys.exit(f"{len(missing)} clip zips missing from tree listing, e.g. {missing[:3]}")
    seed = load_seed_cut_meta(args.seed_cut_meta)
    zipidx, errs = {}, []
    with ThreadPoolExecutor(args.index_workers) as ex:
        futs = {ex.submit(index_zip, p, sizes[p], seed): p for p in paths}
        for k, fu in enumerate(as_completed(futs), 1):
            try:
                zipidx[futs[fu]] = fu.result()
            except Exception as e:
                errs.append(f"{futs[fu]}: {e!r}")
            if k % 200 == 0:
                print(f"indexed {k}/{len(paths)} zips (hf calls {LIMITER.total})", flush=True)
    if errs:
        sys.exit(f"index errors ({len(errs)}), rerun to resume: {errs[:3]}")
    print(f"zip index ready: {sum(z['empty'] for z in zipidx.values())} empty zips, "
          f"{sum(len(z['cut_meta']) for z in zipidx.values())} videos with scene spans", flush=True)

    short_ids = {r["video_id"] for r in read_jsonl(SHORT_VIDEOS)}
    cands, sel_info = select(meta, caps, zipidx, short_ids)
    print(json.dumps(sel_info, indent=1), flush=True)
    if args.select_only:
        return 0

    done = {r["unit_id"]: r for r in read_jsonl(STATE / "done.jsonl") if (VID_DIR / f"{r['unit_id']}.mp4").exists()}
    failed = {} if args.retry_failed else {r["unit_id"]: r for r in read_jsonl(STATE / "failed.jsonl")}
    lock = threading.Lock()
    n_target = min(N_PER_BIN, args.limit or N_PER_BIN)
    transient = collections.Counter()

    def record(c, res):
        with lock:
            if res["ok"]:
                rec = {"unit_id": c["unit_id"], "length_bin": c["length_bin"], **{k: v for k, v in res.items() if k != "ok"}}
                done[c["unit_id"]] = rec
                with open(STATE / "done.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            elif not res.get("transient"):
                rec = {"unit_id": c["unit_id"], "length_bin": c["length_bin"], "reason": res["reason"]}
                failed[c["unit_id"]] = rec
                with open(STATE / "failed.jsonl", "a") as f:
                    f.write(json.dumps(rec) + "\n")
            else:
                transient[c["unit_id"]] += 1

    def plan():
        """Units still to build so that each bin reaches n_target (in candidate order)."""
        todo = []
        for b in BINS:
            have = 0
            for c in cands[b]:
                if have >= n_target:
                    break
                if c["unit_id"] in done:
                    have += 1
                elif c["unit_id"] in failed or transient[c["unit_id"]] >= 3:
                    continue
                else:
                    todo.append(c)
                    have += 1
        return todo

    n_built = 0
    with ThreadPoolExecutor(args.workers) as ex:
        while True:
            todo = plan()
            if not todo:
                break
            print(f"building {len(todo)} units ({len(done)} done, {len(failed)} failed)", flush=True)
            futs = {ex.submit(build_unit, c, zipidx): c for c in todo}
            for fu in as_completed(futs):
                c = futs[fu]
                try:
                    res = fu.result()
                except Exception as e:
                    res = {"ok": False, "reason": f"exception {e!r}"[:300], "transient": True}
                record(c, res)
                n_built += 1
                if not res["ok"]:
                    print(f"  fail {c['unit_id']} [{c['length_bin']}]: {res['reason'][:160]}", flush=True)
                if n_built % 50 == 0:
                    gb = sum(d.get("size_bytes", 0) for d in done.values()) / 1e9
                    print(f"progress: {n_built} processed this run, done {len(done)}, failed {len(failed)}, "
                          f"{gb:.1f} GB, {(time.time() - t0) / 60:.1f} min", flush=True)

    final_units = []
    for b in BINS:
        got = [c for c in cands[b] if c["unit_id"] in done][:n_target]
        if len(got) < n_target:
            print(f"WARNING: bin {b} has only {len(got)} units (reserves exhausted)", flush=True)
        final_units += got
    stats = write_outputs(final_units, done, failed, caps, sel_info, time.time() - t0, args)
    print(json.dumps({k: v for k, v in stats.items() if k != "selection"}, indent=1))
    keep = {c["unit_id"] for c in final_units}
    extra = [p for p in VID_DIR.glob("*.mp4") if p.stem not in keep]
    if extra:
        print(f"note: {len(extra)} mp4s in videos/ are not in the final selection (left in place)")
    shutil.rmtree(TMP, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
