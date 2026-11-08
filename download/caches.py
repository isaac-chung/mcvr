"""Paid-call caches as text shards that git can hold. FREE: no model calls.

    uv run python download/caches.py export   # results/_cache/<kind>.sqlite -> shards
    uv run python download/caches.py import   # shards -> results/_cache/<kind>.sqlite

The shared caches (src.cache: results/_cache/{llm,vlm,embed}.sqlite) hold every paid LLM,
VLM and embedding answer, and two are over GitHub's 100 MB file limit. export writes each one to
results/_cache/shards/<kind>/<kind>.NNN.tsv, one "key<TAB>value" line per row (the cache's own
sha256 key and JSON value, neither ever contains a tab or newline), rows sorted by key, a new
shard before any would pass --max-mb (default 99, in 10^6 bytes), and replaces that kind's old
shards. import adds every shard row to the sqlite file (creating it if needed) without
overwriting rows already there. Both check that the row counts match. The .sqlite files are
gitignored; the shards are committed.
"""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from src.cache import SHARED_KINDS, Cache

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "results/_cache"
SHARDS = CACHE / "shards"


def export(kind: str, max_bytes: int) -> None:
    src = CACHE / f"{kind}.sqlite"
    if not src.exists():
        print(f"{kind}: no {src.relative_to(ROOT)}, skipped")
        return
    out = SHARDS / kind
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob(f"{kind}.*.tsv"):
        old.unlink()
    db = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    n, shard, size, f = 0, 0, 0, None
    for key, value in db.execute("SELECT key, value FROM kv ORDER BY key"):
        assert "\t" not in key + value and "\n" not in key + value, key
        line = f"{key}\t{value}\n".encode()
        if f is None or size + len(line) > max_bytes:
            if f:
                f.close()
                shard += 1
            f, size = (out / f"{kind}.{shard:03d}.tsv").open("wb"), 0
        f.write(line)
        size += len(line)
        n += 1
    if f:
        f.close()
    rows = db.execute("SELECT count(*) FROM kv").fetchone()[0]
    assert n == rows, (kind, n, rows)
    files = sorted(out.glob(f"{kind}.*.tsv"))
    biggest = max((p.stat().st_size for p in files), default=0)
    print(f"{kind}: {n} rows -> {len(files)} shards, largest {biggest / 1e6:.1f} MB")


def import_(kind: str) -> None:
    files = sorted((SHARDS / kind).glob(f"{kind}.*.tsv"))
    if not files:
        print(f"{kind}: no shards, skipped")
        return
    dst = CACHE / f"{kind}.sqlite"
    Cache(dst)  # creates the file and the kv table as src does
    db = sqlite3.connect(dst)
    n = 0
    for p in files:
        with p.open(encoding="utf-8") as f:
            rows = [line.rstrip("\n").split("\t", 1) for line in f]
        db.executemany("INSERT OR IGNORE INTO kv VALUES (?, ?)", rows)
        n += len(rows)
    db.commit()
    have = db.execute("SELECT count(*) FROM kv").fetchone()[0]
    assert have >= n, (kind, have, n)
    print(f"{kind}: {n} shard rows from {len(files)} shards; {dst.relative_to(ROOT)} has {have}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("action", choices=("export", "import"))
    ap.add_argument("--kinds", nargs="+", default=list(SHARED_KINDS), choices=SHARED_KINDS)
    ap.add_argument("--max-mb", type=float, default=99.0)
    args = ap.parse_args()
    for kind in args.kinds:
        if args.action == "export":
            export(kind, int(args.max_mb * 1e6))
        else:
            import_(kind)


if __name__ == "__main__":
    main()
