"""Tiny sqlite key-value cache for expensive calls (LLM, embeddings of queries).

The multi-constraint study keeps one shared cache per kind of call in results/_cache/
(`shared_cache`), so every run reuses what any earlier run paid for. The routing pilot keeps its
per-run and per-dataset caches.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path


class Cache:
    def __init__(self, path: str | Path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        self._db.execute("CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT)")

    @staticmethod
    def key(namespace: str, payload: object) -> str:
        blob = json.dumps([namespace, payload], sort_keys=True, default=str)
        return hashlib.sha256(blob.encode()).hexdigest()

    def get(self, namespace: str, payload: object):
        """The stored JSON value for (namespace, payload), or None."""
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM kv WHERE key = ?", (self.key(namespace, payload),)
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def put_many(self, namespace: str, items: list[tuple[object, object]]) -> None:
        """Store (payload, value) pairs in one transaction."""
        rows = [(self.key(namespace, p), json.dumps(v)) for p, v in items]
        with self._lock:
            self._db.executemany("INSERT OR REPLACE INTO kv VALUES (?, ?)", rows)
            self._db.commit()

    def get_or_compute(self, namespace: str, payload: object, fn: Callable[[], object]):
        """Return the stored JSON value for (namespace, payload), computing it once."""
        key = self.key(namespace, payload)
        with self._lock:
            row = self._db.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is not None:
            return json.loads(row[0])
        value = fn()  # outside the lock, so slow calls from other threads can run in parallel
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO kv VALUES (?, ?)", (key, json.dumps(value)))
            self._db.commit()
        return value


SHARED_KINDS = ("llm", "vlm", "embed")
_shared: dict[Path, Cache] = {}
_shared_lock = threading.Lock()


def shared_cache(kind: str, results_dir: str | Path = "results") -> Cache:
    """results/_cache/<kind>.sqlite (kind: llm, vlm or embed), one connection per file."""
    if kind not in SHARED_KINDS:
        raise ValueError(f"unknown cache kind {kind!r}; choose from {SHARED_KINDS}")
    path = (Path(results_dir) / "_cache" / f"{kind}.sqlite").resolve()
    with _shared_lock:
        if path not in _shared:
            _shared[path] = Cache(path)
        return _shared[path]
