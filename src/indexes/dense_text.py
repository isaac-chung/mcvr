"""Dense retrieval over short texts (ASR transcripts, visual descriptions)."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from src.embedders import Embedder
from src.indexes.base import top_k
from src.types import Cost, Ranking


def texts_fingerprint(texts: dict[str, str]) -> str:
    """sha256 over the ordered (id, text) pairs: what an index was built from."""
    return hashlib.sha256(json.dumps(list(texts.items())).encode()).hexdigest()


class DenseTextIndex:
    def __init__(
        self,
        name: str,
        doc_ids: list[str],
        matrix: np.ndarray,
        embedder: Embedder,
        build_seconds: float = 0.0,
    ):
        self.name, self.doc_ids, self.matrix, self.embedder = name, doc_ids, matrix, embedder
        self.build_seconds = build_seconds  # time spent embedding the documents, kept across reruns

    @classmethod
    def build(cls, name: str, texts: dict[str, str], embedder: Embedder, store: Path):
        """Embed the documents once and keep them on disk next to the results.

        <name>.texts.json records a fingerprint of the ids and texts. A store is reused only when
        it matches; a store that holds another index of the same name raises instead of being
        overwritten. Stores written before fingerprints existed are reused when their ids match
        (as before) and get a fingerprint then."""
        store.mkdir(parents=True, exist_ok=True)
        vectors, ids_file = store / f"{name}.npy", store / f"{name}.ids.json"
        seconds_file, texts_file = store / f"{name}.seconds.json", store / f"{name}.texts.json"
        doc_ids = list(texts)
        fingerprint = texts_fingerprint(texts)
        stored = json.loads(texts_file.read_text())["sha256"] if texts_file.exists() else None
        if stored is not None and stored != fingerprint:
            raise ValueError(
                f"{store} holds another {name} index (other ids or texts); refusing to overwrite it"
            )
        if vectors.exists() and ids_file.exists():
            if json.loads(ids_file.read_text()) != doc_ids:
                raise ValueError(
                    f"{store} holds a {name} index over other documents; refusing to overwrite it"
                )
            matrix = np.load(vectors)
            seconds = json.loads(seconds_file.read_text()) if seconds_file.exists() else 0.0
            if stored is None:
                texts_file.write_text(json.dumps({"sha256": fingerprint, "n_docs": len(doc_ids)}))
        else:
            start = time.perf_counter()
            matrix = embedder.encode([texts[d] for d in doc_ids], "document")
            seconds = time.perf_counter() - start
            np.save(vectors, matrix)
            ids_file.write_text(json.dumps(doc_ids))
            seconds_file.write_text(json.dumps(seconds))
            texts_file.write_text(json.dumps({"sha256": fingerprint, "n_docs": len(doc_ids)}))
        return cls(name, doc_ids, matrix, embedder, seconds)

    def search(self, text: str, k: int) -> tuple[Ranking, Cost]:
        start = time.perf_counter()
        query = self.embedder.encode([text], "query")[0]
        ranking = top_k(self.doc_ids, self.matrix @ query, k)
        return ranking, Cost(searches=1, seconds=time.perf_counter() - start)
