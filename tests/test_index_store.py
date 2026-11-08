"""Index stores are never silently overwritten, and pilot configs over different corpora get
different stores."""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.config import Config
from src.embedders import HashEmbedder
from src.indexes.dense_text import DenseTextIndex


def test_same_texts_reuse_the_store(tmp_path):
    texts = {"a": "a red car", "b": "a blue boat"}
    first = DenseTextIndex.build("DESC", texts, HashEmbedder(), tmp_path)
    again = DenseTextIndex.build("DESC", texts, HashEmbedder(), tmp_path)
    assert np.array_equal(first.matrix, again.matrix)
    assert (tmp_path / "DESC.texts.json").exists()


def test_other_texts_with_the_same_ids_raise(tmp_path):
    DenseTextIndex.build("DESC", {"a": "a red car", "b": "a blue boat"}, HashEmbedder(), tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        DenseTextIndex.build(
            "DESC", {"a": "a green car", "b": "a blue boat"}, HashEmbedder(), tmp_path
        )


def test_other_ids_raise(tmp_path):
    DenseTextIndex.build("DESC", {"a": "a red car"}, HashEmbedder(), tmp_path)
    with pytest.raises(ValueError, match="refusing to overwrite"):
        DenseTextIndex.build("DESC", {"z": "a red car"}, HashEmbedder(), tmp_path)


def test_a_legacy_store_with_matching_ids_is_reused_and_fingerprinted(tmp_path):
    texts = {"a": "a red car", "b": "a blue boat"}
    np.save(tmp_path / "DESC.npy", np.zeros((2, 4), dtype=np.float32))
    (tmp_path / "DESC.ids.json").write_text(json.dumps(list(texts)))
    index = DenseTextIndex.build("DESC", texts, HashEmbedder(), tmp_path)
    assert index.matrix.shape == (2, 4)
    assert (tmp_path / "DESC.texts.json").exists()


def test_pilot_configs_over_different_corpora_get_different_stores():
    multivent = Config(name="pilot-multivent")
    shot2story = Config(name="pilot-shot2story", dataset="shot2story", data_dir="data/shot2story")
    carebench = Config(name="pilot-carebench", dataset="carebench", data_dir="data/carebench")
    dirs = {multivent.index_dir(), shot2story.index_dir(), carebench.index_dir()}
    assert len(dirs) == 3
    assert Config(name="other").index_dir() == multivent.index_dir()  # same corpus, same store
