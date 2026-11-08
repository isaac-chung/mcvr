"""The stratified smoke sample (dev/sample_dev.py, src.datasets.mc.stratified_sample)."""

import json
from collections import Counter
from pathlib import Path

import pytest
import yaml

from src.datasets import mc
from src.datasets.mc import Video, source_key, stratified_sample


def toy_videos():
    """Shot2Story-like clips (two shots per YouTube video) and Vript-long-like spans."""
    s2 = [
        Video(f"yt{i:03d}.{shot}", "", 12.0 if i % 3 else 45.0, dataset="shot2story")
        for i in range(60)
        for shot in (1, 2)
    ]
    s2 = [Video(**{**v.__dict__, "length_bin": mc.length_bin(v.duration_s)}) for v in s2]
    vl = [
        Video(
            f"src{i:03d}__sc001-002",
            "",
            d,
            length_bin=mc.length_bin(d),
            source_video_id=f"src{i:03d}",
            dataset="vript_long",
        )
        for i, d in enumerate([45.0, 90.0, 200.0] * 20)
    ]
    return {"shot2story": s2, "vript_long": vl}


STRATA = {
    "shot2story": {"lt30s": 20, "30to60s": 10},
    "vript_long": {"30to60s": 6, "1to2min": 6, "2to4min": 6},
}


def eligible_all(videos):
    return {ds: {v.video_id for v in vs} for ds, vs in videos.items()}


def test_sample_counts_sources_and_determinism():
    videos = toy_videos()
    a = stratified_sample(videos, eligible_all(videos), STRATA, 7, seed=0)
    b = stratified_sample(videos, eligible_all(videos), STRATA, 7, seed=0)
    c = stratified_sample(videos, eligible_all(videos), STRATA, 7, seed=1)
    assert a == b and a["videos"] != c["videos"]
    assert len(a["videos"]) == 48 == len(set(a["videos"]))
    assert all(g.split("/", 1)[0] in STRATA for g in a["videos"])
    assert set(a["query_sources"]) <= set(a["videos"]) and len(a["query_sources"]) == 14
    # 7 sources per group, spread over the strata, remainder to the first stratum
    assert {k: v["query_sources"] for k, v in a["strata"].items()} == {
        "shot2story/lt30s": 4,
        "shot2story/30to60s": 3,
        "vript_long/30to60s": 3,
        "vript_long/1to2min": 2,
        "vript_long/2to4min": 2,
    }
    assert a["strata"]["vript_long/2to4min"]["length_bins"] == {"2to4min": 6}


def test_one_unit_per_source_video():
    videos = toy_videos()
    sample = stratified_sample(videos, eligible_all(videos), STRATA, 7)
    by_id = {mc.global_id(v.dataset, v.video_id): v for vs in videos.values() for v in vs}
    keys = Counter(source_key(by_id[g]) for g in sample["videos"])
    assert max(keys.values()) == 1


def test_ineligible_videos_are_never_drawn_and_shortfall_is_an_error():
    videos = toy_videos()
    eligible = eligible_all(videos)
    dropped = {v.video_id for v in videos["shot2story"] if v.video_id.endswith(".2")}
    eligible["shot2story"] -= dropped
    sample = stratified_sample(videos, eligible, STRATA, 7)
    assert not any(g.split("/", 1)[1] in dropped for g in sample["videos"])
    with pytest.raises(ValueError, match="only"):
        stratified_sample(videos, eligible, {"shot2story": {"30to60s": 50}}, 2)


def test_source_keys():
    assert source_key(Video("abc.3", "", 1, dataset="shot2story")) == "yt:abc"
    assert source_key(Video("v_0001_2", "", 1, dataset="carebench")) == "fineaction:v_0001"
    assert (
        source_key(Video("abc__sc1", "", 1, source_video_id="abc", dataset="vript_long"))
        == "yt:abc"
    )
    assert source_key(Video("x", "", 1, dataset="vript_short")) == "vript_short:x"


def test_smoke_config_matches_the_plan():
    cfg = yaml.safe_load(Path("configs/mc_smoke.yaml").read_text())
    assert sum(n for bins in cfg["strata"].values() for n in bins.values()) == 500
    assert cfg["sources_per_group"] * len(cfg["strata"]) == 100
    assert cfg["seed"] == 0 and cfg["n_frames"] == 16 and cfg["full_corpus"] == 8603


def test_written_sample_if_present():
    """data/mc/smoke/sample.json, when it exists on this machine, has the agreed shape."""
    path = Path("data/mc/smoke/sample.json")
    if not path.exists():
        pytest.skip("no smoke sample on this machine")
    sample = json.loads(path.read_text())
    assert sample["seed"] == 0 and len(set(sample["videos"])) == 500
    assert len(set(sample["query_sources"])) == 100
    groups = Counter(g.split("/", 1)[0] for g in sample["query_sources"])
    assert set(groups.values()) == {25}
