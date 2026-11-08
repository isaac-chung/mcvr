"""Draw the full-run sample (docs/full_run_plan.md section 1). Free: no API calls.

"videos" is the whole corpus: every video in data/mc/<dataset>/videos.jsonl for the config's
datasets (8,603 global ids). "query_sources" are the test sources, drawn with the config seed per
`test_strata` (dataset -> length bin -> count, 'any' for every bin). Length bins are recomputed
from the duration (`src.datasets.mc.length_bin`).

Eligible: a non-empty caption and Vript's integrity flag not false. Vript long is English by
construction (data/mc/vript_long/README.md: the folder was cut from the English,
integrity-true pool only). Excluded: the dev sources (config `dev_sample`, "query_sources") and
every video that shares a source video (`source_key`: YouTube id for Shot2Story and Vript long,
FineAction id for CaReBench) with a dev source. At most one unit per source video across all
strata. Writes <smoke_dir>/sample.json:
{"seed", "videos", "query_sources", "strata", "eligible", "excluded", "config"}.

    uv run python pipeline/sample_test_pool.py [--config configs/mc_full.yaml]
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import yaml

from src.datasets.mc import (
    ANY_BIN,
    global_id,
    load_captions,
    load_videos,
    source_key,
    split_global_id,
)


def captioned_ids(root: Path, dataset: str) -> set[str]:
    """Videos with a non-empty caption, minus those flagged integrity: false."""
    captions = load_captions(root / dataset)
    return {
        vid for vid, cap in captions.items() if cap.caption.strip() and cap.integrity is not False
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_full.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["data_root"])
    datasets = list(cfg["datasets"])
    strata: dict[str, dict[str, int]] = cfg["test_strata"]
    videos = {ds: load_videos(root / ds) for ds in datasets}
    by_gid = {global_id(ds, v.video_id): v for ds in datasets for v in videos[ds]}

    dev = json.loads(Path(cfg["dev_sample"]).read_text())["query_sources"]
    dev_set = set(dev)
    dev_keys = {source_key(by_gid[g]) for g in dev}

    eligible: dict[str, list] = {}
    excluded: dict[str, dict[str, int]] = {}
    for ds in datasets:
        captioned = captioned_ids(root, ds)
        keep, ex = [], {"no_caption_or_integrity_false": 0, "dev_source": 0, "dev_source_video": 0}
        for v in videos[ds]:
            gid = global_id(ds, v.video_id)
            if v.video_id not in captioned:
                ex["no_caption_or_integrity_false"] += 1
            elif gid in dev_set:
                ex["dev_source"] += 1
            elif source_key(v) in dev_keys:
                ex["dev_source_video"] += 1
            else:
                keep.append(v)
        eligible[ds] = sorted(keep, key=lambda v: v.video_id)
        excluded[ds] = ex

    rng = random.Random(cfg["seed"])
    used: set[str] = set()
    sources: list[str] = []
    counts: dict[str, dict] = {}
    for ds, bins in strata.items():
        for b, n in bins.items():
            cell = [v for v in eligible[ds] if b == ANY_BIN or v.length_bin == b]
            n_eligible = len(cell)
            rng.shuffle(cell)
            picked = []
            for v in cell:
                if len(picked) == n:
                    break
                if source_key(v) not in used:
                    used.add(source_key(v))
                    picked.append(v)
            if len(picked) < n:
                raise ValueError(f"{ds}/{b}: only {len(picked)} eligible sources, need {n}")
            sources += [global_id(ds, v.video_id) for v in picked]
            found: dict[str, int] = {}
            for v in picked:
                found[v.length_bin] = found.get(v.length_bin, 0) + 1
            counts[f"{ds}/{b}"] = {
                "query_sources": len(picked),
                "eligible": n_eligible,
                "length_bins": dict(sorted(found.items())),
            }

    # checks the build relies on
    assert len(set(sources)) == len(sources)
    assert not set(sources) & dev_set
    assert not {source_key(by_gid[g]) for g in sources} & dev_keys
    assert all(split_global_id(g)[0] in datasets for g in sources)

    sample = {
        "seed": cfg["seed"],
        "videos": list(by_gid),
        "query_sources": sources,
        "strata": counts,
        "eligible": {ds: len(vs) for ds, vs in eligible.items()},
        "excluded": excluded,
        "corpus": {ds: len(vs) for ds, vs in videos.items()},
        "config": {"test_strata": strata, "dev_sample": cfg["dev_sample"]},
    }
    out = Path(cfg["smoke_dir"]) / "sample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=1) + "\n")
    print(f"wrote {out}: {len(sample['videos'])} videos, {len(sources)} sources")
    print(f"  eligible {sample['eligible']}")
    print(f"  excluded {excluded}")
    for cell, c in counts.items():
        print(
            f"  {cell:22s} sources {c['query_sources']:3d} of {c['eligible']:5d}  {c['length_bins']}"
        )


if __name__ == "__main__":
    main()
