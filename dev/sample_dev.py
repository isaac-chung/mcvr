"""Draw the 500-video smoke / dev sample (docs/plan.md, "Smoke and dev set"). Free: no API calls.

Strata, seed and source counts come from configs/mc_smoke.yaml. Only captioned videos are
eligible (captions are read here only to know which videos have one, and Vript's integrity
flag); Vript short `integrity: false` is excluded. At most one unit per source video.
Writes data/mc/smoke/sample.json: {"seed", "videos" (500 global ids), "query_sources" (100),
"strata" (counts)}.

    uv run python dev/sample_dev.py [--config configs/mc_smoke.yaml]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from src.datasets.mc import load_captions, load_videos, stratified_sample


def eligible_ids(root: Path, dataset: str) -> set[str]:
    """Videos with a non-empty caption, minus those flagged integrity: false."""
    captions = load_captions(root / dataset)
    return {
        vid for vid, cap in captions.items() if cap.caption.strip() and cap.integrity is not False
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_smoke.yaml")
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = Path(cfg["data_root"])
    datasets = list(cfg["strata"])
    videos = {ds: load_videos(root / ds) for ds in datasets}
    eligible = {ds: eligible_ids(root, ds) for ds in datasets}
    sample = stratified_sample(
        videos, eligible, cfg["strata"], cfg["sources_per_group"], seed=cfg["seed"]
    )
    sample["config"] = {"strata": cfg["strata"], "sources_per_group": cfg["sources_per_group"]}
    sample["eligible"] = {ds: len(ids) for ds, ids in eligible.items()}
    out = Path(cfg["smoke_dir"]) / "sample.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(sample, indent=1) + "\n")
    print(f"wrote {out}: {len(sample['videos'])} videos, {len(sample['query_sources'])} sources")
    for cell, c in sample["strata"].items():
        print(
            f"  {cell:22s} videos {c['videos']:4d}  sources {c['query_sources']:3d}  {c['length_bins']}"
        )


if __name__ == "__main__":
    main()
