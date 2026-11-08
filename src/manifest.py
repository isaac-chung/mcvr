"""manifest.json in a run folder: what produced the outputs, so a result can be traced and reused.

It records the git commit, the config and its hash, the model specs and the versions the APIs
returned, seeds, dates and cost totals. Writing again keeps the first `created` date and
updates the rest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import pandas as pd

from src.config import Config
from src.types import Cost

PACKAGES = ("numpy", "openai", "httpx", "anthropic", "sentence-transformers")


def git_state(repo: str | Path = ".") -> dict:
    """Commit, branch and whether tracked files have uncommitted changes (None outside git)."""

    def git(*args: str) -> str | None:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, text=True, check=False
            )
        except FileNotFoundError:  # no git on this machine
            return None
        return out.stdout.strip() if out.returncode == 0 else None

    status = git("status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": bool(status) if status is not None else None,
    }


def package_versions() -> dict[str, str | None]:
    found = {}
    for name in PACKAGES:
        try:
            found[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            found[name] = None
    return found


def run_models(cfg: Config, embedder: object | None = None) -> dict:
    """Model specs a run used, with the embedder versions the API returned when it reports them."""
    models = {"llm": cfg.llm, "embedder": cfg.embedder, "describe_llm": cfg.describe_llm}
    if getattr(embedder, "versions", None):
        models["embedder_versions"] = sorted(embedder.versions)
    return models


def runs_cost(run_dir: str | Path) -> Cost:
    """Total online cost of every system run in run_dir/runs.parquet (empty Cost without it)."""
    path = Path(run_dir) / "runs.parquet"
    if not path.exists():
        return Cost()
    df = pd.read_parquet(path)
    numeric = {f: type(v) for f, v in asdict(Cost()).items() if not isinstance(v, str)}
    total = Cost(**{f: kind(df[f].sum()) for f, kind in numeric.items() if f in df})
    if "model_version" in df:
        total = sum((Cost(model_version=v) for v in df.model_version.dropna().unique()), total)
    return total


def write_manifest(
    run_dir: str | Path,
    cfg: Config | None = None,
    models: dict | None = None,
    seeds: dict | None = None,
    cost: Cost | None = None,
    extra: dict | None = None,
) -> Path:
    """Write run_dir/manifest.json. Model versions come from cost.model_version (the versions
    the APIs returned); seeds default to the config seed."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    old = json.loads(path.read_text()) if path.exists() else {}
    now = datetime.now(UTC).isoformat(timespec="seconds")
    cost = cost or Cost()
    manifest = {
        **old,
        "created": old.get("created", now),
        "updated": now,
        "git": git_state(Path(__file__).resolve().parent.parent),
        "python": sys.version.split()[0],
        "packages": package_versions(),
        "config": asdict(cfg) if cfg else None,
        "config_hash": cfg.hash() if cfg else None,
        "models": models or {},
        "model_versions": [v for v in cost.model_version.split(",") if v],
        "seeds": seeds if seeds is not None else ({"seed": cfg.seed} if cfg else {}),
        "cost": cost.to_dict(),
        **(extra or {}),
    }
    path.write_text(json.dumps(manifest, indent=2, default=list))
    return path
