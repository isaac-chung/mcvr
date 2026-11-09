"""Rebuttal: the description rerankers on the text-embedding-3-large first stage. PAID.

Runs the paper's S4@50, S5 (run_graded_systems.py --stage frames-all), S3 (data key S3g, run_graded_systems.py --stage
frames-graded), (pipeline/run_systems.py --stage desc) and S2@50 (data key S2g@50,
pipeline/run_graded_systems.py --stage desc-graded) on S0-L's top 50 instead of S0's, with the same
prompts, verifier (gpt-6-sol@low), splits and code, so every (query, candidate) call the paper's
runs already made is a cache hit (results/_cache/llm.sqlite) and only new candidates are paid.
The binary S2 (paper S2b@50) is left out.

Everything goes to results/full_large/ (its own runs, spend.json, caps and manifests); the
paper's results/full/ is read, never written.

    uv run python pipeline/embedder_large_rerank.py setup        # free: runs dir from S0-L
    uv run --env-file .env python pipeline/embedder_large_rerank.py s4 --project-only
    uv run --env-file .env python pipeline/embedder_large_rerank.py s4
    uv run --env-file .env python pipeline/embedder_large_rerank.py s2g --project-only
    uv run --env-file .env python pipeline/embedder_large_rerank.py s2g
    uv run --env-file .env python pipeline/embedder_large_rerank.py s5 --project-only
    uv run --env-file .env python pipeline/embedder_large_rerank.py s3g --project-only

setup writes results/full_large/runs/<set>/{S0.jsonl (= S0-L), S1.jsonl (= S1-L), splits.json}
from results/full/rebuttal/embedder_large/runs and results/full/runs. s4 and s2g pass any further
arguments through to the stage (e.g. --project-only, --no-canary).
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path("results/full_large")
LARGE_RUNS = Path("results/full/rebuttal/embedder_large/runs")
PAPER_RUNS = Path("results/full/runs")
SOURCES = "results/full/select/sources100.json"
SETS = ("natural", "twin")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def setup() -> None:
    for name in SETS:
        d = OUT / "runs" / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(LARGE_RUNS / name / "S0-L.jsonl", d / "S0.jsonl")
        shutil.copyfile(LARGE_RUNS / name / "S1-L.jsonl", d / "S1.jsonl")
        shutil.copyfile(PAPER_RUNS / name / "splits.json", d / "splits.json")
    print(f"wrote {OUT}/runs/<set>/{{S0, S1}}.jsonl (3-large) and splits.json")


def s4(rest: list[str]) -> None:
    """systems.py --stage desc with S2 switched off (S4@50 only)."""
    FS = _load(ROOT / "pipeline/run_systems.py", "_elr_systems")
    base = FS.settings

    def s4_only(args):
        S = base(args)
        S.s2 = 0
        return S

    FS.settings = s4_only
    FS.main(["--stage", "desc", "--out", str(OUT), "--sources-file", SOURCES, *rest])


def tier1(stage: str, rest: list[str]) -> None:
    """run_graded_systems.py --stage <stage> on OUT; the 100-source draw comes from configs/mc_tier1.yaml
    sources_file."""
    T1 = _load(ROOT / "pipeline/run_graded_systems.py", "_elr_tier1")
    sys.argv = ["run_graded_systems.py", "--stage", stage, "--out", str(OUT), *rest]
    T1.main()


def frames_setup() -> None:
    """The frame stages read the frozen pool (subsets.json: S4@50, now OUT's 3-large S4@50) and
    the measured tokens per image (frames/frame_rate.json) from OUT."""
    if not (OUT / "runs" / SETS[0] / "S4@50.jsonl").exists():
        sys.exit("run s4 first: the frame pool is S4@50's top 10")
    for rel in ("select/subsets.json", "frames/frame_rate.json"):
        (OUT / rel).parent.mkdir(parents=True, exist_ok=True)
        if not (OUT / rel).exists():
            shutil.copyfile(Path("results/full") / rel, OUT / rel)


if __name__ == "__main__":
    step, rest = sys.argv[1], sys.argv[2:]
    if not (OUT / "runs").exists() and step != "setup":
        sys.exit("run setup first")
    if step in ("s5", "s3g"):
        frames_setup()
    {
        "setup": lambda: setup(),
        "s4": lambda: s4(rest),
        "s2g": lambda: tier1("desc-graded", rest),  # the paper's S2@50
        "s5": lambda: tier1("frames-all", rest),  # S5 on every test query
        "s3g": lambda: tier1("frames-graded", rest),  # the paper's S3
    }[step]()
