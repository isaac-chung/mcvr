"""Rebuttal: evaluate the description rerankers on the text-embedding-3-large first stage. FREE.

Reads S0-L / S1-L (pipeline/embedder_large.py) and S4@50-L / S2@50-L (the paper's S2@50, data
key S2g@50) / S5-L / S3-L (the paper's S3, data key S3g) (pipeline/embedder_large_rerank.py,
results/full_large/runs) next to the paper's
saved runs, and reports, as pipeline/embedder_large.py: nDCG@10 and R@1 by set and k under
fixed expanded, condensed (unjudged removed) and strict labels, paired cluster-bootstrap CIs over
sources, and unjudged top-10 shares.

    uv run python paper/embedder_large_eval.py

Outputs: results/full/rebuttal/embedder_large/{report_rerank.md, summary_rerank.json}.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EL = _load(ROOT / "pipeline/embedder_large.py", "_ele_large")
SV, FS = EL.SV, EL.FS

SOURCES = "results/full/select/sources100.json"
LARGE_RUNS = {  # system name here -> (folder, file stem)
    "S0-L": (EL.OUT / "runs", "S0-L"),
    "S1-L": (EL.OUT / "runs", "S1-L"),
    "S4@50-L": (Path("results/full_large/runs"), "S4@50"),
    "S2@50-L": (Path("results/full_large/runs"), "S2g@50"),
    "S5-L": (Path("results/full_large/runs"), "S5"),
    "S3-L": (Path("results/full_large/runs"), "S3g"),
}
NEW = list(LARGE_RUNS)
# paper names; the paper's S2@50 and S3 are the data keys S2g@50 and S3g
SYSTEMS = ["S0", "S0-L", "S2g@50", "S2@50-L", "S4@50", "S4@50-L", "S3g", "S3-L", "S5", "S5-L"]
PAIRS = [
    ("S4@50-L", "S4@50"),
    ("S2@50-L", "S2g@50"),
    ("S5-L", "S5"),
    ("S3-L", "S3g"),
    ("S4@50-L", "S2@50-L"),
    ("S4@50", "S2g@50"),
    ("S5-L", "S4@50-L"),
    ("S5", "S4@50"),
    ("S5-L", "S3-L"),
    ("S5", "S3g"),
    ("S4@50-L", "S0-L"),
    ("S0-L", "S0"),
]


def main() -> None:
    t0 = time.time()
    E = SV.load_eval(FS.N_BOOT, SOURCES)
    SV.sanity_summary_only(E)
    ranks = {n: dict(E["ranks"][n]) for n in SV.SETS}
    for name in SV.SETS:
        keep = {r["query_id"] for r in E["sets"][name]}
        for s, (folder, stem) in LARGE_RUNS.items():
            rk = FS.read_ranking(folder / name / f"{stem}.jsonl")
            ranks[name][s] = {q: v for q, v in rk.items() if q in keep}
            missing = keep - set(ranks[name][s])
            if missing:
                sys.exit(f"{s} / {name}: {len(missing)} test queries missing")
    res = SV.evaluate(E, ranks, SYSTEMS, PAIRS, NEW)
    cond = SV.evaluate(E, EL.condensed(ranks, E), SYSTEMS, PAIRS, NEW)
    sm = {
        "sources_file": SOURCES,
        "results": res,
        "condensed": {"labels": {"expanded": cond["labels"]["expanded"]}, "n": cond["n"]},
        "n_boot": E["S"].n_boot,
        "seconds": time.time() - t0,
    }
    out = EL.OUT
    (out / "summary_rerank.json").write_text(json.dumps(sm, indent=1, default=str) + "\n")
    md = [
        "# Description rerankers on the text-embedding-3-large first stage",
        "",
        "S2g@50 is the paper's S2@50; S2@50-L is the same system on S0-L's top 50.",
        "",
        "## Fixed labels (expanded and strict)",
        "",
        *SV.tables(res, SYSTEMS, PAIRS, NEW),
        "## Condensed lists (unjudged candidates removed before scoring), expanded labels",
        "",
    ]
    both = {"expanded": cond["labels"]["expanded"], "strict": cond["labels"]["expanded"]}
    on = False
    for line in SV.tables({"labels": both, "unjudged": {n: {} for n in SV.SETS}}, SYSTEMS, PAIRS, NEW):
        if line.startswith("### "):
            on = "expanded" in line and "Unjudged" not in line
        if on:
            md.append(line)
    (out / "report_rerank.md").write_text("\n".join(md) + "\n")
    print(f"wrote {out}/report_rerank.md ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
