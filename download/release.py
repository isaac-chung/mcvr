"""Download the MCVR Hugging Face release and unpack it into the files the scripts read. FREE: no
model calls (needs network, and HF_TOKEN while the dataset is private).

    uv run python download/release.py --repo <namespace>/MCVR
    uv run python download/release.py --from-dir data/release      # parquet already on disk

The release has three configs (corpus, queries, labels), each as <config>/test-*.parquet. They are
saved under data/release/ as they come, and these files are rebuilt from them:

- data/mc/<part>/descriptions_fps05.jsonl   the index descriptions (video_id, description, prompt,
  model); the describer's cost and frame fields are not in the release
- data/mc/full/sample.json                  "videos" (8,603 corpus ids) and "query_sources" (280)
- data/mc/full/queries.jsonl                nested queries; queries_twin.jsonl, near-miss queries
- data/mc/full/judge/qrels_{strict,expanded}.jsonl
- results/full/select/sources100.json       the 100 test sources of the paper ("in_paper")

The rebuilt query rows keep every field the evaluation reads (query, k, facts with type, scene,
grounding time and index visibility, phrases, HN video, d). They lack fields the release does not
carry (query_template, d_index, a fact's distinctive flag and grounded frame), so they are enough
to evaluate a system and to run pipeline/run_systems.py on new runs, not to rebuild the queries
byte for byte. The per-fact labels stay in data/release/labels/ as released (one row per fact,
video and labeller); the label sheets and the frame judge's raw answers are not in the release.
Existing files are never overwritten unless --force is given.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
MC = ROOT / "data/mc"
FULL = MC / "full"
CONFIGS = ("corpus", "queries", "labels")
SET_NAME = {"nested": "natural", "near_miss": "twin"}  # release name -> run key
DESCRIBER = "gemini:gemini-3.5-flash-lite"
MAIN_PROMPT = "describe_v4.txt"
CORPUS_COLUMNS = ["video_id", "source", "description", "is_query_source", "in_paper"]


def download(repo: str, out: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id=repo,
        repo_type="dataset",
        local_dir=out,
        allow_patterns=[f"{c}/*" for c in CONFIGS] + ["README.md"],
    )


def read(folder: Path, config: str, columns: list[str] | None = None) -> list[dict]:
    files = sorted((folder / config).glob("test-*.parquet"))
    if not files:
        raise SystemExit(f"no {config}/test-*.parquet in {folder}")
    rows: list[dict] = []
    for f in files:  # column subset, so the corpus's video bytes are never loaded
        rows += pq.read_table(f, columns=columns).to_pylist()
    return rows


def write_jsonl(path: Path, rows: list[dict], force: bool) -> None:
    if path.exists() and not force:
        print(f"kept {path.relative_to(ROOT)} (exists; --force to replace)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {path.relative_to(ROOT)} ({len(rows):,} rows)")


def write_json(path: Path, data: dict, force: bool) -> None:
    if path.exists() and not force:
        print(f"kept {path.relative_to(ROOT)} (exists; --force to replace)")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {path.relative_to(ROOT)}")


def query_row(r: dict) -> dict:
    """A released query row back in the shape of queries.jsonl / queries_twin.jsonl."""
    twin = r["set"] == "near_miss"
    facts, phrases = [], {}
    for i, f in enumerate(r["facts"]):
        key = f["fact_id"].split("#", 1)[1]
        fact = {"a_key": key, "role": f["role"]} if twin else {"key": key}
        fact |= {"text": f["text"], "type": f["type"], "scene": f["scene"]}
        if not twin:
            fact["grounded_time"] = f["grounded_time_s"]
        fact["index_visible"] = f["index_visible"]
        facts.append(fact)
        if f["phrase"] is not None:
            phrases[str(i + 1)] = f["phrase"]
    row = {
        "query_id": r["query_id"],
        "query": r["query"],
        "set": SET_NAME[r["set"]],
        "k": r["k"],
        "source": r["source_video_id"],
        "dataset": r["source"],
        "length_bin": r["length_bin"],
        "facts": facts,
        "phrases": phrases,
    }
    if twin:
        row |= {
            "twin": r["hn_video_id"],
            "twin_rank": r["hn_rank"],
            "twin_similarity": r["hn_similarity"],
            "d": r["d"],
            "d_type": r["d_type"],
        }
    return row


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="Hugging Face dataset id, e.g. <namespace>/MCVR")
    src.add_argument("--from-dir", type=Path, help="a folder that already holds the parquet configs")
    ap.add_argument("--out", type=Path, default=ROOT / "data/release", help="download folder")
    ap.add_argument("--force", action="store_true", help="replace files that already exist")
    args = ap.parse_args()

    folder = args.from_dir or args.out
    if args.repo:
        download(args.repo, folder)

    corpus = read(folder, "corpus", CORPUS_COLUMNS)
    by_part: dict[str, list[dict]] = defaultdict(list)
    for r in corpus:
        part, vid = r["video_id"].split("/", 1)
        if r["description"]:  # 14 corpus videos got an empty description; they stay out
            by_part[part].append(
                {"video_id": vid, "description": r["description"], "prompt": MAIN_PROMPT,
                 "model": DESCRIBER}
            )
    for part, rows in sorted(by_part.items()):
        write_jsonl(MC / part / "descriptions_fps05.jsonl", rows, args.force)
    sources = [r["video_id"] for r in corpus if r["is_query_source"]]
    write_json(
        FULL / "sample.json",
        {"videos": [r["video_id"] for r in corpus], "query_sources": sources},
        args.force,
    )

    queries = read(folder, "queries")
    for name, run_key in (("queries.jsonl", "natural"), ("queries_twin.jsonl", "twin")):
        rows = [query_row(r) for r in queries if SET_NAME[r["set"]] == run_key]
        write_jsonl(FULL / name, rows, args.force)
    for labels in ("strict", "expanded"):
        rows = [{"query_id": r["query_id"], "relevant": list(r[f"relevant_{labels}"])} for r in queries]
        write_jsonl(FULL / f"judge/qrels_{labels}.jsonl", rows, args.force)
    paper = sorted({r["source_video_id"] for r in queries if r["in_paper"]})
    write_json(
        ROOT / "results/full/select/sources100.json",
        {"n_sources": len(paper), "sources": paper},
        args.force,
    )
    print(f"labels: {len(read(folder, 'labels', ['fact_id'])):,} rows in {folder / 'labels'}")


if __name__ == "__main__":
    main()
