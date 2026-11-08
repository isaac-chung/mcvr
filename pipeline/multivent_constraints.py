"""Does single-search quality change with the number of constraints in a real human query?

Uses the routing pilot's MultiVENT 2.0 English queries (100 test, 300 dev) over its 2,547-video
corpus and DESC-only search (Jasper over the shipped Qwen3-Omni descriptions; S0 of the pilot).

- Test rankings come from the pilot run (results/pilot-multivent-ad540142/rankings.parquet).
- Dev rankings were not saved by the pilot, so dev queries are embedded locally with Jasper
  (descriptions re-embedded once, see desc_index); no paid calls.
- Constraints per query: gpt-6-luna@low with prompts/count_constraints_v1.txt, one call per
  query, cached in results/_cache/llm.sqlite. Proxy without the API: query length in words.

    uv run --env-file .env python pipeline/multivent_constraints.py
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.cache import shared_cache
from src.llm import make_llm
from src.metrics import ndcg_at_k, recall_at_k

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/multivent"
PILOT = ROOT / "results/pilot-multivent-ad540142"
OUT = ROOT / "results/multivent_k_check"
PROMPT = ROOT / "prompts/count_constraints_v1.txt"
LLM_SPEC = "openai:gpt-6-luna@low"
BINS = ["1", "2", "3", "4+"]
METRICS = ["ndcg10", "r1", "hit1", "r10"]
NAMES = {"ndcg10": "nDCG@10", "r1": "R@1", "hit1": "Hit@1", "r10": "R@10"}


def parse_constraints(text: str) -> list[str]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"no JSON in answer: {text[:200]!r}")
    items = json.loads(match.group(0))["constraints"]
    return [str(c).strip() for c in items if str(c).strip()]


def count_constraints(texts: list[str], workers: int = 8) -> list[list[str]]:
    llm = make_llm(LLM_SPEC, shared_cache("llm", ROOT / "results"))
    template = PROMPT.read_text()

    def one(query: str) -> list[str]:
        answer, _ = llm.complete(template.replace("{query}", query))
        return parse_constraints(answer)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, texts))


def jasper():
    from src.embedders import SentenceTransformerEmbedder

    return SentenceTransformerEmbedder(
        "infgrad/Jasper-Token-Compression-600M",
        batch_size=16,
        encode_kwargs={"compression_ratio": 0.6},
        query_prompt_name="query",
        dtype="float32",
        device="cpu",
    )


def desc_index() -> tuple[list[str], np.ndarray]:
    """Jasper DESC vectors for the pilot corpus. The pilot's own copy (results/_indexes/8fe406e3)
    was overwritten by a later 150-video run, so the 2,547 descriptions are re-embedded locally
    once and kept under OUT."""
    vectors, ids_file = OUT / "desc_jasper.npy", OUT / "desc_jasper.ids.json"
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    ids = [str(d) for d in corpus.doc_id]
    if vectors.exists() and json.loads(ids_file.read_text()) == ids:
        return ids, np.load(vectors)
    matrix = jasper().encode(corpus.desc.fillna("").tolist(), "document")
    np.save(vectors, matrix)
    ids_file.write_text(json.dumps(ids))
    return ids, matrix


def local_rankings(queries: pd.DataFrame, depth: int = 100) -> dict[str, list[str]]:
    """DESC-only rankings with Jasper, as the pilot; query vectors cached under OUT."""
    ids, matrix = desc_index()
    cache = OUT / "query_vectors.npz"
    stored = dict(np.load(cache)) if cache.exists() else {}
    todo = [q for q in queries.id if q not in stored]
    if todo:
        text_of = dict(zip(queries.id, queries.text))
        vectors = jasper().encode([text_of[q] for q in todo], "query")
        stored.update(dict(zip(todo, vectors)))
        np.savez(cache, **stored)
    out = {}
    for q in queries.id:
        scores = matrix @ stored[q]
        out[q] = [ids[i] for i in np.argsort(-scores, kind="stable")[:depth]]
    return out


def pilot_rankings() -> dict[str, list[str]]:
    r = pd.read_parquet(PILOT / "rankings.parquet")
    r = r[r["index"] == "DESC"].sort_values(["query_id", "rank"])
    return {q: list(g.doc_id) for q, g in r.groupby("query_id")}


def score(rankings: dict[str, list[str]], qrels: dict[str, dict[str, int]]) -> pd.DataFrame:
    rows = []
    for q, docs in rankings.items():
        ranking = [(d, 0.0) for d in docs]
        grades = qrels[q]
        rows.append(
            {
                "query_id": q,
                "ndcg10": ndcg_at_k(ranking, grades, 10),
                "r1": recall_at_k(ranking, grades, 1),
                "hit1": float(grades.get(docs[0], 0) > 0),
                "r10": recall_at_k(ranking, grades, 10),
                "n_relevant": sum(g > 0 for g in grades.values()),
            }
        )
    return pd.DataFrame(rows)


def boot_mean(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def boot_spearman(x: np.ndarray, y: np.ndarray, n: int = 5000, seed: int = 0):
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(n):
        i = rng.integers(0, len(x), len(x))
        if len(set(x[i])) > 1 and len(set(y[i])) > 1:
            draws.append(spearmanr(x[i], y[i]).statistic)
    p = float(spearmanr(x, y).pvalue)
    return rho, float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975)), p


def fmt(m: tuple[float, float, float]) -> str:
    return f"{m[0]:.3f} [{m[1]:.3f}, {m[2]:.3f}]"


def bin_of(k: int) -> str:
    return "4+" if k >= 4 else str(k)


def split_report(df: pd.DataFrame, label: str) -> list[str]:
    lines = [f"### {label} (n = {len(df)})", ""]
    lines.append("| constraints | n | " + " | ".join(NAMES[m] for m in METRICS) + " |")
    lines.append("|---|---|" + "---|" * len(METRICS))
    for b in BINS:
        part = df[df.bin == b]
        cells = [fmt(boot_mean(part[m].to_numpy())) if len(part) else "-" for m in METRICS]
        lines.append(f"| {b} | {len(part)} | " + " | ".join(cells) + " |")
    lines += ["", "Spearman correlation (95% bootstrap CI; p from scipy):", ""]
    lines.append("| predictor | " + " | ".join(NAMES[m] for m in METRICS) + " | #relevant |")
    lines.append("|---|" + "---|" * (len(METRICS) + 1))
    for pred, name in [("k", "constraint count"), ("words", "query length (words)")]:
        cells = []
        for m in METRICS + ["n_relevant"]:
            rho, lo, hi, p = boot_spearman(df[pred].to_numpy(), df[m].to_numpy())
            cells.append(f"{rho:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.2f}")
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    rho = spearmanr(df.k, df.words).statistic
    lines += ["", f"Constraint count vs query length: Spearman {rho:+.2f}.", ""]
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-dev", action="store_true", help="test queries only")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    queries = pd.read_parquet(DATA / "queries.parquet")
    qrels_df = pd.read_parquet(DATA / "qrels.parquet")
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_df.itertuples():
        qrels.setdefault(row.query_id, {})[str(row.doc_id)] = int(row.grade)
    test = queries[queries.split == "test"]
    dev = queries[queries.split == "dev"]

    # test: pilot's saved rankings, checked against the pilot's per-query S0 nDCG
    test_scores = score(pilot_rankings(), qrels)
    pq = pd.read_parquet(PILOT / "per_query.parquet")
    saved = pq[pq.system == "S0"].set_index("query_id").ndcg
    gap = (test_scores.set_index("query_id").ndcg10 - saved).abs().max()
    checks = [
        f"Test S0 nDCG@10 recomputed from saved rankings vs pilot per_query: max |diff| {gap:.2e}."
    ]

    frames = [test_scores.assign(split="test")]
    if not args.no_dev:
        local = local_rankings(pd.concat([test, dev]))
        saved_rank = pilot_rankings()
        overlap = np.mean([len(set(local[q][:10]) & set(saved_rank[q][:10])) / 10 for q in test.id])
        local_test = score({q: local[q] for q in test.id}, qrels)
        checks.append(
            f"Local Jasper re-embedding of the 100 test queries: top-10 overlap with saved pilot "
            f"rankings {overlap:.3f}; nDCG@10 {local_test.ndcg10.mean():.3f} vs saved "
            f"{test_scores.ndcg10.mean():.3f}."
        )
        dev_scores = score({q: local[q] for q in dev.id}, qrels)
        sel = json.loads((PILOT / "dev_selection.json").read_text())["mean_ndcg_by_subset"]["DESC"]
        checks.append(
            f"Dev DESC nDCG@10 {dev_scores.ndcg10.mean():.4f} vs pilot dev_selection {sel:.4f}."
        )
        frames.append(dev_scores.assign(split="dev"))
    df = pd.concat(frames, ignore_index=True)

    text_of = dict(zip(queries.id, queries.text))
    df["text"] = df.query_id.map(text_of)
    constraints = count_constraints(df.text.tolist())
    df["constraints"] = [json.dumps(c) for c in constraints]
    df["k"] = [len(c) for c in constraints]
    df["bin"] = df.k.map(bin_of)
    df["words"] = df.text.str.split().str.len()
    df.to_csv(OUT / "per_query.csv", index=False)

    lines = [
        "# MultiVENT k check: single-search quality vs constraints per real query",
        "",
        (
            "Script `pipeline/multivent_constraints.py`; per-query rows with the listed constraints in "
            "`per_query.csv`. System: DESC-only search (Jasper-Token-Compression-600M over the shipped "
            "Qwen3-Omni descriptions; the routing pilot's S0), 2,547-video English corpus. Constraint "
            f"counts by `{LLM_SPEC}` with `prompts/count_constraints_v1.txt` (one call per query, "
            "cached in `results/_cache/llm.sqlite`)."
        ),
        "",
        (
            "R@1 and R@10 are recall over all relevant videos (src.metrics, as the pilot); "
            "MultiVENT queries have several relevant videos, so R@1 is capped at 1/#relevant. Hit@1 "
            "(top video relevant) is added because it is the closer analogue of the smoke set's R@1 "
            "(one target per query). CIs: 5,000 bootstrap resamples over queries."
        ),
        "",
        "## Checks",
        "",
        *[f"- {c}" for c in checks],
        "",
        "## Results",
        "",
    ]
    lines += split_report(df[df.split == "test"], "Test queries (pilot's 100)")
    if not args.no_dev:
        lines += split_report(df[df.split == "dev"], "Dev queries (pilot's 300; locally re-ranked)")
        lines += split_report(df, "Test + dev pooled")
    dist = df.k.value_counts().sort_index().to_dict()
    lines += [
        (
            f"Constraint-count distribution (all): {dist}. Mean words per query "
            f"{df.words.mean():.1f}."
        ),
        "",
        "## Caveats",
        "",
        (
            "- Sparse judgments: judged@10 is only 0.33 to 0.47 in the pilot; unjudged relevant "
            "distractors count as misses. More specific queries may have fewer relevant videos "
            "(see the #relevant column), which moves nDCG and recall mechanically."
        ),
        (
            "- Event-centric news queries: most are short headlines about one event; constraint "
            "count here mostly measures how specific the event description is, not conjunctions of "
            "visual facts as in the smoke set."
        ),
        (
            "- English only; one corpus (2,547 videos); one counting model, one run (gpt-6 models "
            "use the default temperature, so counts may vary slightly on rerun)."
        ),
        (
            "- Observational: queries with more constraints differ in other ways (topic, length, "
            "number of relevant videos); this is a correlation, not a controlled k sweep."
        ),
        "",
    ]
    (OUT / "report.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
