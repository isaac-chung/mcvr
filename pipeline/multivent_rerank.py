"""External check of the paper's text systems on REAL MultiVENT 2.0 queries. PAID (LLM calls).

S0 (one search over the DESC index: shipped Qwen3-Omni descriptions of the routing pilot's
2,547-video English corpus), S2 (per-constraint verification of S0's top N against each
candidate's description; src.systems.verify) and S4 (holistic 0-10 rerank of the top N;
src.systems.rerank), on the pilot's 100 test and 300 dev queries. Verifier
`openai:gpt-6-sol@low`, splitter `openai:gpt-6-luna@low`, prompts as the smoke pilot
(split_v1, verify_desc_v1, rerank_v1). Embedders: Jasper-Token-Compression-600M (the pilot's;
vectors from results/multivent_k_check, re-embedded there) and text-embedding-3-small.

Graded MultiVENT judgments: nDCG@10 with grades as gains; R@1 and R@10 binary at grade > 0
(recall over all relevant videos); Judged@10 = share of the top 10 with any judgment. Constraint
bins (1, 2, 3, 4+) come from results/multivent_k_check/per_query.csv.

Every call goes through results/_cache/llm.sqlite. Before any verifier call the exact prompts
are built and checked against the cache; the uncached ones are projected (chars per token
measured on --measure queries) and the run stops if over --token-budget.

    uv run --env-file .env python pipeline/multivent_rerank.py --measure 5
    uv run --env-file .env python pipeline/multivent_rerank.py --token-budget 4800000
    uv run python pipeline/multivent_rerank.py --eval-only
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from src.cache import shared_cache
from src.embedders import make_embedder
from src.llm import make_llm
from src.metrics import judged_at_k, ndcg_at_k, recall_at_k
from src.systems.base import check_reranked, merge_reranked
from src.systems.decompose import SPLIT_PROMPT, SPLIT_SPEC, QuerySplitter
from src.systems.rerank import (
    DESC_EVIDENCE,
    RERANK_PROMPT,
    DescriptionScore,
    HolisticReranker,
    fill,
)
from src.systems.verify import (
    DESC_PROMPT,
    VERIFIER_SPEC,
    ConstraintVerifier,
    DescriptionCheck,
    numbered,
)
from src.types import Cost, Query, Ranking

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/multivent"
KCHECK = ROOT / "results/multivent_k_check"
OUT = ROOT / "results/multivent_rerank"
RESULTS = ROOT / "results"
EMBEDDERS = {"jasper": "Jasper-Token-Compression-600M", "te3s": "text-embedding-3-small"}
SYSTEMS = ("S0", "S2", "S4")
BINS = ("1", "2", "3", "4+")
METRICS = ("ndcg10", "r1", "r10", "judged10")
NAMES = {"ndcg10": "nDCG@10", "r1": "R@1", "r10": "R@10", "judged10": "Judged@10"}
DEPTH = 100
CHARS_PER_TOKEN_DEFAULT = 3.8


# ---- paid-call plumbing (same pattern as dev/verifiers_dev.py, re-implemented here) --


class Metered:
    """Inside the cache: records the cost of the calls that really go out."""

    def __init__(self, inner, log: list, lock: threading.Lock):
        self.inner, self.log, self.lock = inner, log, lock

    def __getattr__(self, name):
        return getattr(self.inner, name)

    def complete(self, prompt: str):
        text, cost = self.inner.complete(prompt)
        with self.lock:
            self.log.append((len(prompt), cost))
        return text, cost


class SafeCalls:
    """An API error after retries is logged and answered with "" (unclear / unscored); nothing
    is cached for it, so a rerun tries again."""

    def __init__(self, inner, log: list):
        self.inner, self.log = inner, log

    def complete(self, prompt: str):
        try:
            return self.inner.complete(prompt)
        except Exception as error:  # noqa: BLE001
            self.log.append(repr(error)[:300])
            return "", Cost()


def paid_model(spec: str, log: list, errors: list, lock: threading.Lock):
    cached = make_llm(spec, shared_cache("llm", RESULTS))
    cached.inner.client = cached.inner.client.with_options(max_retries=10, timeout=300)
    cached.inner = Metered(cached.inner, log, lock)
    return cached, SafeCalls(cached, errors)


# ---- data -------------------------------------------------------------------------------------


def load():
    queries = pd.read_parquet(DATA / "queries.parquet")
    corpus = pd.read_parquet(DATA / "corpus.parquet")
    descriptions = {str(d): (t or "") for d, t in zip(corpus.doc_id, corpus.desc)}
    qrels: dict[str, dict[str, int]] = {}
    for row in pd.read_parquet(DATA / "qrels.parquet").itertuples():
        qrels.setdefault(row.query_id, {})[str(row.doc_id)] = int(row.grade)
    bins = pd.read_csv(KCHECK / "per_query.csv").set_index("query_id")["bin"].astype(str)
    return queries, corpus, descriptions, qrels, bins.to_dict()


def top(matrix: np.ndarray, ids: list[str], qvec: np.ndarray) -> Ranking:
    scores = matrix @ qvec
    order = np.argsort(-scores, kind="stable")[:DEPTH]
    return [(ids[i], float(scores[i])) for i in order]


def s0_jasper(queries: pd.DataFrame) -> dict[str, Ranking]:
    """The pilot's DESC search: Jasper vectors re-embedded by pipeline/multivent_constraints.py."""
    ids = json.loads((KCHECK / "desc_jasper.ids.json").read_text())
    matrix = np.load(KCHECK / "desc_jasper.npy")
    qv = dict(np.load(KCHECK / "query_vectors.npz"))
    missing = [q for q in queries.id if q not in qv]
    if missing:
        sys.exit(f"{len(missing)} queries lack Jasper vectors; run pipeline/multivent_constraints.py")
    return {q: top(matrix, ids, qv[q]) for q in queries.id}


def s0_openai(queries: pd.DataFrame, corpus: pd.DataFrame, model: str) -> dict[str, Ranking]:
    embedder = make_embedder(f"openai:{model}", cache=shared_cache("embed", RESULTS))
    ids = [str(d) for d in corpus.doc_id]
    matrix = embedder.encode(corpus.desc.fillna("").tolist(), "document")
    qmat = embedder.encode(queries.text.tolist(), "query")
    print(f"{model}: {embedder.cost.tokens_in:,} new embedding tokens, {sorted(embedder.versions)}")
    return {q: top(matrix, ids, v) for q, v in zip(queries.id, qmat, strict=True)}


# ---- running ----------------------------------------------------------------------------------


def write_jsonl(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for q, rec in records.items():
            f.write(json.dumps({"query_id": q, **rec}) + "\n")


def read_jsonl(path: Path) -> dict[str, Ranking]:
    out = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        out[row["query_id"]] = [(d, s) for d, s in row["ranking"]]
    return out


def rerank_all(reranker, queries: list[Query], first: dict[str, Ranking], n: int, workers: int):
    done, start = [0], time.perf_counter()

    def one(q: Query):
        cands = first[q.id][:n]
        result = reranker.rerank(q, cands)
        check_reranked(cands, result.ranking)
        done[0] += 1
        if done[0] % 50 == 0:
            print(
                f"  {reranker.name}: {done[0]}/{len(queries)} ({time.perf_counter() - start:.0f}s)"
            )
        rec = {
            "ranking": merge_reranked(first[q.id], result.ranking)[:DEPTH],
            "trace": {"n": n, **result.trace},
        }
        return q.id, rec

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, queries))


def project(cached_llm, prompts: list[str], chars_per_token: float) -> tuple[int, float]:
    new = [p for p in set(prompts) if not cached_llm.is_cached(p)]
    return len(new), sum(len(p) for p in new) / chars_per_token


# ---- evaluation -------------------------------------------------------------------------------


def query_metrics(ranking: Ranking, grades: dict[str, int]) -> dict[str, float]:
    return {
        "ndcg10": ndcg_at_k(ranking, grades, 10),
        "r1": recall_at_k(ranking, grades, 1),
        "r10": recall_at_k(ranking, grades, 10),
        "judged10": judged_at_k(ranking, grades, 10),
    }


def boot(x: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    if len(x) == 0:
        return (float("nan"),) * 3
    idx = np.random.default_rng(seed).integers(0, len(x), size=(n, len(x)))
    means = x[idx].mean(axis=1)
    return float(x.mean()), float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


def cell(m, signed: bool = False) -> str:
    f = "{:+.3f}" if signed else "{:.3f}"
    return f"{f.format(m[0])} [{f.format(m[1])}, {f.format(m[2])}]"


def tables(df: pd.DataFrame, label: str, systems: list[str]) -> list[str]:
    """df: one row per (query, system) with bin and metrics."""
    lines = [f"### {label}", ""]
    groups = [("all", df)] + [(b, df[df.bin == b]) for b in BINS]
    for metric in ("ndcg10", "r1", "r10"):
        lines.append(f"**{NAMES[metric]}**")
        lines.append("")
        head = "| bin | n | " + " | ".join(systems) + " | S2-S0 | S4-S0 | S2-S4 |"
        lines += [head, "|---|---|" + "---|" * (len(systems) + 3)]
        for name, part in groups:
            wide = part.pivot(index="query_id", columns="system", values=metric)
            if wide.empty:
                continue
            cells = [cell(boot(wide[s].to_numpy())) for s in systems]
            for a, b in (("S2", "S0"), ("S4", "S0"), ("S2", "S4")):
                cells.append(cell(boot((wide[a] - wide[b]).to_numpy()), signed=True))
            lines.append(f"| {name} | {len(wide)} | " + " | ".join(cells) + " |")
        lines.append("")
    wide = df.pivot(index="query_id", columns="system", values="judged10")
    lines.append(
        "Judged@10 (all queries): "
        + "; ".join(f"{s} {cell(boot(wide[s].to_numpy()))}" for s in systems)
    )
    lines.append("")
    return lines


def evaluate(qrels, bins, splits_of: dict[str, str], emb_keys: list[str]) -> list[str]:
    lines: list[str] = []
    summary = {}
    for emb in emb_keys:
        base = OUT / emb
        systems = [s for s in SYSTEMS if (base / f"{s}.jsonl").exists()]
        rows = []
        for s in systems:
            for q, ranking in read_jsonl(base / f"{s}.jsonl").items():
                rows.append(
                    {"query_id": q, "system": s, "bin": bins[q], "split": splits_of[q]}
                    | query_metrics(ranking, qrels[q])
                )
        df = pd.DataFrame(rows)
        if not {"S2", "S4"} <= set(systems):
            lines += [f"## {EMBEDDERS[emb]}: S0 only", ""]
            for split in ("test", "dev"):
                part = df[df.split == split]
                if len(part):
                    vals = "; ".join(
                        f"{NAMES[m]} {cell(boot(part[m].to_numpy()))}" for m in METRICS
                    )
                    lines.append(f"- {split} (n = {len(part)}): {vals}")
            lines.append("")
            continue
        # only queries every system has
        counts = df.groupby("query_id").system.nunique()
        df = df[df.query_id.isin(counts[counts == len(systems)].index)]
        lines += [f"## {EMBEDDERS[emb]} first stage", ""]
        for split, label in (("test", "Test (100)"), ("dev", "Dev (300)")):
            part = df[df.split == split]
            if len(part):
                lines += tables(part, f"{label}, n = {part.query_id.nunique()}", systems)
        if df.split.nunique() > 1:
            lines += tables(df, f"Test + dev pooled, n = {df.query_id.nunique()}", systems)
        summary[emb] = {
            split: df[df.split == split].groupby("system")[list(METRICS)].mean().to_dict("index")
            for split in df.split.unique()
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    return lines


# ---- main -------------------------------------------------------------------------------------


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--splits", default="test,dev")
    p.add_argument("--embedders", default="jasper,te3s", help="S0 for these")
    p.add_argument("--rerank", default="jasper", help="S2/S4 on these embedders' top N")
    p.add_argument("--rerank-splits", default="test,dev", help="S2/S4 on these splits")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--measure", type=int, default=0, help="run S2/S4 on this many test queries")
    p.add_argument("--token-budget", type=float, default=4.8e6, help="cap on new input tokens")
    p.add_argument("--dry-run", action="store_true", help="project only")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--inner", type=int, default=4)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()

    queries, corpus, descriptions, qrels, bins = load()
    splits_of = dict(zip(queries.id, queries.split))
    emb_keys = [e for e in args.embedders.split(",") if e]
    OUT.mkdir(parents=True, exist_ok=True)

    if not args.eval_only:
        qs = queries[queries.split.isin(args.splits.split(","))]
        first: dict[str, dict[str, Ranking]] = {}
        for emb in emb_keys:
            first[emb] = s0_jasper(qs) if emb == "jasper" else s0_openai(qs, corpus, EMBEDDERS[emb])
            if args.measure == 0:
                write_jsonl(
                    OUT / emb / "S0.jsonl",
                    {q: {"ranking": r} for q, r in first[emb].items()},
                )
        if "jasper" in first:  # check against the pilot's test S0
            test_ids = [q for q in first["jasper"] if splits_of[q] == "test"]
            nd = np.mean([ndcg_at_k(first["jasper"][q], qrels[q], 10) for q in test_ids])
            print(f"Jasper S0 test nDCG@10 {nd:.4f} (pilot 0.752)")

        lock = threading.Lock()
        paid: dict[str, list] = {"luna": [], "sol": []}
        errors: list[str] = []
        _, luna = paid_model(SPLIT_SPEC, paid["luna"], errors, lock)
        sol_c, sol = paid_model(VERIFIER_SPEC, paid["sol"], errors, lock)
        splitter = QuerySplitter(luna, SPLIT_PROMPT.read_text())
        desc_t, rerank_t = DESC_PROMPT.read_text(), RERANK_PROMPT.read_text()

        rq = qs[qs.split.isin(args.rerank_splits.split(","))]
        run_q = [Query(r.id, r.text, r.cluster) for r in rq.itertuples()]
        if args.measure:
            run_q = run_q[:: max(1, len(run_q) // args.measure)][: args.measure]
        with ThreadPoolExecutor(args.workers * args.inner) as pool:
            splits = dict(
                zip([q.id for q in run_q], pool.map(lambda q: splitter(q.text)[0], run_q))
            )

        # exact prompts of every check, for the projection
        prompts: list[str] = []
        for emb in args.rerank.split(","):
            for q in run_q:
                cons = numbered(splits[q.id])
                for d, _ in first[emb][q.id][: args.n]:
                    desc = descriptions.get(d, "") or "(no description)"
                    prompts.append(
                        desc_t.replace("{query}", q.text)
                        .replace("{constraints}", cons)
                        .replace("{description}", desc)
                    )
                    prompts.append(
                        fill(rerank_t, q.text, DESC_EVIDENCE.replace("{description}", desc))
                    )
        cpt_file = OUT / "chars_per_token.json"
        cpt = (
            json.loads(cpt_file.read_text())["chars_per_token"]
            if cpt_file.exists()
            else CHARS_PER_TOKEN_DEFAULT
        )
        n_new, tok_new = project(sol_c, prompts, cpt)
        spent = sum(c.tokens_in for _, c in paid["luna"])
        print(
            f"split: {len(paid['luna'])} new calls, {spent:,} in. Verifier/rerank: {len(prompts)} "
            f"calls, {n_new} uncached, projected {tok_new:,.0f} new input tokens "
            f"({cpt:.2f} chars/token); budget {args.token_budget:,.0f}"
        )
        if args.dry_run:
            return
        if tok_new + spent > args.token_budget:
            sys.exit("projected new input tokens exceed --token-budget")

        verifier = ConstraintVerifier(
            "S2", splitter, DescriptionCheck(sol, desc_t, descriptions), args.inner
        )
        holistic = HolisticReranker("S4", DescriptionScore(sol, rerank_t, descriptions), args.inner)
        for emb in args.rerank.split(","):
            for system in (verifier, holistic):
                recs = rerank_all(system, run_q, first[emb], args.n, args.workers)
                if not args.measure:
                    write_jsonl(OUT / emb / f"{system.name}.jsonl", recs)

        sol_in = sum(c.tokens_in for _, c in paid["sol"])
        chars = sum(n for n, _ in paid["sol"])
        if paid["sol"]:
            cpt_file.write_text(json.dumps({"chars_per_token": chars / sol_in}))
        cost = {
            k: {
                "calls": len(v),
                "tokens_in": sum(c.tokens_in for _, c in v),
                "tokens_out": sum(c.tokens_out for _, c in v),
                "models": sorted({c.model_version for _, c in v if c.model_version}),
            }
            for k, v in paid.items()
        }
        cost["errors"] = errors[:20]
        cost["n_errors"] = len(errors)
        cost["projected_new_in"] = tok_new
        tag = "measure" if args.measure else "run"
        log = OUT / "cost.json"
        prev = json.loads(log.read_text()) if log.exists() else []
        log.write_text(json.dumps(prev + [{"tag": tag, "time": time.ctime(), **cost}], indent=1))
        print(json.dumps(cost, indent=1))
        if args.measure:
            return

    lines = evaluate(qrels, bins, splits_of, emb_keys)
    (OUT / "tables.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
