"""Rebuttal: a second first-stage embedder. PAID (embeddings only, about US$0.07).

S0 and S1 are rebuilt over a second DESC index embedded with text-embedding-3-large (S0-L,
S1-L), with the settings of pipeline/run_systems.py (list depth 500 = configs/mc_full.yaml
`depth`, RRF constant 60, the saved splits results/full/runs/<set>/splits.json). The large index
gets its own store (results/_indexes/<hash of settings with the large embedder>); the paper's
text-embedding-3-small index is never touched. Embeddings go through the shared cache
(results/_cache/embed.sqlite), so a rerun is free.

    uv run --env-file .env python pipeline/embedder_large.py \
        --sources-file results/full/select/sources100.json

The expanded labels stay fixed (data/mc/full/judge/qrels_expanded.jsonl): their pool is the top
10 of the paper's systems, all of which start from the small index, so a top-10 video of S0-L or
S1-L that was never judged counts as non-relevant. The size of that bias is reported four ways:

- unjudged top-10 slots per system (paper/fusion_variants.py `unjudged`);
- condensed-list nDCG@10 (Sakai 2007): unjudged candidates are dropped from every system's
  ranking before scoring, so no system is penalised for them;
- expected missed positives: unjudged slots x the rate at which judged-late extras turned out
  relevant (FN_RATE, results/full/review_checks/report.md);
- strict labels (source only), which need no judging.

Also source recall@50 (strict), the candidate pool S2@50 / S4@50 rerank.

Evaluation as paper/fusion_variants.py (shared code). Outputs:
results/full/rebuttal/embedder_large/{report.md, summary.json, runs/<set>/<system>.jsonl}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


SV = _load(ROOT / "paper/fusion_variants.py", "_el_s1v")
FS, PV = SV.FS, SV.PV

from src.cache import shared_cache
from src.datasets.mc import description_index, index_settings, index_store
from src.embedders import MAX_INPUT_TOKENS, _cl100k, make_embedder, truncate_tokens
from src.indexes.base import top_k
from src.systems.decompose import fuse

OUT = Path("results/full/rebuttal/embedder_large")
LARGE = "openai:text-embedding-3-large"
PRICE_PER_M = 0.13  # US$ per 1M input tokens, OpenAI pricing page, checked 2026-11-09
TOKEN_CAP = 1.5e6  # stop before paying for more than this (projection about 0.5M)
FN_RATE = 5 / 812  # judged-late k>=3 extras found relevant (results/full/review_checks/report.md)
NEW = ["S0-L", "S1-L"]
RRF_K = SV.RRF_K


def large_index(E: dict, embedder) -> tuple[list[str], np.ndarray, Path]:
    """The DESC index of the paper's corpus and descriptions, embedded with LARGE."""
    S, ctx = E["S"], E["ctx"]
    cfg, options = S.cfg, ctx["options"]
    settings = index_settings(
        ctx["corpus"].doc_ids,
        LARGE,
        cfg["describer"],
        cfg["describe_prompt"],
        cfg["n_frames"],
        options["frame_policy"],
        options["labels"],
    )
    store = index_store(cfg["results_dir"], settings)
    assert store != ctx["store"], store
    description_index(ctx["corpus"], ctx["descriptions"], embedder, store)
    (store / "settings.json").write_text(json.dumps(settings, indent=1) + "\n")
    ids = json.loads((store / "DESC.ids.json").read_text())
    return ids, np.load(store / "DESC.npy"), store


def uncached_tokens(embedder, texts: list[str]) -> int:
    """cl100k tokens of the distinct non-empty texts not yet in the cache (what would be paid)."""
    n = 0
    for t in dict.fromkeys(texts):
        if not t.strip():
            continue
        sent = truncate_tokens(t, MAX_INPUT_TOKENS)
        if embedder.cache.get(embedder.namespace, sent) is None:
            n += len(_cl100k().encode(sent, disallowed_special=()))
    return n


def retrieve(ids, matrix, qv, cvs, depth) -> tuple[list, list]:
    """(S0 list, S1 list) as pipeline/run_systems.py stage_retrieve."""
    q_list = top_k(ids, matrix @ qv, depth)
    c_lists = [top_k(ids, matrix @ v, depth) for v in cvs]
    return q_list, fuse(c_lists, RRF_K)


def condensed(ranks: dict, E: dict) -> dict:
    """Every ranking with its never-judged candidates removed (the source is always judged)."""
    judged = E["judged"]
    out: dict = {}
    for name in SV.SETS:
        src = {r["query_id"]: r["source"] for r in E["sets"][name]}
        out[name] = {
            s: {
                q: [(d, v) for d, v in r if d == src[q] or d in judged.get(src[q], {})]
                for q, r in rk.items()
                if q in src
            }
            for s, rk in ranks[name].items()
        }
    return out


def source_recall(ranks: dict, E: dict, systems: list[str], pairs: list[tuple]) -> dict:
    """Source in the top 50 / top 10 (strict), means and paired CIs per set."""
    out: dict = {}
    for name in SV.SETS:
        rows = E["sets"][name]
        sc = {
            s: {
                r["query_id"]: {
                    f"src@{n}": float(
                        r["source"] in {d for d, _ in ranks[name][s][r["query_id"]][:n]}
                    )
                    for n in (10, 50)
                }
                for r in rows
                if r["query_id"] in ranks[name][s]
            }
            for s in systems
        }
        out[name] = {
            "means": {
                s: {m: float(np.mean([v[m] for v in sc[s].values()])) for m in ("src@10", "src@50")}
                for s in systems
            },
            "paired": {
                f"{a} - {b}": {
                    m: PV.paired(sc[a], sc[b], rows, m, None, E["S"].n_boot)["all"]
                    for m in ("src@10", "src@50")
                }
                for a, b in pairs
            },
        }
    return out


def overlap(ranks: dict, E: dict) -> dict:
    """Mean share of S0-L's (S1-L's) top 10 also in S0's (S1's) top 10."""
    out = {}
    for name in SV.SETS:
        for new, old in (("S0-L", "S0"), ("S1-L", "S1")):
            shares = [
                len({d for d, _ in ranks[name][new][q][:10]} & {d for d, _ in ranks[name][old][q][:10]})
                / 10
                for q in ranks[name][new]
            ]
            out[f"{name}/{new} vs {old}"] = float(np.mean(shares))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--sources-file", help="the test set (as systems.py --sources-file)")
    ap.add_argument("--dry-run", action="store_true", help="report the tokens to pay and stop")
    args = ap.parse_args(argv)
    t0 = time.time()
    E = SV.load_eval(args.n_boot, args.sources_file)
    check = SV.sanity_summary_only(E) if args.sources_file else SV.sanity(E)
    print("sanity: S1 - S0 and S4@50 - S0 reproduce results/full/eval/summary.json")
    S, ctx = E["S"], E["ctx"]

    splits = {n: json.loads((S.runs / n / "splits.json").read_text()) for n in SV.SETS}
    q_texts = [
        t
        for n in SV.SETS
        for r in E["sets"][n]
        for t in (r["query"], *splits[n][r["query_id"]])
    ]
    doc_texts = [ctx["descriptions"].get(d, "") for d in ctx["corpus"].doc_ids]
    embedder = make_embedder(
        LARGE, cache=shared_cache("embed", S.cfg["results_dir"]), max_tokens=S.cfg["embed_max_tokens"]
    )
    to_pay = uncached_tokens(embedder, doc_texts + q_texts)
    print(f"uncached tokens: {to_pay:,} (about US${to_pay / 1e6 * PRICE_PER_M:.3f})")
    if args.dry_run:
        return
    if to_pay > TOKEN_CAP:
        sys.exit(f"{to_pay:,} tokens to pay, over the cap {TOKEN_CAP:,.0f}")

    ids_l, mat_l, store_l = large_index(E, embedder)
    qvec = dict(zip(q_texts, embedder.encode(q_texts, "query"), strict=True))
    paid = {
        "calls": embedder.cost.llm_calls,
        "tokens": embedder.cost.tokens_in,
        "usd": embedder.cost.tokens_in / 1e6 * PRICE_PER_M,
        "model_versions": sorted(embedder.versions),
        "truncated_inputs": len(embedder.truncated),
    }
    print(json.dumps({"store": str(store_l), "paid": paid}, indent=1))

    # check the retrieval code: the small index and cached small vectors reproduce saved S0 / S1
    store_s = ctx["store"]
    ids_s = json.loads((store_s / "DESC.ids.json").read_text())
    mat_s = np.load(store_s / "DESC.npy")
    small = SV.CachedVectors(S.cfg["embedder"], S.cfg["results_dir"])
    depth = min(S.depth, len(ids_l))
    ranks = {n: dict(E["ranks"][n]) for n in SV.SETS}
    checks: dict = {}
    for name in SV.SETS:
        by_id = {r["query_id"]: r for r in E["sets"][name]}
        recs: dict = {s: {} for s in NEW}
        same = {"S0": 0, "S1": 0}
        n_checked = 0
        for r in E["sets"][name]:
            q, cons = r["query_id"], splits[name][r["query_id"]]
            l0, l1 = retrieve(ids_l, mat_l, qvec[r["query"]], [qvec[c] for c in cons], depth)
            recs["S0-L"][q] = {"ranking": l0, "cost": {}, "trace": {}}
            recs["S1-L"][q] = {"ranking": l1, "cost": {}, "trace": {"constraints": cons}}
            sv = [small.get(t) for t in (r["query"], *cons)]
            if all(v is not None for v in sv):
                n_checked += 1
                s0, s1 = retrieve(ids_s, mat_s, sv[0], sv[1:], depth)
                for s, mine in (("S0", s0), ("S1", s1)):
                    saved = E["ranks"][name][s][q]
                    same[s] += [d for d, _ in mine[:10]] == [d for d, _ in saved[:10]]
        checks[name] = {"queries": len(recs["S0-L"]), "checked": n_checked, "top10_identical": same}
        if any(v != n_checked for v in same.values()) or n_checked == 0:
            sys.exit(f"retrieval does not reproduce the saved S0 / S1 top 10: {checks}")
        for s in NEW:
            SV.write_runs(OUT, name, s, recs[s], by_id)
            ranks[name][s] = {q: rec["ranking"] for q, rec in recs[s].items()}
    print("check:", json.dumps(checks))

    systems = ["S0", "S1", *NEW, "S2@50", "S4@50"]
    pairs = [("S0-L", "S0"), ("S1-L", "S1"), ("S1", "S0"), ("S1-L", "S0-L")]
    res = SV.evaluate(E, ranks, systems, pairs, NEW)
    cond = SV.evaluate(E, condensed(ranks, E), systems, pairs, NEW)
    missed = {
        n: {
            s: {"unjudged": t["all"]["unjudged"], "expected_missed": t["all"]["unjudged"] * FN_RATE}
            for s, t in res["unjudged"][n].items()
        }
        for n in SV.SETS
    }
    summary = {
        "embedder": LARGE,
        "store": str(store_l),
        "paid": paid,
        "check": checks,
        "sanity": check,
        "overlap_top10": overlap(ranks, E),
        "source_recall": source_recall(ranks, E, systems, pairs),
        "expected_missed": missed,
        "fn_rate": FN_RATE,
        "results": res,
        "condensed": {"labels": {"expanded": cond["labels"]["expanded"]}, "n": cond["n"]},
        "sources_file": args.sources_file,
        "n_boot": E["S"].n_boot,
        "seconds": time.time() - t0,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
    (OUT / "report.md").write_text("\n".join(report(summary, systems, pairs)) + "\n")
    print(f"wrote {OUT}/report.md and summary.json ({time.time() - t0:.0f} s)")


def report(sm: dict, systems: list[str], pairs: list[tuple]) -> list[str]:
    md = [
        "# Second first-stage embedder: text-embedding-3-large",
        "",
        f"Index store {sm['store']}; paid {sm['paid']['tokens']:,} tokens in {sm['paid']['calls']} "
        f"calls (about US${sm['paid']['usd']:.3f}); sources file {sm['sources_file']}.",
        f"Retrieval check (small index, cached vectors, saved S0 / S1 top 10): {json.dumps(sm['check'])}.",
        "",
        "## Top-10 overlap with the small embedder",
        "",
        *(f"- {k}: {v:.1%}" for k, v in sm["overlap_top10"].items()),
        "",
        "## Source in the top 10 / top 50 (strict)",
        "",
        "| system | " + " | ".join(f"{SV.SET_LABEL[n]} @10 | {SV.SET_LABEL[n]} @50" for n in SV.SETS) + " |",
        "|---|" + "---:|" * 4,
    ]
    for s in systems:
        vals = [
            f"{sm['source_recall'][n]['means'][s][m]:.3f}" for n in SV.SETS for m in ("src@10", "src@50")
        ]
        md.append(f"| {s} | " + " | ".join(vals) + " |")
    md += ["", "| pair | set | src@10 | src@50 |", "|---|---|---:|---:|"]
    for a, b in pairs:
        for n in SV.SETS:
            p = sm["source_recall"][n]["paired"][f"{a} - {b}"]
            md.append(f"| {a} − {b} | {SV.SET_LABEL[n]} | {SV.ci(p['src@10'])} | {SV.ci(p['src@50'])} |")
    md += ["", "## Fixed labels (expanded and strict)", ""]
    md += SV.tables(sm["results"], systems, pairs, NEW)
    md += [
        "## Condensed lists (unjudged candidates removed before scoring), expanded labels",
        "",
    ]
    cond = {"labels": sm["condensed"]["labels"], "unjudged": {n: {} for n in SV.SETS}}
    cmd = SV.tables(
        {"labels": {"expanded": cond["labels"]["expanded"], "strict": cond["labels"]["expanded"]}, "unjudged": cond["unjudged"]},
        systems,
        pairs,
        NEW,
    )
    # keep only the expanded blocks (tables() always writes both label kinds)
    keep, on = [], False
    for line in cmd:
        if line.startswith("### "):
            on = "expanded" in line
        if on:
            keep.append(line)
    md += keep + [
        "",
        "## Expected missed positives among unjudged slots",
        "",
        f"Unjudged top-10 slots x {sm['fn_rate']:.4f} (rate at which judged-late extras were relevant).",
        "",
        "| set | system | unjudged slots | expected missed positives |",
        "|---|---|---:|---:|",
    ]
    for n in SV.SETS:
        for s, t in sm["expected_missed"][n].items():
            md.append(f"| {SV.SET_LABEL[n]} | {s} | {t['unjudged']} | {t['expected_missed']:.1f} |")
    return md


if __name__ == "__main__":
    main()
