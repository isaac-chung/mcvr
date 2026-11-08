"""Splitter check on the test set (rebuttal): how well the full run's splits (split_v1, luna) match
the mined facts, per set and k, and whether over-splitting explains S1's loss to S0.

Reuses the dev check unchanged (dev/verifiers_dev.py: token_overlap, match_split with
prompts/split_match_v1.txt, split_stats, score_system, paired cluster bootstrap over sources).
Inputs: results/full/runs/<set>/{splits.json, S0.jsonl, S1.jsonl}, data/mc/full/queries*.jsonl
and data/mc/full/judge/qrels_expanded.jsonl. Free parts: constraint counts vs k, token-overlap
recall@0.5, and S1 - S0 nDCG@10 (strict and expanded) for queries split into exactly k vs more
than k constraints. Paid part (--judge): one split_match_v1 call per query with
openai:gpt-6-luna@low through the shared LLM cache (same cache key as the dev check), after a
projection (stops above --token-cap) and a canary; any 429 / quota error stops the run.

    uv run python paper/splitter_errors.py                               # free only
    uv run --env-file .env python paper/splitter_errors.py --judge --canary-only
    uv run --env-file .env python paper/splitter_errors.py --judge
    nohup uv run --env-file .env python -u paper/splitter_errors.py --judge --rpm 9 \
        > results/full/rebuttal/split_check/judge.log 2>&1 &    # throttled (10 RPM key)

Outputs: results/full/rebuttal/split_check/{report.md, summary.json, matches.json}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from src.cache import shared_cache
from src.datasets.mc import read_jsonl
from src.llm import make_llm
from src.systems.decompose import SPLIT_SPEC
from src.systems.verify import numbered
from src.types import Cost

ROOT = Path(__file__).resolve().parents[1]
RUNS = Path("results/full/runs")
DATA = Path("data/mc/full")
QRELS = DATA / "judge" / "qrels_expanded.jsonl"
OUT = Path("results/full/rebuttal/split_check")
SETS = {"natural": "queries.jsonl", "twin": "queries_twin.jsonl"}
LABEL = {"natural": "nested (natural)", "twin": "twin"}
JUDGE_SPEC = SPLIT_SPEC  # openai:gpt-6-luna@low, as in the dev check
DEV_IN_PER_CALL = 0.10e6 / 366  # results/smoke/pilot_report.md: luna split match, 366 calls
CHARS_PER_TOKEN = 3.6
N_BOOT = 10_000


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(
        f"_split_check_{name}", ROOT / "dev" / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PV = load_script("verifiers_dev")


# ---- data -------------------------------------------------------------------------------------


def load(sources_file: str | None = None) -> tuple[dict, dict, dict]:
    """({set: rows}, {set: {qid: constraints}}, {set: {system: {qid: ranking}}}), rows limited
    to the queries the full run split (and, with sources_file as systems.py --sources-file, to
    the queries of the drawn sources)."""
    keep = set(json.loads(Path(sources_file).read_text())["sources"]) if sources_file else None
    rows, splits, ranks = {}, {}, {}
    for name, file in SETS.items():
        splits[name] = json.loads((RUNS / name / "splits.json").read_text())
        rows[name] = [r for r in read_jsonl(DATA / file) if r["query_id"] in splits[name]]
        if keep is not None:
            rows[name] = [r for r in rows[name] if r["source"] in keep]
        ids = {r["query_id"] for r in rows[name]}
        ranks[name] = {
            s: {
                ln["query_id"]: [(d, x) for d, x in ln["ranking"]]
                for ln in read_jsonl(p)
                if ln["query_id"] in ids
            }
            for s in ("S0", "S1")
            if (p := RUNS / name / f"{s}.jsonl").exists()
        }
    return rows, splits, ranks


def prompt_for(template: str, row: dict, constraints: list[str]) -> str:
    """The prompt PV.match_split builds (kept identical, so cache keys match)."""
    return template.replace("{facts}", numbered([f["text"] for f in row["facts"]])).replace(
        "{constraints}", numbered(constraints)
    )


# ---- paid: the split_match_v1 judge ------------------------------------------------------------


class Stop(Exception):
    pass


class Guard:
    """Wraps the cached model. Without rpm, the first rate-limit / quota error stops every later
    call (no retry loop). With rpm (one worker), uncached calls are spaced 60 / rpm s apart; a
    quota error (insufficient_quota, credit) stops at once, an RPM 429 sleeps 60 s and tries again
    (at most MAX_429 in a row, then stops). Other errors are logged and answered with "" (all
    facts count as none), as PV.SafeCalls, and nothing is cached for them."""

    MAX_429 = 5

    def __init__(self, inner, rpm: float = 0.0):
        self.inner, self.errors, self.stopped = inner, [], threading.Event()
        self.rpm, self.last, self.paid, self.in_a_row = rpm, 0.0, 0, 0

    def complete(self, prompt: str):
        if self.stopped.is_set():
            raise Stop("stopped after a rate-limit / quota error")
        if self.rpm and self.inner.is_cached(prompt):
            return self.inner.complete(prompt)
        while True:
            if self.rpm:
                time.sleep(max(0.0, self.last + 60.0 / self.rpm - time.monotonic()))
                self.last = time.monotonic()
            try:
                out = self.inner.complete(prompt)
                self.in_a_row = 0
                self.paid += 1
                if self.rpm and self.paid % 20 == 0:
                    print(f"[judge] {self.paid} paid calls ok", flush=True)
                return out
            except Exception as error:
                text = repr(error)[:300]
                self.errors.append(text)
                low = text.lower()
                quota = any(
                    m in low for m in ("insufficient_quota", "credit", "quota", "per day", "(rpd)")
                )
                limited = "429" in text or "ratelimit" in low or "rate limit" in low
                if quota or (limited and not self.rpm):
                    self.stopped.set()
                    raise Stop(text) from error
                if not limited:
                    return "", Cost()
                self.in_a_row += 1
                if self.in_a_row > self.MAX_429:
                    self.stopped.set()
                    raise Stop(f"{self.in_a_row} consecutive 429s: {text}") from error
                print(f"[judge] RPM 429 ({self.in_a_row} in a row), sleeping 60 s", flush=True)
                self.errors.pop()  # retried, not a lost answer
                time.sleep(60)


def judge_models(retries: int = 2):
    cache = shared_cache("llm", "results")
    cached = make_llm(JUDGE_SPEC, cache)
    cached.inner.client = cached.inner.client.with_options(max_retries=retries, timeout=300)
    log: list = []
    cached.inner = PV.Metered(cached.inner, log)
    return cached, log


def run_judge(guard, template, items, workers) -> dict:
    """{qid: match} for items [(qid, row, constraints)]; raises Stop on a rate-limit error."""

    def one(item):
        qid, row, cons = item
        return qid, PV.match_split(guard, template, [f["text"] for f in row["facts"]], cons)

    with ThreadPoolExecutor(workers) as pool:
        return dict(pool.map(one, items))


def spend(log: list) -> dict:
    total = sum(log, Cost())
    return {
        "calls": total.llm_calls,
        "tokens_in": total.tokens_in,
        "tokens_out": total.tokens_out,
        "model_version": total.model_version,
    }


# ---- analysis ---------------------------------------------------------------------------------


def split_class(n_cons: int, k: int) -> str:
    return "=k" if n_cons == k else (">k" if n_cons > k else "<k")


def s1_by_split(rows, splits, ranks, qrels, matches, n_boot) -> dict:
    """S1 - S0 (nDCG@10, R@1) per k and overall, for all queries and per split class (=k, >k;
    with matches also LLM exact 1:1 vs not), paired cluster bootstrap over sources."""
    sa = PV.score_system(ranks["S1"], qrels)
    sb = PV.score_system(ranks["S0"], qrels)
    out: dict = {}
    for k in ["all", *sorted({r["k"] for r in rows})]:
        rk = [r for r in rows if k == "all" or r["k"] == k]
        groups = {
            "all": rk,
            "=k": [r for r in rk if split_class(len(splits[r["query_id"]]), r["k"]) == "=k"],
            ">k": [r for r in rk if split_class(len(splits[r["query_id"]]), r["k"]) == ">k"],
        }
        if matches:
            ex = {q for q, m in matches.items() if exact(m)}
            groups["exact 1:1"] = [r for r in rk if r["query_id"] in ex]
            groups["not exact"] = [
                r for r in rk if r["query_id"] in matches and r["query_id"] not in ex
            ]
        cell = {}
        for g, rs in groups.items():
            if not rs:
                cell[g] = {"n": 0}
                continue
            cell[g] = {
                "n": len(rs),
                "S0_ndcg@10": float(np.mean([sb[r["query_id"]]["ndcg@10"] for r in rs])),
                "S1_ndcg@10": float(np.mean([sa[r["query_id"]]["ndcg@10"] for r in rs])),
                **{
                    m: PV.paired(sa, sb, rs, m, None, n_boot)["all"]
                    for m in ("ndcg@10", "recall@1")
                },
            }
        out[str(k)] = cell
    return out


def exact(m: dict) -> bool:
    return (
        len(m["constraints"]) == len(m["facts"])
        and all(x == "full" for x in m["facts"])
        and all(c is not None for c in m["constraints"])
    )


def mix_share(rows, splits, ranks, qrels, k=1) -> dict:
    """Decomposition of the k-level S1 - S0 mean into share x mean per split class."""
    sa, sb = PV.score_system(ranks["S1"], qrels), PV.score_system(ranks["S0"], qrels)
    rk = [r for r in rows if r["k"] == k]
    out = {}
    for c in ("=k", ">k", "<k"):
        rs = [r for r in rk if split_class(len(splits[r["query_id"]]), r["k"]) == c]
        d = [sa[r["query_id"]]["ndcg@10"] - sb[r["query_id"]]["ndcg@10"] for r in rs]
        out[c] = {"n": len(rs), "contribution": float(sum(d) / len(rk)) if rk else 0.0}
    return out


# ---- report -----------------------------------------------------------------------------------


def f2(x):
    return "-" if x is None else f"{x:.2f}"


def ci(d: dict) -> str:
    return f"{d['mean']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


def stats_table(stats: dict) -> list[str]:
    head = (
        "| set | k | n | constraints parsed (mean) | = k | > k | < k | facts fully covered | "
        "full or partial | constraints that state a fact | exact 1:1 split | constraints per "
        "covered fact | token-overlap recall@0.5 |"
    )
    lines = [head, "|" + "---|" * 13]
    for name, st in stats.items():
        for g, s in st.items():
            lines.append(
                f"| {LABEL[name]} | {g} | {s['n']} | {s['mean_constraints']:.2f} | "
                f"{s['share_eq_k']:.2f} | {s['share_gt_k']:.2f} | {s['share_lt_k']:.2f} | "
                f"{f2(s.get('llm_fact_full'))} | {f2(s.get('llm_fact_full_or_partial'))} | "
                f"{f2(s.get('llm_constraint_precision'))} | {f2(s.get('llm_exact_split'))} | "
                f"{f2(s.get('llm_constraints_per_matched_fact'))} | "
                f"{s['overlap_fact_recall@0.5']:.2f} |"
            )
    return lines


def s1_table(s1: dict) -> list[str]:
    lines = [
        "| set | labels | k | split | n | S0 | S1 | S1 - S0 nDCG@10 | S1 - S0 R@1 |",
        "|" + "---|" * 9,
    ]
    for name, by_lab in s1.items():
        for lab, by_k in by_lab.items():
            for k, cell in by_k.items():
                for g, d in cell.items():
                    if not d["n"]:
                        continue
                    lines.append(
                        f"| {LABEL[name]} | {lab} | {k} | {g} | {d['n']} | "
                        f"{d['S0_ndcg@10']:.3f} | {d['S1_ndcg@10']:.3f} | {ci(d['ndcg@10'])} | "
                        f"{ci(d['recall@1'])} |"
                    )
    return lines


def report(res: dict, takeaways: list[str]) -> str:
    L = ["# Splitter check on the test set (split_v1, luna)", ""]
    L += ["## Key takeaways", "", *[f"- {t}" for t in takeaways], ""]
    L += [
        "## Splitter match against the mined facts",
        "",
        f"Splits: `results/full/runs/<set>/splits.json` (`split_v1`, `{SPLIT_SPEC}`). Judge: "
        f"`{res['judge_spec']}` with `prompts/split_match_v1.txt`, one call per query"
        + ("" if res["judged"] else " (not run: LLM columns are '-')")
        + ". The token-overlap column counts a fact as matched when its best token F1 against any "
        "parsed constraint is >= 0.5. Same code and columns as the dev table "
        "(docs/results_smoke.md section 5).",
        "",
        *stats_table(res["split_stats"]),
        "",
        "## S1 - S0 by split type",
        "",
        (
            "S1 = one search per parsed constraint, fused with RRF. `=k`: the splitter returned exactly "
            "k constraints; `>k`: more than k (over-split). With the judge, `exact 1:1` is the judge's "
            "exact one-to-one split. Paired differences with 95% cluster-bootstrap CIs over source "
            f"videos ({N_BOOT:,} resamples). Strict labels = the source only; expanded = "
            "`data/mc/full/judge/qrels_expanded.jsonl` (the eval's labels)."
        ),
        "",
        *s1_table(res["s1_minus_s0"]),
        "",
        (
            "Share of the k = 1 S1 - S0 nDCG@10 mean contributed by each split class (sum = the k = 1 "
            "mean):"
        ),
        "",
        "| set | labels | class | n | contribution |",
        "|---|---|---|---|---|",
    ]
    for name, by_lab in res["k1_contribution"].items():
        for lab, d in by_lab.items():
            for c, x in d.items():
                if x["n"]:
                    L.append(
                        f"| {LABEL[name]} | {lab} | {c} | {x['n']} | {x['contribution']:+.4f} |"
                    )
    L += ["", "## Spend", ""]
    sp = res.get("spend")
    if sp:
        L += [
            "| model | calls (paid) | tokens in | tokens out | errors |",
            "|---|---|---|---|---|",
            (
                f"| {res['judge_spec']} | {sp['calls']} | {sp['tokens_in']:,} | {sp['tokens_out']:,} | "
                f"{res['n_errors']} |"
            ),
            "",
            (
                f"Projection before the run: {res['projection']['tokens_in']:,.0f} input tokens for "
                f"{res['projection']['calls']} uncached calls (cap {res['projection']['cap']:,.0f}). "
                "Calls answered from the cache are not counted."
            ),
        ]
        if res.get("canary_spend"):
            c = res["canary_spend"]
            L += [
                "",
                (
                    f"Before this, a `--canary-only` run paid {c['calls']} calls ({c['tokens_in']:,} in, "
                    f"{c['tokens_out']:,} out); those answers are cached and not counted above."
                ),
            ]
    else:
        L += ["No full judge run (free parts only)."]
    if res.get("stopped") and not res["judged"]:
        st = res["stopped"]
        L += [
            "",
            (
                f"The judge run stopped on a rate-limit error after {st['spend']['calls']} paid calls "
                f"({st['spend']['tokens_in']:,} in, {st['spend']['tokens_out']:,} out): "
                f"`{st['error'][:120]}`"
            ),
        ]
    if res.get("canary_spend") and not sp:
        c = res["canary_spend"]
        L += [
            "",
            (
                f"Canary (`--canary-only`): {c['calls']} paid calls ({c['tokens_in']:,} in, "
                f"{c['tokens_out']:,} out)."
            ),
        ]
    return "\n".join(L) + "\n"


def takeaways_of(res: dict) -> list[str]:
    out = []
    st = res["split_stats"]
    n1, t1 = st["natural"][1], st["twin"][1]
    out.append(
        f"At k = 1 the splitter returns more than one constraint for {n1['share_gt_k']:.0%} of "
        f"nested and {t1['share_gt_k']:.0%} of twin queries (mean {n1['mean_constraints']:.2f} / "
        f"{t1['mean_constraints']:.2f}); overall {st['natural']['all']['share_gt_k']:.0%} / "
        f"{st['twin']['all']['share_gt_k']:.0%} are over-split and token-overlap recall@0.5 is "
        f"{st['natural']['all']['overlap_fact_recall@0.5']:.2f} / "
        f"{st['twin']['all']['overlap_fact_recall@0.5']:.2f}."
    )
    if res["judged"]:
        a, b = st["natural"]["all"], st["twin"]["all"]
        out.append(
            f"Judge: facts fully covered {a['llm_fact_full']:.2f} / {b['llm_fact_full']:.2f} "
            f"(nested / twin), constraints that state a fact {a['llm_constraint_precision']:.2f} / "
            f"{b['llm_constraint_precision']:.2f}, {a['llm_constraints_per_matched_fact']:.2f} / "
            f"{b['llm_constraints_per_matched_fact']:.2f} constraints per covered fact, exact 1:1 "
            f"{a['llm_exact_split']:.2f} / {b['llm_exact_split']:.2f}."
        )
    e = res["s1_minus_s0"]["natural"]["expanded"]["1"]
    s = res["s1_minus_s0"]["natural"]["strict"]["1"]
    out.append(
        f"Nested k = 1, expanded labels: S1 - S0 nDCG@10 is {ci(e['all']['ndcg@10'])} overall, "
        f"{ci(e['=k']['ndcg@10'])} when split into one constraint (n = {e['=k']['n']}) and "
        f"{ci(e['>k']['ndcg@10'])} when over-split (n = {e['>k']['n']}); strict: "
        f"{s['=k']['ndcg@10']['mean']:+.3f} vs {s['>k']['ndcg@10']['mean']:+.3f}."
    )
    return out


# ---- main -------------------------------------------------------------------------------------


def main() -> None:
    global OUT
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--judge", action="store_true", help="run the paid split_match_v1 judge")
    ap.add_argument("--canary", type=int, default=10, help="queries judged first as a canary")
    ap.add_argument("--canary-only", action="store_true")
    ap.add_argument("--token-cap", type=float, default=4e6, help="stop if projected input > cap")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument(
        "--rpm",
        type=float,
        nargs="?",
        const=9.0,
        default=0.0,
        help="throttle: one worker, at most this many paid calls per minute (default 9); "
        "sleeps 60 s on an RPM 429 (5 in a row stop the run), stops on a quota error",
    )
    ap.add_argument(
        "--sources-file",
        help="keep only the queries of these sources ({'sources': [...]}), as systems.py",
    )
    ap.add_argument("--out", help=f"output folder (default {OUT})")
    args = ap.parse_args()
    if args.out:
        OUT = Path(args.out)

    rows, splits, ranks = load(args.sources_file)
    template = PV.MATCH_PROMPT.read_text()
    OUT.mkdir(parents=True, exist_ok=True)
    matches_path = OUT / "matches.json"
    res: dict = {"judge_spec": JUDGE_SPEC, "judged": False, "n_errors": 0}
    if args.sources_file:
        res["sources_file"] = args.sources_file
        res["sources"] = len({r["source"] for n in SETS for r in rows[n]})
    matches: dict = {n: {} for n in SETS}

    if args.judge:
        cached, log = judge_models(0 if args.rpm else 2)
        guard = Guard(cached, args.rpm)
        if args.rpm:
            args.workers = 1
        items = [(n, r["query_id"], r, splits[n][r["query_id"]]) for n in SETS for r in rows[n]]
        new = [it for it in items if not cached.is_cached(prompt_for(template, it[2], it[3]))]
        chars = sum(len(prompt_for(template, it[2], it[3])) for it in new)
        proj = max(chars / CHARS_PER_TOKEN, len(new) * DEV_IN_PER_CALL)
        res["projection"] = {"calls": len(new), "tokens_in": proj, "cap": args.token_cap}
        print(
            f"[judge] {len(new)} of {len(items)} calls to pay; projected input {proj:,.0f} tokens "
            f"(chars/{CHARS_PER_TOKEN}: {chars / CHARS_PER_TOKEN:,.0f}; dev rate: "
            f"{len(new) * DEV_IN_PER_CALL:,.0f}); cap {args.token_cap:,.0f}",
            flush=True,
        )
        if proj > args.token_cap:
            sys.exit("[judge] projection above the cap: stopping")
        canary = items[:: max(1, len(items) // args.canary)][: args.canary]
        try:
            done = run_judge(
                guard, template, [(q, r, c) for _, q, r, c in canary], 1 if args.rpm else 4
            )
            if guard.errors:
                sys.exit(f"[judge] canary errors: {guard.errors[:3]}")
            cost = spend(log)
            print(f"[judge] canary ok: {len(done)} queries, {cost}", flush=True)
            for q, m in done.items():
                print("   ", q, m)
            if args.canary_only:
                (OUT / "canary_spend.json").write_text(json.dumps(cost, indent=1))
                return
            for n in SETS:
                part = [(q, r, c) for nn, q, r, c in items if nn == n]
                matches[n] = run_judge(guard, template, part, args.workers)
        except Stop as error:
            print(f"[judge] STOPPED on a rate-limit / quota error: {error}", flush=True)
            print(f"[judge] spend so far: {spend(log)}")
            err = re.sub(r"org-\w+", "<org>", str(error))[:200]
            stop = {"error": err, "spend": spend(log)}
            (OUT / "stopped.json").write_text(json.dumps(stop, indent=1))
            sys.exit(1)
        res["judged"] = True
        res["spend"] = spend(log)
        res["errors"] = guard.errors[:50]
        res["n_errors"] = len(guard.errors)
        matches_path.write_text(json.dumps(matches, indent=1))
        (OUT / "stopped.json").unlink(missing_ok=True)
        print(f"[judge] done: {spend(log)}", flush=True)
    elif matches_path.exists():
        matches = json.loads(matches_path.read_text())
        res["judged"] = all(len(matches[n]) == len(rows[n]) for n in SETS)
        prev = OUT / "summary.json"
        if prev.exists():
            old = json.loads(prev.read_text())
            for key in ("spend", "projection", "n_errors", "errors"):
                if key in old:
                    res[key] = old[key]

    res["split_stats"] = {n: PV.split_stats(rows[n], splits[n], matches[n] or None) for n in SETS}
    res["s1_minus_s0"], res["k1_contribution"] = {}, {}
    for n in SETS:
        res["s1_minus_s0"][n], res["k1_contribution"][n] = {}, {}
        for lab, path in (("strict", None), ("expanded", QRELS)):
            qrels = PV.load_qrels(path, rows[n])
            res["s1_minus_s0"][n][lab] = s1_by_split(
                rows[n], splits[n], ranks[n], qrels, matches[n], args.n_boot
            )
            res["k1_contribution"][n][lab] = mix_share(rows[n], splits[n], ranks[n], qrels)
    if (OUT / "canary_spend.json").exists():
        res["canary_spend"] = json.loads((OUT / "canary_spend.json").read_text())
    if (OUT / "stopped.json").exists():
        res["stopped"] = json.loads((OUT / "stopped.json").read_text())
    takeaways = takeaways_of(res)
    res["takeaways"] = takeaways
    (OUT / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    (OUT / "report.md").write_text(report(res, takeaways))
    print(f"wrote {OUT}/report.md and {OUT}/summary.json")


if __name__ == "__main__":
    main()
