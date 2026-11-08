"""Two separately reported Gemini checks with gemini-3.8-flash (OpenAI out of credits). PAID
(calibrate only; the other stages read existing outputs). Nothing here writes into
results/full/runs or data/mc/full/judge.

    # judge fill with a second Gemini judge (gemini-3.5-flash-lite + gemini-3.8-flash@low, both
    # must agree), into its own folder seeded with a copy of data/mc/full/judge/judged.jsonl:
    uv run --env-file .env python pipeline/judge_labels.py --stage run \
        --config data/mc/full/judge_fill_gemini38/config.yaml \
        --out data/mc/full/judge_fill_gemini38/run \
        --fill-queries results/full/select/frames_subset.json
    # calibration: gemini-3.8-flash@low on 200 seeded already-labelled pool pairs (vs astra)
    uv run --env-file .env python paper/gemini_frame_verifier.py --stage calibrate
    # qrels variants, fill effect on the frame comparisons, report
    uv run python paper/gemini_frame_verifier.py --stage fill
    # S5 with gemini-3.8-flash as frame verifier (after systems.py --frame-verifier ...)
    uv run python paper/gemini_frame_verifier.py --stage s5 --s5-system S5_gemini-3.8-flash-low

Outputs: data/mc/full/judge_fill_gemini38/{qrels_*.jsonl, calibration/, fill_summary.json,
report.md}; results/full/frame_verifier_gemini38/{summary.json, report.md}.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


JD = _load(ROOT / "pipeline/judge_labels.py", "_g38_judge")
FS = JD.FS
jf = JD.jf

from src.grounding_v2 import agreement_table, cohen_kappa
from src.manifest import write_manifest
from src.types import Cost

G35, G38, ASTRA = (
    "gemini:gemini-3.5-flash-lite",
    "gemini:gemini-3.8-flash@low",
    "openai:gpt-6-astra",
)
MODEL = "gemini-3.8-flash"
FILL = Path("data/mc/full/judge_fill_gemini38")
MAIN_JUDGE = Path("data/mc/full/judge")
FV = Path("results/full/frame_verifier_gemini38")
PRICE = {  # USD per 1M tokens, ai.google.dev/gemini-api/docs/pricing (read 2026-10-24)
    G38: {"in": 0.75, "out": 3.75, "note": "through 2026-12-31; $1.50 / $7.50 from 2027-01-01"},
    G35: {"in": 0.30, "out": 2.50, "note": ""},
}
FRAME_SYS = ("S0", "S4@50", "S3", "S5")
SUBSETS = {"twin-both": "twin", "nested sample": "natural"}


def settings(n_boot: int):
    S = FS.settings(
        argparse.Namespace(
            smoke=False, config=str(FILL / "config.yaml"), out=str(FILL / "run"), n_boot=n_boot
        )
    )
    S.runs = Path("results/full/runs")  # read-only
    return S


def dollars(spec: str, cost: Cost) -> float:
    p = PRICE[spec]
    return cost.tokens_in / 1e6 * p["in"] + cost.tokens_out / 1e6 * p["out"]


def write_qrels(path: Path, table: dict) -> None:
    with open(path, "w") as handle:
        handle.writelines(
            json.dumps({"query_id": q, "relevant": rel}) + "\n" for q, rel in table.items()
        )


# ---- calibration ------------------------------------------------------------------------------


def calibration_pairs(judged: dict, rows: list[dict], n: int, seed: str) -> list[tuple]:
    twins = {(r["source"], r.get("twin")) for r in rows}
    pool = sorted(
        (s, d) for s, cands in judged.items() for d in cands if d != s and (s, d) not in twins
    )
    random.Random(seed).shuffle(pool)
    return sorted(pool[:n])


def stage_calibrate(args) -> None:
    S = settings(args.n_boot)
    jcfg, rows, facts_of, _of_query, _ = JD.setup(S)
    main = JD.JP.read_judged(MAIN_JUDGE / "judged.jsonl")
    pairs = calibration_pairs(main, rows, args.sample, "gemini38-calibration|0")
    out = FILL / "calibration"
    out.mkdir(parents=True, exist_ok=True)
    stage = "calibration"
    FS.CAP_OF[stage] = stage
    FS.STAGES_OF_CAP[stage] = [stage]
    S.caps = {**S.caps, stage: args.cap}
    ledger = FS.Ledger(S, stage)
    paid: list = []
    prompt = jf.JUDGE_PROMPT
    judge = JD.Judge(
        S.cfg,
        {**jcfg, "models": [G38]},
        prompt.read_text(),
        max(0.0, args.cap - ledger.total),
        paid,
    )
    proj, calls = judge.projection(pairs, facts_of, check_cache=True)
    print(f"calibration: {len(pairs)} pairs, {calls} calls, {proj:,.0f} tokens projected")
    ledger.check(stage, proj)
    if args.project_only:
        return
    done, left, errors = JD.run_with_passes(judge, pairs, facts_of, args, stage)
    with open(out / "answers.jsonl", "w") as handle:
        handle.writelines(
            json.dumps(
                {
                    "source": s,
                    "candidate": d,
                    "prompt": prompt.name,
                    "n_frames": n,
                    "answers": answers,
                    "cost": {k: c.to_dict() for k, c in cost.items()},
                }
            )
            + "\n"
            for (s, d), (answers, cost, n) in sorted(done.items())
        )
    cost = sum(paid, Cost())
    ledger.add(stage, cost, len(paid), {"pairs": len(done), "failed": len(left), "model": G38})
    write_manifest(
        out,
        models={"judge": [G38], "judge_media_resolution": judge.media, "compared_with": ASTRA},
        cost=cost,
        extra={"prompt": prompt.name, "pairs": len(done), "sample_seed": "gemini38-calibration|0"},
    )
    print(f"calibration: {len(done)} pairs judged, {len(left)} failed {errors[:1]}")


def calibration_stats(rows, facts_of, of_query) -> dict:
    main = JD.JP.read_judged(MAIN_JUDGE / "judged.jsonl")
    new = {}
    for line in JD.read_jsonl(FILL / "calibration" / "answers.jsonl"):
        new.setdefault(line["source"], {})[line["candidate"]] = line["answers"]
    merged: dict = {}
    for s, cands in new.items():
        for d, answers in cands.items():
            m = {k: dict(v) for k, v in main[s][d].items()}
            for k, v in answers.items():
                m.setdefault(k, {}).update(v)
            merged.setdefault(s, {})[d] = m
    pairs = [(s, d) for s, c in merged.items() for d in c]
    items = JD.fact_items(pairs, merged, facts_of)
    facts = {
        f"{a.split(':')[1]} vs {b.split(':')[1]}": agreement_table(items, a, b, by="type")
        for a, b in ((G38, ASTRA), (G35, ASTRA), (G38, G35))
    }
    rel: dict = {}
    for name, (sa, sb) in {
        "gemini-3.8-flash vs gpt-6-astra (alone)": ((G38,), (ASTRA,)),
        "gemini-3.5-flash-lite vs gpt-6-astra (alone)": ((G35,), (ASTRA,)),
        "gemini-3.8-flash vs gemini-3.5-flash-lite (alone)": ((G38,), (G35,)),
        "3.5-lite+3.8-flash vs 3.5-lite+astra (both agree)": ((G35, G38), (G35, ASTRA)),
    }.items():
        xs, ys = [], []
        for r in rows:
            for d, answers in merged.get(r["source"], {}).items():
                if d in (r["source"], r.get("twin")):
                    continue
                xs.append(str(jf.relevant(answers, of_query[r["query_id"]], sa)))
                ys.append(str(jf.relevant(answers, of_query[r["query_id"]], sb)))
        rel[name] = {
            "n": len(xs),
            "raw_agreement": float(np.mean([x == y for x, y in zip(xs, ys, strict=True)])),
            "kappa": cohen_kappa(xs, ys),
            "positive_a": xs.count("True"),
            "positive_b": ys.count("True"),
        }
    versions = sorted(
        {
            c.get("model_version", "")
            for line in JD.read_jsonl(FILL / "calibration" / "answers.jsonl")
            for c in line["cost"].values()
        }
    )
    return {"pairs": len(pairs), "facts": facts, "relevance": rel, "model_versions": versions}


# ---- fill -------------------------------------------------------------------------------------


def fill_judged() -> dict:
    """Answers of the fill pairs only (lines appended to the copy after the main judged.jsonl)."""
    n_main = len((MAIN_JUDGE / "judged.jsonl").read_text().splitlines())
    out: dict = {}
    for i, line in enumerate(JD.read_jsonl(FILL / "judged.jsonl")):
        if i < n_main:
            continue
        out.setdefault(line["source"], {})[line["candidate"]] = line["answers"]
    return out, n_main


def frame_eval(S, sets, qrels_paths: dict, n_boot: int) -> dict:
    sub = FS.read_subsets(S) if (S.out / "select").exists() else None
    ids = {lab: set(sub["frames"][name]) for lab, name in SUBSETS.items()}
    ranks = {n: FS.load_runs(S.runs / n, (10,))[0] for n in sets}
    res: dict = {}
    pairs = [("S5", "S4@50"), ("S3", "S4@50"), ("S5", "S0"), ("S3", "S0"), ("S3", "S5")]
    for kind, path in qrels_paths.items():
        cell: dict = {}
        for lab, name in SUBSETS.items():
            rs = [r for r in sets[name] if r["query_id"] in ids[lab]]
            qrels = FS.PV.load_qrels(path, sets[name])
            sc = {s: FS.PV.score_system(ranks[name][s], qrels) for s in FRAME_SYS}
            cell[lab] = {
                "means": {s: FS.means(sc[s], rs) for s in FRAME_SYS},
                "paired": {
                    f"{a} - {b}": FS.paired_cells(sc[a], sc[b], rs, n_boot) for a, b in pairs
                },
                "positives_per_query": float(np.mean([len(qrels[r["query_id"]]) for r in rs])),
            }
        res[kind] = cell
    return res


def stage_fill(args) -> None:
    S = settings(args.n_boot)
    _jcfg, rows, facts_of, of_query, _ = JD.setup(S)
    sets = FS.load_sets(S, FS.load_context(S, need_index=False))
    main = JD.JP.read_judged(MAIN_JUDGE / "judged.jsonl")
    fill, n_main = fill_judged()
    # qrels variants: strict, main expanded (gemini-3.5-flash-lite + astra), expanded + fill
    # (fill pairs relevant when gemini-3.5-flash-lite AND gemini-3.8-flash say yes to all facts)
    strict, expanded_main = jf.qrels(rows, of_query, main, (G35, ASTRA))
    _, fill_only = jf.qrels(rows, of_query, fill, (G35, G38))
    plus = {
        q: [*expanded_main[q], *[d for d in fill_only[q][1:] if d not in expanded_main[q]]]
        for q in expanded_main
    }
    ref = {
        json.loads(x)["query_id"]: json.loads(x)["relevant"]
        for x in (MAIN_JUDGE / "qrels_expanded.jsonl").read_text().splitlines()
    }
    assert all(sorted(ref[q]) == sorted(expanded_main[q]) for q in ref), "main qrels differ"
    write_qrels(FILL / "qrels_strict.jsonl", strict)
    write_qrels(FILL / "qrels_expanded_main.jsonl", expanded_main)
    write_qrels(FILL / "qrels_expanded_plus_fill_gemini38.jsonl", plus)
    for stale in ("qrels_expanded.jsonl", "judge_stats.json", "judge_report.md"):
        p = FILL / stale  # written by judge.py over the seeded copy with the fill's two Gemini
        if p.exists():  # specs, which cannot label the main pairs (no gemini-3.8-flash answers)
            p.unlink()
    fill_rows = [r for r in rows if r["source"] in fill]
    agree = jf.agreement_stats(fill_rows, facts_of, of_query, fill, (G35, G38))
    added = sum(len(plus[q]) - len(expanded_main[q]) for q in plus)
    ev = frame_eval(
        S,
        sets,
        {
            "strict": None,
            "expanded (main)": MAIN_JUDGE / "qrels_expanded.jsonl",
            "expanded + fill (gemini-3.8-flash)": FILL / "qrels_expanded_plus_fill_gemini38.jsonl",
        },
        args.n_boot,
    )
    calib = (
        calibration_stats(rows, facts_of, of_query)
        if (FILL / "calibration" / "answers.jsonl").exists()
        else None
    )
    spend = {}
    for name, p in (
        ("fill", FILL / "run/judge/spend.json"),
        ("calibration", FILL / "run/calibration/spend.json"),
    ):
        if p.exists():
            e = json.loads(p.read_text())["entries"]
            c = Cost(
                tokens_in=sum(x["tokens_in"] for x in e), tokens_out=sum(x["tokens_out"] for x in e)
            )
            spend[name] = {
                "model": G38,
                "tokens_in": c.tokens_in,
                "tokens_out": c.tokens_out,
                "usd": dollars(G38, c),
                "calls": sum(x["calls"] for x in e),
            }
    versions = sorted(
        {
            c.get("model_version", "")
            for line in JD.read_jsonl(FILL / "judged.jsonl")
            for k, c in line.get("cost", {}).items()
            if k == G38
        }
    )
    res = {
        "judges": [G35, G38],
        "media_resolution": S.cfg.get("grounding_media_resolution"),
        "prompt": jf.JUDGE_PROMPT.name,
        "main_judged_pairs": n_main,
        "fill_pairs": sum(len(v) for v in fill.values()),
        "fill_sources": len(fill),
        "added_query_positives": added,
        "g38_model_versions": versions,
        "agreement_fill": agree,
        "frame_eval": ev,
        "calibration": calib,
        "spend": spend,
        "price": PRICE,
    }
    (FILL / "fill_summary.json").write_text(json.dumps(res, indent=1, default=str))
    (FILL / "report.md").write_text(fill_report(res))
    print(fill_report(res))


def f3(v) -> str:
    return "-" if v is None else f"{v:.3f}"


def ci(c: dict) -> str:
    return f"{c['mean']:+.3f} [{c['ci_low']:+.3f}, {c['ci_high']:+.3f}]"


def fill_report(res: dict) -> str:
    ev = res["frame_eval"]
    kinds = list(ev)
    L = [
        "# Judge fill with gemini-3.8-flash as second judge (separate check, not astra)",
        "",
        (
            f"The {res['fill_pairs']:,} frame-subset (query source, candidate) pairs that had no label "
            f"(top 10 of S0, S1, S2@50, S4@50, S3, S5 on the 380 frame-subset queries, outside the "
            f"main pool plan) judged by **{G35}** and **{G38}** (both must say yes to every fact), "
            f"prompt {res['prompt']}, index frames (at most 16, 512 px), media resolution "
            f"{res['media_resolution']} for both, as the main judge. gemini-3.8-flash replaces "
            "gpt-6-astra here because OpenAI has no credits; this is a separately reported label "
            "variant, not astra labels. Returned model version(s) of gemini-3.8-flash: "
            f"{', '.join(res['g38_model_versions'])}."
        ),
        "",
        (
            f"Qrels in this folder: qrels_strict.jsonl (source only), qrels_expanded_main.jsonl "
            f"(identical to data/mc/full/judge/qrels_expanded.jsonl: gemini-3.5-flash-lite + astra on "
            f"the {res['main_judged_pairs']:,} main pairs), qrels_expanded_plus_fill_gemini38.jsonl "
            f"(main expanded plus fill pairs accepted by gemini-3.5-flash-lite + gemini-3.8-flash: "
            f"{res['added_query_positives']} added (query, video) positives over 1,997 queries)."
        ),
        "",
        "## Spend (gemini-3.8-flash@low; gemini-3.5-flash-lite answers were cache hits)",
        "",
        "| stage | calls | input tokens | output tokens | USD (at $0.75 / $3.75 per 1M) |",
        "|---|---|---|---|---|",
    ]
    for name, s in res["spend"].items():
        L.append(
            f"| {name} | {s['calls']:,} | {s['tokens_in']:,} | {s['tokens_out']:,} | {s['usd']:.2f} |"
        )
    L += ["", "## Frame comparisons: main expanded vs expanded + fill", ""]
    for lab in SUBSETS:
        L += [
            f"### {lab} (n {ev[kinds[0]][lab]['means']['S0']['None']['all']['n']})",
            "",
            "| system | metric | " + " | ".join(kinds) + " | fill - main |",
            "|---|---|" + "---|" * (len(kinds) + 1),
        ]
        for s in FRAME_SYS:
            for m in ("ndcg@10", "recall@1"):
                vals = [ev[k][lab]["means"][s]["None"]["all"][m] for k in kinds]
                L.append(
                    f"| {s} | {m} | "
                    + " | ".join(f3(v) for v in vals)
                    + f" | {vals[2] - vals[1]:+.3f} |"
                )
        L.append(
            "| positives per query | | "
            + " | ".join(f"{ev[k][lab]['positives_per_query']:.2f}" for k in kinds)
            + " | |"
        )
        L += [
            "",
            "Paired differences (cluster-bootstrap 95% CI over source videos), all k:",
            "",
            "| pair | metric | " + " | ".join(kinds) + " |",
            "|---|---|" + "---|" * len(kinds),
        ]
        for pair in ev[kinds[0]][lab]["paired"]:
            for m in ("ndcg@10", "recall@1"):
                L.append(
                    f"| {pair} | {m} | "
                    + " | ".join(ci(ev[k][lab]["paired"][pair][m]["None"]["all"]) for k in kinds)
                    + " |"
                )
        L += [
            "",
            "nDCG@10 by k (main expanded -> expanded + fill):",
            "",
            "| system | " + " | ".join(f"k={g}" for g in ("1", "2", "3", "4")) + " |",
            "|---|" + "---|" * 4,
        ]
        for s in FRAME_SYS:
            cells = []
            for g in ("1", "2", "3", "4"):
                a = ev[kinds[1]][lab]["means"][s]["k"].get(int(g)) or ev[kinds[1]][lab]["means"][s][
                    "k"
                ].get(g)
                b = ev[kinds[2]][lab]["means"][s]["k"].get(int(g)) or ev[kinds[2]][lab]["means"][s][
                    "k"
                ].get(g)
                cells.append(
                    "-" if not a else f"{a['ndcg@10']:.3f} -> {b['ndcg@10']:.3f} (n {a['n']})"
                )
            L.append(f"| {s} | " + " | ".join(cells) + " |")
        L.append("")
    ag = res["agreement_fill"]
    L += [
        "## gemini-3.5-flash-lite vs gemini-3.8-flash on the fill pairs",
        "",
        "Relevance decisions per (query, candidate), source and twin excluded:",
        "",
        "| k | n | raw agreement | kappa | relevant 3.5-lite | relevant 3.8-flash | both |",
        "|---|---|---|---|---|---|---|",
    ]
    for g, c in ag["relevance"].items():
        L.append(
            f"| {g} | {c['n']} | {f3(c['raw_agreement'])} | {f3(c['kappa'])} | "
            f"{c['relevant_a']} | {c['relevant_b']} | {c['relevant_both']} |"
        )
    L += [
        "",
        "Per fact (answer level):",
        "",
        "| fact type | n | raw | kappa | yes 3.5-lite | yes 3.8-flash |",
        "|---|---|---|---|---|---|",
    ]
    for t, c in ag["facts"].items():
        L.append(
            f"| {t} | {c.get('n')} | {f3(c.get('raw_agreement', c.get('agreement')))} | "
            f"{f3(c.get('kappa'))} | {f3(c.get('yes_a'))} | {f3(c.get('yes_b'))} |"
        )
    cal = res["calibration"]
    if cal:
        L += [
            "",
            f"## Calibration: gemini-3.8-flash vs gpt-6-astra on {cal['pairs']} already-labelled pool pairs",
            "",
            (
                "Seeded sample (seed 'gemini38-calibration|0') of main judged pairs other than "
                "(source, source) and (source, twin); gemini-3.8-flash@low judged them fresh with the "
                "same prompt, frames and LOW resolution; astra and gemini-3.5-flash-lite answers are "
                "the main judge's."
            ),
            "",
            "Relevance decisions per (query, candidate):",
            "",
            "| comparison | n | raw agreement | kappa | positives a | positives b |",
            "|---|---|---|---|---|---|",
        ]
        for name, c in cal["relevance"].items():
            L.append(
                f"| {name} | {c['n']} | {f3(c['raw_agreement'])} | {f3(c['kappa'])} | "
                f"{c['positive_a']} | {c['positive_b']} |"
            )
        L += [
            "",
            "Per fact (all types pooled, then by type):",
            "",
            "| comparison | type | n | raw | kappa | yes a | yes b |",
            "|---|---|---|---|---|---|---|",
        ]
        for name, table in cal["facts"].items():
            for t, c in table.items():
                L.append(
                    f"| {name} | {t} | {c.get('n')} | {f3(c.get('raw_agreement', c.get('agreement')))} | "
                    f"{f3(c.get('kappa'))} | {f3(c.get('yes_a'))} | {f3(c.get('yes_b'))} |"
                )
    L.append("")
    return "\n".join(L)


# ---- S5 with gemini-3.8-flash ------------------------------------------------------------------


def stage_s5(args) -> None:
    S = FS.settings(argparse.Namespace(smoke=False, config=None, out=None, n_boot=args.n_boot))
    sets = FS.load_sets(S, FS.load_context(S, need_index=False))
    if args.sources_file:  # the test set: query sets and frozen subsets restricted
        sets = FS.restrict_sets(S, sets, args.sources_file)
    sub = FS.read_subsets(S)
    fv = Path(args.out) if args.out else FV
    name_g = args.s5_system
    runs = Path(args.s5_runs or FV / "runs")
    qpaths = {"strict": None, "expanded": MAIN_JUDGE / "qrels_expanded.jsonl"}
    res: dict = {
        "verifier_system": name_g,
        "sources_file": args.sources_file,
        "labels": {},
        "top1_agreement": {},
        "twin_above_source": {},
    }
    systems = ("S4@50", "S5", name_g)
    pairs = [(name_g, "S5"), (name_g, "S4@50"), ("S5", "S4@50")]
    for kind, path in qpaths.items():
        cell = {}
        for lab, name in SUBSETS.items():
            ranks = FS.load_runs(S.runs / name, (10,))[0]
            ranks[name_g] = FS.read_ranking(runs / name / f"{name_g}.jsonl")
            ids = set(sub["frames"][name]) & set(ranks[name_g])
            rs = [r for r in sets[name] if r["query_id"] in ids]
            if S.keep:
                ranks = {s: {q: v for q, v in r.items() if q in S.keep[name]} for s, r in ranks.items()}
            qrels = FS.PV.load_qrels(path, sets[name])
            sc = {s: FS.PV.score_system(ranks[s], qrels) for s in systems}
            cell[lab] = {
                "n": len(rs),
                "means": {s: FS.means(sc[s], rs) for s in systems},
                "paired": {
                    f"{a} - {b}": FS.paired_cells(sc[a], sc[b], rs, args.n_boot) for a, b in pairs
                },
            }
            if kind == "strict":
                res["top1_agreement"][lab] = FS.P2.by_k(
                    rs,
                    lambda r, R=ranks: float(
                        R[name_g][r["query_id"]][0][0] == R["S5"][r["query_id"]][0][0]
                    ),
                    args.n_boot,
                )
                if name == "twin":
                    res["twin_above_source"] = {
                        s: FS.twin_above(ranks[s], rs, args.n_boot) for s in systems
                    }
        res["labels"][kind] = cell
    fv.mkdir(parents=True, exist_ok=True)
    (fv / "summary.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps({k: res[k] for k in ("top1_agreement", "twin_above_source")}, indent=1)[:3000])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=("calibrate", "fill", "s5"))
    ap.add_argument("--sample", type=int, default=200)
    ap.add_argument("--cap", type=float, default=2e6, help="calibrate: input-token cap")
    ap.add_argument("--project-only", action="store_true")
    ap.add_argument("--gemini-workers", type=int, default=16)
    ap.add_argument("--openai-workers", type=int, default=1)
    ap.add_argument("--passes", type=int, default=4)
    ap.add_argument("--wait", type=float, default=60.0)
    ap.add_argument("--n-boot", type=int, default=FS.N_BOOT)
    ap.add_argument("--s5-system", default="S5_gemini-3.8-flash-low")
    ap.add_argument("--s5-runs")
    ap.add_argument("--sources-file", help="s5: the test set, as systems.py --sources-file")
    ap.add_argument("--out", help=f"s5: output folder (default {FV})")
    args = ap.parse_args(argv)
    {"calibrate": stage_calibrate, "fill": stage_fill, "s5": stage_s5}[args.stage](args)


if __name__ == "__main__":
    main()
