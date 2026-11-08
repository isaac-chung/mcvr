"""Summary of the full-run query build (no model call): data/mc/full/build_report.md.

Reads the build outputs in the config's out_dir (constraints.jsonl, queries.jsonl,
queries_twin.jsonl, twins.jsonl, build_stats.json, gonogo.json, canary.json, yield_check.json,
index_chunks.json, spend.json). Length bins are reported within Vript only (plan 1, 3).

    uv run python pipeline/build_queries_report.py [--config configs/mc_full.yaml]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from src.datasets.mc import read_jsonl
from src.types import Cost

K = (1, 2, 3, 4)
DATASETS = ("shot2story", "carebench", "vript_short", "vript_long")
BINS = ("lt30s", "30to60s", "1to2min", "2to4min")
VRIPT = ("vript_short", "vript_long")
# plan 2.1: expected queries per k for 280 sources
PLAN = {"natural": (277, 272, 252, 176), "twin": (280, 280, 263, 165)}
GEMINI_PRICE = (0.30, 2.50)  # $ per 1M input / output tokens (plan 4.2)


def groups_of(dataset: str, length_bin: str) -> list[str]:
    out = ["all", f"dataset={dataset}"]
    if dataset in VRIPT:
        out += [f"vript bin={length_bin}", f"{dataset}/{length_bin}"]
    return out


def group_order(names) -> list[str]:
    def key(name):
        if name == "all":
            return (0, 0, name)
        if name.startswith("dataset="):
            return (1, DATASETS.index(name.split("=")[1]), name)
        if name.startswith("vript bin="):
            return (2, BINS.index(name.split("=")[1]), name)
        ds, b = name.split("/")
        return (3, VRIPT.index(ds) * 10 + BINS.index(b), name)

    return sorted(names, key=key)


def kind(answers: dict, prefix: str) -> str:
    return next((a["answer"] for s, a in answers.items() if s.startswith(prefix)), "none")


def count_table(rows: list[dict], records: list[dict]) -> list[str]:
    n_src: Counter = Counter()
    cells: dict = defaultdict(Counter)
    meta = {r["source"]: (r["dataset"], r["length_bin"]) for r in records}
    for r in records:
        for g in groups_of(r["dataset"], r["length_bin"]):
            n_src[g] += 1
    for row in rows:
        for g in groups_of(*meta[row["source"]]):
            cells[g][row["k"]] += 1
    lines = ["| group | sources | " + " | ".join(f"k={k}" for k in K) + " | total |"]
    lines.append("|---|---|" + "---|" * (len(K) + 1))
    for g in group_order(n_src):
        c = cells[g]
        lines.append(
            f"| {g} | {n_src[g]} | " + " | ".join(str(c[k]) for k in K) + f" | {sum(c.values())} |"
        )
    return lines


def chunk_lines(path: Path) -> list[str]:
    if not path.exists():
        return ["Not run (index chunk check not included in this repo).", ""]
    data = json.loads(path.read_text())
    lines = [
        (
            f"{data['n_facts']} seeded grounded facts from long-bin Vript long sources "
            f"({data['n_sources']} sources), re-asked with the index frames in chunks of at most "
            f"{data['chunk']} (Gemini LOW, visible = yes in any chunk). Paid: "
            f"{data['paid_in']:,} input tokens."
        ),
        "",
        (
            "| bin | facts | index frames (mean) | visible, one call | visible, chunked | "
            "flips to visible | flips to not visible |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for name, s in data["by_bin"].items():
        lines.append(
            f"| {name} | {s['n']} | {s['frames_mean']:.1f} | {s['single']:.2f} | "
            f"{s['chunked']:.2f} | {s['to_visible']} | {s['to_hidden']} |"
        )
    return lines + [""]


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_full.yaml")
    parser.add_argument("--out", default=None, help="default: <out_dir>/build_report.md")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = Path(cfg["out_dir"])
    records = list(read_jsonl(out_dir / "constraints.jsonl"))
    natural = list(read_jsonl(out_dir / "queries.jsonl"))
    twin_rows = list(read_jsonl(out_dir / "queries_twin.jsonl"))
    twin_recs = list(read_jsonl(out_dir / "twins.jsonl"))
    meta = {r["source"]: (r["dataset"], r["length_bin"]) for r in records}

    def load(name):
        p = out_dir / name
        return json.loads(p.read_text()) if p.exists() else None

    gonogo, canary, spend = load("gonogo.json"), load("canary.json"), load("spend.json")
    manifest = load("manifest.json") or {}

    L = [f"# Query build, full run ({len(records)} test sources)", ""]
    L += [
        (
            f"Config `{args.config}`: smoke v2 pipeline unchanged (facts_mc_v2, twin_facts_v1, "
            f"ground_v3 on about 6 local frames per scene window with `gemini-3.5-flash-lite` at "
            f"`{cfg.get('grounding_media_resolution')}` + `gpt-6-astra`, kept only when both say "
            "yes; query_mc_v2; index visibility on the 0.5 fps index frames). Twin neighbours: "
            "caption embeddings over every captioned corpus video, other YouTube / source video "
            "only. Nothing checked by hand."
        ),
        "",
        (
            f"Natural (nested) queries {len(natural)} from "
            f"{len({r['source'] for r in natural})} sources; twin queries {len(twin_rows)} from "
            f"{len({r['source'] for r in twin_rows})} sources."
        ),
        "",
    ]
    if manifest.get("failed_sources"):
        L += [f"Failed sources (not in the outputs): {manifest['failed_sources']}", ""]
    if manifest.get("max_b_passes_raised"):
        L += [
            (
                f"max_b_passes raised to 5 for {len(manifest['max_b_passes_raised'])} sources "
                "without a k = 4 twin (plan 7.1)."
            ),
            "",
        ]

    # yield
    L += ["## Queries per k against plan 2.1", "", "| set | k=1 | k=2 | k=3 | k=4 | total |"]
    L += ["|---|---|---|---|---|---|"]
    for name, rows in (("natural", natural), ("twin", twin_rows)):
        c = Counter(r["k"] for r in rows)
        plan = PLAN[name]
        L.append(
            f"| {name} | "
            + " | ".join(f"{c[k]} ({plan[k - 1]})" for k in K)
            + f" | {len(rows)} ({sum(plan)}) |"
        )
    L += [
        "",
        "Plan values in brackets. Checks: at least 900 per set and 150 twin queries at k = 4.",
        "",
    ]
    L += ["## Natural nested set: queries per k", ""] + count_table(natural, records) + [""]
    L += ["## Twin set: queries per k", ""] + count_table(twin_rows, records) + [""]

    # grounding
    items = [(r, g) for r in records for g in r["grounding"]]
    L += ["## Grounding: the two-model agreement rule", ""]
    for role in ("natural", "twin"):
        sel = [g for _r, g in items if g["role"] == role]
        if not sel:
            continue
        L += [
            f"**{role}** ({len(sel)} facts grounded on A's local frames). Kept = both yes.",
            "",
            (
                "| type | n | kept | share kept | Gemini yes only | astra yes only | neither yes | "
                "Gemini yes-rate | astra yes-rate |"
            ),
            "|---|---|---|---|---|---|---|---|---|",
        ]
        by = defaultdict(list)
        for g in sel:
            by["all"].append(g)
            by[str(g.get("type"))].append(g)
        for t in sorted(by, key=lambda x: (x != "all", x)):
            gs = by[t]
            gem = [kind(g["answers"], "gemini") == "yes" for g in gs]
            oai = [kind(g["answers"], "openai") == "yes" for g in gs]
            kept = sum(g["confirmed"] for g in gs)
            n = len(gs)
            only_g = sum(a and not b for a, b in zip(gem, oai, strict=True))
            only_o = sum(b and not a for a, b in zip(gem, oai, strict=True))
            neither = sum(not a and not b for a, b in zip(gem, oai, strict=True))
            L.append(
                f"| {t} | {n} | {kept} | {kept / n:.2f} | {only_g} | {only_o} | {neither} | "
                f"{sum(gem) / n:.2f} | {sum(oai) / n:.2f} |"
            )
        L.append("")
    windows = Counter(g["window"]["kind"] for _r, g in items)
    agree = Counter(g["agreement"] for _r, g in items)
    L += [f"Windows: {dict(windows)}. Agreement labels: {dict(agree)}.", ""]
    st = load("build_stats.json") or {}
    kt = st.get("kappa_by_type", {}).get("all")
    if kt:
        L += [
            (
                f"Gemini x astra over all {kt['n']} decisions: raw agreement "
                f"{kt['raw_agreement']:.2f}, kappa {kt['kappa']:.2f} (yes vs not: "
                f"{kt['raw_agreement_yes']:.2f}, kappa {kt['kappa_yes']:.2f})."
            ),
            "",
        ]

    # index visibility
    L += [
        "## Index visibility (grounded facts not visible in the 0.5 fps index frames, Gemini)",
        "",
        "| group | grounded | not visible in index | share | index frames (mean) |",
        "|---|---|---|---|---|",
    ]
    vis = defaultdict(lambda: [0, 0, 0])
    for r, g in items:
        if g["confirmed"] and g.get("index") is not None:
            names = groups_of(r["dataset"], r["length_bin"]) + [f"type={g.get('type')}"]
            for name in names:
                vis[name][0] += 1
                vis[name][1] += bool(g["index_dropped"])
                vis[name][2] += g["index"].get("n_frames", 0)
    order = group_order([n for n in vis if not n.startswith("type=")])
    order += sorted(n for n in vis if n.startswith("type="))
    for name in order:
        n, d, f = vis[name]
        L.append(f"| {name} | {n} | {d} | {d / n:.2f} | {f / n:.1f} |")
    L += ["", "### Chunked re-check on long bins (plan 3)", ""] + chunk_lines(
        out_dir / "index_chunks.json"
    )

    # twins
    with_twin = [t for t in twin_recs if t.get("chosen")]
    shared = [len(t["chosen"]["order"]) - 1 for t in with_twin]
    passes = [sum(1 for e in t.get("log", []) if e.get("b_pass")) for t in twin_recs]
    sims = [t["chosen"]["similarity"] for t in with_twin]
    cross = Counter(t["chosen"]["b"].split("/")[0] == t["source"].split("/")[0] for t in with_twin)
    L += [
        "## Twin statistics",
        "",
        (
            f"- Sources with a twin: {len(with_twin)} of {len(twin_recs)} (by dataset "
            f"{dict(Counter(meta[t['source']][0] for t in with_twin))})."
        ),
        (
            f"- Sources with a k = 4 twin query: {sum(1 for r in twin_rows if r['k'] == 4)}; "
            f"shared facts used per twin: mean {sum(shared) / max(len(shared), 1):.2f}, "
            f"distribution {dict(sorted(Counter(shared).items()))}."
        ),
        f"- d types: {dict(Counter(t['chosen']['d'].get('type') for t in with_twin).most_common())}.",
        (
            f"- Chosen neighbour rank: {dict(sorted(Counter(t['chosen']['rank'] for t in with_twin).items()))}; "
            f"caption similarity mean {sum(sims) / max(len(sims), 1):.3f}; twin from the same "
            f"dataset {cross[True]}, other dataset {cross[False]}."
        ),
        f"- B passes run per source: mean {sum(passes) / max(len(passes), 1):.2f}.",
        "",
    ]

    # cost
    total: dict[tuple[str, str], Cost] = defaultdict(Cost)
    for r in records:
        for stage, by in r["cost"].items():
            for spec, c in by.items():
                total[(stage, spec)] = total[(stage, spec)] + Cost(**c)
    L += [
        "## Tokens per step (recorded per source, cached calls included)",
        "",
        "| stage | model | calls | images | tokens in | tokens out |",
        "|---|---|---|---|---|---|",
    ]
    grand = Cost()
    for (stage, spec), c in sorted(total.items()):
        L.append(
            f"| {stage} | {spec} | {c.llm_calls:,} | {c.images:,} | {c.tokens_in:,} | "
            f"{c.tokens_out:,} |"
        )
        grand = grand + c
    L += [
        f"| total | | {grand.llm_calls:,} | {grand.images:,} | {grand.tokens_in:,} | {grand.tokens_out:,} |",
        "",
    ]
    per_model: dict[str, Cost] = defaultdict(Cost)
    for (_stage, spec), c in total.items():
        per_model[spec] = per_model[spec] + c
    n = max(len(records), 1)
    L += [
        "Per source (recorded): "
        + "; ".join(
            f"{s} {c.tokens_in / n / 1e3:.1f}k in, {c.tokens_out / n / 1e3:.2f}k out"
            for s, c in sorted(per_model.items())
        )
        + ".",
        "",
    ]
    if spend:
        L += ["## Spend (calls actually paid, all build runs; `spend.json`)", ""]
        L += ["| stage | model | calls | tokens in | tokens out |", "|---|---|---|---|---|"]
        for stage, by in spend.get("totals", {}).items():
            for spec, c in by.items():
                if isinstance(c, dict):
                    L.append(
                        f"| {stage} | {spec} | {c['llm_calls']:,} | {c['tokens_in']:,} | "
                        f"{c['tokens_out']:,} |"
                    )
        g = spend.get("gemini_usd", {})
        L += [
            "",
            (
                f"Build stage paid input {spend.get('totals', {}).get('build', {}).get('tokens_in', 0):,} "
                f"against the cap {float(cfg['caps']['build']):,.0f}. Gemini (all stages in this "
                f"file) {g.get('tokens_in', 0):,} in / {g.get('tokens_out', 0):,} out = "
                f"${g.get('usd', 0):.2f} at ${GEMINI_PRICE[0]} / ${GEMINI_PRICE[1]} per 1M."
            ),
            "",
        ]
        runs = [
            f"{r['what']} {r.get('wall_s', '-')} s" for r in spend["runs"] if r["stage"] == "build"
        ]
        L += [f"Build runs and wall clock: {'; '.join(runs)}.", ""]

    # checks
    L += ["## Go/no-go (plan 7.2 G2)", ""]
    if gonogo:
        e = gonogo["caption_embed_tokens"]
        L += [
            (
                f"- Test sources {gonogo['n_sources']}, dev {gonogo['n_dev']}: shared ids "
                f"{len(gonogo['shared_ids'])}, shared source videos "
                f"{len(gonogo['shared_source_keys'])}, shared YouTube ids "
                f"{len(gonogo['shared_youtube_ids'])}; strata {gonogo['strata']}."
            ),
            f"- Vript bins derived from duration: {gonogo['bins_ok']}.",
            (
                f"- Twin pool: {gonogo['n_pool']} captioned videos. Caption embedding inputs "
                f"(first {8000:,} characters, as on the smoke run): max {e['max']:,} tokens "
                f"(limit 8,191), {e['over_limit']} over; {e['truncated_at_chars']} captions cut."
            ),
            f"- Pass: {gonogo['pass']}.",
        ]
    if canary:
        L += [
            (
                f"- Canary ({len(canary['sources'])} sources): input per source "
                + "; ".join(
                    f"{s.split(':')[1]} {canary['per_source_in'][s] / 1e3:.1f}k vs "
                    f"{canary['expected_per_source_in'][s] / 1e3:.1f}k "
                    f"(x{canary['ratio'][s]:.2f})"
                    for s in canary["per_source_in"]
                )
                + f"; projection for all sources {canary['projection_all_in_by_mean'] / 1e6:.1f}M "
                f"(mean) / {canary['projection_all_in_by_estimate'] / 1e6:.1f}M (estimate x "
                f"{canary['actual_over_estimate']:.2f}) against the cap "
                f"{canary['cap'] / 1e6:.0f}M. Pass: {canary['pass']}."
            )
        ]
    L.append("")
    text = "\n".join(L) + "\n"
    path = Path(args.out) if args.out else out_dir / "build_report.md"
    path.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
