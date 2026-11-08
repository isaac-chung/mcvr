"""Gemini media-resolution test for local grounding (ground_v3). PAID (Gemini, gpt-6-astra).

Five seeded smoke query sources (one each from Shot2Story, CaReBench and Vript short, two Vript
long from different length bins). Their natural fact candidates (mined facts, cached; selected
as in build_v2) are grounded on the same local frames with Gemini at the default resolution and
at each tested media resolution, and once with gpt-6-astra (whose answers are shared by every
condition). Per condition: Gemini image tokens per frame (from usageMetadata
promptTokensDetails), Gemini yes-rate, agreement with Gemini default, Gemini x astra agreement
and kappa, facts kept by the both-yes rule, and the fact types that flip. Applies the decision
rule and writes results/smoke_v2/media_resolution_test.{md,json}.

Only natural candidates are grounded here (build_v2 grounds them together with the twin facts,
so these calls are not reused by the build).

    uv run --env-file .env python dev/media_resolution_dev.py --levels low medium
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import threading
from collections import defaultdict
from pathlib import Path

import httpx
import yaml

from src import constraints, grounding_v2
from src.cache import shared_cache
from src.frames import FramePolicy
from src.grounding_v2 import FrameStore, Grounder, cohen_kappa
from src.llm import CachedVision, GeminiVision, media_level, parse_spec
from src.types import Cost

OUT = Path("results/smoke_v2")
TYPES = ("attribute", "on_screen_text", "count", "object", "scene")


def load_build_v2():
    path = Path(__file__).resolve().parent / "build_queries_dev.py"
    spec = importlib.util.spec_from_file_location("build_v2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class UsageLog:
    """httpx response hook that keeps Gemini usageMetadata (with the per-modality details)."""

    def __init__(self):
        self.records: list[dict] = []
        self.lock = threading.Lock()

    def __call__(self, response: httpx.Response) -> None:
        response.read()
        if response.status_code == 200:
            usage = response.json().get("usageMetadata", {})
            with self.lock:
                self.records.append(usage)

    def image_tokens(self) -> int:
        return sum(
            d.get("tokenCount", 0)
            for u in self.records
            for d in u.get("promptTokensDetails", [])
            if d.get("modality") == "IMAGE"
        )


def pick_sources(sample: dict, lookup, seed: int = 0) -> list[str]:
    """One Shot2Story, one CaReBench, one Vript short, two Vript long of different bins."""
    rng = random.Random(constraints.seed_for(f"seed{seed}", "media_resolution_test"))
    by: dict[str, list[str]] = defaultdict(list)
    for gid in sample["query_sources"]:
        by[gid.split("/", 1)[0]].append(gid)
    chosen = [rng.choice(sorted(by[d])) for d in ("shot2story", "carebench", "vript_short")]
    long_by_bin: dict[str, list[str]] = defaultdict(list)
    for gid in sorted(by["vript_long"]):
        long_by_bin[lookup(gid)[1].length_bin].append(gid)
    bins = rng.sample(sorted(long_by_bin), 2)
    chosen += [rng.choice(long_by_bin[b]) for b in bins]
    return chosen


def gemini_model(spec: str, level: str | None, log: UsageLog):
    _, model, effort = parse_spec(spec)
    client = httpx.Client(timeout=180, event_hooks={"response": [log]})
    inner = GeminiVision(model, effort, media_resolution=level, client=client)
    return CachedVision(inner, shared_cache("vlm"), spec)


def yes(answer: str | None) -> bool:
    return answer == "yes"


def condition_stats(facts, gem, astra, default) -> dict:
    n = len(facts)
    g = [gem[f["key"]] for f in facts]
    a = [astra[f["key"]] for f in facts]
    d = [default[f["key"]] for f in facts]
    flips = defaultdict(lambda: {"n": 0, "answer_flips": 0, "decision_flips": 0})
    for f, x, y in zip(facts, g, d, strict=True):
        t = f.get("type")
        flips[t]["n"] += 1
        flips[t]["answer_flips"] += x != y
        flips[t]["decision_flips"] += yes(x) != yes(y)
    kept = [yes(x) and yes(y) for x, y in zip(g, a, strict=True)]
    return {
        "n": n,
        "yes_rate": sum(map(yes, g)) / n,
        "answers": {w: g.count(w) for w in grounding_v2.ANSWERS},
        "agree_default_answer": sum(x == y for x, y in zip(g, d, strict=True)) / n,
        "agree_default_decision": sum(yes(x) == yes(y) for x, y in zip(g, d, strict=True)) / n,
        "agree_astra_answer": sum(x == y for x, y in zip(g, a, strict=True)) / n,
        "kappa_astra_answer": cohen_kappa(g, a),
        "agree_astra_decision": sum(yes(x) == yes(y) for x, y in zip(g, a, strict=True)) / n,
        "kappa_astra_decision": cohen_kappa([str(yes(x)) for x in g], [str(yes(y)) for y in a]),
        "kept_both_yes": sum(kept),
        "kept_by_type": {
            t: sum(k for f, k in zip(facts, kept, strict=True) if f.get("type") == t) for t in TYPES
        },
        "flips_by_type": {t: dict(v) for t, v in sorted(flips.items(), key=lambda kv: str(kv[0]))},
    }


def decide(conds: dict, levels: list[str]) -> tuple[str | None, dict]:
    """The rule: (a) decisions match default on >= 90% of facts, (b) Gemini x astra agreement
    within 5 points of default, (c) on_screen_text facts kept by both-yes drop by at most 1.
    LOW first; MEDIUM only if LOW fails."""
    base = conds["default"]
    checks = {}
    for level in levels:
        c = conds[level]
        a = c["agree_default_decision"] >= 0.90
        b = abs(c["agree_astra_answer"] - base["agree_astra_answer"]) <= 0.05
        drop = base["kept_by_type"]["on_screen_text"] - c["kept_by_type"]["on_screen_text"]
        checks[level] = {
            "a": a,
            "b": b,
            "c": drop <= 1,
            "ost_drop": drop,
            "pass": a and b and drop <= 1,
        }
    for level in levels:
        if checks[level]["pass"]:
            return level, checks
    return None, checks


def fmt(x, digits=2):
    return "-" if x is None else f"{x:.{digits}f}"


def report(out: dict) -> str:
    conds, levels = out["conditions"], out["levels"]
    names = ["default", *levels]
    lines = ["# Gemini media-resolution test (local grounding, ground_v3)", ""]
    lines += [
        f"Date {out['date']}. Sources ({len(out['sources'])}, seeded): "
        + ", ".join(f"`{s}` ({b})" for s, b in out["sources"])
        + f". Facts: {out['n_facts']} natural candidates (types {out['types']}), "
        f"{out['n_windows']} scene windows, {out['n_frames']} local frames per condition. "
        "Gemini `gemini-3.5-flash-lite` per condition (media resolution sent per image part); "
        "`gpt-6-astra` once, shared by all conditions. Both-yes rule as in build_v2.",
        "",
        (
            "| condition | Gemini image tokens / frame | Gemini input tokens | yes-rate | answers yes/no/unsure "
            "| = default (yes vs not) | = default (3-way) | Gemini x astra agreement (3-way) | kappa "
            "| agreement (yes vs not) | kappa (yes vs not) | kept (both yes) | on_screen_text kept |"
        ),
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name in names:
        c, u = conds[name], out["usage"][name]
        ans = c["answers"]
        lines.append(
            f"| {name} | {fmt(u['image_tokens_per_frame'], 0)} | {u['tokens_in']:,} | "
            f"{fmt(c['yes_rate'])} | {ans['yes']}/{ans['no']}/{ans['unsure']} | "
            f"{fmt(c['agree_default_decision'])} | {fmt(c['agree_default_answer'])} | "
            f"{fmt(c['agree_astra_answer'])} | {fmt(c['kappa_astra_answer'])} | "
            f"{fmt(c['agree_astra_decision'])} | {fmt(c['kappa_astra_decision'])} | "
            f"{c['kept_both_yes']} / {c['n']} | {c['kept_by_type']['on_screen_text']} |"
        )
    a = out["astra"]
    lines += [
        "",
        (
            f"gpt-6-astra: yes-rate {fmt(a['yes_rate'])}, input tokens {a['tokens_in']:,} "
            f"({fmt(a['tokens_per_frame'], 0)} per frame incl. prompt)."
        ),
        "",
        "## Flips against Gemini default, by fact type (answer flips / decision flips / n)",
        "",
        "| type | " + " | ".join(levels) + " |",
        "|---|" + "---|" * len(levels),
    ]
    for t in sorted({t for lv in levels for t in conds[lv]["flips_by_type"]}, key=str):
        cells = []
        for lv in levels:
            f = conds[lv]["flips_by_type"].get(t, {"n": 0, "answer_flips": 0, "decision_flips": 0})
            cells.append(f"{f['answer_flips']} / {f['decision_flips']} / {f['n']}")
        lines.append(f"| {t} | " + " | ".join(cells) + " |")
    lines += ["", "## Kept by the both-yes rule, by type", ""]
    lines += ["| type | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for t in TYPES:
        lines.append(
            f"| {t} | " + " | ".join(str(conds[n]["kept_by_type"][t]) for n in names) + " |"
        )
    lines += ["", "## Decision", ""]
    lines += [
        (
            "Rule: use a lower resolution if (a) its yes / not-yes decisions match Gemini default on "
            ">= 90% of facts, (b) its Gemini x astra agreement rate (3-way) is within 5 points of "
            "default's, and (c) on_screen_text facts kept by the both-yes rule drop by at most 1. "
            "LOW first; MEDIUM only if LOW fails."
        ),
        "",
        "| level | (a) | (b) | (c) on_screen_text drop | pass |",
        "|---|---|---|---|---|",
    ]
    for lv, ch in out["checks"].items():
        lines.append(
            f"| {lv} | {ch['a']} | {ch['b']} | {ch['ost_drop']} ({ch['c']}) | {ch['pass']} |"
        )
    chosen = out["decision"]
    lines += [
        "",
        (
            f"**Decision: {'`' + chosen + '`' if chosen else 'keep the default (unset)'}** for Gemini "
            "in grounding and the frame judge (`grounding_media_resolution` in "
            "`configs/mc_smoke_v2.yaml`)."
        ),
        "",
        (
            f"Paid this run: {out['paid']}. Caveat: {out['n_facts']} facts from 5 sources; one "
            "astra sample (default temperature, not deterministic)."
        ),
    ]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_smoke_v2.yaml")
    parser.add_argument("--levels", nargs="+", default=["low", "medium"])
    parser.add_argument("--max-input-tokens", type=float, default=3e6)
    parser.add_argument("--dry", action="store_true", help="list sources and facts, no paid call")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    b2 = load_build_v2()
    root = cfg["data_root"]
    sample = json.loads((Path(cfg["smoke_dir"]) / "sample.json").read_text())
    sources = pick_sources(sample, lambda g: b2.lookup(root, g), cfg.get("seed", 0))
    budget = b2.Budget(args.max_input_tokens)
    llm = b2.GuardedLLM(b2.make_llm(constraints.LLM_SPEC, shared_cache("llm")), budget, "llm")
    lf = cfg.get("local_frames", {})
    store = FrameStore(
        root,
        lf.get("max_side", grounding_v2.MAX_SIDE),
        index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8)),
    )
    fallback = FramePolicy.from_config(cfg["fallback_policy"])
    template = grounding_v2.GROUND_PROMPT.read_text()
    gem_spec, astra_spec = cfg["grounders"]
    levels = [media_level(x) for x in args.levels]
    short = {media_level(x): x for x in args.levels}
    logs = {name: UsageLog() for name in ["default", *levels]}
    models = {
        "default": gemini_model(gem_spec, None, logs["default"]),
        **{lv: gemini_model(gem_spec, lv, logs[lv]) for lv in levels},
        "astra": b2.make_grounder_model(astra_spec),
    }
    guarded = {
        n: b2.GuardedVision(m, budget, gem_spec if n != "astra" else astra_spec)
        for n, m in models.items()
    }

    facts_all, answers = [], defaultdict(dict)
    costs: dict[str, Cost] = defaultdict(Cost)
    n_windows = n_frames = 0
    for gid in sources:
        dataset, video, caption = b2.lookup(root, gid)
        prompt_text = constraints.FACTS_PROMPT.read_text()
        if not llm.is_cached(
            prompt_text.replace("{caption}", constraints.scene_caption(caption)[0])
        ):
            raise SystemExit(f"mining for {gid} is not cached")
        facts, _n, _c, _ = constraints.mine_facts(llm, caption, prompt_text)
        cands = [
            {**f, "key": f"{gid}|n{f['id']}"} for f in constraints.select_candidates(facts, gid)
        ]
        facts_all += [{**f, "source": gid} for f in cands]
        if args.dry:
            plans = Grounder({}, template, store, lf.get("n", 6), fallback=fallback).plan_local(
                dataset, video, caption, cands
            )
            print(gid, video.length_bin, f"{video.duration_s:.0f}s", len(cands), "facts",
                  [f.get("type") for f in cands], "windows", len(plans),
                  "frames", sum(len(p["times"]) for p in plans))  # fmt: skip
            continue
        for name, model in guarded.items():
            g = Grounder({name: model}, template, store, lf.get("n", 6), fallback=fallback,
                         min_offset=lf.get("min_offset_s", 0.5))  # fmt: skip
            records, cost = g.local(dataset, video, caption, cands)
            costs[name] = costs[name] + cost.get(name, Cost())
            for key, rec in records.items():
                answers[name][key] = rec["answers"][name]["answer"]
            if name == "default":
                n_windows += len({json.dumps(r["window"]) for r in records.values()})
                n_frames += sum(
                    len(p["times"]) for p in g.plan_local(dataset, video, caption, cands)
                )

    if args.dry:
        return
    conds = {
        name: condition_stats(facts_all, answers[name], answers["astra"], answers["default"])
        for name in ["default", *levels]
    }
    usage = {}
    for name in ["default", *levels]:
        c, log = costs[name], logs[name]
        img = log.image_tokens()
        usage[name] = {
            "tokens_in": c.tokens_in,
            "tokens_out": c.tokens_out,
            "calls": c.llm_calls,
            "images": c.images,
            "image_tokens": img,
            "image_tokens_per_frame": (img / c.images) if c.images and log.records else None,
            "paid_calls": len(log.records),
        }
    ac = costs["astra"]
    astra = {
        "yes_rate": sum(map(yes, answers["astra"].values())) / len(facts_all),
        "tokens_in": ac.tokens_in,
        "tokens_per_frame": ac.tokens_in / ac.images if ac.images else None,
        "images": ac.images,
    }
    decision, checks = decide(conds, levels)
    from datetime import date

    out = {
        "date": date.today().isoformat(),
        "sources": [(s, b2.lookup(root, s)[1].length_bin) for s in sources],
        "n_facts": len(facts_all),
        "types": {t: sum(f.get("type") == t for f in facts_all) for t in TYPES},
        "n_windows": n_windows,
        "n_frames": n_frames,
        "levels": [short[lv] for lv in levels],
        "conditions": {("default" if n == "default" else short[n]): v for n, v in conds.items()},
        "usage": {("default" if n == "default" else short[n]): v for n, v in usage.items()},
        "astra": astra,
        "checks": {short[lv]: v for lv, v in checks.items()},
        "decision": short.get(decision) if decision else None,
        "paid": f"{budget.calls} calls, projected {budget.spent:,.0f} input tokens",
        "facts": [
            {
                "key": f["key"],
                "type": f.get("type"),
                "text": f["text"],
                **{
                    ("default" if n == "default" else short.get(n, n)): answers[n][f["key"]]
                    for n in answers
                },
            }
            for f in facts_all
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "media_resolution_test.json").write_text(json.dumps(out, indent=1))
    (OUT / "media_resolution_test.md").write_text(report(out))
    print(report(out))


if __name__ == "__main__":
    main()
