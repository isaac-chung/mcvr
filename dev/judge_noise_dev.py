"""Label noise of the pooled frame judge on (source, source) pairs. PAID: two vision models.

Each of the 100 v2 sources is judged against its own facts (the union over its natural and twin
queries, as `dev/judge_labels_dev.py` does), whose facts were already confirmed during
construction by local-frame grounding (Gemini + gpt-6-astra on the fact's own scene). Every
rejection is therefore label noise. Conditions (prompt judge_frames_v1, both models must say yes
to every fact of a query):

    base  the existing judge answers (data/mc/smoke_v2/judge/judged.jsonl; Gemini LOW, up to 16
          uniform index frames, 512 px); no call
    A     Gemini at default media resolution, the same 16 frames at 768 px (astra the same)
    B     as base (Gemini LOW, 512 px), plus the local grounding frames of each fact's scene,
          merged in time order
    C     A + B: default resolution, 768 px, 16 uniform + grounding frames
    D     deployable analogue of B: every index frame (no 16 cap), Gemini LOW, 512 px

Reads src modules read-only (judge_frames, grounding_v2, llm via build_v2). Writes only to
results/smoke_v2/judge_noise/: <condition>.jsonl (per source: frames with paths, answers, costs),
summary.json, report.md.

    uv run --env-file .env python dev/judge_noise_dev.py --estimate
    uv run --env-file .env python dev/judge_noise_dev.py --conditions A B --max-input-tokens 5e6
    uv run --env-file .env python dev/judge_noise_dev.py --report-only
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import yaml

from src import judge_frames as jf
from src.datasets.mc import read_jsonl
from src.frames import FramePolicy, saved_frames
from src.grounding_v2 import LOCAL_FOLDER, FrameStore, Grounder
from src.metrics import ndcg_at_k
from src.twins import norm_text
from src.types import Cost

OUT = Path("results/smoke_v2/judge_noise")
GEM, AST = jf.JUDGES
CONDITIONS = {  # name: (Gemini media resolution, max side, cap on uniform frames, local frames)
    "A": (None, 768, jf.MAX_FRAMES, False),
    "B": ("low", jf.MAX_SIDE, jf.MAX_FRAMES, True),
    "C": (None, 768, jf.MAX_FRAMES, True),
    "D": ("low", jf.MAX_SIDE, None, False),
    "R": ("low", jf.MAX_SIDE, jf.MAX_FRAMES, False),  # base repeated, fresh cache (test-retest)
}
REPEAT_CACHE = OUT / "repeat_cache.sqlite"
PROJ_OPENAI = {512: 230, 768: 450}  # input tokens per astra image (base measured; 768 build_v2)


def load_script(name: str):
    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


JP = load_script("judge_labels_dev")
B2 = JP.load_build_v2()


# ---- inputs ------------------------------------------------------------------------------------


def load_inputs(cfg: dict) -> dict:
    out_dir = Path(cfg["out_dir"])
    rows = JP.read_queries(out_dir)
    cons = {r["source"]: r for r in read_jsonl(out_dir / "constraints.jsonl")}
    judged = {}
    for line in read_jsonl(Path(cfg["judge"]["out_dir"]) / "judged.jsonl"):
        if line["source"] == line["candidate"]:
            judged[line["source"]] = line
    sources = list(dict.fromkeys(r["source"] for r in rows))
    facts_of, of_query, scene_times, types = {}, {}, {}, {}
    for s in sources:
        mine = [r for r in rows if r["source"] == s]
        facts_of[s], oq = jf.source_facts(mine)
        of_query.update(oq)
        assert [f["text"] for f in facts_of[s]] == [f["text"] for f in judged[s]["facts"]], s
        grounding = {g["key"]: g for g in cons[s]["grounding"]}
        key_of = {norm_text(f["text"]): f["key"] for f in facts_of[s]}
        per_fact: dict[str, dict] = {}
        for r in mine:
            for fact in r["facts"]:
                g = grounding[fact.get("key") or fact.get("a_key")]
                cell = per_fact.setdefault(key_of[norm_text(fact["text"])], {"times": set()})
                cell["times"].update(g["frame_times"])
                cell["window"] = g["window"]
        scene_times[s] = per_fact
        types.update({(s, f["key"]): f["type"] for f in facts_of[s]})
    return {
        "rows": rows,
        "cons": cons,
        "judged": judged,
        "sources": sources,
        "facts_of": facts_of,
        "of_query": of_query,
        "scene_times": scene_times,
        "types": types,
    }


def index_paths(store: FrameStore, dataset: str, video) -> tuple[list[Path], list[float]]:
    out = store.index_dir(dataset, video)
    p = store.index_policy
    files = saved_frames(store.video_path(dataset, video), p.n, out, fps=p.fps, max_n=p.max_n)
    data = json.loads((out / "frames.json").read_text())
    return files, [data["times"][i] for i in data["saved"]]


def frame_set(store, root, source, local_times, cond):
    """[(time, path, kind)] of a condition's frames, in time order, and the JPEG bytes."""
    _, side, cap, with_local = CONDITIONS[cond] if cond != "base" else (None, 512, 16, False)
    dataset, v = JP.video(root, source)
    files, times = index_paths(store, dataset, v)
    keep = jf.subsample(len(files), cap) if cap else list(range(len(files)))
    items = [(times[i], files[i], "index") for i in keep]
    if with_local:
        folder = store.root / dataset / LOCAL_FOLDER / v.video_id
        have = {round(t, 3) for t, _, _ in items}
        for t in sorted(local_times):
            file = folder / f"t{round(t * 1000):07d}.jpg"
            if round(t, 3) not in have and file.exists():
                items.append((t, file, "local"))
                have.add(round(t, 3))
    items.sort(key=lambda x: x[0])
    jpegs = [jf.resize_jpeg(path.read_bytes(), side) for _, path, _ in items]
    return items, jpegs


# ---- projection --------------------------------------------------------------------------------


def project(inp, store, root, cond) -> float:
    media, side, _, _ = CONDITIONS[cond]
    template = jf.JUDGE_PROMPT.read_text()
    total = 0.0
    for s in inp["sources"]:
        local = set().union(*(c["times"] for c in inp["scene_times"][s].values()))
        dataset, v = JP.video(root, s)
        n_index = len(store.index_times(dataset, v))
        cap = CONDITIONS[cond][2]
        n = min(cap, n_index) if cap else n_index
        if CONDITIONS[cond][3]:
            n += len(local)
        text = B2.text_tokens(template) + 15 * len(inp["facts_of"][s]) + 8 * n
        total += text + B2.image_tokens(GEM, n, media or "high")
        total += text + n * PROJ_OPENAI[side]
    return total


# ---- running -----------------------------------------------------------------------------------


def repeat_model(spec: str):
    """The base judge model (as build_v2.make_grounder_model) over a fresh cache, so the call is
    really repeated."""
    from src.cache import Cache
    from src.llm import CachedVision, make_vision
    from src.systems.verify import LabelledOpenAIVision

    cache = Cache(REPEAT_CACHE)
    if spec.startswith("openai"):
        inner = LabelledOpenAIVision(spec.split(":", 1)[1])
        inner.send_temperature = False
        inner.max_tokens = 2500
        return CachedVision(inner, cache, spec)
    return make_vision(spec, cache, media_resolution="low")


def run_condition(cond, inp, store, root, budget, workers) -> list[dict]:
    media = CONDITIONS[cond][0]
    if cond == "R":
        models = {spec: B2.GuardedVision(repeat_model(spec), budget, spec) for spec in (GEM, AST)}
    else:
        models = {
            GEM: B2.GuardedVision(B2.make_grounder_model(GEM, media), budget, GEM),
            AST: B2.GuardedVision(B2.make_grounder_model(AST, None), budget, AST),
        }
    grounder = Grounder(models, jf.JUDGE_PROMPT.read_text(), store)
    prepared = {}
    for s in inp["sources"]:
        local = set().union(*(c["times"] for c in inp["scene_times"][s].values()))
        prepared[s] = frame_set(store, root, s, local, cond)

    def call(item):
        s, spec = item
        items, jpegs = prepared[s]
        answers, cost = grounder.on_frames(
            jpegs, [t for t, _, _ in items], inp["facts_of"][s], specs=(spec,)
        )
        return item, answers, cost

    pools = {spec: ThreadPoolExecutor(workers) for spec in (GEM, AST)}
    futures = [pools[spec].submit(call, (s, spec)) for s in inp["sources"] for spec in (GEM, AST)]
    results = {}
    for f in futures:
        (s, spec), answers, cost = f.result()
        results[(s, spec)] = (answers, cost)
    for pool in pools.values():
        pool.shutdown()
    lines = []
    for s in inp["sources"]:
        answers: dict[str, dict] = {}
        cost: dict[str, Cost] = {}
        for spec in (GEM, AST):
            a, c = results[(s, spec)]
            for key, by_model in a.items():
                answers.setdefault(key, {}).update(by_model)
            cost.update(c)
        items, _ = prepared[s]
        lines.append(
            {
                "source": s,
                "condition": cond,
                "frames": [{"t": t, "path": str(p), "kind": k} for t, p, k in items],
                "facts": inp["facts_of"][s],
                "answers": answers,
                "cost": {spec: c.to_dict() for spec, c in cost.items()},
            }
        )
    return lines


def base_lines(inp, store, root) -> list[dict]:
    lines = []
    for s in inp["sources"]:
        items, _ = frame_set(store, root, s, set(), "base")
        j = inp["judged"][s]
        assert len(items) == j["n_frames"], s
        lines.append(
            {
                "source": s,
                "condition": "base",
                "frames": [{"t": t, "path": str(p), "kind": k} for t, p, k in items],
                "facts": j["facts"],
                "answers": j["answers"],
                "cost": j["cost"],
            }
        )
    return lines


# ---- analysis ----------------------------------------------------------------------------------


def yes(ans: dict, spec: str) -> bool:
    return ans.get(spec, {}).get("answer") == "yes"


def rate(xs: list[bool]) -> dict:
    return {"n": len(xs), "rate": round(sum(xs) / len(xs), 4) if xs else None}


def analyse(cond_lines: dict[str, list[dict]], inp) -> dict:
    out = {}
    for cond, lines in cond_lines.items():
        by_s = {line["source"]: line for line in lines}
        groups: dict[str, dict[str, list[bool]]] = {}
        rej = {"gemini_only": 0, "astra_only": 0, "both": 0}
        for r in inp["rows"]:
            ans = by_s[r["source"]]["answers"]
            keys = inp["of_query"][r["query_id"]]
            ok = jf.relevant(ans, keys, jf.JUDGES)
            g_ok, a_ok = jf.relevant(ans, keys, (GEM,)), jf.relevant(ans, keys, (AST,))
            if not ok:
                rej[
                    "both" if not g_ok and not a_ok else "gemini_only" if not g_ok else "astra_only"
                ] += 1
            for name, value in (
                ("all", "all"),
                ("k", str(r["k"])),
                ("set", r["set"]),
                ("dataset", r["dataset"]),
                ("length_bin", r["length_bin"]),
            ):
                groups.setdefault(f"{name}={value}", []).append(ok)
            groups.setdefault("gemini_alone=all", []).append(g_ok)
            groups.setdefault("astra_alone=all", []).append(a_ok)
        facts: dict[str, list[bool]] = {}
        model_no = {GEM: 0, AST: 0}
        n_facts = 0
        for line in by_s.values():
            for f in line["facts"]:
                a = line["answers"][f["key"]]
                ok = yes(a, GEM) and yes(a, AST)
                facts.setdefault("all", []).append(ok)
                facts.setdefault(f["type"], []).append(ok)
                model_no[GEM] += not yes(a, GEM)
                model_no[AST] += not yes(a, AST)
                n_facts += 1
        cost = {}
        for spec in (GEM, AST):
            cs = [line["cost"].get(spec, {}) for line in lines]
            calls = sum(c.get("llm_calls", 0) for c in cs) or 1
            cost[spec] = {
                "calls": calls,
                "tokens_in": sum(c.get("tokens_in", 0) for c in cs),
                "tokens_out": sum(c.get("tokens_out", 0) for c in cs),
                "images": sum(c.get("images", 0) for c in cs),
                "in_per_call": round(sum(c.get("tokens_in", 0) for c in cs) / calls),
                "out_per_call": round(sum(c.get("tokens_out", 0) for c in cs) / calls),
                "in_per_image": round(
                    sum(c.get("tokens_in", 0) for c in cs)
                    / max(1, sum(c.get("images", 0) for c in cs))
                ),
                "fit": fit([(c.get("images", 0), c.get("tokens_in", 0)) for c in cs if c]),
            }
        out[cond] = {
            "queries": {k: rate(v) for k, v in sorted(groups.items())},
            "query_rejections_by_model": rej,
            "facts": {k: rate(v) for k, v in sorted(facts.items())},
            "fact_not_yes_by_model": {"n_facts": n_facts, GEM: model_no[GEM], AST: model_no[AST]},
            "frames_per_call": round(sum(len(x["frames"]) for x in lines) / len(lines), 2),
            "cost": cost,
        }
    return out


def fit(points: list[tuple[int, int]]) -> dict:
    """Least squares tokens_in = text + per_image x images over calls."""
    n = len(points)
    mx = sum(x for x, _ in points) / n
    my = sum(y for _, y in points) / n
    var = sum((x - mx) ** 2 for x, _ in points)
    slope = sum((x - mx) * (y - my) for x, y in points) / var if var else 0.0
    return {"text": round(my - slope * mx), "per_image": round(slope, 1)}


def coverage(base: list[dict], inp) -> dict:
    """Base fact acceptance split by whether any of the base judge frames lies inside the fact's
    grounding window (the scene used in construction)."""
    cells: dict[str, list[bool]] = {}
    for line in base:
        times = [f["t"] for f in line["frames"]]
        for f in line["facts"]:
            w = inp["scene_times"][line["source"]][f["key"]]["window"]
            inside = any(w["start"] - 0.25 <= t <= w["end"] + 0.25 for t in times)
            a = line["answers"][f["key"]]
            cells.setdefault("scene_in_frames" if inside else "scene_missed", []).append(
                yes(a, GEM) and yes(a, AST)
            )
    return {k: rate(v) for k, v in cells.items()}


def evidence_source(lines: list[dict], base: list[dict]) -> dict:
    """For facts rejected in base and accepted in this condition: share whose cited frames
    (both models' first frame) are local grounding frames."""
    b = {x["source"]: x for x in base}
    n = local = 0
    for line in lines:
        for f in line["facts"]:
            a, a0 = line["answers"][f["key"]], b[line["source"]]["answers"][f["key"]]
            if (yes(a, GEM) and yes(a, AST)) and not (yes(a0, GEM) and yes(a0, AST)):
                n += 1
                kinds = [
                    line["frames"][a[s]["frames"][0] - 1]["kind"]
                    for s in (GEM, AST)
                    if a[s]["frames"]
                ]
                local += any(k == "local" for k in kinds)
    return {"rescued_facts": n, "cited_local_frame": local}


def persistence(cond_lines: dict[str, list[dict]]) -> dict:
    """Over facts rejected (not both yes) in at least one condition: in how many conditions each
    is rejected, and the share of base rejections also rejected in each other condition."""
    rej: dict[tuple[str, str], set[str]] = {}
    for cond, lines in cond_lines.items():
        for line in lines:
            for f in line["facts"]:
                a = line["answers"][f["key"]]
                if not (yes(a, GEM) and yes(a, AST)):
                    rej.setdefault((line["source"], f["key"]), set()).add(cond)
    hist: dict[str, int] = {}
    for conds in rej.values():
        hist[str(len(conds))] = hist.get(str(len(conds)), 0) + 1
    base = [c for c in rej.values() if "base" in c]
    also = {c: sum(c in x for x in base) for c in cond_lines if c != "base"}
    return {
        "conditions": list(cond_lines),
        "facts_rejected_somewhere": len(rej),
        "by_number_of_conditions": dict(sorted(hist.items())),
        "base_rejected": len(base),
        "base_rejected_also_in": also,
    }


def examples(cond_lines, inp, n=10) -> list[dict]:
    base = {x["source"]: x for x in cond_lines["base"]}
    others = {c: {x["source"]: x for x in v} for c, v in cond_lines.items() if c != "base"}
    picks = []
    for s, line in base.items():
        for f in line["facts"]:
            a = line["answers"][f["key"]]
            if not (yes(a, GEM) and yes(a, AST)):
                picks.append((s, f))
    random.Random(0).shuffle(picks)
    out = []
    for s, f in picks[:n]:
        st = inp["scene_times"][s][f["key"]]
        dataset, vid = s.split("/", 1)
        folder = Path("data/mc") / dataset / LOCAL_FOLDER / vid
        a = base[s]["answers"][f["key"]]
        out.append(
            {
                "source": s,
                "fact": f["text"],
                "type": f["type"],
                "window": st["window"],
                "grounding_frames": [
                    str(folder / f"t{round(t * 1000):07d}.jpg") for t in sorted(st["times"])
                ],
                "base_frames": [x["path"] for x in base[s]["frames"]],
                "base_frame_times": [round(x["t"], 1) for x in base[s]["frames"]],
                "base": {
                    m.split(":")[0]: (a[m]["model_answer"], a[m]["evidence"]) for m in (GEM, AST)
                },
                **{
                    c: {
                        m.split(":")[0]: (
                            v[s]["answers"][f["key"]][m]["answer"],
                            v[s]["answers"][f["key"]][m]["evidence"],
                        )
                        for m in (GEM, AST)
                    }
                    for c, v in others.items()
                },
            }
        )
    return out


# ---- full-pool projection ----------------------------------------------------------------------


def transitions(base: list[dict], best: list[dict]) -> dict[str, float]:
    """P(both yes under best | base answer pair) over the source facts."""
    b = {x["source"]: x for x in base}
    cnt: dict[str, list[int]] = {}
    for line in best:
        for f in line["facts"]:
            a0 = b[line["source"]]["answers"][f["key"]]
            a1 = line["answers"][f["key"]]
            pair = f"{a0[GEM]['answer']}|{a0[AST]['answer']}"
            c = cnt.setdefault(pair, [0, 0])
            c[0] += yes(a1, GEM) and yes(a1, AST)
            c[1] += 1
    return {k: v[0] / v[1] for k, v in cnt.items()} | {"_n": {k: v[1] for k, v in cnt.items()}}


def pool_projection(cfg, inp, trans, sims=200) -> dict:
    """Expected extra positives and expanded-label nDCG@10 if the whole pool were judged under
    the best condition, by a per-fact transition model from base answer pairs (estimated on the
    source facts, which are true positives). 'upper' applies the transitions to every pattern;
    'lower' keeps facts both models said "no" to as rejected (they are mostly true negatives in
    other candidates)."""
    judged_all: dict[str, dict[str, dict]] = {}
    for line in read_jsonl(Path(cfg["judge"]["out_dir"]) / "judged.jsonl"):
        judged_all.setdefault(line["source"], {})[line["candidate"]] = line["answers"]
    pairs = []  # (query_id, candidate, [base pair per fact])
    for r in inp["rows"]:
        for doc, ans in judged_all.get(r["source"], {}).items():
            if doc in (r["source"], r.get("twin")):
                continue
            pats = [
                f"{ans[k][GEM]['answer']}|{ans[k][AST]['answer']}"
                for k in inp["of_query"][r["query_id"]]
            ]
            pairs.append((r["query_id"], doc, pats))
    base_pos = sum(all(p == "yes|yes" for p in pats) for _, _, pats in pairs)
    rankings = {}
    for split in ("natural", "twin"):
        for sysname in ("S0", "S1", "S2", "S4"):
            path = Path("results/smoke_v2/pilot") / split / f"{sysname}.jsonl"
            rankings[(split, sysname)] = {x["query_id"]: x["ranking"] for x in read_jsonl(path)}
    src = {r["query_id"]: r["source"] for r in inp["rows"]}

    def ndcg(pos: set[tuple[str, str]]) -> dict[str, float]:
        rel: dict[str, dict[str, int]] = {q: {s: 1} for q, s in src.items()}
        for q, d in pos:
            rel[q][d] = 1
        out = {}
        for (split, sysname), by_q in rankings.items():
            vals = [ndcg_at_k(rk, rel[q], 10) for q, rk in by_q.items() if q in rel]
            out[f"{split}/{sysname}"] = sum(vals) / len(vals)
        return out

    base_nd = ndcg({(q, d) for q, d, pats in pairs if all(p == "yes|yes" for p in pats)})
    result = {"base_extra_positives": base_pos, "judged_pairs": len(pairs), "base_ndcg": base_nd}
    for variant in ("lower", "upper"):

        def p_of(pat, variant=variant):
            if variant == "lower" and pat == "no|no":
                return 0.0
            return trans.get(pat, 0.0)

        probs = []
        for q, d, pats in pairs:
            p = 1.0
            for pat in pats:
                p *= p_of(pat)
            probs.append(p)
        rng = random.Random(0)
        nd_sum: dict[str, float] = {}
        for _ in range(sims):
            pos = {(q, d) for (q, d, _), p in zip(pairs, probs, strict=True) if rng.random() < p}
            for k, v in ndcg(pos).items():
                nd_sum[k] = nd_sum.get(k, 0.0) + v / sims
        result[variant] = {
            "expected_extra_positives": round(sum(probs), 1),
            "expected_ndcg": {k: round(v, 4) for k, v in nd_sum.items()},
            "ndcg_change": {k: round(nd_sum[k] - base_nd[k], 4) for k in base_nd},
        }
    return result


def rejudge_cost(cfg, store, root, cond, per_image: dict[str, float], text_per_call: dict) -> dict:
    """Projected input tokens to re-judge every pooled (source, candidate) of judged.jsonl under
    a condition, from measured per-image and per-call text tokens (B and C: no grounding frames
    exist for other candidates; projected with D's frames, i.e. all index frames)."""
    cap = CONDITIONS.get(cond, (None, 512, 16, False))[2]
    if cond in ("B", "C"):
        cap = None
    n_pairs = images = 0
    for line in read_jsonl(Path(cfg["judge"]["out_dir"]) / "judged.jsonl"):
        dataset, v = JP.video(root, line["candidate"])
        n = len(store.index_times(dataset, v))
        images += min(cap, n) if cap else n
        n_pairs += 1
    total = {s: images * per_image[s] + n_pairs * text_per_call[s] for s in per_image}
    return {
        "pairs": n_pairs,
        "images_per_model": images,
        "tokens_in": total,
        "sum": sum(total.values()),
    }


# ---- report ------------------------------------------------------------------------------------


def fmt(cell) -> str:
    return "-" if cell["rate"] is None else f"{cell['rate']:.3f} ({cell['n']})"


def report(summary: dict) -> str:
    st = summary["conditions"]
    conds = list(st)
    lines = ["# Frame judge label noise on (source, source) pairs", ""]
    lines += [summary["setup"], ""]
    lines += ["## Source acceptance per query (both models, all facts)", ""]
    lines += ["| group | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    keys = list(st["base"]["queries"])
    order = sorted(keys, key=lambda k: (k.split("=")[0] != "all", k))
    for k in order:
        lines.append(f"| {k} | " + " | ".join(fmt(st[c]["queries"][k]) for c in conds) + " |")
    lines += ["", "## Fact acceptance by fact type (both models yes)", ""]
    lines += ["| type | " + " | ".join(conds) + " |", "|---|" + "---|" * len(conds)]
    for k in st["base"]["facts"]:
        lines.append(f"| {k} | " + " | ".join(fmt(st[c]["facts"][k]) for c in conds) + " |")
    lines += ["", "## Which model rejects", ""]
    lines += [
        "| condition | queries rejected: Gemini only / astra only / both | facts not yes: Gemini / astra (of n) |"
    ]
    lines += ["|---|---|---|"]
    for c in conds:
        r, m = st[c]["query_rejections_by_model"], st[c]["fact_not_yes_by_model"]
        lines.append(
            f"| {c} | {r['gemini_only']} / {r['astra_only']} / {r['both']} | "
            f"{m[GEM]} / {m[AST]} ({m['n_facts']}) |"
        )
    lines += ["", "## Token cost per call", ""]
    lines += [
        "| condition | frames / call | Gemini in / out per call (per image) | astra in / out per call (per image) | total in |"
    ]
    lines += ["|---|---|---|---|---|"]
    for c in conds:
        g, a = st[c]["cost"][GEM], st[c]["cost"][AST]
        lines.append(
            f"| {c} | {st[c]['frames_per_call']} | {g['in_per_call']} / {g['out_per_call']} "
            f"({g['in_per_image']}) | {a['in_per_call']} / {a['out_per_call']} ({a['in_per_image']}) "
            f"| {g['tokens_in'] + a['tokens_in']:,} |"
        )
    lines += ["", "## Cause diagnostics", ""]
    lines += [
        f"Base fact acceptance by whether a base judge frame falls in the fact's grounding scene: {summary['coverage']}",
        "",
    ]
    for c, e in summary["evidence_source"].items():
        lines.append(
            f"- {c}: facts rejected in base and accepted here: {e['rescued_facts']}; cited a local grounding frame: {e['cited_local_frame']}"
        )
    lines += ["", f"Persistence of fact rejections across conditions: {summary['persistence']}"]
    lines += ["", "## Rejection examples (base; answers as (answer, evidence))", ""]
    for i, ex in enumerate(summary["examples"], 1):
        lines.append(
            f'{i}. `{ex["source"]}` [{ex["type"]}] "{ex["fact"]}" scene {ex["window"]["start"]:.1f}-{ex["window"]["end"]:.1f} s; base frame times {ex["base_frame_times"]}'
        )
        g = ex["grounding_frames"]
        more = f" (+{len(g) - 6} more)" if len(g) > 6 else ""
        lines.append(f"   - grounding frames: {', '.join(g[:6])}{more}")
        lines.append(
            f"   - base frames: {ex['base_frames'][0]} ... ({len(ex['base_frames'])} files in that folder)"
        )
        for c in ["base"] + [c for c in conds if c != "base"]:
            if c in ex:
                lines.append(f"   - {c}: {ex[c]}")
    if "projection" in summary:
        p = summary["projection"]
        lines += ["", f"## Full-pool projection (best condition {p['best']})", ""]
        lines += [p["method"], ""]
        lines += [f"Transition P(both yes under {p['best']} | base pair): {p['transitions']}", ""]
        lines += [
            f"Judged (query, non-source, non-twin) pairs: {p['pool']['judged_pairs']}; base extra positives {p['pool']['base_extra_positives']}",
            "",
        ]
        lines += [
            "| variant | expected extra positives | " + " | ".join(p["pool"]["base_ndcg"]) + " |"
        ]
        lines += ["|---|---|" + "---|" * len(p["pool"]["base_ndcg"])]
        lines.append(
            "| base | "
            + str(p["pool"]["base_extra_positives"])
            + " | "
            + " | ".join(f"{v:.3f}" for v in p["pool"]["base_ndcg"].values())
            + " |"
        )
        for var in ("lower", "upper"):
            v = p["pool"][var]
            lines.append(
                f"| {var} | {v['expected_extra_positives']} | "
                + " | ".join(
                    f"{v['expected_ndcg'][k]:.3f} ({v['ndcg_change'][k]:+.3f})"
                    for k in p["pool"]["base_ndcg"]
                )
                + " |"
            )
        lines += ["", f"Re-judge cost: {p['rejudge_cost']}", ""]
    return "\n".join(lines) + "\n"


# ---- main --------------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/mc_smoke_v2.yaml")
    parser.add_argument("--conditions", nargs="*", default=[])
    parser.add_argument("--estimate", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    parser.add_argument("--max-input-tokens", type=float, default=5e6)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--best", default=None, help="condition used for the projection")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    root = cfg["data_root"]
    store = FrameStore(
        root, index_policy=FramePolicy.from_config(cfg.get("frame_policy"), cfg.get("n_frames", 8))
    )
    inp = load_inputs(cfg)
    OUT.mkdir(parents=True, exist_ok=True)
    spent_file = OUT / "spent.json"
    spent = json.loads(spent_file.read_text()) if spent_file.exists() else {}
    if args.estimate:
        for c in CONDITIONS:
            print(f"{c}: projected {project(inp, store, root, c) / 1e6:.2f}M input tokens")
        print(f"spent so far (actual): {spent}")
        return
    for cond in args.conditions:
        projected = project(inp, store, root, cond)
        used = sum(v["tokens_in"] for v in spent.values())
        if used + projected > args.max_input_tokens:
            raise SystemExit(
                f"{cond}: {used:,.0f} spent + {projected:,.0f} > {args.max_input_tokens:,.0f}"
            )
        budget = B2.Budget(args.max_input_tokens - used)
        lines = run_condition(cond, inp, store, root, budget, args.workers)
        with open(OUT / f"{cond}.jsonl", "w") as handle:
            handle.writelines(json.dumps(x) + "\n" for x in lines)
        spent[cond] = {
            "projected": projected,
            "tokens_in": sum(c.get("tokens_in", 0) for x in lines for c in x["cost"].values()),
            "tokens_out": sum(c.get("tokens_out", 0) for x in lines for c in x["cost"].values()),
        }
        spent_file.write_text(json.dumps(spent, indent=1))
        print(cond, spent[cond])
    cond_lines = {"base": base_lines(inp, store, root)}
    for c in CONDITIONS:
        if (OUT / f"{c}.jsonl").exists():
            cond_lines[c] = list(read_jsonl(OUT / f"{c}.jsonl"))
    summary = {
        "setup": (
            f"{len(inp['sources'])} sources, {len(inp['rows'])} queries (natural + twin), "
            f"{sum(len(v) for v in inp['facts_of'].values())} distinct source facts; prompt "
            "judge_frames_v1 unchanged (it says 'sampled at equal intervals' also when local "
            "frames are added). base = existing judged.jsonl (Gemini LOW, <=16 uniform index "
            "frames, 512 px). A = Gemini default resolution, same frames at 768 px. B = base + "
            "local grounding frames of each fact's scene (512 px, LOW). C = A + B. D = all index "
            "frames, no cap (512 px, LOW). Spend this script (actual input tokens): "
            f"{ {k: v['tokens_in'] for k, v in spent.items()} }."
        ),
        "conditions": analyse(cond_lines, inp),
        "coverage": coverage(cond_lines["base"], inp),
        "evidence_source": {
            c: evidence_source(v, cond_lines["base"]) for c, v in cond_lines.items() if c != "base"
        },
        "persistence": persistence(cond_lines),
        "examples": examples(cond_lines, inp),
        "spent": spent,
    }
    st = summary["conditions"]
    deployable = [c for c in cond_lines if c in ("A", "D")]
    best = args.best or max(
        deployable or ["base"], key=lambda c: st[c]["queries"]["all=all"]["rate"]
    )
    if best != "base":
        trans = transitions(cond_lines["base"], cond_lines[best])
        per_image = {s: st[best]["cost"][s]["fit"]["per_image"] for s in (GEM, AST)}
        text = {s: st[best]["cost"][s]["fit"]["text"] for s in (GEM, AST)}
        summary["projection"] = {
            "best": best,
            "method": (
                "Per-fact transition model: P(both yes under the best deployable condition | the "
                "base answer pair), estimated on the source facts (true positives), applied to "
                "every judged (query, candidate) pair of the full pool (non-source, non-twin); a "
                "pair is relevant when all its facts are. 'lower' keeps no|no facts rejected; "
                "'upper' applies the source rates to every pattern (an overestimate: no|no on "
                "other candidates is mostly a true negative). Expected nDCG@10 over 200 draws; "
                "source always relevant; S0/S1/S2/S4 rankings of results/smoke_v2/pilot. Only "
                "conditions that need no grounding frames (A, D) are deployable for the pool."
            ),
            "transitions": {
                k: (round(v, 3) if not isinstance(v, dict) else v) for k, v in trans.items()
            },
            "pool": pool_projection(cfg, inp, trans),
            "rejudge_cost": rejudge_cost(cfg, store, root, best, per_image, text),
        }
    (OUT / "summary.json").write_text(json.dumps(summary, indent=1))
    (OUT / "report.md").write_text(report(summary))
    print(report(summary))


if __name__ == "__main__":
    main()
