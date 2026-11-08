"""Human label sheets (human_label_plan.md; human_labeling_100src.md for the 100-source test
set). FREE: no model calls; reads the query files, results/full/runs, results/full/select/
{subsets,sources100}.json, data/mc/<dataset>/<descriptions file> and data/mc/full/judge only.

    uv run python human/make_label_sheets.py [--out data/mc/full/human] [--limit 3]

    # the 100-source test set (docs/human_labeling_100src.md, "Tooling changes")
    uv run python human/make_label_sheets.py \
        --sources-file results/full/select/sources100.json --taskb-all \
        --prefill data/mc/full/human --overlap 25 --overlap-bd 10 --overlap-e 10 \
        --out data/mc/full/human100

Writes into --out one static HTML page per task, a media/ folder of symlinks to the raw .mp4s
under opaque names (so neither the page nor the file names tell which video is the source) and
the hidden key.json:

- taskA.html: the twin-both queries where S4@50 and S5 put a different video at rank 1, one
  item per source (a source's twin queries nest, so its queries share facts and often both
  top-1 videos): the longest query, the union of its queries' facts as constraints, and every
  video any of them needs, the two top-1s of each query plus the source and the twin, under
  shuffled letters (system names and roles hidden). Every constraint is marked met / not met /
  can't tell on every video. Per query, "which top-1 meets all" (A, B, both, neither) is read
  off the grid, not asked: a video meets all when every one of the query's constraints is met.
  Twin check, also off the grid: d in the source, d absent from the twin and the shared facts
  in the twin. With --sources-file the nested-set disagreements are merged in the same way
  (one item per source, the source video added, no twin) and Task N is dropped; each item
  records its set ("set_label": nested / near-miss).
- taskA_overlap.html: --overlap Task A items (seed draw; with --sources-file stratified by set
  and forced to include the --prefill folder's overlap items that are unchanged) with the same
  item ids and video letters, for the second annotator (Cohen's kappa on the answers).
- taskB.html: --taskb-nested nested + --taskb-twin twin queries whose source the two frame
  judges reject (not every judge says yes to every fact), equal allocation over k (shortfall
  moved to the other k), seed draw. Every fact of the query is asked present / absent / can't
  tell on the source video (the rejected ones are only in the key).
- taskBD.html (--taskb-all, Task B over all sources; Task D is derived): one item per source:
  the source video and the union of ALL its queries' facts, each present / absent / can't
  tell. The source's queries are shown read-only with their fact numbers; a query is accepted
  when every one of its facts is present (computed in the analysis, not asked). The key
  records which facts a judge rejected and which queries the judges reject.
- taskC.html (--sources-file): each near-miss source without a Task A item: the source and
  its twin (HN) video under shuffled letters, the union of its near-miss queries' facts (the
  longest query's first), met / not met / can't tell on both; the key records the letters,
  d and its index.
- taskE.html (--sources-file): one item per source: the index description (the text S0 / S2 /
  S4 read, data/mc/<dataset>/<descriptions_file>) and the source's facts, each stated /
  implied / absent. No video; the description stands in as "video" D.
- taskBD_overlap.html / taskE_overlap.html: --overlap-bd / --overlap-e items for the second
  annotator.
- taskN.html (optional task, without --sources-file): nested-sample queries where S4@50 and
  S5 disagree at rank 1, Task A's form without the twin videos.

--prefill DIR (repeatable) copies the answers of DIR/{taskA,taskB,taskBD,taskC,taskN}.csv
(first annotator) and DIR/{taskA,taskBD}_overlap.csv (second annotator, into the overlap
pages) into the new sheets where (source, video id, fact key) match, mapping met / not met to
present / absent across tasks on the source video. The --out folder itself is read the same
way, so a page exported into it (Task B, Task C) pre-fills and hints the pages still to do
(Task A's source column comes from Task B); rebuild with --force after each export. Pre-filled cells render as already selected, the item shows a "pre-filled"
badge and key.json lists the cells under "prefill". Every other earlier answer on the same
video (a fact worded differently, or a fact of another query) is NOT copied but shown as a
hint: next to the new fact when the old wording is close (HINT_SIM), else under the video
as "earlier labels on this video"; key.json lists them under "hints". A browser's saved
answers are keyed on the draw (items without the pre-fill), so re-running with more old CSVs
keeps them; pages built before 2026-11-02 keyed on the pre-fill too and are migrated once.
human/carry_over.py reports what an old folder can and cannot pre-fill.

Each page autosaves to the browser and exports <task>.csv (annotator, task, item, question,
video, constraint, answer; question is constraint / fact / stated per page, "note" for free
text; a page built before 2026-11-02 could also hold "askable" rows, which the reader skips);
save it into --out. key.json holds the item order, the video letters, each video's systems
and role (source / twin / other / description), the fact keys and the judges' answers.
--limit N writes only the first N items of each task (pilot); use another --out.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
from collections import Counter
from pathlib import Path

import yaml

from src.datasets.mc import read_jsonl
from src.judge_frames import relevant, source_facts

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/mc_full.yaml"
RUNS = ROOT / "results/full/runs"
SUBSETS = ROOT / "results/full/select/subsets.json"
PAIR = ("S4@50", "S5")
SET_LABEL = {"natural": "nested", "twin": "twin"}
SET_NAME = {"natural": "nested", "twin": "near-miss"}
KS = (1, 2, 3, 4)
PREFIX = {"taskA": "A", "taskB": "B", "taskBD": "B", "taskC": "C", "taskE": "E", "taskN": "N"}


# ---- inputs ------------------------------------------------------------------------------------


def load(cfg: dict, sources_file: str | Path | None = None) -> dict:
    """Queries, fact keys, judge answers and the two systems' top-1s. sources_file ({"sources":
    [...]}, the draw of pipeline/draw_test_set.py) keeps only the queries of those sources,
    as systems.py --sources-file."""
    out_dir = ROOT / cfg["out_dir"]
    rows = read_jsonl(out_dir / "queries.jsonl") + read_jsonl(out_dir / "queries_twin.jsonl")
    sources = None
    if sources_file:
        sources = list(json.loads(Path(sources_file).read_text())["sources"])
        keep = set(sources)
        rows = [r for r in rows if r["source"] in keep]
    by_source: dict[str, list[dict]] = {}
    for r in rows:
        by_source.setdefault(r["source"], []).append(r)
    facts_of, of_query = {}, {}
    for s, rs in by_source.items():  # fact keys exactly as pipeline/judge_labels.py setup
        facts_of[s], oq = source_facts(rs)
        of_query.update(oq)
    judged: dict[str, dict[str, dict]] = {}
    for line in read_jsonl(ROOT / cfg["judge"]["out_dir"] / "judged.jsonl"):
        judged.setdefault(line["source"], {})[line["candidate"]] = line["answers"]
    top1 = {
        name: {
            s: {
                ln["query_id"]: ln["ranking"][0][0] for ln in read_jsonl(RUNS / name / f"{s}.jsonl")
            }
            for s in PAIR
        }
        for name in SET_LABEL
    }
    return {
        "rows": {r["query_id"]: r for r in rows},
        "by_source": by_source,
        "facts_of": facts_of,
        "of_query": of_query,
        "judged": judged,
        "specs": tuple(cfg["judge"]["models"]),
        "top1": top1,
        "subsets": json.loads(SUBSETS.read_text()),
        "sources": sources,
        "descriptions": {
            "file": cfg.get("descriptions_file", "descriptions.jsonl"),
            "prompt": Path(cfg["describe_prompt"]).name,
        },
    }


def load_descriptions(D: dict, sources: list[str]) -> dict[str, str]:
    """{source: index description} from data/mc/<dataset>/<descriptions_file>, the texts the
    description systems read (src.datasets.mc.corpus_descriptions), the describer prompt's
    records only."""
    want: dict[str, set[str]] = {}
    for s in sources:
        dataset, name = s.split("/", 1)
        want.setdefault(dataset, set()).add(name)
    out = {}
    for dataset, names in want.items():
        path = ROOT / "data/mc" / dataset / D["descriptions"]["file"]
        for r in read_jsonl(path):
            if r["video_id"] in names and r.get("prompt") == D["descriptions"]["prompt"]:
                out[f"{dataset}/{r['video_id']}"] = r["description"]
    missing = [s for s in sources if s not in out]
    assert not missing, f"no description for {missing[:5]} in {D['descriptions']['file']}"
    return out


def video_path(vid: str) -> Path:
    dataset, name = vid.split("/", 1)
    path = ROOT / "data/mc" / dataset / "videos" / f"{name}.mp4"
    assert path.exists(), path
    return path


def opaque(vid: str) -> str:
    return hashlib.sha256(f"human-label|{vid}".encode()).hexdigest()[:12]


def judge_answers(D: dict, source: str, vid: str, keys: list[str]) -> list[dict] | None:
    """Per fact {judge: answer} on (source, vid), None when the pair was never judged."""
    answers = D["judged"].get(source, {}).get(vid)
    if answers is None:
        return None
    return [{s: answers.get(k, {}).get(s, {}).get("answer") for s in D["specs"]} for k in keys]


# ---- items -------------------------------------------------------------------------------------


def facts_of(D: dict, r: dict) -> list[dict]:
    keys = D["of_query"][r["query_id"]]
    return [
        {"text": f["text"], "type": f["type"], "key": key, "role": f.get("role")}
        for f, key in zip(r["facts"], keys, strict=True)
    ]


def merged_item(
    D: dict,
    rs: list[dict],
    rng: random.Random,
    top1: bool,
    source: bool,
    twin: bool,
) -> dict:
    """One item for the queries rs of one source: the union of their facts (the longest query's
    first) on the union of their top-1 videos (with top1) plus the source and the twin video
    (with source / twin), so each fact is asked once per video. Letters are shuffled over all
    of the item's videos; each query's top-1 letters per system are in item["queries"]."""
    t = D["top1"][rs[0]["set"]]
    rs = sorted(rs, key=lambda r: -r["k"])
    r0 = rs[0]
    assert len({r.get("twin") for r in rs}) == 1, r0["source"]
    facts, at = [], {}
    for r in rs:
        for f in facts_of(D, r):
            if f["key"] not in at:
                at[f["key"]] = len(facts)
                facts.append(f)
    vids = list(dict.fromkeys(t[s][r["query_id"]] for r in rs for s in PAIR)) if top1 else []
    extra = ([r0["source"]] if source else []) + ([r0["twin"]] if twin else [])
    vids += [v for v in extra if v not in vids]
    rng.shuffle(vids)
    letter = {v: chr(ord("A") + i) for i, v in enumerate(vids)}
    role = {r0["source"]: "source"} | ({r0["twin"]: "twin"} if twin else {})
    keys = [f["key"] for f in facts]
    return {
        "set": r0["set"],
        "set_label": SET_NAME[r0["set"]],
        "source": r0["source"],
        "twin": r0.get("twin"),
        "dataset": r0["dataset"],
        "query": r0["query"],
        "facts": facts,
        "queries": [
            {
                "query_id": r["query_id"],
                "k": r["k"],
                "d_index": r.get("d_index"),
                "facts": [at[k] + 1 for k in D["of_query"][r["query_id"]]],
                "top1": {s: letter[t[s][r["query_id"]]] for s in PAIR} if top1 else None,
            }
            for r in sorted(rs, key=lambda r: r["k"])
        ],
        "videos": {
            letter[v]: {
                "video": v,
                "systems": sorted(
                    {
                        f"{s}:{r['query_id']}"
                        for r in rs
                        for s in PAIR
                        if top1 and t[s][r["query_id"]] == v
                    }
                ),
                "role": role.get(v, "other"),
                "judges": judge_answers(D, r0["source"], v, keys),
            }
            for v in vids
        },
    }


def disagreement_items(
    D: dict,
    name: str,
    pool: list[str],
    rng: random.Random,
    twin: bool,
    merge: bool,
    source: bool | None = None,
) -> list:
    """The queries of pool (one set) where the two systems' top-1 differ: one item per query, or
    with merge one per source (merged_item). source adds the source video (default: with twin,
    the 279-source Task A / Task N behaviour)."""
    t = D["top1"][name]
    ids = [q for q in pool if t[PAIR[0]][q] != t[PAIR[1]][q]]
    groups: dict[str, list[dict]] = {}
    for q in ids:
        groups.setdefault(D["rows"][q]["source"] if merge else q, []).append(D["rows"][q])
    order = list(groups)
    rng.shuffle(order)
    source = twin if source is None else source
    return [merged_item(D, groups[g], rng, top1=True, source=source, twin=twin) for g in order]


def hn_items(D: dict, sources: list[str], rng: random.Random) -> list:
    """Task C: for each near-miss source in sources, its source and twin video (no top-1s) with
    the union of its near-miss queries' facts, the longest query's first."""
    order = list(sources)
    rng.shuffle(order)
    items = []
    for s in order:
        rs = [r for r in D["by_source"][s] if r["set"] == "twin"]
        it = merged_item(D, rs, rng, top1=False, source=True, twin=True)
        longest = it["queries"][-1]  # sorted by k
        d_key = D["of_query"][longest["query_id"]][longest["d_index"] - 1]
        it["d_index"] = next(i + 1 for i, f in enumerate(it["facts"]) if f["key"] == d_key)
        it["d"] = D["rows"][longest["query_id"]]["d"]
        items.append(it)
    return items


def allocate(n: int, avail: dict[int, int]) -> dict[int, int]:
    """Equal split of n over k (remainder to the lowest k), shortfall moved to the other k."""
    take = {k: 0 for k in avail}
    left = n
    while left and any(take[k] < avail[k] for k in avail):
        open_ = [k for k in sorted(avail) if take[k] < avail[k]]
        share, rem = divmod(left, len(open_))
        for i, k in enumerate(open_):
            add = min(share + (i < rem), avail[k] - take[k])
            take[k] += add
            left -= add
    return take


def own_rejected(D: dict, r: dict) -> bool | None:
    """Whether the two frame judges reject the query's own source (None: never judged)."""
    own = D["judged"].get(r["source"], {}).get(r["source"])
    if own is None:
        return None
    return not relevant(own, D["of_query"][r["query_id"]], D["specs"])


def rejected_items(D: dict, quota: dict[str, int], rng: random.Random) -> tuple[list, dict]:
    pop: dict[str, dict[int, list[str]]] = {n: {k: [] for k in KS} for n in quota}
    judged_n: dict[str, Counter] = {n: Counter() for n in quota}
    for q, r in sorted(D["rows"].items()):
        rej = own_rejected(D, r)
        if r["set"] not in quota or rej is None:
            continue
        judged_n[r["set"]][r["k"]] += 1
        if rej:
            pop[r["set"]][r["k"]].append(q)
    strata, picked = {}, []
    for name, n in quota.items():
        take = allocate(n, {k: len(v) for k, v in pop[name].items()})
        strata[SET_LABEL[name]] = {
            str(k): {"judged": judged_n[name][k], "rejected": len(pop[name][k]), "drawn": take[k]}
            for k in KS
        }
        for k in KS:
            picked += rng.sample(pop[name][k], take[k])
    rng.shuffle(picked)
    items = []
    for q in picked:
        r = D["rows"][q]
        keys = D["of_query"][q]
        judges = judge_answers(D, r["source"], r["source"], keys)
        items.append(
            {
                "query_id": q,
                "set": r["set"],
                "source": r["source"],
                "k": r["k"],
                "dataset": r["dataset"],
                "query": r["query"],
                "facts": facts_of(D, r),
                "rejected": [
                    i + 1 for i, a in enumerate(judges) if any(v != "yes" for v in a.values())
                ],
                "videos": {"S": {"video": r["source"], "role": "source", "judges": judges}},
            }
        )
    return items, strata


def source_fact_items(D: dict, sources: list[str], rng: random.Random) -> list:
    """Task B+D: one item per source with queries: every fact of every query of the source (the
    judge.py fact keys, first-seen order) on the source video, and every query with its fact
    numbers for the "could a person ask this" yes / no. The key records the facts a judge
    rejected and the queries the judges reject; the page shows neither."""
    order = [s for s in sources if D["by_source"].get(s)]
    rng.shuffle(order)
    items = []
    for s in order:
        rs = D["by_source"][s]
        facts = D["facts_of"][s]
        keys = [f["key"] for f in facts]
        judges = judge_answers(D, s, s, keys)
        at = {k: i + 1 for i, k in enumerate(keys)}
        queries = [
            {
                "query_id": r["query_id"],
                "set": r["set"],
                "k": r["k"],
                "d_index": r.get("d_index"),
                "query": r["query"],
                "facts": [at[k] for k in D["of_query"][r["query_id"]]],
                "rejected": own_rejected(D, r),
            }
            for r in sorted(rs, key=lambda r: (r["set"], r["k"], r["query_id"]))
        ]
        items.append(
            {
                "source": s,
                "dataset": rs[0]["dataset"],
                "query": "",
                "facts": facts,
                "queries": queries,
                "rejected": [
                    i + 1 for i, a in enumerate(judges or []) if any(v != "yes" for v in a.values())
                ],
                "rejected_queries": [i + 1 for i, q in enumerate(queries) if q["rejected"]],
                "videos": {"S": {"video": s, "role": "source", "judges": judges}},
            }
        )
    return items


def description_items(D: dict, sources: list[str], rng: random.Random) -> list:
    """Task E: one item per source: the index description as "video" D and the source's facts."""
    desc = load_descriptions(D, sources)
    order = [s for s in sources if D["by_source"].get(s)]
    rng.shuffle(order)
    return [
        {
            "source": s,
            "dataset": D["by_source"][s][0]["dataset"],
            "query": "",
            "facts": D["facts_of"][s],
            "videos": {"D": {"video": s, "role": "description", "text": desc[s]}},
        }
        for s in order
    ]


# ---- pre-fill and overlap ----------------------------------------------------------------------

TO_PRESENT = {"met": "present", "not_met": "absent"}
TO_MET = {"present": "met", "absent": "not_met"}


OLD_CSVS = (
    ("taskA", "first"),
    ("taskB", "first"),
    ("taskBD", "first"),
    ("taskC", "first"),
    ("taskN", "first"),
    ("taskA_overlap", "second"),
    ("taskBD_overlap", "second"),
)


def read_old_answers(folder: Path, unexported: bool = True) -> dict[str, dict]:
    """{"first": {(source, video id, fact key): answer}, "second": ..., "by_video": {video id:
    [{"source", "key", "fact", "answer", "who", "item"}]}, "unexported": {old page: videos}}
    from the folder's sheets: taskA, taskB, taskBD, taskC, taskN .csv (first annotator),
    taskA_overlap, taskBD_overlap .csv (second); Task E (stated / implied / absent on the
    description) is not read. A page whose CSV is missing is listed under "unexported"
    (unless unexported=False: the folder being built, whose pages are simply not done)."""
    key = json.loads((folder / "key.json").read_text())
    items = {it["item"]: it for t in key["tasks"].values() for it in t}
    out: dict[str, dict] = {"first": {}, "second": {}, "by_video": {}, "unexported": {}}
    for name, who in OLD_CSVS:
        path = folder / f"{name}.csv"
        if not path.exists():
            page = name.replace("_overlap", "")
            if unexported and page in key["tasks"] and (folder / f"{name}.html").exists():
                out["unexported"][name] = sorted(
                    {
                        v["video"]
                        for it in key["tasks"][page]
                        for v in it["videos"].values()
                        if name != "taskA_overlap" or it["item"] in key.get("taskA_overlap", [])
                    }
                )
            continue
        for r in csv.DictReader(path.open(newline="")):
            if r["question"] in ("note", "askable", "stated"):
                continue
            it = items[r["item"]]
            vid = it["videos"][r["video"]]["video"]
            f = it["facts"][int(r["constraint"]) - 1]
            out[who][(it["source"], vid, f["key"])] = r["answer"]
            out["by_video"].setdefault(vid, []).append(
                {
                    "source": it["source"],
                    "key": f["key"],
                    "fact": f["text"],
                    "answer": r["answer"],
                    "who": who,
                    "item": r["item"],
                }
            )
    return out


def read_answer_folders(folders: list[Path], out: Path | None = None) -> dict[str, dict]:
    """read_old_answers over several folders merged (a later folder's answer wins on the same
    (source, video, fact key)); the folder being built (out) contributes its exported CSVs
    but no "unexported" pages. An unexported page is keyed "<folder name>/<page>"."""
    merged: dict[str, dict] = {"first": {}, "second": {}, "by_video": {}, "unexported": {}}
    for f in folders:
        own = out is not None and f.resolve() == out.resolve()
        one = read_old_answers(f, unexported=not own)
        merged["first"].update(one["first"])
        merged["second"].update(one["second"])
        for vid, rows in one["by_video"].items():
            merged["by_video"].setdefault(vid, []).extend(rows)
        for page, vids in one["unexported"].items():
            merged["unexported"][f"{f.name}/{page}"] = vids
    return merged


HINT_SIM = 0.6  # token Jaccard / sequence ratio at which an old fact is shown next to a new one


def fact_sim(a: str, b: str) -> float:
    """Max of token Jaccard and difflib ratio on lower-cased alphanumeric text."""
    import difflib

    norm = lambda s: re.sub(r"[^a-z0-9 ]", "", s.lower()).split()
    A, B = set(norm(a)), set(norm(b))
    j = len(A & B) / len(A | B) if A | B else 0.0
    return max(j, difflib.SequenceMatcher(None, " ".join(norm(a)), " ".join(norm(b))).ratio())


def hint_items(
    items: list[dict], old: dict, who: str, opts: str, prefill: dict[str, dict] | None = None
) -> dict[str, dict[str, list]]:
    """{item id: {video label: [{"fact", "answer", "row"}]}}: the annotator's earlier answers
    on the same video that are not pre-filled (another fact key). "row" is the 1-based new
    fact the old wording resembles (fact_sim >= HINT_SIM) when that cell is not pre-filled,
    else None (listed under the video). The same fact with the same answer from two folders
    (or two old pages) is shown once; with different answers both are shown."""
    conv = TO_MET if opts == "met" else TO_PRESENT
    out: dict[str, dict] = {}
    for it in items:
        keys = {f["key"] for f in it["facts"]}
        filled = (prefill or {}).get(it["item"], {})
        for lab, v in it["videos"].items():
            if v.get("role") == "description":
                continue
            hints: list[dict] = []
            seen: set[tuple[str, str]] = set()
            for o in old["by_video"].get(v["video"], []):
                if o["who"] != who or (o["source"] == it["source"] and o["key"] in keys):
                    continue
                ans = conv.get(o["answer"], o["answer"])
                if (o["fact"], ans) in seen:
                    continue
                seen.add((o["fact"], ans))
                sims = [(fact_sim(o["fact"], f["text"]), j) for j, f in enumerate(it["facts"], 1)]
                best = max(sims) if sims else (0.0, None)
                row = (
                    best[1] if best[0] >= HINT_SIM and f"c|{lab}|{best[1]}" not in filled else None
                )
                hints.append({"fact": o["fact"], "answer": ans, "row": row})
            if hints:
                out.setdefault(it["item"], {})[lab] = hints
    return out


def prefill_items(items: list[dict], old: dict, opts: str) -> dict[str, dict[str, str]]:
    """{item id: {cell: answer}} for the cells of items whose (source, video, fact) the old
    answers cover, mapped to the page's option set (met <-> present)."""
    conv = TO_MET if opts == "met" else TO_PRESENT
    out = {}
    for it in items:
        cells = {}
        for lab, v in it["videos"].items():
            for j, f in enumerate(it["facts"], 1):
                a = old.get((it["source"], v["video"], f["key"]))
                if a is not None:
                    cells[f"c|{lab}|{j}"] = conv.get(a, a)
        if cells:
            out[it["item"]] = cells
    return out


def unchanged_overlap(items: list[dict], folder: Path) -> list[str]:
    """New Task A item ids whose source was in the old overlap with the same video set and the
    same fact keys (the old second-annotator answers then cover the item)."""
    key = json.loads((folder / "key.json").read_text())
    old = {
        it["source"]: it
        for it in key["tasks"]["taskA"]
        if it["item"] in key.get("taskA_overlap", [])
    }
    out = []
    for it in items:
        o = old.get(it["source"])
        if (
            o
            and {v["video"] for v in o["videos"].values()}
            == {v["video"] for v in it["videos"].values()}
            and [f["key"] for f in o["facts"]] == [f["key"] for f in it["facts"]]
        ):
            out.append(it["item"])
    return out


def draw_overlap(
    items: list[dict], n: int, rng: random.Random, forced: list[str] = ()
) -> list[str]:
    """n item ids, stratified by set_label (largest remainder), forced ids first in their set."""
    n = min(n, len(items))
    groups: dict[str, list[str]] = {}
    for it in items:
        groups.setdefault(it.get("set_label", ""), []).append(it["item"])
    total = len(items)
    quota = {g: n * len(ids) // total for g, ids in groups.items()}
    rest = sorted(groups, key=lambda g: -(n * len(groups[g]) % total))
    for g in rest[: n - sum(quota.values())]:
        quota[g] += 1
    picked = []
    for g, ids in groups.items():
        take = [i for i in forced if i in ids][: quota[g]]
        pool = [i for i in ids if i not in take]
        take += rng.sample(pool, quota[g] - len(take))
        picked += take
    return sorted(picked, key=lambda i: int(i[1:]))


# ---- pages -------------------------------------------------------------------------------------

INTRO = {
    "taskA": (
        "Each item has a search query split into constraints, and two to seven videos. Watch "
        "each. For each video mark every constraint met, not met or can't tell (can't tell = "
        "the video does not let you decide, e.g. too dark or too fast). A constraint is met if "
        "it holds anywhere in the video."
    ),
    "taskB": (
        "Each item has a search query split into facts, and one video. Watch it and mark each "
        "fact present (it holds somewhere in the video), absent, or can't tell (the video does "
        "not let you decide)."
    ),
    "taskBD": (
        "Each item has one video and a list of facts. Watch it and mark each fact present (it "
        "holds somewhere in the video), absent, or can't tell (the video does not let you "
        "decide). The search queries built from these facts are listed under the video for "
        "context only; nothing is asked about them."
    ),
    "taskC": (
        "Each item has a list of constraints and two videos. Watch both. For each video mark "
        "every constraint met, not met or can't tell (can't tell = the video does not let you "
        "decide). A constraint is met if it holds anywhere in the video."
    ),
    "taskE": (
        "Each item has a short text describing a video (you do not see the video) and a list of "
        "facts. For each fact say whether the text states it (said outright), implies it (a "
        "reader would conclude it from the text), or leaves it absent (nothing in the text "
        "supports it)."
    ),
}
INTRO["taskN"] = INTRO["taskA"].replace("two to seven videos. Watch each", "two videos. Watch both")

PAGES = {
    "taskA": ("Task A: top-1 disagreements", "taskA", "met", "constraint"),
    "taskA_overlap": ("Task A (overlap)", "taskA", "met", "constraint"),
    "taskB": ("Task B: fact check", "taskB", "present", "fact"),
    "taskBD": ("Task B: facts on every source", "taskBD", "present", "fact"),
    "taskBD_overlap": ("Task B (overlap)", "taskBD", "present", "fact"),
    "taskC": ("Task C: pairs", "taskC", "met", "constraint"),
    "taskE": ("Task E: text check", "taskE", "stated", "stated"),
    "taskE_overlap": ("Task E (overlap)", "taskE", "stated", "stated"),
    "taskN": ("Optional: nested disagreements", "taskN", "met", "constraint"),
}
OPTIONS = {
    "met": [["met", "met"], ["not_met", "not met"], ["cant_tell", "can't tell"]],
    "present": [["present", "present"], ["absent", "absent"], ["cant_tell", "can't tell"]],
    "stated": [["stated", "stated"], ["implied", "implied"], ["absent", "absent"]],
}
ASK = "The queries built from these facts (nothing to answer)"


def page_items(items: list[dict], prefill: dict[str, dict] | None = None) -> list[dict]:
    """What the page shows: item id, query, constraints, videos (src, or text for Task E), the
    per-query yes / no of Task B+D, and the pre-filled cells. No source, role or system."""
    out = []
    for it in items:
        p = {
            "id": it["item"],
            "query": it["query"],
            "constraints": [f["text"] for f in it["facts"]],
            "videos": [
                {"label": lab, "text": v["text"]}
                if v.get("role") == "description"
                else {"label": lab, "src": f"media/{opaque(v['video'])}.mp4"}
                for lab, v in it["videos"].items()
            ],
        }
        if it.get("queries") and "query" in it["queries"][0]:
            p["queries"] = [{"text": q["query"], "facts": q["facts"]} for q in it["queries"]]
        if prefill and prefill.get(it["item"]):
            p["prefill"] = prefill[it["item"]]
        out.append(p)
    return out


def draw_id(page_items_: list[dict], with_prefill: bool) -> str:
    """The id a browser keys its saved answers on. Without the pre-fill (the default since
    2026-11-02) a rebuild that pre-fills more keeps them; with it is the pre-2026-11-02 id."""
    its = (
        page_items_
        if with_prefill
        else [{k: v for k, v in it.items() if k != "prefill"} for it in page_items_]
    )
    return hashlib.sha256(json.dumps(its).encode()).hexdigest()[:8]


def write_page(
    out: Path,
    page: str,
    items: list[dict],
    prefill: dict | None = None,
    hints: dict | None = None,
) -> None:
    title, intro, opts, question = PAGES[page]
    data = {
        "task": page,
        "title": title,
        "intro": INTRO[intro],
        "options": OPTIONS[opts],
        "question": question,
        "ask": ASK,
        "items": page_items(items, prefill),
        "hints": hints or {},
    }
    # saved answers are tied to this draw, so a regenerated sheet never reuses them; the page
    # copies answers saved under the legacy id (draw with the pre-fill) once
    data["draw"] = draw_id(data["items"], with_prefill=False)
    data["legacy_draws"] = [draw_id(data["items"], with_prefill=True)]
    blob = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    html = TEMPLATE.replace("__TITLE__", title).replace("__DATA__", blob)
    (out / f"{page}.html").write_text(html)


def link_media(out: Path, vids: set[str]) -> None:
    media = out / "media"
    media.mkdir(parents=True, exist_ok=True)
    for v in sorted(vids):
        link = media / f"{opaque(v)}.mp4"
        target = os.path.relpath(video_path(v), media)
        if link.is_symlink() or link.exists():
            if os.readlink(link) == target:
                continue
            link.unlink()
        link.symlink_to(target)


def cells(items: list[dict]) -> int:
    """Answers a page asks: one per (video, fact)."""
    return sum(len(it["videos"]) * len(it["facts"]) for it in items)


# ---- main --------------------------------------------------------------------------------------


def build(args: argparse.Namespace) -> dict:
    """The tasks, overlap draws and pre-fill of one run; writes nothing. Returns {"tasks",
    "overlap", "prefill", "key", "strata"}."""
    cfg = yaml.safe_load(Path(args.config).read_text())
    D = load(cfg, args.sources_file)
    rng = lambda: random.Random(args.seed)  # one fresh draw per task
    tasks: dict[str, list] = {}
    strata = None
    if D["sources"] is None:
        tasks["taskA"] = disagreement_items(
            D, "twin", D["subsets"]["twin_both"]["twin"], rng(), twin=True, merge=True
        )
        tasks["taskN"] = disagreement_items(
            D, "natural", D["subsets"]["nested_sample"]["natural"], rng(), twin=False, merge=False
        )
    else:
        pool = {n: [q for q, r in D["rows"].items() if r["set"] == n] for n in SET_LABEL}
        r = rng()
        nested = disagreement_items(
            D, "natural", pool["natural"], r, twin=False, merge=True, source=True
        )
        near = disagreement_items(D, "twin", pool["twin"], r, twin=True, merge=True)
        tasks["taskA"] = nested + near
        r.shuffle(tasks["taskA"])
        in_a = {it["source"] for it in near}
        hn_sources = sorted({D["rows"][q]["source"] for q in pool["twin"]} - in_a)
        tasks["taskC"] = hn_items(D, hn_sources, rng())
    if args.taskb_all:
        sources = D["sources"] or sorted(D["by_source"])
        tasks["taskBD"] = source_fact_items(D, sources, rng())
    else:
        tasks["taskB"], strata = rejected_items(
            D, {"natural": args.taskb_nested, "twin": args.taskb_twin}, rng()
        )
    if D["sources"] is not None:
        tasks["taskE"] = description_items(D, D["sources"], rng())
    if args.limit:
        tasks = {t: its[: args.limit] for t, its in tasks.items()}
    for t, its in tasks.items():
        for i, it in enumerate(its, 1):
            it["item"] = f"{PREFIX[t]}{i:02d}"

    # the folder being built is read too: pages exported into it (Task B, C) feed the others
    folders = [Path(f) for f in args.prefill or []]
    out_dir = Path(args.out)
    if (
        folders
        and out_dir.resolve() not in {f.resolve() for f in folders}
        and (out_dir / "key.json").exists()
    ):
        folders.append(out_dir)
    old = read_answer_folders(folders, out_dir) if folders else None
    forced = unchanged_overlap(tasks["taskA"], folders[0]) if old else []
    overlap = {"taskA": draw_overlap(tasks["taskA"], args.overlap, rng(), forced)}
    if "taskBD" in tasks and args.overlap_bd:
        overlap["taskBD"] = draw_overlap(tasks["taskBD"], args.overlap_bd, rng())
    if "taskE" in tasks and args.overlap_e:
        overlap["taskE"] = draw_overlap(tasks["taskE"], args.overlap_e, rng())
    pages = dict(tasks)
    for t, ids in overlap.items():
        pages[f"{t}_overlap"] = [it for it in tasks[t] if it["item"] in ids]

    prefill: dict[str, dict] = {}
    hints: dict[str, dict] = {}
    if old:
        for page, its in pages.items():
            opts = PAGES[page][2]
            if opts == "stated":
                continue
            who = "second" if page.endswith("_overlap") else "first"
            prefill[page] = prefill_items(its, old[who], opts)
            hints[page] = hint_items(its, old, who, opts, prefill[page])
        for t, its in tasks.items():
            for it in its:
                it["prefilled"] = len(prefill.get(t, {}).get(it["item"], {}))
                it["hinted"] = sum(len(h) for h in hints.get(t, {}).get(it["item"], {}).values())

    vids = {
        v["video"]
        for its in tasks.values()
        for it in its
        for v in it["videos"].values()
        if v.get("role") != "description"
    }
    key = {
        "plan": "human_labeling_100src.md" if D["sources"] is not None else "human_label_plan.md",
        "seed": args.seed,
        "sources_file": args.sources_file,
        "systems": list(PAIR),
        "judges": list(D["specs"]),
        "limit": args.limit,
        "taskB_strata": strata,
        "taskA_overlap": overlap["taskA"],
        "overlap": overlap,
        "overlap_forced": forced,
        "prefill_from": [str(f) for f in folders],
        "prefill_counts": {p: sum(len(c) for c in v.values()) for p, v in prefill.items()},
        "prefill": prefill,
        "hint_counts": {
            p: sum(len(h) for i in v.values() for h in i.values()) for p, v in hints.items()
        },
        "hints": hints,
        "unexported": old["unexported"] if old else {},
        "media": {opaque(v): v for v in sorted(vids)},
        "tasks": tasks,
    }
    return {
        "tasks": tasks,
        "pages": pages,
        "overlap": overlap,
        "prefill": prefill,
        "hints": hints,
        "key": key,
        "strata": strata,
        "vids": vids,
    }


def write(args: argparse.Namespace, B: dict) -> Path:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    key_path = out / "key.json"
    text = json.dumps(B["key"], indent=1, ensure_ascii=False)
    if key_path.exists() and key_path.read_text() != text and not args.force:
        raise SystemExit(f"{key_path} exists with a different draw; pass --force to replace it")
    key_path.write_text(text)
    link_media(out, B["vids"])
    for page, its in B["pages"].items():
        write_page(out, page, its, B["prefill"].get(page), B["hints"].get(page))
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", default=str(CONFIG))
    ap.add_argument("--out", default="data/mc/full/human")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sources-file", help="restrict every task to these sources ({'sources'})")
    ap.add_argument("--overlap", type=int, default=15, help="Task A items for the second annotator")
    ap.add_argument("--overlap-bd", type=int, default=0, help="Task B+D overlap items")
    ap.add_argument("--overlap-e", type=int, default=0, help="Task E overlap items")
    ap.add_argument("--taskb-nested", type=int, default=30)
    ap.add_argument("--taskb-twin", type=int, default=20)
    ap.add_argument("--taskb-all", action="store_true", help="Task B+D: every fact of every source")
    ap.add_argument(
        "--prefill",
        action="append",
        help="earlier sheets folder (key.json + CSVs) to copy answers from; repeatable, the "
        "first one also forces its unchanged overlap items; --out itself is read too",
    )
    ap.add_argument("--limit", type=int, help="first N items of each task only (pilot)")
    ap.add_argument("--force", action="store_true", help="overwrite a different key.json")
    return ap


def main() -> None:
    args = parser().parse_args()
    B = build(args)
    out = write(args, B)
    pages = B["pages"]
    print(f"wrote {out}/{{{','.join(pages)}}}.html, key.json, media/ ({len(B['vids'])} videos)")
    for t, its in pages.items():
        n_vid = sum(len(it["videos"]) for it in its)
        line = f"{t}: {len(its)} items, {n_vid} video slots, {cells(its)} cells"
        if B["prefill"].get(t):
            line += f", {B['key']['prefill_counts'][t]} pre-filled"
        if B["key"]["hint_counts"].get(t):
            line += f", {B['key']['hint_counts'][t]} hints"
        if t == "taskA":
            qs = [q for it in its for q in it["queries"]]
            line += (
                f"; {len(qs)} queries, k {dict(sorted(Counter(q['k'] for q in qs).items()))}, "
                f"sets {dict(Counter(it['set_label'] for it in its))}, "
                f"constraints {sum(len(it['facts']) for it in its)}, videos per item "
                f"{dict(sorted(Counter(len(it['videos']) for it in its).items()))}"
            )
        if t == "taskBD":
            line += (
                f"; facts {sum(len(it['facts']) for it in its)}, rejected facts "
                f"{sum(len(it['rejected']) for it in its)}, queries "
                f"{sum(len(it['queries']) for it in its)}, rejected queries "
                f"{sum(len(it['rejected_queries']) for it in its)}"
            )
        if t == "taskB":
            line += f"; strata {json.dumps(B['strata'])}"
        print(line)
    print(f"overlap: {json.dumps(B['overlap'])}; forced {B['key']['overlap_forced']}")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #fff; --fg: #1d1d1f; --muted: #6e6e73; --line: #d2d2d7; --card: #f5f5f7;
          --accent: #0a66c2; --ok: #1a7f37; --todo: #b35900; --pre: #6b4fbb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161618; --fg: #f2f2f2; --muted: #a1a1a6; --line: #3a3a3c; --card: #222225;
            --accent: #4ea1ff; --ok: #4ac26b; --todo: #f0a050; --pre: #b39ddb; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg);
         font: 15px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
  header { position: sticky; top: 0; z-index: 2; background: var(--bg);
           border-bottom: 1px solid var(--line); padding: 10px 16px;
           display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center; }
  header h1 { font-size: 16px; margin: 0 8px 0 0; }
  button, input[type=text] { font: inherit; padding: 5px 10px; border-radius: 6px;
           border: 1px solid var(--line); background: var(--card); color: var(--fg); }
  button { cursor: pointer; }
  button.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
  main { padding: 16px; max-width: 1500px; margin: 0 auto; }
  .intro { color: var(--muted); margin: 0 0 14px; }
  .query { font-size: 18px; margin: 4px 0 6px; }
  ol.cons { margin: 0 0 14px; padding-left: 22px; }
  .videos { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr)); gap: 14px; }
  .vid { background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 10px; }
  .vid h2 { font-size: 15px; margin: 0 0 6px; }
  video { width: 100%; max-height: 48vh; background: #000; border-radius: 6px; }
  .desc { font-size: 16px; line-height: 1.5; padding: 10px 12px; border: 1px solid var(--line);
          border-radius: 6px; background: var(--bg); margin: 0 0 6px; }
  table { width: 100%; border-collapse: collapse; margin-top: 8px; }
  td { padding: 4px 4px; border-top: 1px solid var(--line); vertical-align: top; }
  td.o { width: 1%; white-space: nowrap; padding-left: 10px; }
  td.o label { display: block; margin: 0; line-height: 1.7; }
  td.o.pre label { color: var(--pre); }
  tr.hint td { color: var(--muted); font-size: 12px; border-top: 0; padding: 0 4px 6px 18px; }
  tr.hint b { color: var(--pre); font-weight: normal; }
  .hints { color: var(--muted); font-size: 12px; margin: 6px 0 0; }
  .hints summary { cursor: pointer; }
  .hints ul { margin: 4px 0 0; padding-left: 18px; }
  .hints b { color: var(--pre); font-weight: normal; }
  label { margin-right: 10px; cursor: pointer; }
  .badge { display: inline-block; font-size: 12px; padding: 1px 8px; border-radius: 10px;
           border: 1px solid var(--pre); color: var(--pre); margin-left: 8px; }
  .bd { display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 14px;
        align-items: start; }
  .bd .qs { margin-top: 14px; }
  .bd .vid table { margin-top: 0; }
  .qs { margin-top: 14px; background: var(--card); border: 1px solid var(--line);
        border-radius: 10px; padding: 10px; }
  .qs h2 { font-size: 15px; margin: 0 0 6px; }
  .qs td.f { color: var(--muted); white-space: nowrap; width: 1%; padding-right: 10px; }
  textarea { width: 100%; min-height: 50px; font: inherit; background: var(--card); color: var(--fg);
             border: 1px solid var(--line); border-radius: 6px; padding: 6px; }
  .nav { display: flex; gap: 8px; margin: 14px 0; flex-wrap: wrap; }
  .dots { display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px; }
  .dot { width: 34px; padding: 2px 0; text-align: center; font-size: 11px; border-radius: 4px;
         border: 1px solid var(--line); cursor: pointer; color: var(--todo); }
  .dot.done { color: var(--ok); }
  .dot.pre { border-color: var(--pre); }
  .dot.cur { outline: 2px solid var(--accent); }
  .status { color: var(--muted); }
</style>
</head>
<body>
<header>
  <h1 id="title"></h1>
  <label>Annotator <input type="text" id="annot" size="10"></label>
  <span class="status" id="progress"></span>
  <button class="primary" id="dl">Download CSV</button>
  <label><button id="loadbtn" type="button">Load CSV</button>
    <input type="file" id="load" accept=".csv" hidden></label>
</header>
<main>
  <p class="intro" id="intro"></p>
  <div id="item"></div>
  <div class="nav">
    <button id="prev">&larr; Previous</button>
    <button id="next">Next &rarr;</button>
    <button id="todo">Next unanswered</button>
  </div>
  <div class="dots" id="dots"></div>
</main>
<script>
const DATA = __DATA__;
const STORE = "human-label/" + DATA.task + "/" + DATA.draw;
let S = { annot: "", cur: 0, a: {} };
try {
  let saved = localStorage.getItem(STORE);
  // answers saved by this page before 2026-11-02 sit under the draw id that included the pre-fill
  for (const d of (saved ? [] : DATA.legacy_draws || [])) {
    saved = localStorage.getItem("human-label/" + DATA.task + "/" + d);
    if (saved) { localStorage.setItem(STORE, saved); break; }
  }
  S = Object.assign(S, JSON.parse(saved || "{}"));
} catch (e) {}
const save = () => { try { localStorage.setItem(STORE, JSON.stringify(S)); } catch (e) {} };
const $ = id => document.getElementById(id);
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
$("title").textContent = DATA.title;
document.title = DATA.title;
$("intro").textContent = DATA.intro;
$("annot").value = S.annot;
$("annot").oninput = e => { S.annot = e.target.value.trim(); save(); };

function needed(it) {
  const keys = [];
  it.videos.forEach(v => it.constraints.forEach((_, j) => keys.push(`c|${v.label}|${j + 1}`)));
  return keys;
}
// the answer of a cell: the annotator's, else the pre-filled one (shown selected)
const ans = (it, k) => (S.a[it.id] || {})[k] || (it.prefill || {})[k];
const done = it => needed(it).every(k => ans(it, k));
function set(id, k, v) { (S.a[id] = S.a[id] || {})[k] = v; save(); status(); }

function radios(name, opts, cur, id, k) {
  return opts.map(([val, lab]) =>
    `<label><input type="radio" name="${esc(name)}" value="${val}" ${cur === val ? "checked" : ""}
      onchange="set('${id}', '${k}', '${val}')"> ${esc(lab)}</label>`).join("");
}

function render() {
  const it = DATA.items[S.cur], a = S.a[it.id] || {}, pre = it.prefill || {};
  const hints = (DATA.hints || {})[it.id] || {};
  const npre = Object.keys(pre).length, nhint = Object.values(hints).reduce((n, l) => n + l.length, 0);
  let h = `<div class="status">Item ${S.cur + 1} of ${DATA.items.length} &middot; ${esc(it.id)}`;
  if (npre) h += `<span class="badge">pre-filled: ${npre} of ${needed(it).length} answers from your earlier sheet; check and change if needed</span>`;
  if (nhint) h += `<span class="badge">${nhint} earlier answers on these videos shown as hints (not copied)</span>`;
  h += `</div>`;
  if (it.query) h += `<p class="query">${esc(it.query)}</p>`;
  h += `<ol class="cons">`;
  it.constraints.forEach(c => h += `<li>${esc(c)}</li>`);
  // Task B+D: the video and the queries on the left, the facts on the right
  const facts = v => {
    const vh = hints[v.label] || [];
    let t = `<table>`;
    it.constraints.forEach((c, j) => {
      const k = `c|${v.label}|${j + 1}`;
      const cls = pre[k] && !a[k] ? "o pre" : "o";
      t += `<tr><td class="c">${j + 1}. ${esc(c)}</td><td class="${cls}">${radios(it.id + k, DATA.options, ans(it, k), it.id, k)}</td></tr>`;
      vh.filter(x => x.row === j + 1).forEach(x =>
        t += `<tr class="hint"><td colspan="2">earlier label on this video: &ldquo;${esc(x.fact)}&rdquo; &rarr; <b>${esc(x.answer)}</b></td></tr>`);
    });
    t += `</table>`;
    const rest = vh.filter(x => !x.row);
    if (rest.length) t += `<details class="hints"${rest.length <= 4 ? " open" : ""}><summary>${rest.length} earlier label${rest.length > 1 ? "s" : ""} on this video (other facts, not copied)</summary><ul>` + rest.map(x =>
      `<li>&ldquo;${esc(x.fact)}&rdquo; &rarr; <b>${esc(x.answer)}</b></li>`).join("") + `</ul></details>`;
    return t;
  };
  const media = v => v.text !== undefined
    ? `<h2>Text ${esc(v.label)}</h2><p class="desc">${esc(v.text)}</p>`
    : `<h2>Video ${esc(v.label)}</h2><video controls preload="metadata" src="${esc(v.src)}"></video>`;
  const queries = () => {
    let t = `<div class="qs"><h2>${esc(DATA.ask)}</h2><table>`;
    it.queries.forEach(q => {
      t += `<tr><td class="f">facts ${q.facts.join(", ")}</td><td class="c">${esc(q.text)}</td></tr>`;
    });
    return t + `</table></div>`;
  };
  h += `</ol>`;
  if (it.queries) {
    const v = it.videos[0];
    h += `<div class="bd"><div><div class="vid">${media(v)}</div>${queries()}</div><div class="vid">${facts(v)}</div></div>`;
  } else {
    h += `<div class="videos">`;
    it.videos.forEach(v => h += `<div class="vid">${media(v)}${facts(v)}</div>`);
    h += `</div>`;
  }
  h += `<label>Note (optional)<textarea id="note">${esc(a.note || "")}</textarea></label>`;
  $("item").innerHTML = h;
  $("note").oninput = e => set(it.id, "note", e.target.value);
  // one video plays at a time
  document.querySelectorAll("video").forEach(v => v.onplay = () =>
    document.querySelectorAll("video").forEach(o => { if (o !== v) o.pause(); }));
  status();
  window.scrollTo(0, 0);
}

function status() {
  const n = DATA.items.filter(done).length;
  $("progress").textContent = `${n} / ${DATA.items.length} complete`;
  $("dots").innerHTML = DATA.items.map((it, i) =>
    `<span class="dot ${done(it) ? "done" : ""} ${it.prefill ? "pre" : ""} ${i === S.cur ? "cur" : ""}" onclick="go(${i})">${esc(it.id)}</span>`).join("");
}
function go(i) { S.cur = Math.max(0, Math.min(DATA.items.length - 1, i)); save(); render(); }
$("prev").onclick = () => go(S.cur - 1);
$("next").onclick = () => go(S.cur + 1);
$("todo").onclick = () => {
  const n = DATA.items.length;
  for (let d = 1; d <= n; d++) { const i = (S.cur + d) % n; if (!done(DATA.items[i])) return go(i); }
  alert("All items are complete.");
};

const COLS = ["annotator", "task", "item", "question", "video", "constraint", "answer"];
const cell = v => /[",\n]/.test(v) ? `"${String(v).replace(/"/g, '""')}"` : String(v);
$("dl").onclick = () => {
  if (!S.annot) { alert("Enter your annotator name first."); return $("annot").focus(); }
  const rows = [COLS];
  DATA.items.forEach(it => {
    const a = S.a[it.id] || {};
    it.videos.forEach(v => it.constraints.forEach((_, j) => {
      const k = `c|${v.label}|${j + 1}`, x = ans(it, k);
      if (x) rows.push([S.annot, DATA.task, it.id, DATA.question, v.label, j + 1, x]);
    }));
    if (a.note) rows.push([S.annot, DATA.task, it.id, "note", "", "", a.note]);
  });
  const blob = new Blob([rows.map(r => r.map(cell).join(",")).join("\n") + "\n"], {type: "text/csv"});
  const url = URL.createObjectURL(blob), link = document.createElement("a");
  link.href = url; link.download = DATA.task + ".csv"; link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
};

function parseCSV(text) {
  const out = []; let row = [], f = "", q = false;
  for (let i = 0; i < text.length; i++) {
    const c = text[i];
    if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
    else if (c === '"') q = true;
    else if (c === ",") { row.push(f); f = ""; }
    else if (c === "\n") { row.push(f); out.push(row); row = []; f = ""; }
    else if (c !== "\r") f += c;
  }
  if (f || row.length) { row.push(f); out.push(row); }
  return out;
}
$("loadbtn").onclick = () => $("load").click();
$("load").onchange = async e => {
  const file = e.target.files[0]; if (!file) return;
  const [head, ...rows] = parseCSV(await file.text());
  const ix = Object.fromEntries(head.map((h, i) => [h, i]));
  let n = 0;
  rows.forEach(r => {
    if (r[ix.task] !== DATA.task) return;
    const id = r[ix.item], qn = r[ix.question], a = (S.a[id] = S.a[id] || {});
    if (qn === "note") a.note = r[ix.answer];
    else if (qn === "askable") a[`q|${r[ix.constraint]}`] = r[ix.answer];
    else a[`c|${r[ix.video]}|${r[ix.constraint]}`] = r[ix.answer];
    if (!S.annot) S.annot = r[ix.annotator];
    n++;
  });
  $("annot").value = S.annot; save(); render();
  alert(`Loaded ${n} answers.`);
};
render();
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
