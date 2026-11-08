"""Smoke pilot v2 (dev/run_systems_dev.py): subset, ledger, twin statistics, evaluation;
offline."""

import importlib.util
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.types import Cost

SCRIPT = Path(__file__).resolve().parents[1] / "dev/run_systems_dev.py"


@pytest.fixture(scope="module")
def p2():
    spec = importlib.util.spec_from_file_location("pilot_v2", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BINS = ("lt30s", "30to60s", "1to2min", "2to4min")


def make_sets(n_sources=24, twin_drop_k4=(0, 1)):
    natural, twin = [], []
    for i in range(n_sources):
        src = f"vript_long/v{i}" if i % 2 else f"shot2story/v{i}"
        b = BINS[i % 4]
        for k in (1, 2, 3, 4):
            base = {"source": src, "k": k, "length_bin": b, "dataset": src.split("/")[0]}
            natural.append({**base, "query_id": f"{src}#k{k}", "query": f"n {i} {k}"})
            if k == 4 and i in twin_drop_k4:
                continue
            twin.append(
                {
                    **base,
                    "query_id": f"{src}#twin-k{k}",
                    "query": f"t {i} {k}",
                    "twin": f"other/b{i}",
                }
            )
    return {"natural": natural, "twin": twin}


def test_allocate_floor_and_proportion(p2):
    alloc = p2.allocate({"a": 50, "b": 30, "c": 2, "d": 0}, 10)
    assert sum(alloc.values()) == 10
    assert alloc["c"] >= 1 and alloc["d"] == 0
    assert alloc["a"] > alloc["b"] > alloc["c"]
    assert p2.allocate({"a": 2, "b": 1}, 10) == {"a": 2, "b": 1}  # capped by what exists


def test_matched_subset_is_balanced_shared_and_seeded(p2):
    sets = make_sets()
    keys = p2.matched_subset(sets, 16, seed=0)
    assert keys == p2.matched_subset(sets, 16, seed=0)
    assert Counter(k for _, k in keys) == {1: 4, 2: 4, 3: 4, 4: 4}
    both = {(r["source"], r["k"]) for r in sets["twin"]}
    assert all(key in both for key in keys)  # k=4 pairs missing from twin are never drawn
    bin_of = {(r["source"], r["k"]): r["length_bin"] for r in sets["natural"]}
    for k in (1, 2, 3, 4):
        assert {bin_of[x] for x in keys if x[1] == k} == set(BINS)  # every bin covered
    ids = p2.subset_query_ids(sets["twin"], keys)
    assert len(ids) == 16 and all("#twin-k" in q for q in ids)


def test_ledger_stops_over_cap(p2, tmp_path):
    ledger = p2.Ledger(tmp_path / "spend.json", cap=1000)
    ledger.check("a", 400)
    ledger.add("a", Cost(tokens_in=600, tokens_out=5), calls=3)
    again = p2.Ledger(tmp_path / "spend.json", cap=1000)
    assert again.total == 600
    again.check("b", 300)
    with pytest.raises(SystemExit):
        again.check("b", 300, reserve=200)


def test_diff_of_means_ci(p2):
    a, b = [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]
    cl = ["s1", "s1", "s2", "s2"]
    m, lo, hi = p2.diff_of_means_ci(a, cl, b, cl)
    assert m == pytest.approx(0.5)
    assert 0.0 <= lo <= m <= hi <= 1.0
    m, lo, hi = p2.diff_of_means_ci([0.2] * 4, cl, [0.2] * 4, cl)
    assert (m, lo, hi) == pytest.approx((0.0, 0.0, 0.0))


def test_twin_win_rate(p2):
    rows = [
        {"query_id": "q1", "k": 1, "source": "a", "twin": "b"},
        {"query_id": "q2", "k": 2, "source": "a", "twin": "b"},
        {"query_id": "q3", "k": 2, "source": "a", "twin": "z"},  # twin not ranked: no win
    ]
    rankings = {
        "q1": [("b", 1.0), ("a", 0.5)],
        "q2": [("a", 1.0), ("b", 0.5)],
        "q3": [("a", 1.0)],
    }
    out = p2.twin_win_rate(rankings, rows)
    assert out["1"] == {"n": 1, "rate": 1.0}
    assert out["2"] == {"n": 2, "rate": 0.0}
    assert out["all"]["rate"] == pytest.approx(1 / 3)
    assert p2.twin_win_rate(rankings, rows, ids={"q2"})["all"] == {"n": 1, "rate": 0.0}


def write_system(path: Path, rows: list[dict], order_fn, system: str, extra_trace=None):
    with path.open("w") as f:
        for r in rows:
            ranking = order_fn(r)
            trace = {"first_stage": [d for d, _ in ranking][:20]}
            if extra_trace:
                trace.update(extra_trace(r, ranking))
            f.write(
                json.dumps(
                    {
                        "query_id": r["query_id"],
                        "system": system,
                        "ranking": [[d, s] for d, s in ranking],
                        "cost": Cost(llm_calls=2, tokens_in=10).to_dict(),
                        "trace": trace,
                    }
                )
                + "\n"
            )


def test_evaluate_end_to_end(p2, tmp_path):
    sets = make_sets(8, twin_drop_k4=())
    docs = [f"d{i}" for i in range(12)]

    def s0(r):  # source at rank 3, twin at rank 1 for twin queries
        others = [d for d in docs if d not in (r["source"], r.get("twin"))]
        head = [r.get("twin") or others.pop(0), others.pop(0), r["source"]]
        return [(d, 1.0 - i / 100) for i, d in enumerate(head + others)]

    def best(r):  # source first
        rest = [d for d, _ in s0(r) if d != r["source"]]
        return [(d, 1.0 - i / 100) for i, d in enumerate([r["source"], *rest])]

    for name, rows in sets.items():
        out = tmp_path / name
        out.mkdir()
        write_system(out / "S0.jsonl", rows, s0, "S0")
        write_system(out / "S1.jsonl", rows, s0, "S1")
        # S2 verdicts: the source gets one more yes than everyone else in the twin set only
        write_system(
            out / "S2.jsonl",
            rows,
            best if name == "twin" else s0,
            "S2",
            lambda r, rk, name=name: {
                "verdicts": {
                    d: ["yes"] * (2 if d == r["source"] and name == "twin" else 1) for d, _ in rk
                }
            },
        )
        write_system(
            out / "S4.jsonl",
            rows,
            best,
            "S4",
            lambda r, rk: {"scores": {d: 10.0 if d == r["source"] else 1.0 for d, _ in rk}},
        )
        sub = [r for r in rows if r["k"] in (1, 2)]
        write_system(out / "S3.jsonl", sub, best, "S3")
        write_system(out / "S5.jsonl", sub, best, "S5")
    ev = p2.evaluate(sets, tmp_path, tmp_path / "no_judge")
    strict = ev["labels"]["strict"]
    assert "expanded" not in ev["labels"]  # no judge qrels yet
    assert strict["means"]["natural"]["S0"]["overall"]["all"]["recall@1"] == 0.0
    assert strict["means"]["natural"]["S4"]["overall"]["all"]["recall@1"] == 1.0
    assert strict["means"]["twin"]["S2@10"]["overall"]["all"]["recall@1"] == 1.0
    gd = strict["gain_diff"]["recall@1"]["S2"]["all"]
    assert gd["gain_twin"] == 1.0 and gd["gain_natural"] == 0.0 and gd["mean"] == 1.0
    assert strict["gain_diff"]["recall@1"]["S4"]["all"]["mean"] == 0.0
    assert ev["subset_sizes"] == {"natural": 16, "twin": 16}
    assert ev["twin_win"]["S0"]["all"]["rate"] == 1.0
    assert ev["twin_win"]["S4"]["all"]["rate"] == 0.0
    assert ev["twin_win_subset"]["S3"]["all"]["n"] == 16
    pair = strict["paired"]["twin"]["S3 - S0"]
    assert pair["on"] == "subset" and pair["recall@1"]["overall"]["all"]["mean"] == 1.0
    vript = strict["means"]["natural"]["S0"]["length_bin_vript"]
    assert set(vript) == {"30to60s", "2to4min"}  # only the vript_long sources
    assert ev["costs"]["natural"]["S2"]["calls"] == 2.0
    text = p2.report(
        ev, sets, "notes here", {"store": "x", "n_docs": 12, "missing": [], "spend": []}
    )
    assert "notes here" in text and "Twin win rate" in text


def test_judge_pool_merges_per_model_results():
    path = Path(__file__).resolve().parents[1] / "dev/judge_labels_dev.py"
    spec = importlib.util.spec_from_file_location("judge_pool", path)
    jp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jp)
    tasks = [("s", "a"), ("s", "b")]
    specs = ("gemini:g", "openai:o")
    results = {
        (i, m): ({"f0": {m: {"answer": "yes" if i == 0 else "no"}}}, {m: Cost(tokens_in=i + 1)}, 8)
        for i in range(2)
        for m in specs
    }
    done = jp.merge_model_results(tasks, specs, results)
    assert [d[:2] for d in done] == tasks
    _, _, answers, cost, n = done[0]
    assert answers == {"f0": {"gemini:g": {"answer": "yes"}, "openai:o": {"answer": "yes"}}}
    assert set(cost) == set(specs) and n == 8


# ---- --query-ids (explicit S3 / S5 subset, merged into the saved rankings) ---------------------


def test_read_query_ids_checks_sets_and_ids(p2, tmp_path):
    sets = make_sets(4)
    path = tmp_path / "ids.json"
    path.write_text(json.dumps({"twin": ["shot2story/v2#twin-k1", "shot2story/v0#twin-k3"]}))
    assert p2.read_query_ids(path, sets) == {
        "natural": [],
        "twin": ["shot2story/v0#twin-k3", "shot2story/v2#twin-k1"],
    }
    path.write_text(json.dumps({"twin": ["nope#twin-k1"]}))
    with pytest.raises(SystemExit):
        p2.read_query_ids(path, sets)
    path.write_text(json.dumps({"other": []}))
    with pytest.raises(SystemExit):
        p2.read_query_ids(path, sets)


def test_ledger_stage_cap_replaces_overall_cap(p2, tmp_path):
    ledger = p2.Ledger(tmp_path / "spend.json", cap=1000)
    ledger.add("a", Cost(tokens_in=900), calls=1)
    ledger.check("b", 500, stage_cap=600)  # overall cap would stop it; the stage cap does not
    with pytest.raises(SystemExit):
        ledger.check("b", 700, stage_cap=600)
    with pytest.raises(SystemExit):
        ledger.check("b", 500)


class FakeReranker:
    name = "fake"

    def rerank(self, query, candidates):
        from src.systems.base import Reranked

        return Reranked(list(reversed(candidates)), Cost(llm_calls=1, tokens_in=7))


def test_rerank_set_merge_keeps_other_queries(p2, tmp_path):
    rows = make_sets(3)["twin"]
    out = tmp_path / "twin"
    out.mkdir()
    docs = ["a", "b", "c", "d"]
    write_system(
        out / "S0.jsonl", rows, lambda r: [(d, 1.0 - i / 10) for i, d in enumerate(docs)], "S0"
    )
    old = rows[:2]
    write_system(out / "S3.jsonl", old, lambda r: [("z", 9.0)], "S3")
    (out / "cost.json").write_text(json.dumps({"S3": {"queries": 2}}))
    args = SimpleNamespace(out=str(tmp_path), workers=2)
    models = SimpleNamespace(errors=[])
    new = [rows[1]["query_id"], rows[2]["query_id"]]  # one overlaps the saved lines
    plan = {"S3": (FakeReranker(), new, 3), "S5": (FakeReranker(), [], 3)}
    costs = p2.rerank_set(models, "twin", rows, plan, args, "frames-ids", merge=True)
    lines = {line["query_id"]: line for line in p2.read_jsonl(out / "S3.jsonl")}
    assert set(lines) == {rows[0]["query_id"], *new}
    assert lines[rows[0]["query_id"]]["ranking"] == [["z", 9.0]]  # kept as saved
    assert [d for d, _ in lines[new[0]]["ranking"]] == ["c", "b", "a", "d"]  # replaced
    assert costs["S3"] == {"queries": 2} and costs["S3 (frames-ids)"]["queries"] == 2
    assert not (out / "S5.jsonl").exists()  # nothing planned, nothing written


def test_evaluate_subset_stays_the_seeded_one(p2, tmp_path):
    sets = make_sets(4, twin_drop_k4=())
    docs = [f"d{i}" for i in range(6)]
    for name, rows in sets.items():
        out = tmp_path / name
        out.mkdir()
        for system in ("S0", "S1", "S3", "S5"):
            write_system(
                out / f"{system}.jsonl",
                rows,
                lambda r: [(d, 1.0 - i / 10) for i, d in enumerate([r["source"], *docs])],
                system,
            )
    seeded = {name: [r["query_id"] for r in rows if r["k"] == 1] for name, rows in sets.items()}
    (tmp_path / "frame_subset.json").write_text(json.dumps(seeded))
    ev = p2.evaluate(sets, tmp_path, tmp_path / "no_judge")
    assert ev["subset_sizes"] == {"natural": 4, "twin": 4}


def test_judge_pool_fill_tasks_only_unjudged():
    path = Path(__file__).resolve().parents[1] / "dev/judge_labels_dev.py"
    spec = importlib.util.spec_from_file_location("judge_pool", path)
    jp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(jp)
    rows = [
        {"query_id": "q1", "source": "s", "twin": "t", "k": 3},
        {"query_id": "q2", "source": "s", "twin": "t", "k": 4},
        {"query_id": "q3", "source": "u", "twin": "v", "k": 1},
    ]
    rankings = {"S3": {"q1": ["s", "a", "t", "b"], "q2": ["b", "c"], "q3": ["x"]}}
    judged = {"s": {"s": {}, "t": {}, "a": {}}}
    tasks, per = jp.fill_tasks(rows, rankings, ["q1", "q2"], judged, top=3)
    assert per == {"q1": 0, "q2": 2} and tasks == [("s", "b"), ("s", "c")]


# ---- --stage eval-ids (conditional twin set) ----------------------------------------------------


def twin_row(qid="s#twin-k2", k=2):
    return {
        "query_id": qid,
        "source": "s",
        "twin": "t",
        "k": k,
        "d_index": 2,
        "d": "A red basket is present.",
        "phrases": {"1": "a man cooks", "2": "a red basket"},
        "facts": [{"text": "A man is cooking."}, {"text": "A red basket is present."}],
    }


def test_d_mapping_by_phrase_and_match(p2):
    row = twin_row()
    assert p2.d_by_phrase(row, ["A man cooks food.", "A red basket appears."]) == [1]
    assert p2.d_by_phrase(row, ["A man cooks beside a red basket."]) is None  # merged: ambiguous
    assert p2.d_by_phrase(row, []) is None
    answer = json.dumps(
        {
            "facts": [{"fact": 2, "match": "full", "constraints": [1, 3]}],
            "constraints": [{"constraint": 2, "fact": 1}],
        }
    )
    assert p2.d_by_match(answer, 2, 3) == [0, 2]
    answer = json.dumps(
        {"constraints": [{"constraint": 1, "fact": None}, {"constraint": 2, "fact": 2}]}
    )
    assert p2.d_by_match(answer, 2, 2) == [1]
    assert p2.d_by_match("not json", 2, 2) == []
    assert p2.d_verdict(["yes", "no"], [1]) == "no"
    assert p2.d_verdict(["yes", "unclear", "yes"], [0, 2]) == "yes"
    assert p2.d_verdict(["yes", "unclear"], [0, 1]) == "unclear"
    assert p2.d_verdict(None, [0]) == "missing" and p2.d_verdict(["no"], []) == "missing"


def test_ids_evaluate_twin_rates_and_d(p2):
    rows = [twin_row("s#twin-k2", 2), twin_row("s#twin-k3", 3)]
    good = [("s", 1.0), ("t", 0.5), ("x", 0.1)]
    bad = [("t", 1.0), ("s", 0.5), ("x", 0.1)]
    systems = {s: {r["query_id"]: good for r in rows} for s in p2.IDS_SYSTEMS}
    systems["S0"] = {r["query_id"]: bad for r in rows}
    qrels = {"strict": {r["query_id"]: {"s": 1} for r in rows}}
    verdicts = {"s": ["yes", "yes"], "t": ["yes", "no"], "x": ["no", "no"]}
    trace = {r["query_id"]: {"constraints": ["c1", "c2"], "verdicts": verdicts} for r in rows}
    d_map = {r["query_id"]: {"idx": [1], "how": "phrase"} for r in rows}
    ev = p2.ids_evaluate(rows, systems, qrels, {"S3": trace}, d_map)
    assert ev["twin_above_source"]["S0"]["all"]["rate"] == 1.0
    assert ev["twin_above_source"]["S5"]["3"]["rate"] == 0.0
    s5 = ev["labels"]["strict"]["paired"]["S5 - S0"]["recall@1"]["all"]
    assert s5["mean"] == 1.0 and s5["n"] == 2
    assert ev["d"]["S3"]["twin_rejects"]["all"]["rate"] == 1.0
    assert ev["d"]["S3"]["source_accepts"]["all"]["rate"] == 1.0
    assert ev["d"]["mapping"] == {"phrase": 2}
    text = p2.ids_report(ev, "t", {"notes": ["n"]})
    assert "Twin ranked above the source" in text and "S5 - S0" in text
