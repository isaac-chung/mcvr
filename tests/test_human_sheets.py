"""The 100-source human label sheets (docs/human_labeling_100src.md, tooling changes 1-6, 9):
item builders of human/make_label_sheets.py on the real test-set inputs, offline."""

import csv
import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "human/make_label_sheets.py"
SOURCES = ROOT / "results/full/select/sources100.json"
OLD = ROOT / "data/mc/full/human"
NEEDED = [
    SOURCES,
    OLD / "key.json",
    OLD / "taskA.csv",
    ROOT / "data/mc/full/queries_twin.jsonl",
    ROOT / "data/mc/full/judge/judged.jsonl",
    ROOT / "results/full/runs/natural/S5.jsonl",
    ROOT / "data/mc/shot2story/descriptions_fps05.jsonl",
]
pytestmark = pytest.mark.skipif(
    not all(p.exists() for p in NEEDED), reason="full-run inputs not present"
)


@pytest.fixture(scope="module")
def ms():
    spec = importlib.util.spec_from_file_location("make_label_sheet", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def built(ms, tmp_path_factory):
    """The full 100-source build (no --limit) written to a temporary folder."""
    out = tmp_path_factory.mktemp("human100")
    args = ms.parser().parse_args(
        [
            "--sources-file",
            str(SOURCES),
            "--taskb-all",
            "--prefill",
            str(OLD),
            "--overlap",
            "25",
            "--overlap-bd",
            "10",
            "--overlap-e",
            "10",
            "--out",
            str(out),
        ]
    )
    B = ms.build(args)
    ms.write(args, B)
    return B, out


def page_data(path: Path) -> dict:
    m = re.search(r"const DATA = (\{.*?\});\n", path.read_text(), re.DOTALL)
    return json.loads(m.group(1).replace("<\\/", "</"))


def test_item_counts(built):
    B, _ = built
    T = B["tasks"]
    assert {t: len(v) for t, v in T.items()} == {
        "taskA": 127,
        "taskC": 39,
        "taskBD": 100,
        "taskE": 100,
    }
    A = T["taskA"]
    assert dict(__import__("collections").Counter(it["set_label"] for it in A)) == {
        "nested": 68,
        "near-miss": 59,
    }
    assert sum(len(it["videos"]) for it in A) == 494
    assert sum(len(it["facts"]) for it in A) == 353
    assert sum(len(it["queries"]) for it in A) == 235
    # every Task A item carries its source; near-miss items also the twin (HN) video
    for it in A:
        roles = {v["role"] for v in it["videos"].values()}
        assert "source" in roles
        assert ("twin" in roles) == (it["set_label"] == "near-miss")
        assert all(v["judges"] is not None for v in it["videos"].values())
    # Task C: the near-miss sources without a Task A item, source + HN only
    in_a = {it["source"] for it in A if it["set_label"] == "near-miss"}
    for it in T["taskC"]:
        assert it["source"] not in in_a
        assert sorted(v["role"] for v in it["videos"].values()) == ["source", "twin"]
        assert it["facts"][it["d_index"] - 1]["text"] == it["d"]
    assert sum(len(it["facts"]) * 2 for it in T["taskC"]) == 292
    # Task B+D and E: the same facts on every source
    assert sum(len(it["facts"]) for it in T["taskBD"]) == 715
    assert sum(len(it["queries"]) for it in T["taskBD"]) == 722
    assert sum(len(it["rejected_queries"]) for it in T["taskBD"]) == 145
    assert sum(len(it["rejected"]) for it in T["taskBD"]) == 74
    bd = {it["source"]: [f["key"] for f in it["facts"]] for it in T["taskBD"]}
    e = {it["source"]: [f["key"] for f in it["facts"]] for it in T["taskE"]}
    assert bd == e
    assert all(len(it["videos"]["D"]["text"].split()) > 5 for it in T["taskE"])


def test_prefill_and_overlap(built):
    B, _ = built
    counts = B["key"]["prefill_counts"]
    assert counts["taskA"] == 208 and counts["taskC"] == 3
    assert counts["taskBD"] >= 41  # taskB.csv cells plus the cross-task cells of taskA.csv
    assert sum(it["prefilled"] > 0 for it in B["tasks"]["taskA"]) == 27  # 6 full + 21 partial
    # pre-filled values are in the page's option set (met <-> present mapped)
    for cells in B["prefill"]["taskBD"].values():
        assert set(cells.values()) <= {"present", "absent", "cant_tell"}
    for cells in B["prefill"]["taskA"].values():
        assert set(cells.values()) <= {"met", "not_met", "cant_tell"}
    # the overlap: 25 Task A items stratified by set, forced to the unchanged old overlap items
    ov = B["overlap"]
    assert len(ov["taskA"]) == 25 and len(ov["taskBD"]) == 10 and len(ov["taskE"]) == 10
    a = {it["item"]: it for it in B["tasks"]["taskA"]}
    sets = __import__("collections").Counter(a[i]["set_label"] for i in ov["taskA"])
    assert sets == {"nested": 13, "near-miss": 12}
    old = json.loads((OLD / "key.json").read_text())
    old_src = {it["item"]: it["source"] for it in old["tasks"]["taskA"]}
    forced = {a[i]["source"] for i in B["key"]["overlap_forced"]}
    assert forced == {old_src[i] for i in ("A08", "A25", "A37")}
    assert forced <= {a[i]["source"] for i in ov["taskA"]}
    # the second annotator's answers pre-fill the overlap page only
    assert counts["taskA_overlap"] == 44
    # earlier answers on the same video under another fact key are hints, never copied
    H = B["hints"]
    assert B["key"]["hint_counts"] == {
        t: sum(len(h) for i in v.values() for h in i.values()) for t, v in H.items()
    }
    assert B["key"]["hint_counts"]["taskC"] == 19 and B["key"]["hint_counts"]["taskA"] >= 150
    c02 = next(it for it in B["tasks"]["taskC"] if it["source"] == "carebench/v_00006449_0")
    src = next(lab for lab, v in c02["videos"].items() if v["role"] == "source")
    hs = H["taskC"][c02["item"]][src]
    assert [h["fact"] for h in hs if h["row"]] == ["A yellow ladder is visible."]
    assert (
        hs[0]["row"] == [f["text"] for f in c02["facts"]].index("A yellow ladder is present.") + 1
    )
    assert all(h["answer"] in ("met", "not_met", "cant_tell") for h in hs)
    for page, per_item in H.items():
        for item, per_video in per_item.items():
            for lab, hs in per_video.items():
                assert not {f"c|{lab}|{h['row']}" for h in hs if h["row"]} & set(
                    B["prefill"][page].get(item, {})
                )
    # the old optional page was never exported: reported, not silently skipped
    assert list(B["key"]["unexported"]) == ["human/taskN"]
    second = {r["item"] for r in csv.DictReader((OLD / "taskA_overlap.csv").open())}
    assert {"A08", "A25", "A37"} <= second


def test_pages(built, ms):
    B, out = built
    secrets = ["S4@50", "S5:", '"role"', '"source"', '"twin"', "near-miss", "nested", "judges"]
    expect = {
        "taskA": (127, "met"),
        "taskA_overlap": (25, "met"),
        "taskC": (39, "met"),
        "taskBD": (100, "present"),
        "taskBD_overlap": (10, "present"),
        "taskE": (100, "stated"),
        "taskE_overlap": (10, "stated"),
    }
    for page, (n, opts) in expect.items():
        html = (out / f"{page}.html").read_text()
        assert not [s for s in secrets if s in html], page
        data = page_data(out / f"{page}.html")
        assert len(data["items"]) == n
        assert [o[0] for o in data["options"]] == [o[0] for o in ms.OPTIONS[opts]]
        for it in data["items"]:
            for v in it["videos"]:
                if "src" in v:
                    link = out / v["src"]
                    assert link.is_symlink() and link.resolve().exists()
                else:
                    assert page.startswith("taskE") and v["text"]
            assert ("queries" in it) == page.startswith("taskBD")
        if page.startswith("taskBD"):  # queries are context only: no askable cells
            data = page_data(out / f"{page}.html")
            assert all(it["queries"] for it in data["items"])
            assert (
                "YESNO" not in html and '"askable", ""' not in html
            )  # no yes / no asked or exported
    # letters are shuffled: the source is not always A (or always last)
    a_letters = [
        next(lab for lab, v in it["videos"].items() if v["role"] == "source")
        for it in B["tasks"]["taskA"]
    ]
    assert len(set(a_letters)) > 1 and a_letters.count("A") < len(a_letters) / 2
    c_letters = [
        next(lab for lab, v in it["videos"].items() if v["role"] == "source")
        for it in B["tasks"]["taskC"]
    ]
    assert set(c_letters) == {"A", "B"}
    # the autosave id ignores the pre-fill, and the pre-2026-11-02 id is kept for migration
    for page in expect:
        data = page_data(out / f"{page}.html")
        assert data["draw"] == ms.draw_id(data["items"], with_prefill=False)
        assert data["legacy_draws"] == [ms.draw_id(data["items"], with_prefill=True)]
        assert (data["draw"] == data["legacy_draws"][0]) == (
            page.startswith("taskE") or page == "taskBD_overlap"
        )
        assert set(data["hints"]) <= {it["id"] for it in data["items"]}
    # the overlap pages share ids and letters with the main pages
    main = {it["id"]: it for it in page_data(out / "taskA.html")["items"]}
    for it in page_data(out / "taskA_overlap.html")["items"]:
        assert it["videos"] == main[it["id"]]["videos"]
    key = json.loads((out / "key.json").read_text())
    assert key["plan"] == "human_labeling_100src.md"
    assert set(key["tasks"]) == {"taskA", "taskC", "taskBD", "taskE"}


def test_limit_pilot(ms, tmp_path):
    args = ms.parser().parse_args(
        ["--sources-file", str(SOURCES), "--taskb-all", "--limit", "2", "--out", str(tmp_path)]
    )
    B = ms.build(args)
    assert {t: len(v) for t, v in B["tasks"].items()} == {
        "taskA": 2,
        "taskC": 2,
        "taskBD": 2,
        "taskE": 2,
    }
    assert B["prefill"] == {} and B["key"]["prefill_counts"] == {}
    assert B["hints"] == {} and B["key"]["unexported"] == {}
    assert len(B["overlap"]["taskA"]) == 2 and "taskBD" not in B["overlap"]


def test_old_sheets_unchanged(ms):
    """Without the new flags the 279-source draw (data/mc/full/human/key.json) is reproduced:
    the same items, facts, videos and letters (the judge answers may have grown since)."""
    args = ms.parser().parse_args(["--out", "/nonexistent"])
    B = ms.build(args)
    old = json.loads((OLD / "key.json").read_text())

    def strip(it):
        it = {k: v for k, v in it.items() if k not in ("set_label", "prefilled")}
        it["videos"] = {
            lab: {k: v for k, v in vid.items() if k != "judges"}
            for lab, vid in it["videos"].items()
        }
        return it

    for t, its in old["tasks"].items():
        assert [strip(it) for it in B["tasks"][t]] == [strip(it) for it in its], t
    assert B["overlap"]["taskA"] == old["taskA_overlap"]


def test_own_exports_feed_task_a(ms, built):
    """The CSVs exported into the 100-source folder (Task B on every source, Task C, the
    Task B overlap) are read like an earlier folder: every Task A source-video cell is then
    pre-filled from Task B, the overlap page takes the second annotator's Task B answers, and
    a hint list never repeats the same fact with the same answer."""
    B, _ = built
    new = SCRIPT.parents[1] / "data" / "mc" / "full" / "human100"
    if not (new / "taskBD.csv").exists():
        pytest.skip("taskBD.csv not exported yet")
    old = ms.read_answer_folders([OLD, new], new)
    assert list(old["unexported"]) == ["human/taskN"]  # the folder being built adds none
    assert len(old["first"]) >= 715 + 292  # at least Task B and Task C on every source
    items = B["tasks"]["taskA"]
    pre = ms.prefill_items(items, old["first"], "met")
    for it in items:
        for lab, v in it["videos"].items():
            if v["role"] == "source":
                assert all(
                    f"c|{lab}|{j}" in pre[it["item"]] for j in range(1, len(it["facts"]) + 1)
                )
    hints = ms.hint_items(items, old, "first", "met", pre)
    for per_video in hints.values():
        for hs in per_video.values():
            pairs = [(h["fact"], h["answer"]) for h in hs]
            assert len(pairs) == len(set(pairs))
            assert all(h["answer"] in ("met", "not_met", "cant_tell") for h in hs)
    ov = [it for it in items if it["item"] in B["overlap"]["taskA"]]
    pre2 = ms.prefill_items(ov, old["second"], "met")
    assert sum(len(c) for c in pre2.values()) >= 52
