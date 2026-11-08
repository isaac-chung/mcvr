"""What an earlier label folder can pre-fill in a newer one (docs/human_labeling_100src.md).
FREE: reads key.json and the exported CSVs of both folders only.

    uv run python human/carry_over.py \\
        [--old data/mc/full/human]... [--new data/mc/full/human100] [--report <new>/carryover.md]

--old is repeatable; the new folder's own exported CSVs (Task B, Task C done in it) are read
as well, as make_label_sheets.py --prefill does.

For every video of every item of the new sheets it finds the earlier answers on the same
video and sorts them into

- exact: same source, video and fact key -> copied into the page by make_label_sheets.py
  --prefill (rendered as selected, "pre-filled" badge);
- hint: the same video answered on another fact (worded differently, or a fact of another
  query) -> shown next to the closest new fact, or under the video, never copied;
- unexported: the video is on an old page whose CSV was never exported (e.g. taskN.html) ->
  open that page, "Download CSV" into the old folder, rebuild the new sheets;
- new: never labelled.

It also says whether the new pages on disk carry the current pre-fill and hints (rebuild
with make_label_sheets.py ... --prefill <old> --force when not). Exit code 0 always.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sheet_module():
    spec = importlib.util.spec_from_file_location("make_label_sheet", HERE / "make_label_sheets.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def page_data(path: Path) -> dict | None:
    if not path.exists():
        return None
    m = re.search(r"const DATA = (\{.*?\});\n", path.read_text(), re.DOTALL)
    return json.loads(m.group(1).replace("<\\/", "</"))


def pages_of(key: dict) -> dict[str, list[dict]]:
    pages = dict(key["tasks"])
    for t, ids in key.get("overlap", {}).items():
        pages[f"{t}_overlap"] = [it for it in key["tasks"][t] if it["item"] in ids]
    return pages


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--old", action="append", help="earlier folder(s); default data/mc/full/human")
    ap.add_argument("--new", default="data/mc/full/human100")
    ap.add_argument("--report", help="markdown report (default <new>/carryover.md)")
    args = ap.parse_args()
    ms = sheet_module()
    new_dir = Path(args.new)
    old_dirs = [Path(f) for f in args.old or ["data/mc/full/human"]]
    if new_dir.resolve() not in {f.resolve() for f in old_dirs}:
        old_dirs.append(new_dir)
    old_dir = ", ".join(f"`{f}`" for f in old_dirs)
    old = ms.read_answer_folders(old_dirs, new_dir)
    key = json.loads((new_dir / "key.json").read_text())
    unexported = {v: p for p, vs in old["unexported"].items() for v in vs}

    lines = [
        f"# Carry-over from {old_dir} into `{new_dir}`",
        "",
        (
            f"Old answers on disk: {len(old['first'])} first-annotator cells, "
            f"{len(old['second'])} second-annotator cells over {len(old['by_video'])} videos."
        ),
    ]
    if old["unexported"]:
        for p, vs in old["unexported"].items():
            lines.append(
                f"Old page `{p}.html` has no `{p}.csv` ({len(vs)} videos): if it was labelled, "
                f"open it, click Download CSV, save the file into its folder and rebuild."
            )
    lines += [
        "",
        "| page | items | videos | seen before | cells | exact (pre-filled) | hints: close fact | hints: other fact | cells on unexported old pages | rebuild? |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    detail = []
    totals = Counter()
    for page, items in pages_of(key).items():
        opts = ms.PAGES[page][2]
        if opts == "stated":
            continue
        who = "second" if page.endswith("_overlap") else "first"
        pre = ms.prefill_items(items, old[who], opts)
        hints = ms.hint_items(items, old, who, opts, pre)
        n_cells = sum(len(it["videos"]) * len(it["facts"]) for it in items)
        vids = [(it, lab, v) for it in items for lab, v in it["videos"].items()]
        seen = sum(1 for _, _, v in vids if v["video"] in old["by_video"])
        n_pre = sum(len(c) for c in pre.values())
        n_row = sum(1 for i in hints.values() for l in i.values() for h in l if h["row"])
        n_oth = sum(1 for i in hints.values() for l in i.values() for h in l if not h["row"])
        n_unx = sum(len(it["facts"]) for it, _, v in vids if v["video"] in unexported)
        data = page_data(new_dir / f"{page}.html")
        on_disk = (
            data is not None
            and sum(len(it.get("prefill", {})) for it in data["items"]) == n_pre
            and sum(len(l) for i in data.get("hints", {}).values() for l in i.values())
            == n_row + n_oth
        )
        lines.append(
            f"| {page} | {len(items)} | {len(vids)} | {seen} | {n_cells} | {n_pre} | {n_row} | "
            f"{n_oth} | {n_unx} | {'no' if on_disk else 'YES'} |"
        )
        totals.update(cells=n_cells, pre=n_pre, row=n_row, oth=n_oth, unx=n_unx)
        for it, lab, v in vids:
            src = []
            if v["video"] in old["by_video"]:
                src = sorted({o["item"] for o in old["by_video"][v["video"]] if o["who"] == who})
            if v["video"] in unexported:
                src.append(f"{unexported[v['video']]} (unexported)")
            if not src:
                continue
            npre = sum(1 for c in pre.get(it["item"], {}) if c.startswith(f"c|{lab}|"))
            hs = hints.get(it["item"], {}).get(lab, [])
            detail.append(
                f"| {page} | {it['item']} | {lab} | {', '.join(src)} | {npre} of {len(it['facts'])} | "
                f"{sum(1 for h in hs if h['row'])} | {sum(1 for h in hs if not h['row'])} |"
            )
    lines += [
        "",
        (
            f"Total: {totals['pre']} of {totals['cells']} cells pre-filled, {totals['row']} close-fact "
            f"hints, {totals['oth']} other hints, {totals['unx']} cells whose video sits on an "
            "unexported old page."
        ),
        "",
        "## Videos seen before, per item",
        "",
        "| page | item | video | old item(s) | pre-filled | close-fact hints | other hints |",
        "|---|---|---|---|---|---|---|",
        *detail,
    ]
    report = Path(args.report) if args.report else new_dir / "carryover.md"
    report.write_text("\n".join(lines) + "\n")
    print("\n".join(lines[: lines.index("## Videos seen before, per item")]))
    print(f"\nreport: {report} ({len(detail)} videos seen before)")


if __name__ == "__main__":
    main()
