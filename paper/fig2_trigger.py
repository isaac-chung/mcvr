"""Figure C of the AAAI-27 workshop paper: the frame trigger's trade-off curve.

    uv run --offline --with matplotlib python paper/fig2_trigger.py [CSV]

Reads results/paper/data_trigger_sweep.csv (rebuilt offline from the saved S4@50 / S5 runs
by paper/fig2_trigger_sweep.py; no model or API call) and plots, over all 722 test queries, the share
of S5's gain over S4@50 that is kept against the share of queries sent to frames:
  - the S4@50 top-two margin sweep (send iff margin <= t) and the S4@50 top-score sweep
    (send iff top score <= t), test set pooled;
  - the margin sweep on the development set (120 queries), dashed;
  - the diagonal, which is what a random trigger keeps in expectation;
  - named operating points: development threshold (margin <= 2), five-fold cross-fit
    (keep >= 0.8), top score < 8, k >= 3, random at the margin rule's share;
each annotated with its input tokens per query (S4@50 base plus the frame step when sent).
Points are numbered and a key gives input tokens per query (sent/kept dropped 2026-11-08).
2026-11-06 (merge of the cost-quality frontier into this figure): two relabelled axes, no new data.
  - right y-axis: absolute nDCG@10 (expanded, 722 test queries). Gain kept is defined as
    (nDCG - nDCG(S4@50)) / (nDCG(S5) - nDCG(S4@50)), so nDCG = .6337 + kept x (.7110 - .6337) is an
    exact affine relabelling (checked below against the CSV's ndcg10 column: max error 5e-5,
    the rounding of gain_kept to 4 decimals).
  - top x-axis: input tokens per query, approximately: tokens = 10.4k + share x 80.3k (mean frame
    step over all queries). The real cost depends on which queries are sent; over every test row of
    the CSV it deviates from this line by at most 2.3k tokens (top-score sweep; 1.9k on the margin
    sweep), so the axis is marked "approx." and exact tokens stay in the key.
Writes results/full/figures/fig_trigger.{pdf,png} and copies the PDF to results/paper/figures/.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "results/paper/data_trigger_sweep.csv"
# 2026-11-05 re-review: pointwise 95% cluster-bootstrap band (10,000 resamples of source videos) on
# the gain kept along the pooled test margin sweep, from paper/trigger_analysis.py.
BAND_CSV = ROOT / "results/full/review_20261105/o3_trigger_curve.csv"
OUT_DIR = ROOT / "results/full/figures"
PAPER_DIR = ROOT / "results/paper/figures"
STEM = "fig_trigger"

import json  # noqa: E402

# 2026-11-06: 95% cluster-bootstrap intervals (10,000 resamples of sources) of the named points.
# dev threshold / cross-fit / S4@50 / S5 from paper/trigger_analysis.py and fig2_frontier.py;
# top score < 8, k >= 3, random from paper/fig2_trigger_cis.py (o3_named_points_ci.json).
O3 = json.loads((ROOT / "results/full/review_20261105/o3_trigger.json").read_text())["groups"]["pooled"]
O4 = json.loads((ROOT / "results/full/review_20261105/o4_frontier.json").read_text())
CI_NAMED = json.loads((ROOT / "results/full/review_20261105/o3_named_points_ci.json").read_text())

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_MARGIN, C_TOP = "#5b3fa8", "#199e70"


def rows(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def sweep(R: list[dict], group: str, rule: str) -> list[tuple[float, float, float]]:
    out = [
        (100 * float(r["share_sent"]), 100 * float(r["gain_kept"]), float(r["tokens_per_query"]))
        for r in R
        if r["group"].startswith(group) and r["kind"] == "sweep" and r["rule"] == rule
    ]
    return sorted(out)


def point(R: list[dict], kind: str, rule: str, group: str = "test pooled") -> tuple[float, float, float]:
    for r in R:
        if r["group"] == group and r["kind"] == kind and r["rule"] == rule:
            return 100 * float(r["share_sent"]), 100 * float(r["gain_kept"]), float(r["tokens_per_query"])
    raise KeyError((kind, rule))


def main() -> int:
    R = rows(Path(sys.argv[1]) if len(sys.argv) > 1 else CSV)
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times", "Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.unicode_minus": True,
        }
    )
    never = point(R, "operating point", "never (S4@50)")
    always = point(R, "operating point", "always (S5)")
    nd = {(r["kind"], r["rule"]): float(r["ndcg10"]) for r in R if r["group"] == "test pooled" and r["ndcg10"]}
    n0, n1 = nd[("operating point", "never (S4@50)")], nd[("operating point", "always (S5)")]
    t0, t1 = never[2], always[2]
    err_n = max(abs(n0 + float(r["gain_kept"]) * (n1 - n0) - float(r["ndcg10"]))
                for r in R if r["group"] == "test pooled" and r["ndcg10"])
    err_t = max(abs(t0 + float(r["share_sent"]) * (t1 - t0) - float(r["tokens_per_query"]))
                for r in R if r["group"] == "test pooled")
    assert err_n < 1e-4, err_n  # the right axis is an exact relabelling
    print(f"nDCG axis: {n0:.4f} + kept x {n1 - n0:.4f} (max error {err_n:.1e}); "
          f"token axis: {t0:.0f} + share x {t1 - t0:.0f} (max deviation {err_t:.0f} tokens)")

    fig, ax = plt.subplots(figsize=(3.3, 2.59))
    ax.plot([0, 100], [0, 100], color=INK_2, lw=0.7, ls=":", zorder=1)
    ax.text(30, 25.5, "chance", fontsize=6.0, color=INK_2, rotation=37, ha="center", va="center")

    m = sweep(R, "test pooled", "margin <= t")
    t = sweep(R, "test pooled", "S4 top score <= t")
    dv = sweep(R, "dev", "margin <= t")
    ax.plot([p[0] for p in dv], [p[1] for p in dv], color=C_MARGIN, lw=0.9, ls="--", alpha=0.55, zorder=2,
            label="margin, development (120)")
    ax.plot([p[0] for p in t], [p[1] for p in t], color=C_TOP, lw=1.2, marker="s", ms=2.4, zorder=3,
            label="top score, test (722)")
    band = sorted(
        (100 * float(r["share"]), 100 * float(r["kept_lo"]), 100 * float(r["kept_hi"]))
        for r in rows(BAND_CSV)
        if r["group"] == "pooled" and r["kind"] == "threshold"
    )
    # the band's curve must be the plotted margin sweep
    assert all(abs(a[0] - b[0]) < 0.05 for a, b in zip(band, m)) and len(band) == len(m), (band, m)
    ax.fill_between([b[0] for b in band], [b[1] for b in band], [b[2] for b in band], color=C_MARGIN,
                    alpha=0.16, lw=0, zorder=2.5, label="95% band, margin, test")
    ax.plot([p[0] for p in m], [p[1] for p in m], color=C_MARGIN, lw=1.2, marker="o", ms=2.4, zorder=4,
            label="margin, test (722)")

    # Points are numbered next to their markers; the key (label: sent % / kept % / input tokens
    # per query, explained in the caption) sits in the empty triangle below the diagonal.
    # (label, point, colour, marker, number offset in points)
    named = [
        ("margin$\\,\\leq 2$ (dev threshold)", point(R, "operating point", "margin <= 2 (dev threshold)"), C_MARGIN, "o", (6, 6)),
        ("top score$\\,<8$", point(R, "operating point", "S4 top score < 8"), C_TOP, "s", (-5, 5)),
        ("cross-fit (held-out sources)", point(R, "cross-fit", "margin, keep >= 0.8"), C_MARGIN, "o", (-6, 1)),
        ("S5 (frames always)", always, INK, "*", (-7, -6)),
        ("random trigger", point(R, "operating point", "random, share 0.771 (pooled margin share)"), INK_2, "X", (6, 0)),
        ("$k\\geq 3$", point(R, "operating point", "k >= 3"), INK_2, "D", (6, -2)),
    ]
    # intervals on gain kept for the trigger rules (the right axis is the same affine map). S4@50 and S5
    # get none: they define 0% and 100% kept, and their absolute nDCG@10 intervals (about +-0.03) are not
    # comparable with these paired intervals.
    cis = {
        "margin$\\,\\leq 2$ (dev threshold)": [100 * v for v in CI_NAMED["margin <= 2 (dev threshold)"]["kept_ci"]],
        "top score$\\,<8$": [100 * v for v in CI_NAMED["S4 top score < 8"]["kept_ci"]],
        "cross-fit (held-out sources)": [100 * v for v in O3["crossfit_keep0.8"]["sampling_ci(bootstrap over sources)"]["kept_ci"]],
        "random trigger": [100 * v for v in CI_NAMED["random (pooled margin share)"]["kept_ci"]],
        "$k\\geq 3$": [100 * v for v in CI_NAMED["k >= 3"]["kept_ci"]],
    }
    for label, (x, y, tok), color, marker, off in [(n[0], n[1], n[2], n[3], n[4]) for n in named]:
        if label not in cis:
            continue
        lo, hi = cis[label]
        ax.errorbar([x], [y], yerr=[[y - lo], [hi - y]], color=color, lw=0.55, capsize=1.3, capthick=0.55,
                    fmt="none", zorder=5, alpha=0.9)
        print(f"interval {label:32s} kept {y:6.1f}  [{lo:6.1f}, {hi:6.1f}]")
    key = []
    for i, (label, (x, y, tok), color, marker, (dx, dy)) in enumerate(named, 1):
        ax.plot([x], [y], marker=marker, ms=5.2 if marker != "*" else 7, color=color, mec="white", mew=0.6,
                lw=0, zorder=6)
        ax.annotate(str(i), (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.0,
                    fontweight="bold", color=INK, ha="center", va="center", zorder=7)
        key.append(f"{i}   {label}: {tok / 1000:.0f}k")  # 2026-11-08: sent/kept dropped from the key (readable off the axes)
        print(f"{label:32s} sent {x:5.1f}%  kept {y:5.1f}%  {tok:8.0f} tok/query")
    # S4@50 (never send) is the origin; mark it as the fixed system it is.
    ax.plot([0], [0], marker="o", ms=4.6, color=INK, mec="white", mew=0.6, lw=0, zorder=6, clip_on=False)
    ax.annotate("S4@50", (0, 0), xytext=(15, 1.0), textcoords="offset points", fontsize=6.0, fontweight="bold",
                color=INK, ha="left", va="bottom", zorder=7,
                arrowprops={"arrowstyle": "-", "color": INK_2, "lw": 0.5, "shrinkA": 0.5, "shrinkB": 2.5})
    ax.text(101, 3, "\n".join(key), fontsize=5.6, color=INK, ha="right", va="bottom", multialignment="left", linespacing=1.25,
            zorder=8, bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": INK, "linewidth": 0.6})

    ax.set_xlim(0, 103)
    # Relabelled axes: right = absolute nDCG@10 (exact), top = input tokens per query (approximate).
    ay = ax.secondary_yaxis("right", functions=(lambda k: n0 + k / 100 * (n1 - n0), lambda v: 100 * (v - n0) / (n1 - n0)))
    ay.set_yticks([n0, 0.65, 0.67, 0.69, n1])
    ay.set_yticklabels([f"{n0:.3f}"[1:], ".65", ".67", ".69", f"{n1:.3f}"[1:]])
    ay.set_ylabel("nDCG@10 (expanded labels)", fontsize=7, color=INK, labelpad=2)
    axx = ax.secondary_xaxis("top", functions=(lambda x: (t0 + x / 100 * (t1 - t0)) / 1000,
                                               lambda v: 100 * (1000 * v - t0) / (t1 - t0)))
    axx.set_xticks([20, 40, 60, 80])
    axx.set_xlabel("$\\approx$ input tokens per query (thousands)", fontsize=7, color=INK, labelpad=2)
    for sec in (ay, axx):
        for sp in sec.spines.values():
            sp.set_color(INK_2)
            sp.set_linewidth(0.6)
        sec.tick_params(colors=INK_2, labelcolor=INK, labelsize=6.5, width=0.6, length=2.5)
    ax.set_ylim(0, 106)
    ax.set_xlabel("queries sent to frames (%)", fontsize=7, color=INK, labelpad=1)
    ax.set_ylabel("frame gain kept (%)", fontsize=7, color=INK)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelcolor=INK, labelsize=6.5, width=0.6, length=2.5)
    ax.legend(loc="upper left", frameon=False, fontsize=6.2, handlelength=1.8, borderaxespad=0.2,
              labelspacing=0.35)
    fig.subplots_adjust(left=0.115, right=0.885, top=0.875, bottom=0.115)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pdf = OUT_DIR / f"{STEM}.pdf"
    fig.savefig(pdf)
    fig.savefig(OUT_DIR / f"{STEM}.png", dpi=300)
    PAPER_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(pdf, PAPER_DIR / pdf.name)
    print(f"wrote {pdf}; copied to {PAPER_DIR / pdf.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
