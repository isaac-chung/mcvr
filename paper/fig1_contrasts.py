"""Figure A of the AAAI-27 workshop paper: the RQ1/RQ2 contrasts as a forest plot, pooled and per k.

    uv run --offline --with matplotlib python paper/fig1_contrasts.py

Reads only saved outputs (no model or API call):
  results/full/eval/summary.json   95% cluster-bootstrap intervals (expanded labels), pooled and per k
  results/full/tier1/summary.json  90% intervals of S3g - S5 for the equivalence row, pooled and per k
Writes results/full/figures/fig_contrasts.{pdf,png} and copies the PDF to results/paper/figures/.

Rows (paper names after the 2026-11-04 rename; data keys in brackets):
  split search          S1 - S0
  per-constraint, desc. S2@50 - S0         [S2g@50 - S0]
  whole-query rerank    S4@50 - S0
  check form, desc.     S4@50 - S2@50      [S4@50 - S2g@50]
  check form, frames    S5 - S3            [-(S3g - S5)], 90% CI drawn, +-0.02 margin shaded
  add frames            S5 - S4@50
Per row and query set (nested: circle; near-miss: square), a large filled marker gives the
estimate pooled over k = 1-4 and a "k-trail" of four small markers below it gives k = 1, 2, 3, 4
top to bottom, hollow and joined by a thin line, each with its own interval as a faint hairline.
Replaces the old gain-vs-k figure (figure1_graded.py), which is no longer used by the paper.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "results/full/eval/summary.json"
TIER1 = ROOT / "results/full/tier1/summary.json"
OUT_DIR = ROOT / "results/full/figures"
PAPER_DIR = ROOT / "results/paper/figures"
STEM = "fig_contrasts"
LABELS = "expanded"
KS = ["1", "2", "3", "4"]

INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
SET_STYLE = {
    "nested": {"color": "#2a78d6", "marker": "o", "label": "Nested ($n=350$)"},
    "near-miss": {"color": "#eb6834", "marker": "s", "label": "Near-miss ($n=372$)"},
}
MARGIN = 0.02


def rc() -> None:
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


def cell(block: dict) -> tuple[tuple, list]:
    """(pooled (mean, lo, hi), [per-k (mean, lo, hi)]) from an eval block, 95% intervals."""
    r = block["ndcg@10"]
    a = r["None"]["all"]
    return (a["mean"], a["ci_low"], a["ci_high"]), [(r["k"][k]["mean"], r["k"][k]["ci_low"], r["k"][k]["ci_high"]) for k in KS]


def eq_cell(block: dict) -> tuple[tuple, list]:
    """Same from a tier1 S3g - S5 block, 90% intervals, sign flipped to S5 - S3g."""
    def flip(a: dict) -> tuple[float, float, float]:
        return -a["mean"], -a["ci90_high"], -a["ci90_low"]

    return flip(block["all"]), [flip(block[f"k={k}"]) for k in KS]


def main() -> int:
    ev = json.loads(EVAL.read_text())["labels"][LABELS]
    t1 = json.loads(TIER1.read_text())["labels"][LABELS]["frames_graded"]["subsets"]
    P, SP = ev["paired"], ev["subset_paired"]
    sets = {"nested": ("natural", "all nested"), "near-miss": ("twin", "all near-miss")}

    rows = []  # (label, {set: (pooled, perk)}, is_equivalence_row)
    rows.append(("Split the search\n(S1 $-$ S0)", {s: cell(P[a]["S1 - S0"]) for s, (a, _) in sets.items()}, False))
    rows.append(
        ("Per-constraint check, desc.\n(S2@50 $-$ S0)", {s: cell(P[a]["S2g@50 - S0"]) for s, (a, _) in sets.items()}, False)
    )
    rows.append(
        ("Whole-query rerank, desc.\n(S4@50 $-$ S0)", {s: cell(P[a]["S4@50 - S0"]) for s, (a, _) in sets.items()}, False)
    )
    rows.append(
        (
            "Whole-query vs per-constr.,\ndesc. (S4@50 $-$ S2@50)",
            {s: cell(P[a]["S4@50 - S2g@50"]) for s, (a, _) in sets.items()},
            False,
        )
    )
    rows.append(
        (
            "Whole-query vs per-constr.,\nframes (S5 $-$ S3; 90% CI)",
            {s: eq_cell(t1[sub]["paired"]["S3g - S5"]) for s, (_, sub) in sets.items()},
            True,
        )
    )
    rows.append(("Add frames\n(S5 $-$ S4@50)", {s: cell(SP[sub]["S5 - S4@50"]) for s, (_, sub) in sets.items()}, False))

    rc()
    # Full text width (figure*): one panel per query set, rows shared; per-k trail under each
    # pooled marker. 2026-11-05 re-review: bigger markers and fonts, darker k-trails.
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.75), dpi=300, sharey=True)
    n = len(rows)
    ys = list(range(n))[::-1]
    y_pool = 0.30  # pooled marker height within a row
    step = 0.14  # vertical step between k-trail markers
    ext = [v for _, vals, _ in rows for (pm, perk) in vals.values() for t in [pm, *perk] for v in t[1:]]
    XL, XR = min(ext) - 0.01, max(ext) + 0.01
    for ax, s in zip(axes, SET_STYLE):
        st = SET_STYLE[s]
        ax.axvline(0, color=INK_2, lw=0.8, zorder=1)
        for y, (label, vals, is_eq) in zip(ys, rows):
            if is_eq:
                ax.axvspan(-MARGIN, MARGIN, ymin=0, ymax=1, color=GRID, lw=0, zorder=0) if False else \
                    ax.fill_betweenx([y - 0.5, y + 0.5], -MARGIN, MARGIN, color=GRID, lw=0, zorder=0)
            (m, lo, hi), perk = vals[s]
            y0 = y + y_pool
            ax.errorbar([m], [y0], xerr=[[m - lo], [hi - m]], fmt=st["marker"], color=st["color"],
                        mfc=st["color"], mec=st["color"], ms=5.5, mew=1.0, elinewidth=1.4, capsize=2.2,
                        capthick=1.1, zorder=4)
            ky = [y0 - step * (i + 1.25) for i in range(len(KS))]
            kx = [p[0] for p in perk]
            for (_, plo, phi), yk in zip(perk, ky):
                ax.plot([plo, phi], [yk, yk], color=st["color"], lw=0.8, alpha=0.6, zorder=2,
                        solid_capstyle="butt")
            ax.plot(kx, ky, color=st["color"], lw=0.9, zorder=3)
            ax.plot(kx, ky, marker=st["marker"], ms=3.6, mew=0.9, mec=st["color"], mfc="white", lw=0,
                    zorder=3.5)
            trail = "  ".join(f"k{k} {p[0]:+.3f}" for k, p in zip(KS, perk))
            print(f"{label.splitlines()[0]:28s} {s:9s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}]   {trail}")
        ax.set_title(st["label"], fontsize=8.5, color=st["color"], pad=3)
        ax.set_ylim(ys[-1] - 0.52, ys[0] + 0.52)
        ax.set_xlim(XL, XR)
        ax.set_xlabel(r"$\Delta$nDCG@10 (95% CI; expanded labels)", fontsize=7.8, color=INK, labelpad=1.5)
        ax.grid(axis="x", color=GRID, lw=0.5, zorder=0)
        for y in ys[:-1]:
            ax.axhline(y - 0.5, color=GRID, lw=0.5, zorder=0)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(INK_2)
        ax.spines["bottom"].set_linewidth(0.6)
        ax.tick_params(axis="x", colors=INK_2, labelcolor=INK, labelsize=7.5, width=0.6, length=2.5)
        ax.tick_params(axis="y", length=0, pad=3)
    axes[0].set_yticks(ys)
    axes[0].set_yticklabels([r[0] for r in rows], fontsize=7.8, color=INK, linespacing=1.05)
    handles = [
        Line2D([], [], color=INK_2, marker="o", mfc=INK_2, mec=INK_2, ms=5.0, lw=0, label="all queries ($k=1$ to 4)"),
        Line2D([], [], color=INK_2, marker="o", mfc="white", mec=INK_2, ms=3.6, lw=0.9,
               label="$k=1,2,3,4$, top to bottom"),
        Line2D([], [], color=GRID, lw=6, label=r"$\pm0.02$ equivalence margin (90% CI row)"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.6, 0.0), ncol=3, frameon=False,
               fontsize=7.3, handlelength=1.6, columnspacing=1.4, handletextpad=0.5, borderaxespad=0.1)
    fig.subplots_adjust(left=0.205, right=0.99, top=0.925, bottom=0.215, wspace=0.06)

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
