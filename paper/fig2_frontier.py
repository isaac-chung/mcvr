"""Review item O4 (2026-11-05): cost-quality frontier, mean input tokens per query against
nDCG@10 (expanded labels), all 722 test queries pooled, with 95% cluster-bootstrap intervals over
source videos. No model calls.

    uv run --offline --with matplotlib python paper/fig2_frontier.py

Systems (paper names; data keys in brackets): S0, S2@50 [S2g@50], S4@50, S6 (margin <= 2, dev
threshold), S3 [S3g], S5. Tokens as Table 1 (logged input tokens; the frame systems add S4@50's
pass; the splitter's ~0.18k input tokens for S2 / S3 are not in the logged verifier cost, as in
Table 1). The faint line is the margin trigger's threshold sweep from S4@50 (never send) to S5
(always send), from results/full/review_20261105/o3_trigger_curve.csv if present.
Style follows paper/fig2_trigger.py (serif, colours, 3.3 in wide).
Writes results/full/figures/fig_frontier.{pdf,png} and copies the PDF to results/paper/figures/.
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]

OUT_DIR = C.ROOT / "results/full/figures"
PAPER_DIR = ROOT / "results/paper/figures"
STEM = "fig_frontier"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#d9d8d4"
C_MARGIN, C_TOP = "#5b3fa8", "#199e70"
C_DESC, C_FRAME = "#2a78d6", "#eb6834"


def main() -> int:
    S, sets, sub, ranks, lines, qrels = C.load()
    rows = [(n, r) for n, rs in sets.items() for r in rs]
    q = qrels["expanded"]
    t_m = float(S.tier["trigger"]["margin_threshold"])
    sc = {n: {s: C.score(ranks[n][s], q[n]) for s in ("S0", "S2g@50", "S4@50", "S3g", "S5")} for n in sets}

    def tok(n, s, qid):
        return float(lines[n][s][qid]["cost"]["tokens_in"]) if s != "S0" else 0.0

    pts = {}
    cl = [r["source"] for _, r in rows]
    for name, key, frames in (("S0", "S0", False), ("S2@50", "S2g@50", False), ("S4@50", "S4@50", False),
                              ("S3", "S3g", True), ("S5", "S5", True)):
        y = [sc[n][key][r["query_id"]]["ndcg@10"] for n, r in rows]
        x = [tok(n, key, r["query_id"]) + (tok(n, "S4@50", r["query_id"]) if frames else 0.0) for n, r in rows]
        pts[name] = (C.mean_ci(y, cl), C.mean_ci(x, cl))
    y6, x6 = [], []
    for n, r in rows:
        qid = r["query_id"]
        send = C.TS.margin(ranks[n]["S4@50"][qid]) <= t_m
        y6.append(sc[n]["S5" if send else "S4@50"][qid]["ndcg@10"])
        x6.append(tok(n, "S4@50", qid) + (tok(n, "S5", qid) if send else 0.0))
    pts["S6"] = (C.mean_ci(y6, cl), C.mean_ci(x6, cl))
    for s, (y, x) in pts.items():
        print(f"{s:6s} tokens {x['mean'] / 1000:6.1f}k [{x['ci_low'] / 1000:.1f}, {x['ci_high'] / 1000:.1f}]  "
              f"nDCG@10 {y['mean']:.3f} [{y['ci_low']:.3f}, {y['ci_high']:.3f}]")
    C.dump("o4_frontier.json", {s: {"ndcg10": y, "tokens_in": x} for s, (y, x) in pts.items()})

    sweep = []
    cpath = C.OUT / "o3_trigger_curve.csv"
    if cpath.exists():
        with cpath.open() as fh:
            for r in csv.DictReader(fh):
                if r["group"] == "pooled" and r["kind"] == "threshold":
                    sweep.append((float(r["tokens_in_per_query"]) / 1000, float(r["ndcg10"])))
        sweep.sort()

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times", "Times New Roman", "STIX Two Text", "STIXGeneral", "DejaVu Serif"],
        "mathtext.fontset": "stix", "pdf.fonttype": 42, "ps.fonttype": 42, "axes.unicode_minus": True,
    })
    fig, ax = plt.subplots(figsize=(3.3, 2.35))
    if sweep:
        ax.plot([p[0] for p in sweep], [p[1] for p in sweep], color=C_MARGIN, lw=0.8, alpha=0.45, marker="o",
                ms=1.6, zorder=2, label="margin trigger, threshold sweep")
    style = {  # colour, marker, label offset (points), ha
        "S0": (INK_2, "o", (5, -1), "left"),
        "S2@50": (C_DESC, "s", (5, -3), "left"),
        "S4@50": (C_DESC, "o", (5, 1), "left"),
        "S6": (C_MARGIN, "D", (-4, 6), "right"),
        "S5": (C_FRAME, "o", (-10, 7), "center"),
        "S3": (C_FRAME, "s", (6, -8), "center"),
    }
    for s, (col, mk, (dx, dy), ha) in style.items():
        y, x = pts[s]
        xm, ym = x["mean"] / 1000, y["mean"]
        ax.errorbar([xm], [ym], yerr=[[ym - y["ci_low"]], [y["ci_high"] - ym]], fmt=mk, color=col, mfc=col, mec="white",
                    ms=4.6 if mk != "D" else 4.0, mew=0.6, elinewidth=0.9, capsize=1.6, capthick=0.8, zorder=5)
        lab = s if s != "S6" else "S6 (margin$\\,\\leq 2$)"
        ax.annotate(lab, (xm, ym), xytext=(dx, dy), textcoords="offset points", fontsize=6.2, color=INK,
                    ha=ha, va="center", zorder=7)
    ax.set_xlim(-3, 100)
    ax.set_ylim(0.36, 0.78)
    ax.set_xlabel("input tokens per query (thousands)", fontsize=7, color=INK, labelpad=1)
    ax.set_ylabel("nDCG@10 (expanded labels)", fontsize=7, color=INK)
    ax.grid(color=GRID, lw=0.5, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
        ax.spines[side].set_linewidth(0.6)
    ax.tick_params(colors=INK_2, labelcolor=INK, labelsize=6.5, width=0.6, length=2.5)
    if sweep:
        ax.legend(loc="lower right", frameon=False, fontsize=6.0, handlelength=1.6, borderaxespad=0.3)
    fig.subplots_adjust(left=0.13, right=0.98, top=0.98, bottom=0.15)
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
