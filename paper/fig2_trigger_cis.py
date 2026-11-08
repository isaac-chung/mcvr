"""95% cluster-bootstrap intervals (10,000 resamples of sources) of the named operating points of
Figure 2 that results/full/review_20261105/o3_trigger.json does not store: top score < 8, k >= 3,
random at the margin rule's share (same fire rules as paper/fig2_trigger_sweep.py; same bootstrap as
paper/trigger_analysis.py policy_stats). Offline, no model call.
    uv run --offline --with matplotlib python paper/fig2_trigger_cis.py
Writes results/full/review_20261105/o3_named_points_ci.json.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import paper_common as C  # noqa: E402
import trigger_analysis as O3  # noqa: E402


def main():
    S, sets, sub, ranks, lines, qrels = C.load()
    rows = C.T1.trigger_rows(S, sets, ranks, lines, qrels["expanded"], sub)
    T = S.tier["trigger"]
    t_m, t_k, t_s = float(T["margin_threshold"]), int(T["k_threshold"]), float(T["top_score_threshold"])
    share_m = sum(r["s4_margin"] <= t_m for r in rows) / len(rows)
    cidx, n_c = O3.cluster_index(rows)
    W = O3.boot_picks(n_c)
    rules = {
        "margin <= 2 (dev threshold)": lambda r: r["s4_margin"] <= t_m,
        "S4 top score < 8": lambda r: r["s4_top"] < t_s,
        "k >= 3": lambda r: r["k"] >= t_k,
        "random (pooled margin share)": lambda r: r["u"] < share_m,
    }
    out = {name: O3.policy_stats(rows, [bool(f(r)) for r in rows], W, cidx, n_c) for name, f in rules.items()}
    p = C.dump("o3_named_points_ci.json", out)
    for k, v in out.items():
        print(f"{k:32s} share {v['share']:.3f} kept {v['kept']:.4f} {v['kept_ci']}")
    print(p)


if __name__ == "__main__":
    main()
