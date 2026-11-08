"""S6 trigger simulation (dev/trigger_dev.py): sweep logic on synthetic rows; offline."""

import importlib.util
import math
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "dev/trigger_dev.py"


@pytest.fixture(scope="module")
def ts():
    spec = importlib.util.spec_from_file_location("trigger_sim", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def rows():
    # four queries; S5 helps on a and c, hurts on d; feature f: low = uncertain
    spec = [
        ("a", "s1", 0.0, 0.5, 1.0, 100.0),
        ("b", "s2", 1.0, 1.0, 1.0, 200.0),
        ("c", "s3", 2.0, 0.0, 1.0, 100.0),
        ("d", "s4", 3.0, 1.0, 0.5, 100.0),
    ]
    return [
        {
            "query_id": q,
            "source": s,
            "f": f,
            "q4": q4,
            "q5": q5,
            "r4": q4,
            "r5": q5,
            "c_base": 10.0,
            "c5": c5,
        }
        for q, s, f, q4, q5, c5 in spec
    ]


def test_helpers(ts):
    assert ts.margin([("x", 9.0), ("y", 5.0), ("z", 1.0)]) == 4.0
    assert ts.unclear_share({"x": ["yes", "unclear", "unclear", "no"]}, "x") == 0.5
    assert ts.unclear_share({}, "x") is None
    assert ts.fires(1.0, 1.0, "le") and not ts.fires(1.1, 1.0, "le")
    assert ts.fires(2.0, 2.0, "ge") and not ts.fires(1.0, 2.0, "ge")
    assert ts.thresholds([2.0, 1.0, 2.0], "le") == [-math.inf, 1.0, 2.0]
    assert ts.thresholds([2.0, 1.0], "ge") == [math.inf, 2.0, 1.0]
    assert math.isnan(ts.kept(0.5, 0.5, 0.4))
    assert ts.kept(0.75, 0.5, 1.0) == 0.5


def test_sweep_endpoints_and_order(ts):
    pts = ts.sweep(rows(), "f", "le")
    assert [p["share"] for p in pts] == [0.0, 0.25, 0.5, 0.75, 1.0]
    # never: S4 only; always: S5 everywhere
    assert pts[0]["ndcg10"] == pytest.approx(0.625) and pts[0]["cost"] == 10.0
    assert pts[-1]["ndcg10"] == pytest.approx(0.875) and pts[-1]["cost"] == pytest.approx(135.0)
    assert pts[0]["kept_ndcg"] == 0.0 and pts[-1]["kept_ndcg"] == pytest.approx(1.0)
    # f <= 0 sends only a: gain .5 of 1.0 total gain
    assert pts[1]["kept_ndcg"] == pytest.approx(0.5) and pts[1]["cost"] == pytest.approx(35.0)
    # f <= 2 sends a, b, c: kept 1.5 / 1.0 (d, where S5 hurts, stays on S4)
    assert pts[3]["kept_ndcg"] == pytest.approx(1.5)
    ge = ts.sweep(rows(), "f", "ge")
    assert [p["share"] for p in ge] == [0.0, 0.25, 0.5, 0.75, 1.0]
    assert ge[1]["kept_ndcg"] == pytest.approx(-0.5)  # f >= 3 sends only d


def test_operating_point_is_cheapest_meeting_target(ts):
    pts = ts.sweep(rows(), "f", "le")
    op = ts.operating_point(pts, 0.8)
    # f <= 2 (kept 1.5, cost 110) is cheaper than always (135); f <= 1 has kept .5
    assert op["t"] == 2.0 and op["cost"] == pytest.approx(110.0)
    assert ts.operating_point(pts, 2.0) is None
    flat = [{**r, "q5": r["q4"]} for r in rows()]
    assert ts.operating_point(ts.sweep(flat, "f", "le"), 0.8) is None


def test_auc(ts):
    assert ts.auc([0.0, 1.0], [0.0, 1.0]) == pytest.approx(0.5)
    assert ts.auc([0.0, 0.5, 1.0], [0.0, 1.0, 1.0]) == pytest.approx(0.75)
    assert ts.auc([0.0, 0.5, 1.0], [0.0, math.nan, 1.0]) == pytest.approx(0.5)
    # a random-like linear curve has zero excess
    lin = [
        {"share": x, "cost": 10 + 100 * x, "ndcg10": 0.5 + 0.2 * x, "kept_ndcg": x}
        for x in (0.0, 0.25, 0.5, 1.0)
    ]
    a = ts.auc_summary(lin)
    assert a["auc_share"] == pytest.approx(0.5) and a["auc_cost"] == pytest.approx(0.5)
    assert a["excess"] == pytest.approx(0.0, abs=1e-12)


def test_ratio_ci_and_crossfit(ts):
    lo, hi = ts.ratio_ci([1.0, 2.0, 3.0], [2.0, 4.0, 6.0], ["a", "b", "c"], n_boot=200)
    assert lo == pytest.approx(0.5) and hi == pytest.approx(0.5)
    many = [
        {**r, "query_id": f"{r['query_id']}{i}", "source": f"{r['source']}{i}"}
        for i in range(50)
        for r in rows()
    ]
    cf = ts.crossfit(many, "f", "le", 0.8, folds=5, reps=3)
    assert set(cf) == {"kept", "share", "cost"}
    assert 0.0 <= cf["share"][0] <= 1.0
    # near-balanced folds: f <= 2 picked on every training split, held-out kept 1.5
    assert cf["kept"][0] == pytest.approx(1.5)
