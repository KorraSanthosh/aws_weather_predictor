import json, sys
from datetime import date
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_range as rr
R = json.loads((HERE / "results.json").read_text())


def test_calibration_strictly_before_test():
    for t in R["tests"].values():
        assert t["calibration_before_test"] and t["calibration_last_day"] < t["first"]


def test_method_chosen_without_test_results_and_gate_declared():
    assert R["method_choice_on_2024_calibration_split"]["chosen"] in ("M1", "M2", "M3")
    assert json.loads((HERE / "config.json").read_text())["declared_before_running"] is True


def test_ranges_contain_point_and_are_ordered():
    for t in R["tests"].values():
        d = t["daily_chosen"]
        lo, p, hi, y = map(np.array, (d["lo"], d["point"], d["hi"], d["actual"]))
        assert (lo <= p + 1e-6).all() and (p <= hi + 1e-6).all() and (lo >= 0).all()
        assert len(y) == t["test_days"]


def test_conformal_quantiles_have_nominal_coverage_on_synthetic_exchangeable_data():
    rng = np.random.default_rng(0)
    p = rng.uniform(10, 200, 4000); y = p * np.exp(rng.normal(0, 0.3, 4000))
    cal = {"y": y[:2000], "p": p[:2000]}; te = {"y": y[2000:], "p": p[2000:]}
    m = rr.m1(cal, 0.10); lo, hi = rr.apply(m, te | {"qlog": np.zeros((2000, 2))}, 0.10)
    assert abs(((te["y"] >= lo) & (te["y"] <= hi)).mean() - 0.90) < 0.03


def test_category_range_logic():
    d = {"y": np.array([20.0, 100.0]), "p": np.array([25.0, 70.0]), "pm0": np.array([20.0, 80.0]), "days": [date(2025, 1, 1), date(2025, 1, 2)]}
    sc = rr.score(d, np.array([10.0, 50.0]), np.array([40.0, 130.0]), 0.1)
    assert sc["arrays"]["cont"].tolist() == [True, True] and sc["arrays"]["span"].tolist() == [2, 4]


def test_no_forecast_substitution_calibration_uses_archive_only():
    assert R["sizes"]["first_calibration_day"] >= "2024-01-19"
