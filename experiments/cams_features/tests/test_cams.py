import json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_cams as rc
R = json.loads((HERE / "results.json").read_text()); P = json.loads((HERE / "probe.json").read_text())


def test_probe_shows_no_archived_cams_forecast():
    assert P["summary"]["previous_day_values_present_anywhere"] is False and P["summary"]["pm2_5_has_values_in_all_answered"]


def test_results_are_labelled_non_realistic():
    assert "NON-REALISTIC" in R["label"]


def test_lag_features_only_use_days_up_to_origin():
    cams = {date(2025, 1, d): float(d) for d in range(1, 10)}
    t = date(2025, 1, 6)
    out = rc.with_cams({t: {"y": 1.0, "row": {}}}, cams, rc.ARMS["O_lags_only"])[t]["row"]
    assert out["cams_d0"] == 5.0 and out["cams_d1"] == 4.0                      # origin day and the day before, never t or later
    assert not set(rc.ARMS["O_lags_only"]) & {"cams_t", "cams_chg"}


def test_missing_cams_stays_nan_not_filled():
    out = rc.with_cams({date(2025, 1, 6): {"y": 1.0, "row": {}}}, {}, ["cams_d0"])[date(2025, 1, 6)]["row"]
    assert np.isnan(out["cams_d0"]) and np.isnan(out["cams_t"])


def test_same_days_for_baseline_and_candidate_and_baseline_reproduces_earlier_numbers():
    c = R["windows"]["calendar_2025 (train <= 2024-12-31)"]["result"]
    assert c["n_days"] == 344 and abs(c["O_target_day"]["models"]["hybrid"]["mae"] - 19.5675) < 1e-3          # equals experiments/backtest_2025
    w = R["windows"]["winter_2025_26 (train < 2025-10-15)"]["result"]
    assert w["n_days"] == 104 and abs(w["O_target_day"]["models"]["hybrid"]["mae"] - 28.6597) < 1e-3


def test_training_cutoffs_precede_tests():
    assert all(a < b for (a, b, c) in rc.WINDOWS.values())
