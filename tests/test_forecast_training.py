"""Tests for the forecast-training experiment (model/forecast_training_experiment.py)."""
import copy
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "tests"):
    sys.path.insert(0, str(p))

import forecast_training_experiment as fte  # noqa: E402
import winter_backtest as wb  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from test_backtest import synth  # noqa: E402


def _fc(hist, days, shift=2.0):
    return {t: {k: hist[t][k] + shift for k in WEATHER_KEYS} for t in days}


def test_stat_intervals_match_the_backtest_bootstrap():
    rng = np.random.default_rng(1)
    n = 80
    dates = [date(2024, 11, 1) + timedelta(days=i) for i in range(n)]
    y = rng.uniform(40, 300, n)
    pa = y + rng.normal(0, 30, n)
    pb = y + rng.normal(0, 25, n)
    st = fte.Stat(y, {"persistence": pa, "a": pa, "b": pb}, dates, y)
    d = st.diff("a", "b")
    ref = wb.mbb_mae_diff(y, pa, pb, dates, block=7, n_boot=5000, seed=0)
    assert np.allclose(d["mae_ci95_7d"], ref["ci95_diff"], atol=1e-9)
    assert abs(d["mae"] - ref["diff"]) < 1e-9


def test_identical_predictions_are_never_called_conclusive():
    dates = [date(2024, 11, 1) + timedelta(days=i) for i in range(30)]
    y = np.linspace(50, 200, 30)
    st = fte.Stat(y, {"persistence": y + 5, "a": y + 3, "b": y + 3}, dates, y)
    d = st.diff("a", "b")
    assert d["identical"] and not d["excludes_zero_both"]


def test_calibration_uses_only_days_before_the_window():
    hist, _ = synth(last=date(2023, 6, 1))
    days = [d for d in sorted(hist) if date(2022, 1, 1) <= d <= date(2023, 5, 31)]
    fc = _fc(hist, days)
    before = date(2023, 1, 1)
    c1 = fte.fit_calibration(hist, fc, before)
    fc2 = copy.deepcopy(fc)
    for d in days:
        if d >= before:
            for k in WEATHER_KEYS:
                fc2[d][k] += 1000.0                          # wreck every day from the window start on
    assert fte.fit_calibration(hist, fc2, before) == c1
    assert all(n == sum(1 for d in days if d < before) for _, _, n in c1.values())


def test_calibration_output_stays_physical():
    cal = {k: (-50.0, 1.0, 100) for k in WEATHER_KEYS}
    out = fte.apply_calibration({k: 1.0 for k in WEATHER_KEYS}, cal)
    assert out["wind"] == 0.0 and out["precip"] == 0.0 and out["rh"] == 0.0
    cal = {k: (500.0, 1.0, 100) for k in WEATHER_KEYS}
    assert fte.apply_calibration({k: 1.0 for k in WEATHER_KEYS}, cal)["rh"] == 100.0


def test_forecast_only_rows_never_substitute_observed_weather():
    hist, live = synth(last=date(2023, 6, 1))
    days = [date(2023, 4, 10), date(2023, 4, 11)]
    fc = _fc(hist, days[:1], shift=7.0)
    strict = fte.build(hist, live, lambda t: fc.get(t), view="offline")
    assert list(strict) == days[:1]                          # the day without a forecast is absent, not filled
    mixed = fte.build(hist, live, lambda t: fc.get(t) or {k: hist[t][k] for k in WEATHER_KEYS}, view="offline")
    assert {days[0], days[1]} <= set(mixed)
    assert mixed[days[0]]["row"]["wx1_temp"] == hist[days[0]]["temp"] + 7.0
    assert mixed[days[1]]["row"]["wx1_temp"] == hist[days[1]]["temp"]


def test_today_weather_override_only_changes_wx0():
    hist, live = synth(last=date(2023, 6, 1))
    t = date(2023, 4, 10)
    d0 = t - timedelta(days=1)
    obs = lambda tt: {k: hist[tt][k] for k in WEATHER_KEYS}  # noqa: E731
    a = fte.build(hist, live, obs)[t]["row"]
    shifted = {k: hist[d0][k] + 3.0 for k in WEATHER_KEYS}
    b = fte.build(hist, live, obs, wx0_for=lambda d: shifted if d == d0 else None)[t]["row"]
    assert b["wx0_temp"] == hist[d0]["temp"] + 3.0 and a["wx1_temp"] == b["wx1_temp"]
    assert hist[d0]["temp"] == synth(last=date(2023, 6, 1))[0][d0]["temp"]      # history not mutated


def test_the_untouched_winter_is_never_built():
    hist, live = synth(last=date(2026, 11, 30))
    fc = _fc(hist, [d for d in hist if d >= date(2026, 10, 1)])
    rows = fte.build(hist, live, lambda t: fc.get(t))
    assert all(t < date(2026, 10, 15) for t in rows)


def test_arm_definitions_match_the_documented_design():
    assert fte.ARMS["A0_era5_trained"] == ("obs", "fc")
    assert fte.ARMS["A1_forecast_only"][0] == "fc_strict" and fte.MIN_TRAIN == 365
    assert set(fte.CANDIDATES) == {"A1_forecast_only", "A2_mixed", "A3_calibrated_input"}
    assert all(wb.winter_window(Y)[1] <= fte.HARD_STOP for Y in fte.WINTERS)


def test_prevruns_fetch_merges_chunks_and_names_variables():
    import fetch_prevruns as fp
    calls = []

    def fake(url, params):
        calls.append(params)
        n = (date.fromisoformat(params["end_date"]) - date.fromisoformat(params["start_date"])).days + 1
        t = [(date.fromisoformat(params["start_date"]) + timedelta(days=i)).isoformat() + "T00:00" for i in range(n)]
        return {"latitude": 28.5, "hourly": {"time": t, **{v: [1.0] * n for v in fp.variables()}}}

    out = fp.fetch(date(2024, 1, 1), date(2025, 1, 1), get=fake)
    assert len(calls) == 3 and len(out["hourly"]["time"]) == 367
    assert "temperature_2m_previous_day1" in out["hourly"] and "surface_pressure_previous_day2" in out["hourly"]
