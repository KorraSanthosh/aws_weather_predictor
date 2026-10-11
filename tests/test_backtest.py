"""Tests of the historical winter back-test (model/winter_backtest.py) on synthetic data: leakage, frozen fit,
eligibility ladder, hybrid routing and the moving-block bootstrap."""
import copy
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "model", ROOT / "pipeline"):
    sys.path.insert(0, str(p))

import winter_backtest as wb  # noqa: E402
from core.daily import DP_KEYS  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402

START = date(2024, 10, 15)
BOOT = {"block_days": 7, "sensitivity_block_days": 14, "n_boot": 200, "seed": 0}


def synth(first=date(2023, 1, 1), last=date(2025, 3, 1), dp_from=None, seed=0):
    """A daily history with an AR(1) log PM2.5, smooth weather, and (optionally) day-part values from `dp_from`."""
    rng = np.random.default_rng(seed)
    hist, live = {}, {}
    lp = 4.5
    d = first
    while d <= last:
        lp = 0.85 * lp + 0.15 * 4.6 + rng.normal(0, 0.25)
        pm = float(np.exp(lp))
        rec = {"pm25": pm}
        for i, k in enumerate(WEATHER_KEYS):
            rec[k] = float(10 + 3 * np.sin(d.toordinal() / (20 + i)) + rng.normal(0, 0.5))
        has_dp = dp_from is not None and d >= dp_from
        for k in DP_KEYS:
            rec[k] = float(pm * rng.uniform(0.8, 1.2)) if has_dp else None
        hist[d] = rec
        live[d] = {"pm25": pm, **{k: (rec[k] if has_dp else float("nan")) for k in DP_KEYS}}
        d += timedelta(days=1)
    return hist, live


# ------------------------------------------------------------------ bootstrap
def test_mbb_identical_predictions_have_zero_interval():
    y = np.arange(50, 100.0)
    p = y + 3
    dates = [START + timedelta(days=i) for i in range(len(y))]
    r = wb.mbb_mae_diff(y, p, p.copy(), dates, n_boot=100)
    assert r["diff"] == 0 and r["ci95_diff"] == [0.0, 0.0] and r["identical_predictions"] and not r["excludes_zero"]


def test_mbb_detects_a_consistent_difference_and_contains_the_estimate():
    rng = np.random.default_rng(1)
    y = rng.uniform(50, 300, 120)
    pa = y + rng.normal(0, 10, 120)
    pb = y + rng.normal(0, 40, 120)                       # B is clearly worse
    dates = [START + timedelta(days=i) for i in range(120)]
    r = wb.mbb_mae_diff(y, pa, pb, dates, n_boot=500)
    assert r["diff"] > 0 and r["excludes_zero"]
    lo, hi = r["ci95_diff"]
    assert lo <= r["diff"] <= hi


def test_mbb_no_difference_is_mostly_inconclusive():
    dates = [START + timedelta(days=i) for i in range(100)]
    hits = 0
    for seed in range(20):                                   # equally good models: a 95% interval rarely excludes 0
        rng = np.random.default_rng(100 + seed)
        y = rng.uniform(50, 300, 100)
        pa, pb = y + rng.normal(0, 30, 100), y + rng.normal(0, 30, 100)
        hits += wb.mbb_mae_diff(y, pa, pb, dates, n_boot=300)["excludes_zero"]
    assert hits <= 4
    y = np.linspace(60, 200, 100)
    r = wb.mbb_mae_diff(y, y + 12, y - 12, dates, n_boot=200)       # same |error| every day
    assert abs(r["diff"]) < 1e-9 and not r["excludes_zero"]


def test_longer_blocks_widen_the_interval_for_autocorrelated_differences():
    n = 200
    rng = np.random.default_rng(3)
    drift = np.zeros(n)
    for i in range(1, n):
        drift[i] = 0.95 * drift[i - 1] + rng.normal(0, 3)      # strongly serially correlated error difference
    y = np.full(n, 100.0)
    pa, pb = y + 10, y + 10 + drift
    dates = [START + timedelta(days=i) for i in range(n)]
    w1 = np.ptp(wb.mbb_mae_diff(y, pa, pb, dates, block=1, n_boot=2000)["ci95_diff"])
    w14 = np.ptp(wb.mbb_mae_diff(y, pa, pb, dates, block=14, n_boot=2000)["ci95_diff"])
    assert w14 > w1


def test_mbb_handles_gaps_in_the_calendar():
    rng = np.random.default_rng(4)
    days = [START + timedelta(days=i) for i in list(range(0, 30)) + list(range(60, 90))]
    y = rng.uniform(50, 200, len(days))
    r = wb.mbb_mae_diff(y, y + 5, y + 9, days, n_boot=300)
    assert r["n"] == 60 and r["diff"] == pytest.approx(4.0)


# ------------------------------------------------------------------ eligibility ladder
def test_eligibility_ladder_reasons_in_order():
    hist, live = synth(last=date(2023, 3, 1))
    t = date(2023, 2, 10)
    assert wb.classify_target(t, hist, live) is None
    h = copy.deepcopy(hist); h[t]["pm25"] = None
    assert wb.classify_target(t, h, live, {t: 1})[0].startswith("1_") and "only 1 station" in wb.classify_target(t, h, live, {t: 1})[1]
    assert "no station" in wb.classify_target(t, h, live, {})[1]
    h = copy.deepcopy(hist); h[t - timedelta(days=1)]["pm25"] = None
    assert wb.classify_target(t, h, live)[0].startswith("2_")
    h = copy.deepcopy(hist); h[t]["wind"] = None
    assert wb.classify_target(t, h, live)[0].startswith("3_")
    h = copy.deepcopy(hist); h[t - timedelta(days=1)]["wind"] = None
    assert wb.classify_target(t, h, live)[0].startswith("4_")
    lv = dict(live); lv.pop(t - timedelta(days=1))
    assert wb.classify_target(t, hist, lv)[0].startswith("5_")
    h = copy.deepcopy(hist); h[t]["pm25"] = None; h[t]["wind"] = None          # first failing rung is reported
    assert wb.classify_target(t, h, live)[0].startswith("1_")


def test_compress_ranges_merges_consecutive_days_with_the_same_reason():
    items = [(date(2025, 1, d), "a") for d in (1, 2, 3)] + [(date(2025, 1, 4), "b"), (date(2025, 1, 6), "b")]
    out = wb.compress_ranges(items)
    assert [(r["from"], r["to"], r["days"], r["reason"]) for r in out] == [
        ("2025-01-01", "2025-01-03", 3, "a"), ("2025-01-04", "2025-01-04", 1, "b"), ("2025-01-06", "2025-01-06", 1, "b")]


# ------------------------------------------------------------------ no leakage in the feature rows
def test_rows_for_a_target_ignore_the_target_day_pm25_and_the_future():
    hist, live = synth(last=date(2023, 6, 1))
    t = date(2023, 4, 10)
    a = wb.build_rows(hist, live)[t]["rows"]
    h2 = copy.deepcopy(hist)
    for k in range(0, 40):                                        # scramble PM2.5 on the target day and after it
        h2[t + timedelta(days=k)]["pm25"] = 999.0
    b = wb.build_rows(h2, live)[t]["rows"]
    for reg in wb.REGIMES:
        for k, v in a[reg].items():
            assert (v == b[reg][k]) or (v != v and b[reg][k] != b[reg][k]), (reg, k)


def test_persistence_weather_regime_never_reads_tomorrows_weather():
    hist, live = synth(last=date(2023, 6, 1))
    t = date(2023, 4, 10)
    a = wb.build_rows(hist, live)[t]["rows"]
    h2 = copy.deepcopy(hist)
    for k in WEATHER_KEYS:
        h2[t][k] = hist[t][k] + 50.0                              # change ONLY tomorrow's observed weather
    b = wb.build_rows(h2, live)[t]["rows"]
    assert a["live23h_wx_persistence"] == b["live23h_wx_persistence"]
    assert a["live23h_wx_observed"] != b["live23h_wx_observed"]   # while the diagnostic regime does see it
    for k in WEATHER_KEYS:
        assert a["live23h_wx_persistence"][f"wx1_{k}"] == hist[t - timedelta(days=1)][k]


# ------------------------------------------------------------------ frozen fit
def _eval(hist, live, **kw):
    data = wb.build_rows(hist, live)
    return data, wb.evaluate_winter(2024, data, hist, live, boot=BOOT, **kw)


def test_training_uses_only_targets_before_the_window_and_window_outcomes_never_change_predictions():
    hist, live = synth()
    data, res = _eval(hist, live)
    assert res["status"] == "scored"
    assert date.fromisoformat(res["training"]["target_dates_through"]) < START
    assert res["training"]["origins"] == sum(1 for t in data if t < START)
    assert res["training"]["refit_inside_window"] is False
    # wreck the OUTCOMES of the window (targets only): frozen predictions must not move
    data2 = copy.deepcopy(data)
    for t in data2:
        if t >= START:
            data2[t]["y"] *= 7.0
    res2 = wb.evaluate_winter(2024, data2, hist, live, boot=BOOT)
    for reg in wb.REGIMES:
        assert res["regimes"][reg]["daily_predictions"] == res2["regimes"][reg]["daily_predictions"]
    assert res["regimes"]["live23h_wx_persistence"]["sets"]["all_matched_days"]["models"]["v1_25_baseline"]["mae_ug_m3"] != \
        res2["regimes"]["live23h_wx_persistence"]["sets"]["all_matched_days"]["models"]["v1_25_baseline"]["mae_ug_m3"]


def test_all_models_are_scored_on_exactly_the_same_days():
    hist, live = synth()
    _, res = _eval(hist, live)
    g = res["regimes"]["live23h_wx_persistence"]
    n = {len(v) for v in g["daily_predictions"].values()}
    assert n == {res["scored_days"]}
    assert g["sets"]["all_matched_days"]["n_days"] == res["scored_days"]
    s = g["sets"]
    assert s["days_with_day_parts"]["n_days"] + s["days_without_day_parts"]["n_days"] == res["scored_days"]


def test_hybrid_equals_v1_without_day_parts_and_never_imputes():
    hist, live = synth(dp_from=None)
    _, res = _eval(hist, live)
    g = res["regimes"]["live23h_wx_persistence"]
    assert g["n_with_day_parts"] == 0
    assert g["daily_predictions"]["v3_hybrid"] == g["daily_predictions"]["v1_25_baseline"]
    assert g["sets"]["all_matched_days"]["paired"]["v3_hybrid_minus_v1_25_baseline"]["identical_predictions"]
    assert g["sets"]["days_with_day_parts"]["n_days"] == 0


def test_hybrid_uses_v2_exactly_on_days_with_all_day_parts():
    hist, live = synth(dp_from=date(2024, 12, 1))
    _, res = _eval(hist, live)
    g = res["regimes"]["live23h_wx_persistence"]
    hp = g["daily_has_day_parts"]
    assert 0 < sum(hp) < len(hp)
    for i, flag in enumerate(hp):
        want = g["daily_predictions"]["v2_35_candidate" if flag else "v1_25_baseline"][i]
        assert g["daily_predictions"]["v3_hybrid"][i] == want


def test_a_winter_with_too_little_training_data_is_reported_not_scored():
    hist, live = synth(first=date(2024, 6, 1))
    _, res = _eval(hist, live)
    assert res["status"].startswith("not run") and "regimes" not in res


def test_baselines_use_only_information_at_forecast_time():
    hist, live = synth()
    data, res = _eval(hist, live)
    g = res["regimes"]["live23h_wx_persistence"]
    first = date.fromisoformat(res["daily_dates"][0])
    d0 = first - timedelta(days=1)
    assert g["daily_predictions"]["persistence"][0] == pytest.approx(hist[d0]["pm25"], abs=0.01)
    m7 = np.mean([hist[d0 - timedelta(days=k)]["pm25"] for k in range(7)])
    assert g["daily_predictions"]["mean7"][0] == pytest.approx(m7, abs=0.01)


def test_pooled_stacks_winters_without_mixing_blocks():
    hist, live = synth(first=date(2022, 1, 1), last=date(2026, 3, 1))
    data = wb.build_rows(hist, live)
    ws = [wb.evaluate_winter(y, data, hist, live, boot=BOOT) for y in (2024, 2025)]
    assert [w["status"] for w in ws] == ["scored", "scored"]
    out = wb.pooled(ws, "live23h_wx_persistence")
    assert out["sets"]["all_matched_days"]["n_days"] == sum(w["scored_days"] for w in ws)
