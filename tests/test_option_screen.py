"""Tests for the exploratory option screen (model/option_screen.py): forecast-weather rows and pooling."""
import copy
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "tests"):
    sys.path.insert(0, str(p))

import option_screen as osc  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402
from test_backtest import synth  # noqa: E402


def test_forecast_rows_use_the_supplied_forecast_and_never_the_observed_target_weather():
    hist, live = synth(last=date(2023, 6, 1))
    t = date(2023, 4, 10)
    fc = {t: {k: 1.0 + i for i, k in enumerate(WEATHER_KEYS)}}
    a = osc.build(hist, live, lambda tt, d0: fc.get(tt))
    assert list(a) == [t]                                          # only days with a forecast are built
    h2 = copy.deepcopy(hist)
    for k in WEATHER_KEYS:
        h2[t][k] += 50.0                                           # change tomorrow's OBSERVED weather
    b = osc.build(h2, live, lambda tt, d0: fc.get(tt))
    assert a[t]["row"] == b[t]["row"]
    for i, k in enumerate(WEATHER_KEYS):
        assert a[t]["row"][f"wx1_{k}"] == 1.0 + i


def test_offline_and_live_views_differ_only_through_today():
    hist, live = synth(last=date(2023, 6, 1))
    t = date(2023, 4, 10)
    live2 = copy.deepcopy(live)
    live2[t - timedelta(days=1)]["pm25"] = 1.0                     # the 23:00 value of today
    obs = lambda tt, d0: {k: hist[tt][k] for k in WEATHER_KEYS}  # noqa: E731
    off = osc.build(hist, live2, obs, view="offline")[t]["row"]
    liv = osc.build(hist, live2, obs, view="live")[t]["row"]
    assert liv["pm_d0"] == 1.0 and off["pm_d0"] == hist[t - timedelta(days=1)]["pm25"]


def test_pooled_dates_keep_winters_apart():
    w1 = [date(2024, 10, 15) + timedelta(days=i) for i in range(10)]
    w2 = [date(2025, 10, 15) + timedelta(days=i) for i in range(10)]
    ds = osc.pooled_dates([w1, w2])
    assert len(ds) == 20 and (ds[10] - ds[9]).days > 150


def test_hybrid_routes_by_day_part_availability():
    rows = [{f: 1.0 for f in osc.FEATURE_SETS[osc.V2]}, {f: float("nan") for f in osc.FEATURE_SETS[osc.V2]}]
    out = osc.hybrid(np.array([10.0, 10.0]), np.array([20.0, 20.0]), rows)
    assert list(out) == [20.0, 10.0]
