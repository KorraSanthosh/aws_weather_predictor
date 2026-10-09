import math
import random
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.aqi import (aqi_to_category, is_alert, pm25_to_aqi, pm25_to_category,  # noqa: E402
                      school_guidance)
from core.daily import IST, city_daily_mean, readings_to_daily  # noqa: E402
from core.features import FEATURE_NAMES, build_features  # noqa: E402
from core.weather import WEATHER_KEYS, hourly_to_daily_weather  # noqa: E402


# ---------- AQI (CPCB breakpoints) ----------
@pytest.mark.parametrize("pm,aqi", [(0, 0), (30, 50), (31, 51), (60, 100), (90, 200),
                                    (120, 300), (250, 400), (380, 500), (900, 500)])
def test_pm25_breakpoints(pm, aqi):
    assert pm25_to_aqi(pm) == aqi


def test_pm25_interpolates_and_is_monotonic():
    # CPCB formula in band 61-90 -> 101-200: (200-101)/(90-61) * (75-61) + 101 = 148.8 -> 149
    assert pm25_to_aqi(75) == 149
    vals = [pm25_to_aqi(x / 2) for x in range(0, 900)]
    assert vals == sorted(vals)


def test_categories_and_alerts():
    assert pm25_to_category(20) == "Good"
    assert pm25_to_category(45) == "Satisfactory"
    assert pm25_to_category(75) == "Moderate"
    assert pm25_to_category(100) == "Poor"
    assert pm25_to_category(200) == "Very Poor"
    assert pm25_to_category(300) == "Severe"
    assert not is_alert("Moderate") and is_alert("Poor") and is_alert("Severe")
    assert aqi_to_category(200) == "Moderate" and aqi_to_category(201) == "Poor"


def test_school_guidance_levels():
    assert school_guidance("Good")["decision"] == "GO"
    assert school_guidance("Moderate")["decision"] == "CAUTION"
    assert school_guidance("Poor")["decision"] == "INDOOR"
    assert school_guidance("Severe")["decision"] == "INDOOR"


def test_pm25_rejects_nan():
    with pytest.raises(ValueError):
        pm25_to_aqi(float("nan"))


# ---------- daily aggregation ----------
def _hours(day_ist: date, n: int, value: float):
    start = datetime(day_ist.year, day_ist.month, day_ist.day, tzinfo=IST)
    return [(start + timedelta(hours=h), value) for h in range(n)]


def test_daily_needs_16_valid_hours():
    d = date(2026, 1, 10)
    assert d not in readings_to_daily(_hours(d, 15, 100))
    assert readings_to_daily(_hours(d, 16, 100))[d] == (100.0, 16)


def test_daily_uses_india_time_for_day_boundaries():
    # 20:00 UTC on Jan 9 is 01:30 IST on Jan 10 -> belongs to Jan 10
    ts = datetime(2026, 1, 9, 20, 0, tzinfo=timezone.utc)
    readings = [(ts + timedelta(hours=h), 50.0) for h in range(24)]
    out = readings_to_daily(readings)
    assert date(2026, 1, 10) in out


def test_daily_drops_bad_values_and_averages_subhourly():
    d = date(2026, 1, 10)
    good = _hours(d, 20, 80)
    bad = [(good[0][0], -5.0), (good[1][0], 5000.0), (good[2][0], float("nan"))]
    out = readings_to_daily(good + bad)
    assert out[d][0] == pytest.approx(80.0)
    # two readings in the same hour are averaged, not double counted
    sub = [(t + timedelta(minutes=30), 120.0) for t, _ in good[:1]]
    out2 = readings_to_daily(good + sub)
    assert out2[d][1] == 20


def test_city_mean():
    d = date(2026, 1, 10)
    out = city_daily_mean([{d: (100.0, 20)}, {d: (200.0, 22)}])
    assert out[d] == (150.0, 2)


# ---------- weather reduction ----------
def test_hourly_to_daily_weather_wind_components():
    times = [f"2026-01-10T{h:02d}:00" for h in range(24)]
    hourly = {"time": times,
              "temperature_2m": [10.0] * 24, "relative_humidity_2m": [50.0] * 24,
              "wind_speed_10m": [10.0] * 24, "wind_direction_10m": [270.0] * 24,   # from the west
              "precipitation": [0.5] * 24, "surface_pressure": [1000.0] * 24}
    w = hourly_to_daily_weather(hourly)[date(2026, 1, 10)]
    assert w["precip"] == pytest.approx(12.0)
    assert w["wind_u"] == pytest.approx(10.0, abs=1e-6)      # west wind blows toward +x (east)
    assert w["wind_v"] == pytest.approx(0.0, abs=1e-6)


# ---------- features: no leakage ----------
def _history(n=40, seed=1):
    rnd = random.Random(seed)
    start = date(2026, 1, 1)
    h = {}
    for i in range(n):
        rec = {"pm25": rnd.uniform(40, 300)}
        rec.update({k: rnd.uniform(1, 30) for k in WEATHER_KEYS})
        h[start + timedelta(days=i)] = rec
    return h, start


def test_features_ignore_the_future():
    hist, start = _history()
    d0 = start + timedelta(days=30)
    wx1 = {k: 5.0 for k in WEATHER_KEYS}
    full = build_features(hist, d0, wx1)
    truncated = {d: r for d, r in hist.items() if d <= d0}
    assert build_features(truncated, d0, wx1) == pytest.approx(full, nan_ok=True)
    # tomorrow's PM2.5 (the label) must never appear among the features
    tomorrow = hist[d0 + timedelta(days=1)]["pm25"]
    assert tomorrow not in full


def test_features_shape_and_missing_handling():
    hist, start = _history()
    d0 = start + timedelta(days=30)
    del hist[d0 - timedelta(days=1)]                 # a gap in the history
    row = build_features(hist, d0, {k: 1.0 for k in WEATHER_KEYS})
    assert len(row) == len(FEATURE_NAMES)
    assert math.isnan(row[FEATURE_NAMES.index("pm_d1")])
    assert not math.isnan(row[FEATURE_NAMES.index("pm_mean7")])
