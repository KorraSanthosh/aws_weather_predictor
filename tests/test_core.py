import json
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


# ---------------------------------------------------------------- day-part features and domain flags
def test_daypart_stats_use_the_right_hours():
    from core.daily import daypart_stats
    # PM2.5 = 100 at night (0-5), 200 commute (6-8), 50 school (9-15), 150 evening (16-19), 300 late (20-23)
    hours = {**{h: 100.0 for h in range(0, 6)}, **{h: 200.0 for h in range(6, 9)}, **{h: 50.0 for h in range(9, 16)},
             **{h: 150.0 for h in range(16, 20)}, **{h: 300.0 for h in range(20, 24)}}
    s = daypart_stats(hours)
    assert (s["dp_night"], s["dp_commute"], s["dp_school"], s["dp_evening"], s["dp_late"]) == (100, 200, 50, 150, 300)
    assert s["dp_last3"] == 300
    assert abs(s["dp_day"] - sum(hours.values()) / 24) < 1e-9


def test_daypart_stats_need_a_mostly_complete_day():
    import math
    from core.daily import daypart_stats
    s = daypart_stats({h: 80.0 for h in range(0, 10)})          # only 10 hours: below the 16-hour rule
    assert all(math.isnan(v) for v in s.values())
    assert all(math.isnan(v) for v in daypart_stats(None).values())


def test_daypart_features_ignore_tomorrow():
    """The features for target day D+1 read only day D's hours: changing D+1's day-parts must change nothing."""
    from datetime import date
    from core.features import FEATURE_SETS, build_features
    V2 = FEATURE_SETS["v2_35_candidate"]
    d0 = date(2026, 1, 10)
    dp = {"dp_night": 90.0, "dp_commute": 120.0, "dp_school": 60.0, "dp_evening": 110.0, "dp_late": 180.0,
          "dp_last3": 200.0, "dp_day": 100.0}
    hist = {d0 - timedelta(days=i): {"pm25": 100.0 + i, "temp": 8.0, "rh": 70.0, "wind": 4.0, "precip": 0.0,
                                     "pressure": 1010.0, "wind_u": 1.0, "wind_v": -1.0} for i in range(16)}
    hist[d0].update(dp)
    wx1 = {"temp": 8.0, "rh": 90.0, "wind": 3.0, "precip": 0.0, "pressure": 1010.0, "wind_u": 2.0, "wind_v": -2.0}
    a = build_features(hist, d0, wx1, names=V2)
    hist[d0 + timedelta(days=1)] = {"pm25": 999.0, **{k: 999.0 for k in dp}}
    assert build_features(hist, d0, wx1, names=V2) == a
    row = dict(zip(V2, a))
    assert abs(row["pm_buildup"] - 3.0) < 1e-9 and abs(row["pm_last3h_ratio"] - 2.0) < 1e-9


def test_domain_flags():
    from datetime import date
    from core.features import FEATURE_SETS, build_features
    V2 = FEATURE_SETS["v2_35_candidate"]

    def flags(d0, wx1):
        hist = {d0: {"pm25": 100.0, "temp": 10, "rh": 60, "wind": 8, "precip": 0, "pressure": 1000, "wind_u": 0, "wind_v": 0}}
        return dict(zip(V2, build_features(hist, d0, wx1, names=V2)))

    calm_fog = {"temp": 6.0, "rh": 92.0, "wind": 2.0, "precip": 0.0, "pressure": 1015.0, "wind_u": -1.0, "wind_v": 1.0}
    f = flags(date(2025, 12, 19), calm_fog)                         # target Sat 20 Dec 2025
    assert (f["f_inversion"], f["f_weekend"], f["f_stubble"]) == (1.0, 1.0, 0.0)
    assert (f["f_stagnant"], f["f_fog"], f["f_rain"], f["f_wind_nw"]) == (1.0, 1.0, 0.0, 0.0)
    wet_nw = {"temp": 22.0, "rh": 60.0, "wind": 15.0, "precip": 3.0, "pressure": 1000.0, "wind_u": 5.0, "wind_v": -5.0}
    g = flags(date(2025, 10, 19), wet_nw)                           # target 20 Oct 2025 = the day after Diwali
    assert (g["f_stubble"], g["f_diwali"], g["f_rain"], g["f_wind_nw"], g["f_stagnant"], g["f_fog"]) == (1, 1, 1, 1, 0, 0)


# ---------------------------------------------------------------- model versioning (v1 baseline / v2 candidate)
def test_feature_registry_has_the_two_documented_sets():
    from core.features import ALL_FEATURES, DEFAULT_MODEL_VERSION, FEATURE_NAMES, FEATURE_SETS
    assert set(FEATURE_SETS) == {"v1_25_baseline", "v2_35_candidate"}
    assert len(FEATURE_SETS["v1_25_baseline"]) == 25 and len(FEATURE_SETS["v2_35_candidate"]) == 35
    assert all(len(set(v)) == len(v) for v in FEATURE_SETS.values())          # no duplicates
    assert all(f in ALL_FEATURES for v in FEATURE_SETS.values() for f in v)
    assert DEFAULT_MODEL_VERSION == "v1_25_baseline" and FEATURE_NAMES == FEATURE_SETS["v1_25_baseline"]


@pytest.mark.parametrize("version", ["v1_25_baseline", "v2_35_candidate"])
def test_stored_model_meta_matches_feature_registry(version):
    from core.features import FEATURE_SETS
    meta_path = Path(__file__).resolve().parent.parent / "model" / version / "model_meta.json"
    if not meta_path.exists():
        pytest.skip("artifact not present")
    assert json.loads(meta_path.read_text())["features"] == FEATURE_SETS[version]


def test_stored_artifacts_match_checksums():
    """The preserved model files must never change silently (model/CHECKSUMS.sha256 lists them)."""
    import hashlib
    root = Path(__file__).resolve().parent.parent / "model"
    sums = root / "CHECKSUMS.sha256"
    if not sums.exists():
        pytest.skip("no checksum file")
    for line in sums.read_text().splitlines():
        digest, rel = line.split(maxsplit=1)
        assert hashlib.sha256((root / rel.strip()).read_bytes()).hexdigest() == digest, f"{rel} changed"


def test_v1_features_do_not_depend_on_hourly_day_parts():
    from core.features import FEATURE_SETS
    hist, start = _history()
    d0 = start + timedelta(days=30)
    wx1 = {k: 5.0 for k in WEATHER_KEYS}
    plain = build_features(hist, d0, wx1, names=FEATURE_SETS["v1_25_baseline"])
    hist[d0].update({"dp_night": 1.0, "dp_commute": 2.0, "dp_school": 3.0, "dp_evening": 4.0, "dp_late": 5.0,
                     "dp_last3": 6.0, "dp_day": 7.0})
    assert build_features(hist, d0, wx1, names=FEATURE_SETS["v1_25_baseline"]) == plain


def test_v2_day_parts_stay_missing_when_there_is_no_hourly_data():
    """No hourly data -> the 7 day-part features are NaN (XGBoost treats NaN as missing); nothing is invented."""
    from core.features import DAYPART_FEATURES, FEATURE_SETS
    hist, start = _history()
    d0 = start + timedelta(days=30)
    names = FEATURE_SETS["v2_35_candidate"]
    row = dict(zip(names, build_features(hist, d0, {k: 5.0 for k in WEATHER_KEYS}, names=names)))
    assert len(DAYPART_FEATURES) == 7
    assert all(math.isnan(row[f]) for f in DAYPART_FEATURES)
    assert not any(math.isnan(v) for k, v in row.items() if k not in DAYPART_FEATURES and not k.startswith("pm_d")
                   and k not in ("pm_d0", "pm_d1", "pm_d2", "pm_d3"))


def test_shared_features_are_identical_across_versions():
    from core.features import FEATURE_SETS
    hist, start = _history()
    d0 = start + timedelta(days=30)
    wx1 = {k: 5.0 for k in WEATHER_KEYS}
    v1, v2 = FEATURE_SETS["v1_25_baseline"], FEATURE_SETS["v2_35_candidate"]
    a = dict(zip(v1, build_features(hist, d0, wx1, names=v1)))
    b = dict(zip(v2, build_features(hist, d0, wx1, names=v2)))
    for k in set(v1) & set(v2):
        assert a[k] == pytest.approx(b[k], nan_ok=True), k


def test_daypart_cutoff_hour_drops_later_hours():
    from core.daily import daypart_stats
    hours = {h: 100.0 for h in range(0, 23)} | {23: 900.0}
    full, live = daypart_stats(hours), daypart_stats(hours, max_hour=22)
    assert full["dp_late"] > live["dp_late"] == 100.0        # hour 23 (900) is excluded from the live view
    assert live["dp_day"] == pytest.approx(100.0)
    assert all(math.isnan(v) for v in daypart_stats({h: 1.0 for h in range(0, 17)}, max_hour=10).values())


def test_hybrid_rule_is_all_or_nothing_on_the_seven_daypart_inputs():
    from core.features import DAYPART_FEATURES
    from core.hybrid import HYBRID_FALLBACK, HYBRID_WITH_DAYPARTS, daypart_valid, hybrid_pick
    row = {f: 100.0 for f in DAYPART_FEATURES}
    assert daypart_valid(row) and hybrid_pick(row) == HYBRID_WITH_DAYPARTS == "v2_35_candidate"
    for f in DAYPART_FEATURES:                                   # any single gap -> fall back
        bad = {**row, f: float("nan")}
        assert not daypart_valid(bad) and hybrid_pick(bad) == HYBRID_FALLBACK == "v1_25_baseline"
    assert hybrid_pick({}) == HYBRID_FALLBACK and hybrid_pick({**row, "pm_night": None}) == HYBRID_FALLBACK
