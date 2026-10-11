"""Feature construction for 'tomorrow's PM2.5' - pure Python (no pandas).

Used identically by training, by the experiments and by the Lambda, so there is
no train/serve skew.

Given data up to day D (today) and the weather expected on D+1 (tomorrow),
produce one feature row. Nothing after day D is ever read, except tomorrow's
weather, which in live use comes from a weather FORECAST (in training it is the
observed weather of that day - see README "weather forecast caveat").

`ALL_FEATURES` is every candidate that was computed; `FEATURE_SETS` holds the two model versions
(see model/MODEL_VERSIONS.md); `FEATURE_NAMES` is the default (baseline) version's list. Day-part
features (v2 only) are NaN on days without hourly data - they are never filled in.
"""
from __future__ import annotations

import math
from datetime import date, timedelta

from .weather import WEATHER_KEYS
from .daily import DP_KEYS

NAN = float("nan")

# ---------- candidate feature groups (used for the ablation study) ----------
BASE_PM = ["pm_d0", "pm_d1", "pm_d2", "pm_d3", "pm_d6", "pm_mean3", "pm_mean7", "pm_std7", "pm_delta1"]
BASE_WX = [f"wx0_{k}" for k in WEATHER_KEYS] + [f"wx1_{k}" for k in WEATHER_KEYS]
BASE_SEASON = ["doy_sin", "doy_cos"]
BASE = BASE_PM + BASE_WX + BASE_SEASON                       # the original (v1) model

DAYPART_FEATURES = ["pm_night", "pm_commute", "pm_school", "pm_evening", "pm_late", "pm_buildup", "pm_last3h_ratio"]
FLAG_FEATURES = ["f_stubble", "f_diwali", "f_inversion", "f_weekend", "f_rain", "f_stagnant", "f_fog", "f_wind_nw"]

GROUPS = {
    "lags": ["pm_d7", "pm_d13"],
    "rolling": ["pm_mean14", "pm_min7", "pm_max7", "pm_delta3", "pm_delta7", "pm_slope7",
                "pm_logratio_mean7", "pm_logratio_mean14", "pm_logdelta1"],
    "weather": ["wx_dtemp", "wx_drh", "wx_dpressure", "wx_dwind", "wx0_rain", "wx1_rain",
                "wx1_dir_u", "wx1_dir_v", "wx1_stagnant"],
    "seasonal": ["month", "dow", "weekend", "season", "winter", "diwali_days"],
    "spatial": ["n_stations", "pm_sd0", "pm_spread0", "pm_maxratio0"],
    "dayparts": DAYPART_FEATURES,
    "flags": FLAG_FEATURES,
}
ALL_FEATURES = BASE + [f for g in GROUPS.values() for f in g]

# ---------- model versions: each is a fixed, named feature list (never edited after it is trained) ----------
# v1_25_baseline  : the original 25 features - the BASELINE and the default deployed model.
# v2_35_candidate : v1-style PM2.5 history + weather + season, plus 7 day-part PM2.5 features and 8 binary domain
#                   flags - the EXPERIMENTAL CANDIDATE (not default until a forward-looking test supports it).
FEATURES_V1 = list(BASE)
FEATURES_V2 = (
    ["pm_d0", "pm_d1", "pm_d2", "pm_d3", "pm_mean3", "pm_mean7", "pm_std7", "pm_delta1"]          # 8 PM2.5 history
    + [f"wx1_{k}" for k in ("temp", "rh", "wind", "precip", "pressure", "wind_u", "wind_v")]       # 7 tomorrow's weather
    + ["wx0_wind", "wx0_rh", "wx0_precip"]                                                        # 3 today's weather
    + BASE_SEASON                                                                                 # 2 day-of-year cycle
    + DAYPART_FEATURES                                                                            # 7 day-part PM2.5
    + FLAG_FEATURES                                                                               # 8 binary domain flags
)
assert len(FEATURES_V1) == 25 and len(FEATURES_V2) == 35
FEATURE_SETS = {"v1_25_baseline": FEATURES_V1, "v2_35_candidate": FEATURES_V2}
DEFAULT_MODEL_VERSION = "v1_25_baseline"
FEATURE_NAMES = FEATURE_SETS[DEFAULT_MODEL_VERSION]      # what the default (deployed) model uses

# Diwali (firecracker night) - a public calendar date, known years in advance.
DIWALI = [date(2016, 10, 30), date(2017, 10, 19), date(2018, 11, 7), date(2019, 10, 27),
          date(2020, 11, 14), date(2021, 11, 4), date(2022, 10, 24), date(2023, 11, 12),
          date(2024, 10, 31), date(2025, 10, 20), date(2026, 11, 8), date(2027, 10, 29),
          date(2028, 10, 17)]


def _num(v):
    return NAN if v is None or v != v else float(v)


def _mean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else NAN


def _std(vals):
    vals = [v for v in vals if v == v]
    if len(vals) < 2:
        return NAN
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / (len(vals) - 1))


def _slope(vals):
    """OLS slope of log(PM) over the last days (vals[0] = today, vals[i] = i days ago), per day."""
    pts = [(-i, math.log1p(v)) for i, v in enumerate(vals) if v == v]
    if len(pts) < 4:
        return NAN
    mx = sum(p[0] for p in pts) / len(pts)
    my = sum(p[1] for p in pts) / len(pts)
    sxx = sum((p[0] - mx) ** 2 for p in pts)
    return sum((p[0] - mx) * (p[1] - my) for p in pts) / sxx if sxx else NAN


def _season(month: int) -> int:
    # India Meteorological Department seasons
    if month in (12, 1, 2):
        return 0          # winter
    if month in (3, 4, 5):
        return 1          # summer / pre-monsoon
    if month in (6, 7, 8, 9):
        return 2          # monsoon
    return 3              # post-monsoon (Oct, Nov)


def feature_dict(history: dict[date, dict], d0: date, wx1: dict) -> dict[str, float]:
    """All candidate features for target day d0 + 1. history: {date: record}, only days <= d0 are read.
    A record has 'pm25' and may have the weather keys and 'n', 'pm25_sd', 'pm25_max', 'pm25_min'."""
    def lag(n):
        rec = history.get(d0 - timedelta(days=n))
        return _num(rec.get("pm25")) if rec else NAN

    pm = [lag(n) for n in range(0, 15)]
    ok = lambda *xs: all(x == x for x in xs)  # noqa: E731
    row = {
        "pm_d0": pm[0], "pm_d1": pm[1], "pm_d2": pm[2], "pm_d3": pm[3], "pm_d6": pm[6],
        "pm_d7": pm[7], "pm_d13": pm[13],
        "pm_mean3": _mean(pm[:3]), "pm_mean7": _mean(pm[:7]), "pm_mean14": _mean(pm[:14]),
        "pm_std7": _std(pm[:7]),
        "pm_min7": min([v for v in pm[:7] if v == v], default=NAN),
        "pm_max7": max([v for v in pm[:7] if v == v], default=NAN),
        "pm_delta1": pm[0] - pm[1] if ok(pm[0], pm[1]) else NAN,
        "pm_delta3": pm[0] - pm[3] if ok(pm[0], pm[3]) else NAN,
        "pm_delta7": pm[0] - pm[7] if ok(pm[0], pm[7]) else NAN,
        "pm_logdelta1": math.log1p(pm[0]) - math.log1p(pm[1]) if ok(pm[0], pm[1]) else NAN,
        "pm_slope7": _slope(pm[:7]),
    }
    m7, m14 = row["pm_mean7"], row["pm_mean14"]
    row["pm_logratio_mean7"] = math.log1p(pm[0]) - math.log1p(m7) if ok(pm[0], m7) else NAN
    row["pm_logratio_mean14"] = math.log1p(pm[0]) - math.log1p(m14) if ok(pm[0], m14) else NAN

    today = history.get(d0, {})
    for k in WEATHER_KEYS:
        row[f"wx0_{k}"] = _num(today.get(k))
        row[f"wx1_{k}"] = _num(wx1.get(k))
    diff = lambda k: row[f"wx1_{k}"] - row[f"wx0_{k}"] if ok(row[f"wx1_{k}"], row[f"wx0_{k}"]) else NAN  # noqa: E731
    row["wx_dtemp"], row["wx_drh"] = diff("temp"), diff("rh")
    row["wx_dpressure"], row["wx_dwind"] = diff("pressure"), diff("wind")
    row["wx0_rain"] = float(row["wx0_precip"] >= 1.0) if ok(row["wx0_precip"]) else NAN
    row["wx1_rain"] = float(row["wx1_precip"] >= 1.0) if ok(row["wx1_precip"]) else NAN
    u, v, w = row["wx1_wind_u"], row["wx1_wind_v"], row["wx1_wind"]
    sp = math.hypot(u, v) if ok(u, v) else NAN
    row["wx1_dir_u"] = u / sp if ok(sp) and sp > 0.1 else NAN       # wind direction, independent of speed
    row["wx1_dir_v"] = v / sp if ok(sp) and sp > 0.1 else NAN
    row["wx1_stagnant"] = float(w < 5.0 and row["wx1_precip"] < 0.5) if ok(w, row["wx1_precip"]) else NAN

    target = d0 + timedelta(days=1)
    doy = target.timetuple().tm_yday
    row["doy_sin"] = math.sin(2 * math.pi * doy / 365.25)
    row["doy_cos"] = math.cos(2 * math.pi * doy / 365.25)
    row["month"] = float(target.month)
    row["dow"] = float(target.weekday())
    row["weekend"] = float(target.weekday() >= 5)
    row["season"] = float(_season(target.month))
    row["winter"] = float((target.month, target.day) >= (10, 15) or (target.month, target.day) <= (2, 15))
    nearest = min(DIWALI, key=lambda d: abs((target - d).days))
    gap = (target - nearest).days
    row["diwali_days"] = float(gap) if abs(gap) <= 10 else 11.0

    row["n_stations"] = _num(today.get("n"))
    sd, mx, mn = _num(today.get("pm25_sd")), _num(today.get("pm25_max")), _num(today.get("pm25_min"))
    row["pm_sd0"] = sd / pm[0] if ok(sd, pm[0]) and pm[0] > 0 else NAN          # relative spread
    row["pm_spread0"] = (mx - mn) / pm[0] if ok(mx, mn, pm[0]) and pm[0] > 0 else NAN
    row["pm_maxratio0"] = mx / pm[0] if ok(mx, pm[0]) and pm[0] > 0 else NAN

    # ---- day-part PM2.5 from today's hourly data (NaN when only daily files exist, e.g. CPCB years) ----
    dp = {k: _num(today.get(k)) for k in DP_KEYS}
    row["pm_night"], row["pm_commute"], row["pm_school"] = dp["dp_night"], dp["dp_commute"], dp["dp_school"]
    row["pm_evening"], row["pm_late"] = dp["dp_evening"], dp["dp_late"]
    row["pm_buildup"] = dp["dp_late"] / dp["dp_school"] if ok(dp["dp_late"], dp["dp_school"]) and dp["dp_school"] > 0 else NAN
    row["pm_last3h_ratio"] = dp["dp_last3"] / dp["dp_day"] if ok(dp["dp_last3"], dp["dp_day"]) and dp["dp_day"] > 0 else NAN

    # ---- binary domain flags about the TARGET day (calendar flags are known in advance) ----
    md = (target.month, target.day)
    row["f_weekend"] = float(target.weekday() >= 5)               # traffic and industry differ on weekends
    row["f_stubble"] = float((10, 15) <= md <= (11, 30))          # crop-residue burning season upwind
    row["f_diwali"] = float(-1 <= gap <= 4)                       # firecrackers and the smoke that lingers after
    row["f_inversion"] = float(target.month in (12, 1))           # coldest months: shallow inversion layer
    t1, rh1, w1, p1 = row["wx1_temp"], row["wx1_rh"], row["wx1_wind"], row["wx1_precip"]
    row["f_rain"] = float(p1 >= 1.0) if ok(p1) else NAN           # rain washes particles out
    row["f_stagnant"] = float(w1 < 5.0 and p1 < 0.5) if ok(w1, p1) else NAN
    row["f_fog"] = float(rh1 > 85 and t1 < 12) if ok(rh1, t1) else NAN   # cold humid air: fog / smog traps pollution
    row["f_wind_nw"] = float(u > 0.5 and v < -0.5) if ok(u, v) else NAN  # wind FROM the north-west, the smoke direction
    return row


def build_features(history: dict[date, dict], d0: date, wx1: dict, names: list[str] | None = None) -> list[float]:
    """Feature row aligned with `names` (default: the deployed FEATURE_NAMES)."""
    row = feature_dict(history, d0, wx1)
    return [row[n] for n in (names or FEATURE_NAMES)]
