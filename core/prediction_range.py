"""Calibrated 90 % prediction range around the point forecast (method "M1": split-conformal in log1p space).

This is DERIVED data, a pure function of two things only:
    * the point forecast (PM2.5 in ug/m3) that the served model produced, and
    * the packaged calibration file core/range_calibration.json (built from past out-of-sample forecast errors by
      experiments/prediction_range/build_calibration.py).
It never changes the point forecast, the AQI category, the school decision or the alert, and it never uses today's
actual value or any future information. If there is no calibration for a model version, or the input is not a
usable number, it returns None - a range is never invented.

How the range is made (all in log1p space, where the model's errors are roughly symmetric):
    residual = log1p(actual PM2.5) - log1p(point forecast)          # from past out-of-sample forecasts
    lo_offset = 5 % quantile of the residuals                       # alpha / 2
    hi_offset = 95 % quantile with the split-conformal finite-sample correction  q(level * (1 + 1/n))
    range     = expm1(log1p(point) + offset), then clamped so the point forecast is always inside and lo >= 0
The bounds are shown as whole numbers rounded OUTWARD (floor of the low end, ceil of the high end), so the shown range is
never narrower than the computed one and every number on screen matches its category.

What the range means: in retrospective tests the actual PM2.5 fell inside such a range on about 9 days in 10 (see
experiments/prediction_range/report.md). It is a statement about past errors, not a guarantee, and it is not a measure
of the point forecast's accuracy (that is unchanged).
"""
from __future__ import annotations

import json
import logging
import math
import os

from core.aqi import CATEGORIES, pm25_to_category

log = logging.getLogger(__name__)

CALIBRATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "range_calibration.json")
METHOD = "M1_split_conformal_log1p"
LEVEL = 0.90
SCHEMA = 1

_CAL: dict = {}        # path -> parsed calibration (loaded once per process)


# ------------------------------------------------------------------ calibration maths (also used by the builder and tests)
def quantile(values, q: float) -> float:
    """q-quantile with linear interpolation between order statistics (identical to numpy.quantile's default)."""
    xs = sorted(float(v) for v in values)
    if not xs:
        raise ValueError("no values")
    q = min(max(q, 0.0), 1.0)
    pos = q * (len(xs) - 1)
    i = int(math.floor(pos))
    j = min(i + 1, len(xs) - 1)
    return xs[i] + (xs[j] - xs[i]) * (pos - i)


def residuals(actual, forecast) -> list[float]:
    """log1p(actual) - log1p(forecast) for each past day (negatives are clipped to 0 like the training code does)."""
    return [math.log1p(max(float(a), 0.0)) - math.log1p(max(float(f), 0.0)) for a, f in zip(actual, forecast)]


def conformal_offsets(res, level: float = LEVEL) -> tuple[float, float]:
    """(low, high) offsets in log1p space for a central `level` range, exactly as the validated experiment (M1) did:
    the low side is the plain alpha/2 quantile, the high side is the (1 - alpha/2) quantile with the (1 + 1/n) correction."""
    n = len(res)
    if n < 2:
        raise ValueError("need at least 2 residuals")
    alpha = 1.0 - level
    lo = quantile(res, alpha / 2)
    hi = quantile(res, min(1.0, (1 - alpha / 2) * (1 + 1 / n)))
    return lo, hi


def interval(point: float, lo_off: float, hi_off: float) -> tuple[float, float]:
    """Unrounded PM2.5 range for `point`. The point forecast is always inside it and the low end is never negative."""
    p = max(float(point), 0.0)
    base = math.log1p(p)
    lo = math.expm1(base + lo_off)
    hi = math.expm1(base + hi_off)
    return min(max(lo, 0.0), p), max(hi, p)


# ------------------------------------------------------------------ calibration file
def load_calibration(path: str | None = None) -> dict:
    path = path or CALIBRATION_FILE
    if path not in _CAL:
        with open(path) as f:
            cal = json.load(f)
        if cal.get("schema") != SCHEMA or cal.get("method") != METHOD or not isinstance(cal.get("versions"), dict):
            raise ValueError(f"{path}: not a v{SCHEMA} {METHOD} calibration file")
        if abs(cal.get("level", 0) - LEVEL) > 1e-9:
            raise ValueError(f"{path}: level {cal.get('level')} is not {LEVEL}")
        _CAL[path] = cal
    return _CAL[path]


def prediction_range(pm25, model_version, calibration: dict | None = None) -> dict | None:
    """The 90 % range for a point forecast made by `model_version`, or None when it cannot be given honestly
    (unknown/missing version, no calibration for it, or `pm25` is not a finite non-negative number)."""
    try:
        p = float(pm25)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(p) or p < 0:
        return None
    cal = calibration if calibration is not None else load_calibration()
    entry = cal["versions"].get(model_version) if isinstance(model_version, str) else None
    if not entry:
        return None
    lo, hi = interval(p, entry["lo"], entry["hi"])
    low, high = int(math.floor(lo)), int(math.ceil(hi))
    cat_low, cat_high = pm25_to_category(low), pm25_to_category(high)
    i, j = CATEGORIES.index(cat_low), CATEGORIES.index(cat_high)
    return {
        "level": cal["level"],
        "method": cal["method"],
        "pm25_low": low,
        "pm25_high": high,
        "category_low": cat_low,
        "category_high": cat_high,
        "categories": CATEGORIES[i:j + 1],
        "basis": "retrospective",
        "calibration_id": cal["calibration_id"],
        "calibration_days": entry["n"],
        "calibration_period": [entry["first"], entry["last"]],
    }


def attach_prediction_range(forecast: dict) -> dict:
    """Return a copy of a stored forecast record with `prediction_range` added (key omitted when it cannot be given).
    `forecast` needs `pm25` and `model_version`. Never raises: the range is extra information and must never be able to
    break the forecast API."""
    out = dict(forecast)
    try:
        rng = prediction_range(forecast.get("pm25"), forecast.get("model_version"))
    except Exception:                                   # pragma: no cover - defensive
        log.exception("prediction range unavailable")
        rng = None
    if rng is not None:
        out["prediction_range"] = rng
    return out
