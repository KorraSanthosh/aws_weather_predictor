"""India (CPCB) AQI from 24-hour average PM2.5, plus school guidance.

CPCB PM2.5 breakpoints (ug/m3, 24-hour average) -> AQI sub-index:
    0-30 -> 0-50 Good | 31-60 -> 51-100 Satisfactory | 61-90 -> 101-200 Moderate
    91-120 -> 201-300 Poor | 121-250 -> 301-400 Very Poor | 251-380 -> 401-500 Severe
"""
from __future__ import annotations

# (conc_lo, conc_hi, aqi_lo, aqi_hi)
_BANDS = [
    (0.0, 30.0, 0, 50),
    (31.0, 60.0, 51, 100),
    (61.0, 90.0, 101, 200),
    (91.0, 120.0, 201, 300),
    (121.0, 250.0, 301, 400),
    (251.0, 380.0, 401, 500),
]

CATEGORIES = ["Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe"]
CATEGORY_COLORS = {
    "Good": "#009865",
    "Satisfactory": "#7FBF3F",
    "Moderate": "#E8C100",
    "Poor": "#F28C28",
    "Very Poor": "#D7263D",
    "Severe": "#7E0023",
}

# Index of the first category that triggers an alert ("Poor")
ALERT_FROM = CATEGORIES.index("Poor")
# PM2.5 value above which the AQI category is Poor or worse (AQI > 200)
POOR_PM25_THRESHOLD = 90.0


def pm25_to_aqi(pm25: float) -> int:
    """Convert a 24h-average PM2.5 (ug/m3) to a CPCB AQI value (0-500)."""
    if pm25 is None or pm25 != pm25:  # None or NaN
        raise ValueError("pm25 must be a number")
    c = max(0.0, float(pm25))
    if c >= _BANDS[-1][1]:
        return 500
    for lo, hi, ilo, ihi in _BANDS:
        if c <= hi:
            c = max(c, lo)  # CPCB bands have a 1-unit gap (e.g. 30 -> 31)
            if hi == lo:
                return ihi
            return int(round((ihi - ilo) / (hi - lo) * (c - lo) + ilo))
    return 500  # unreachable


def aqi_to_category(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    if aqi <= 100:
        return "Satisfactory"
    if aqi <= 200:
        return "Moderate"
    if aqi <= 300:
        return "Poor"
    if aqi <= 400:
        return "Very Poor"
    return "Severe"


def pm25_to_category(pm25: float) -> str:
    return aqi_to_category(pm25_to_aqi(pm25))


def is_alert(category: str) -> bool:
    return CATEGORIES.index(category) >= ALERT_FROM


def school_guidance(category: str) -> dict:
    """Simple go / caution / indoor decision for school outdoor activities.

    General guidance only, not medical advice. Closure decisions rest with
    school management and the local authority.
    """
    if category in ("Good", "Satisfactory"):
        return {
            "decision": "GO",
            "headline": "Outdoor activities are fine",
            "advice": "Normal assembly, sports and recess can go ahead.",
        }
    if category == "Moderate":
        return {
            "decision": "CAUTION",
            "headline": "Keep outdoor time short",
            "advice": (
                "Shorten outdoor sports and avoid strenuous exercise. "
                "Students with asthma or breathing problems should take it easy."
            ),
        }
    if category == "Poor":
        return {
            "decision": "INDOOR",
            "headline": "Move activities indoors",
            "advice": (
                "Hold PE, assembly and recess indoors. "
                "Limit time outside, especially for younger children and "
                "students with breathing problems."
            ),
        }
    return {
        "decision": "INDOOR",
        "headline": "No outdoor activities",
        "advice": (
            "Keep all activities indoors with windows closed where possible. "
            "Follow advisories from school management and the local authority "
            "on any closure."
        ),
    }
