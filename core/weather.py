"""Open-Meteo weather helpers (free, no API key for non-commercial use).

Hourly data is requested in India time and reduced to daily values here, so
training (archive API) and serving (forecast API) share one code path.
"""
from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from collections import defaultdict
from datetime import date

HOURLY_VARS = [
    "temperature_2m", "relative_humidity_2m", "wind_speed_10m",
    "wind_direction_10m", "precipitation", "surface_pressure",
]
WEATHER_KEYS = ["temp", "rh", "wind", "precip", "pressure", "wind_u", "wind_v"]

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

MIN_HOURS = 12


def http_get_json(url: str, params: dict, retries: int = 3, timeout: int = 30) -> dict:
    full = f"{url}?{urllib.parse.urlencode(params)}"
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(full, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:  # network / 5xx / 429
            last = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed after {retries} tries: {last}")


def hourly_to_daily_weather(hourly: dict) -> dict[date, dict]:
    """hourly: Open-Meteo 'hourly' block (local-time ISO strings)."""
    times = hourly["time"]
    rows: dict[date, list[tuple]] = defaultdict(list)
    for i, t in enumerate(times):
        vals = {k: hourly.get(k, [None] * len(times))[i] for k in HOURLY_VARS}
        if any(v is None for v in vals.values()):
            continue
        rows[date.fromisoformat(t[:10])].append(vals)

    out = {}
    for day, hrs in rows.items():
        if len(hrs) < MIN_HOURS:
            continue
        n = len(hrs)
        u = v = 0.0
        for h in hrs:
            rad = math.radians(h["wind_direction_10m"])
            u += -h["wind_speed_10m"] * math.sin(rad)   # meteorological: direction wind comes FROM
            v += -h["wind_speed_10m"] * math.cos(rad)
        out[day] = {
            "temp": sum(h["temperature_2m"] for h in hrs) / n,
            "rh": sum(h["relative_humidity_2m"] for h in hrs) / n,
            "wind": sum(h["wind_speed_10m"] for h in hrs) / n,
            "precip": sum(h["precipitation"] for h in hrs),
            "pressure": sum(h["surface_pressure"] for h in hrs) / n,
            "wind_u": u / n,
            "wind_v": v / n,
        }
    return out


def fetch_archive_weather(lat: float, lon: float, start: date, end: date) -> dict[date, dict]:
    data = http_get_json(ARCHIVE_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": ",".join(HOURLY_VARS), "timezone": "Asia/Kolkata",
    })
    return hourly_to_daily_weather(data["hourly"])


def fetch_forecast_weather(lat: float, lon: float, past_days: int = 10,
                           forecast_days: int = 3) -> dict[date, dict]:
    """Recent past (analysis) + upcoming days from the forecast API."""
    data = http_get_json(FORECAST_URL, {
        "latitude": lat, "longitude": lon,
        "past_days": past_days, "forecast_days": forecast_days,
        "hourly": ",".join(HOURLY_VARS), "timezone": "Asia/Kolkata",
    })
    return hourly_to_daily_weather(data["hourly"])


def fetch_cams_daily_pm25(lat: float, lon: float, start: date, end: date) -> dict[date, float]:
    """Open-Meteo's CAMS global model PM2.5, daily mean (benchmark only)."""
    data = http_get_json(AIR_QUALITY_URL, {
        "latitude": lat, "longitude": lon,
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "hourly": "pm2_5", "timezone": "Asia/Kolkata",
    })
    h = data["hourly"]
    acc: dict[date, list[float]] = defaultdict(list)
    for t, v in zip(h["time"], h["pm2_5"]):
        if v is not None:
            acc[date.fromisoformat(t[:10])].append(v)
    return {d: sum(v) / len(v) for d, v in acc.items() if len(v) >= MIN_HOURS}
