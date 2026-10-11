"""Turn raw PM2.5 readings into valid daily means (India Standard Time days).

Shared by the offline dataset builder and the Lambda, so training and
serving use exactly the same rules.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from typing import Iterable

IST = timezone(timedelta(hours=5, minutes=30))
MIN_STATIONS = 3           # a "city" day needs at least this many stations, so the definition stays consistent
MIN_HOURS_PER_DAY = 16     # CPCB needs at least 16 valid hours for a 24h average
MAX_VALID_PM25 = 1000.0    # anything above is treated as a sensor error


def to_ist(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(IST)


def readings_to_daily(readings: Iterable[tuple[datetime, float]],
                      min_hours: int = MIN_HOURS_PER_DAY) -> dict[date, tuple[float, int]]:
    """readings: (timestamp, pm25) pairs for ONE station (any interval).

    Returns {ist_date: (daily_mean, valid_hours)} for days with >= min_hours
    valid hours. Multiple readings in the same hour are averaged first.
    """
    by_hour: dict[datetime, list[float]] = defaultdict(list)
    for ts, val in readings:
        if val is None or val != val or val < 0 or val > MAX_VALID_PM25:
            continue
        local = to_ist(ts)
        by_hour[local.replace(minute=0, second=0, microsecond=0)].append(float(val))

    by_day: dict[date, list[float]] = defaultdict(list)
    for hour, vals in by_hour.items():
        by_day[hour.date()].append(sum(vals) / len(vals))

    out = {}
    for day, hourly in by_day.items():
        if len(hourly) >= min_hours:
            out[day] = (sum(hourly) / len(hourly), len(hourly))
    return out


def city_daily_mean(per_station: Iterable[dict[date, tuple[float, int]]],
                    min_stations: int = 1) -> dict[date, tuple[float, int]]:
    """Average the stations' daily means. Returns {date: (mean, n_stations)}."""
    acc: dict[date, list[float]] = defaultdict(list)
    for station in per_station:
        for day, (mean, _n) in station.items():
            acc[day].append(mean)
    return {d: (sum(v) / len(v), len(v)) for d, v in acc.items() if len(v) >= min_stations}


def city_daily_stats(per_station: Iterable[dict[date, tuple[float, int]]],
                     min_stations: int = 1) -> dict[date, dict]:
    """Like city_daily_mean, plus the spread between stations (spatial information).

    Returns {date: {"pm25": mean, "n": stations, "pm25_sd": sd, "pm25_max": max, "pm25_min": min}}.
    Only same-day values are used, so nothing from the future leaks in."""
    acc: dict[date, list[float]] = defaultdict(list)
    for station in per_station:
        for day, (mean, _n) in station.items():
            acc[day].append(mean)
    out = {}
    for d, v in acc.items():
        if len(v) < min_stations:
            continue
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5 if len(v) > 1 else float("nan")
        out[d] = {"pm25": m, "n": len(v), "pm25_sd": sd, "pm25_max": max(v), "pm25_min": min(v)}
    return out


# ---------------------------------------------------------------------------
# Day-part PM2.5 (Delhi's daily cycle). Hours are IST. Domain meaning:
#   night    00-05  inversion build-up, cold still air, little traffic
#   commute  06-08  morning rush / school run
#   school   09-15  school hours; the sun lifts the mixing layer, so PM2.5 is usually lowest
#   evening  16-19  evening rush hour, boundary layer collapses at sunset
#   late     20-23  night-time build-up; the best clue to tomorrow morning's level
# ---------------------------------------------------------------------------
DAYPARTS = {"night": range(0, 6), "commute": range(6, 9), "school": range(9, 16),
            "evening": range(16, 20), "late": range(20, 24)}
LAST3_HOURS = range(21, 24)
DP_KEYS = ["dp_night", "dp_commute", "dp_school", "dp_evening", "dp_late", "dp_last3", "dp_day"]
_NAN = float("nan")


def readings_to_hourly(readings: Iterable[tuple[datetime, float]]) -> dict[date, dict[int, float]]:
    """ONE station: {ist_date: {hour: mean PM2.5 in that hour}} (same validity rules as readings_to_daily)."""
    by_hour: dict[datetime, list[float]] = defaultdict(list)
    for ts, val in readings:
        if val is None or val != val or val < 0 or val > MAX_VALID_PM25:
            continue
        local = to_ist(ts)
        by_hour[local.replace(minute=0, second=0, microsecond=0)].append(float(val))
    out: dict[date, dict[int, float]] = defaultdict(dict)
    for hour, vals in by_hour.items():
        out[hour.date()][hour.hour] = sum(vals) / len(vals)
    return dict(out)


def city_hourly(per_station: Iterable[dict[date, dict[int, float]]]) -> dict[date, dict[int, float]]:
    """Average the stations that reported each hour: {ist_date: {hour: city mean}}."""
    acc: dict[date, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for station in per_station:
        for day, hours in station.items():
            for h, v in hours.items():
                acc[day][h].append(v)
    return {d: {h: sum(v) / len(v) for h, v in hs.items()} for d, hs in acc.items()}


def daypart_stats(hours: dict[int, float] | None, max_hour: int | None = None) -> dict[str, float]:
    """Day-part means from one day's city hourly series. NaN when the day has < 16 hours or a part is too sparse.
    Uses only that day's own hours, so nothing from the future leaks in.

    max_hour: ignore hours after this one. The nightly Lambda runs at 23:00 IST, so for the day it forecasts from only
    hours 0-22 exist yet; max_hour=22 reproduces exactly what is visible then (used by the evaluation to measure the
    train/serve gap). None (default) keeps the full-day definition the stored models were trained with."""
    out = {k: _NAN for k in DP_KEYS}
    if hours and max_hour is not None:
        hours = {h: v for h, v in hours.items() if h <= max_hour}
    if not hours or len(hours) < MIN_HOURS_PER_DAY:
        return out

    def part(rng):
        v = [hours[h] for h in rng if h in hours]
        return sum(v) / len(v) if len(v) >= max(2, len(rng) // 2) else _NAN

    for name, rng in DAYPARTS.items():
        out[f"dp_{name}"] = part(rng)
    out["dp_last3"] = part(LAST3_HOURS)
    out["dp_day"] = sum(hours.values()) / len(hours)
    return out
