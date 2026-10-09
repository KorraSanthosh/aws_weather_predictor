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
