#!/usr/bin/env python3
"""Read-only check: can the live OpenAQ hourly feed supply the seven day-part inputs at the 23:00 IST run?

    export OPENAQ_API_KEY=...            # in YOUR shell; this script reads only that variable and never prints it
    python scripts/check_hourly_feed.py [--sensors 1,2,3] [--days 8] [--out feed_check.json]
    python scripts/check_hourly_feed.py --selftest      # offline, synthetic data, proves the analysis itself works

It only issues GET requests to api.openaq.org (the same call the Lambda makes). It writes nothing to AWS or to the project.

The question it answers. The Hybrid (core/hybrid.py) uses V2 only when ALL seven day-part inputs are finite. At the 23:00 IST run
hour 23 does not exist, so `dp_late` needs two of hours 20-22 and `dp_last3` needs BOTH hours 21 and 22 (core/daily.py:
max(2, len//2)). So the Hybrid has its inputs at 23:00 only if the feed already holds the hour that ended at 23:00 (period
22:00-23:00) - i.e. practically zero publication lag. The models were validated offline on that assumption; the live lag is
what this script measures. Run it close to 23:00 IST (22:58-23:10) for the most direct answer; the nightly provenance record
(pk="provenance") then measures the same thing every night automatically.

It does not fabricate anything: a missing hour stays missing.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core import openaq                                                    # noqa: E402
from core.daily import IST, city_hourly, daypart_stats, readings_to_hourly, to_ist     # noqa: E402
from core.features import feature_dict                                     # noqa: E402
from core.forward_provenance import HOUR, aware_utc, valid_value          # noqa: E402
from core.hybrid import daypart_valid                                      # noqa: E402

TOLERANCE_HOURS = (22, 21, 20, 19)          # newest hour the feed holds when the run starts (22 = no lag at all)


def dp_valid(city_hours: dict, max_hour: int | None) -> bool:
    """Would all seven day-part inputs exist? Uses the production code (daypart_stats -> feature_dict -> daypart_valid)."""
    st = daypart_stats(city_hours, max_hour=max_hour)
    d = date(2000, 1, 1)
    row = feature_dict({d: {"pm25": st["dp_day"], **st}}, d, {})
    return daypart_valid(row)


RUN_OFFSET_MINUTES = 0.5     # the Lambda starts a few seconds after the cron time and its first API call follows at once


def analyse(feed: dict, now_utc: datetime, n_days: int = 8, run_offset_minutes: float = RUN_OFFSET_MINUTES) -> dict:
    """feed: {sensor_id: [(timestamp, pm25), ...]} exactly as core.openaq.fetch_sensor_hours returns it."""
    now_utc = aware_utc(now_utc)
    sensors, finished = [], {}
    for sid, rows in feed.items():
        valid = [(aware_utc(t), v) for t, v in rows if valid_value(v)]
        done = [(t, v) for t, v in valid if t + HOUR <= now_utc]
        finished[sid] = done
        latest = max((t for t, _ in done), default=None)
        sensors.append({
            "sensor_id": sid, "n_rows": len(rows), "n_valid": len(valid),
            "rows_hour_not_finished": sum(1 for t, _ in valid if t + HOUR > now_utc),
            "rows_started_after_now": sum(1 for t, _ in valid if t > now_utc),
            "stamp_minutes_utc": sorted({t.minute for t, _ in valid}),
            "latest_finished_hour_end_ist": to_ist(latest + HOUR).isoformat() if latest else None,
            "lag_minutes": round((now_utc - (latest + HOUR)).total_seconds() / 60, 1) if latest else None,
        })
    lags = [s["lag_minutes"] for s in sensors if s["lag_minutes"] is not None]
    freshest = min(lags) if lags else None
    # The newest IST hour a run started at 23:00 would hold if every hour reaches the feed `freshest` minutes after it ends
    # (the run itself starts run_offset_minutes after 23:00, so that much delay is harmless). 22 = the hour that ends at 23:00.
    last_hour_at_2300 = None if freshest is None else min(22, math.floor(22 - max(0.0, freshest - run_offset_minutes) / 60))

    by_sensor_hourly = {sid: readings_to_hourly(done) for sid, done in finished.items()}
    today = now_utc.astimezone(IST).date()
    days = []
    for i in range(n_days, 0, -1):
        d = today - timedelta(days=i)                       # complete days only
        per_sensor = [h[d] for h in by_sensor_hourly.values() if d in h]
        city = city_hourly([{d: h} for h in per_sensor]).get(d, {})
        days.append({"date": d.isoformat(), "stations": len(per_sensor), "city_hours": len(city),
                     "missing_hours": [h for h in range(24) if h not in city],
                     "all_7_inputs_full_day": dp_valid(city, None) if city else False,
                     "all_7_inputs_if_newest_hour_is": {str(m): (dp_valid(city, m) if city else False) for m in TOLERANCE_HOURS}})
    stats = {str(m): sum(d["all_7_inputs_if_newest_hour_is"][str(m)] for d in days) for m in TOLERANCE_HOURS}
    # today so far: stations per hour for the newest hours
    today_city = {}
    for sid, h in by_sensor_hourly.items():
        for hh, v in h.get(today, {}).items():
            today_city.setdefault(hh, []).append(v)
    newest = sorted(today_city)[-3:]
    return {
        "now_ist": to_ist(now_utc).isoformat(),
        "n_sensors": len(feed),
        "n_sensors_with_data": sum(1 for s in sensors if s["n_valid"]),
        "sensors": sensors,
        "future_observations": {"rows_hour_not_finished": sum(s["rows_hour_not_finished"] for s in sensors),
                                "rows_started_after_now": sum(s["rows_started_after_now"] for s in sensors),
                                "meaning": "rows the API returned whose hour had not ended yet; the forward logger drops them"},
        "stamp_minutes_utc_seen": sorted({m for s in sensors for m in s["stamp_minutes_utc"]}),
        "freshest_sensor_lag_minutes": freshest,
        "median_sensor_lag_minutes": round(sorted(lags)[len(lags) // 2], 1) if lags else None,
        "expected_newest_hour_at_2300_ist_with_this_lag": last_hour_at_2300,
        "hybrid_inputs_expected_at_2300_ist_with_this_lag": (last_hour_at_2300 is not None and last_hour_at_2300 >= 22),
        "today_newest_hours_stations": {str(h): len(today_city[h]) for h in newest},
        "days": days,
        "days_checked": len(days),
        "days_with_all_7_inputs_full_day": sum(d["all_7_inputs_full_day"] for d in days),
        "days_with_all_7_inputs_by_newest_hour": stats,
    }


def render(rep: dict) -> str:
    L = [f"Hourly feed check at {rep['now_ist']}   sensors with data: {rep['n_sensors_with_data']} of {rep['n_sensors']}",
         f"  hour stamps (minute, UTC)      : {rep['stamp_minutes_utc_seen']}   ([30] = hours start on the IST hour, as expected)",
         f"  future/unfinished rows returned: {rep['future_observations']['rows_hour_not_finished']} "
         f"(started after now: {rep['future_observations']['rows_started_after_now']})  <- 0 means the API honoured datetime_to",
         f"  feed lag now (min)             : freshest sensor {rep['freshest_sensor_lag_minutes']}, median {rep['median_sensor_lag_minutes']}",
         f"  newest hours today, stations   : {rep['today_newest_hours_stations']}"]
    n = rep["days_checked"]
    L.append(f"  last {n} complete days: all 7 day-part inputs exist on {rep['days_with_all_7_inputs_full_day']} of {n} (full 24 h of data)")
    for m in TOLERANCE_HOURS:
        L.append(f"    if the newest hour in the feed at the run is {m:2d}:00-{m + 1:02d}:00 -> inputs complete on "
                 f"{rep['days_with_all_7_inputs_by_newest_hour'][str(m)]} of {n} days")
    h = rep["expected_newest_hour_at_2300_ist_with_this_lag"]
    verdict = ("the Hybrid WOULD have its day-part inputs at 23:00" if rep["hybrid_inputs_expected_at_2300_ist_with_this_lag"]
               else "the Hybrid would NOT have its day-part inputs at 23:00 and would fall back to V1")
    L.append(f"  with a lag of {rep['freshest_sensor_lag_minutes']} min the newest hour at 23:00 would be {h}: {verdict}")
    L.append("  (one snapshot of a varying delay - run near 23:00 IST on several nights, or read the nightly provenance records)")
    return "\n".join(L)


# ----------------------------------------------------------------------------- live fetch and self test
def sensor_ids(arg: str | None) -> list[int]:
    if arg:
        return [int(x) for x in arg.split(",") if x.strip()]
    if os.environ.get("SENSOR_IDS"):
        return [int(x) for x in os.environ["SENSOR_IDS"].split(",") if x.strip()]
    return [s["pm25_sensor_id"] for s in json.loads((ROOT / "config" / "stations.json").read_text())]


def fetch_live(ids: list[int], now_utc: datetime, n_days: int, api_key: str) -> dict:
    feed = {}
    for sid in ids:
        feed[sid] = openaq.fetch_sensor_hours(sid, now_utc - timedelta(days=n_days + 1), now_utc, api_key)
        time.sleep(0.4)                                      # stay well inside the 60 requests/minute limit
    return feed


def synthetic_feed(now_utc: datetime, lag_hours: int, n_sensors: int = 6, days: int = 10) -> dict:
    """Hour-aligned (:30Z) readings; the newest finished hour is `lag_hours` older than the newest possible one."""
    feed = {}
    end = aware_utc(now_utc).astimezone(IST).replace(minute=0, second=0, microsecond=0) - timedelta(hours=lag_hours)
    for sid in range(1, n_sensors + 1):
        rows, t = [], end - timedelta(hours=1) - timedelta(days=days)
        while t + HOUR <= end:
            rows.append((t.astimezone(timezone.utc), 60.0 + sid + 3 * abs(12 - t.hour)))
            t += HOUR
        feed[sid] = rows
    return feed


def selftest() -> int:
    now = datetime(2026, 10, 14, 17, 30, 5, tzinfo=timezone.utc)             # 23:00:05 IST
    ok = analyse(synthetic_feed(now, lag_hours=0), now)
    lagged = analyse(synthetic_feed(now, lag_hours=2), now)
    assert ok["hybrid_inputs_expected_at_2300_ist_with_this_lag"] and ok["days_with_all_7_inputs_full_day"] == 8, ok
    assert ok["days_with_all_7_inputs_by_newest_hour"]["22"] == 8 and ok["days_with_all_7_inputs_by_newest_hour"]["21"] == 0
    assert not lagged["hybrid_inputs_expected_at_2300_ist_with_this_lag"] and abs(lagged["freshest_sensor_lag_minutes"] - 120.1) < 0.2
    assert lagged["expected_newest_hour_at_2300_ist_with_this_lag"] == 20 and ok["expected_newest_hour_at_2300_ist_with_this_lag"] == 22
    assert ok["future_observations"]["rows_hour_not_finished"] == 0 and ok["stamp_minutes_utc_seen"] == [30]
    print(render(ok))
    print("\n--- same data, feed 2 h behind ---")
    print(render(lagged))
    print("\nselftest ok")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sensors", help="comma-separated OpenAQ PM2.5 sensor ids (default: $SENSOR_IDS or config/stations.json)")
    ap.add_argument("--days", type=int, default=8, help="complete days to examine (default 8)")
    ap.add_argument("--out", help="also write the report as JSON to this file")
    ap.add_argument("--selftest", action="store_true", help="offline check of the analysis on synthetic data")
    a = ap.parse_args(argv)
    if a.selftest:
        return selftest()
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        print("OPENAQ_API_KEY is not set in this shell. Export it (do not paste it anywhere) and run again.", file=sys.stderr)
        return 2
    now = datetime.now(timezone.utc)
    try:
        feed = fetch_live(sensor_ids(a.sensors), now, a.days, key)
    except Exception as e:                                   # the key is only ever in a request header, not in the message
        print(f"OpenAQ request failed: {str(e).replace(key, '***')}", file=sys.stderr)
        return 3
    rep = analyse(feed, now, a.days)
    print(render(rep))
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=2))
        print(f"report written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
