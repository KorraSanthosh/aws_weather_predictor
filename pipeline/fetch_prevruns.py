"""Download archived weather forecasts (Open-Meteo Previous Runs API) for the forecast-training experiment.

    python pipeline/fetch_prevruns.py [--start 2024-01-01] [--end 2026-02-16] [--out data/raw/weather_prevruns_2024_2026.json]

For every hour it stores the value that was forecast 1 day and 2 days earlier (`<var>_previous_day1`, `_previous_day2`),
at the Delhi grid point the project uses.  Data begins in January 2024.  The output is under data/raw/ (git-ignored).
The API is free but has a daily request limit; the download is split into half-year chunks to keep requests small.
The file already in data/raw was produced this way; this script exists so the experiment can be reproduced.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.weather import HOURLY_VARS, http_get_json  # noqa: E402

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LEADS = (1, 2)
DELHI = (28.57, 77.18)                 # same point the project's other weather files use (grid snaps to 28.576, 77.187)


def variables():
    return [f"{v}_previous_day{n}" for n in LEADS for v in HOURLY_VARS]


def chunks(start: date, end: date, days: int = 183):
    d = start
    while d <= end:
        e = min(end, d + timedelta(days=days - 1))
        yield d, e
        d = e + timedelta(days=1)


def fetch(start: date, end: date, get=http_get_json):
    merged, meta = {"time": []}, {}
    for a, b in chunks(start, end):
        data = get(URL, {"latitude": DELHI[0], "longitude": DELHI[1], "start_date": a.isoformat(), "end_date": b.isoformat(),
                         "hourly": ",".join(variables()), "timezone": "Asia/Kolkata"})
        h = data["hourly"]
        if not meta:
            meta = {k: v for k, v in data.items() if k != "hourly"}
        merged["time"] += h["time"]
        for k in variables():
            merged.setdefault(k, []).extend(h[k])
    return {**meta, "hourly": merged}


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2024-01-01")
    ap.add_argument("--end", default="2026-02-16")
    ap.add_argument("--out", default="data/raw/weather_prevruns_2024_2026.json")
    a = ap.parse_args()
    out = fetch(date.fromisoformat(a.start), date.fromisoformat(a.end))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out))
    print(f"wrote {a.out}: {len(out['hourly']['time'])} hours")
