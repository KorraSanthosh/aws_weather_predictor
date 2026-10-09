"""Step 2b: backfill older PM2.5 history from the OpenAQ API.

The S3 archive only starts around Feb 2025 for these stations, which gives a single
winter. This script fetches hourly data for the period BEFORE each station's first
archive reading, so the model can learn from several winters.

    export OPENAQ_API_KEY=...
    python3 pipeline/backfill_openaq_api.py --start 2022-01-01

Needs config/stations.json (location_id + pm25_sensor_id) and, ideally, the archive files
from download_openaq.py already in data/raw/openaq/. Writes data/raw/openaq_api/<location_id>.csv.
Free-tier limit is 60 requests/minute; the script paces itself (~1 request/second).
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.openaq import fetch_sensor_hours  # noqa: E402


def earliest_archive_reading(path: Path):
    first = None
    if not path.exists():
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["datetime"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if first is None or ts < first:
                first = ts
    return first


def latest_reading(path: Path):
    last = None
    if not path.exists():
        return None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["datetime"].replace("Z", "+00:00"))
            except (ValueError, KeyError):
                continue
            ts = ts.astimezone(timezone.utc) if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
            if last is None or ts > last:
                last = ts
    return last


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--stations", default="config/stations.json")
    ap.add_argument("--chunk-days", type=int, default=30)
    ap.add_argument("--pause", type=float, default=1.1, help="seconds between requests")
    args = ap.parse_args()

    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY first.")
    start = datetime.combine(date.fromisoformat(args.start), datetime.min.time(), tzinfo=timezone.utc)
    stations = json.loads(Path(args.stations).read_text())
    outdir = Path("data/raw/openaq_api")
    outdir.mkdir(parents=True, exist_ok=True)

    for st in stations:
        lid, sensor = st["location_id"], st.get("pm25_sensor_id")
        if not sensor:
            print(f"[{lid}] no pm25_sensor_id in stations.json - skipped")
            continue
        end = earliest_archive_reading(Path(f"data/raw/openaq/{lid}.csv")) or datetime.now(timezone.utc)
        if start >= end:
            print(f"[{lid}] archive already starts before {args.start} - nothing to backfill")
            continue
        print(f"[{lid}] {st.get('name')}: backfilling {start.date()} -> {end.date()}")
        total, cur, chunks = 0, start, 0
        with open(outdir / f"{lid}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["datetime", "pm25"])
            while cur < end:
                nxt = min(cur + timedelta(days=args.chunk_days), end)
                rows = fetch_sensor_hours(sensor, cur, nxt, key)
                for ts, val in rows:
                    w.writerow([ts.isoformat(), val])
                total += len(rows)
                chunks += 1
                if chunks == 1:
                    print(f"   first request returned {len(rows)} hourly values"
                          + ("" if rows else "  <-- empty; the API may hold no data for this period"))
                cur = nxt
                time.sleep(args.pause)
        print(f"   saved {total} hourly values -> {outdir / (str(lid) + '.csv')}")
        if total == 0:
            print("   WARNING: nothing returned for this station. If it is the same for all stations, "
                  "the API may not serve history for these sensors - tell me what you see.")
    print("done. Now run: python3 pipeline/download_weather.py --start " + args.start +
          "  then  python3 pipeline/build_dataset.py")


if __name__ == "__main__":
    main()
