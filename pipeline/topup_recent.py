"""Step 2c: top up the newest days from the live OpenAQ API.

The S3 archive lags by several days (its newest day is usually ~4 days old). This fetches everything
after each station's last archive reading, up to now, so the model/demo use today's data.

    export OPENAQ_API_KEY=...
    python pipeline/topup_recent.py
    python pipeline/download_weather.py       # refresh weather up to today
    python pipeline/build_dataset.py && python model/train.py

Writes data/raw/openaq_recent/<location_id>.csv (picked up by build_dataset.py).
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.openaq import fetch_sensor_hours  # noqa: E402
from backfill_openaq_api import latest_reading  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stations", default="config/stations.json")
    ap.add_argument("--pause", type=float, default=1.1)
    args = ap.parse_args()
    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY first.")
    now = datetime.now(timezone.utc)
    out = Path("data/raw/openaq_recent")
    out.mkdir(parents=True, exist_ok=True)
    for st in json.loads(Path(args.stations).read_text()):
        lid, sensor = st["location_id"], st.get("pm25_sensor_id")
        if not sensor:
            continue
        last = latest_reading(Path(f"data/raw/openaq/{lid}.csv")) or now - timedelta(days=14)
        start = last - timedelta(hours=24)                      # small overlap; duplicates are averaged
        rows = fetch_sensor_hours(sensor, start, now, key)
        with open(out / f"{lid}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["datetime", "pm25"])
            for ts, val in rows:
                w.writerow([ts.isoformat(), val])
        newest = max((t for t, _ in rows), default=None)
        print(f"[{lid}] {st.get('name')}: {len(rows)} hourly values, newest {newest}")
        time.sleep(args.pause)


if __name__ == "__main__":
    main()
