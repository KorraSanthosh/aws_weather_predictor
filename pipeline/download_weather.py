"""Step 3: download daily weather (and the CAMS benchmark) from Open-Meteo.

Free for non-commercial use, no API key. Usage:
    python pipeline/download_weather.py --start 2023-01-01
Writes data/raw/weather_daily.csv and data/raw/cams_daily.csv
"""
import argparse
import csv
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core import weather as wx  # noqa: E402

DELHI = (28.6139, 77.2090)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--lat", type=float, default=DELHI[0])
    ap.add_argument("--lon", type=float, default=DELHI[1])
    args = ap.parse_args()
    start = date.fromisoformat(args.start)
    today = date.today()

    # ERA5 archive lags a few days, so use it up to today-7 and top up from the forecast API
    archive_end = today - timedelta(days=7)
    print(f"archive weather {start} -> {archive_end}")
    daily = {}
    cur = start
    while cur <= archive_end:                       # one year per request
        nxt = min(date(cur.year, 12, 31), archive_end)
        daily.update(wx.fetch_archive_weather(args.lat, args.lon, cur, nxt))
        cur = nxt + timedelta(days=1)
    print("top-up from forecast API (last ~2 weeks)")
    for d, w in wx.fetch_forecast_weather(args.lat, args.lon, past_days=14, forecast_days=1).items():
        if d > archive_end:
            daily[d] = w

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    with open("data/raw/weather_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + wx.WEATHER_KEYS)
        for d in sorted(daily):
            w.writerow([d.isoformat()] + [round(daily[d][k], 4) for k in wx.WEATHER_KEYS])
    print(f"saved {len(daily)} days -> data/raw/weather_daily.csv")

    # CAMS global model PM2.5 for the same dates (history starts Aug 2022). Optional benchmark.
    try:
        cams = {}
        cur = max(start, date(2022, 8, 1))
        while cur <= today:
            nxt = min(cur + timedelta(days=89), today)
            cams.update(wx.fetch_cams_daily_pm25(args.lat, args.lon, cur, nxt))
            cur = nxt + timedelta(days=1)
        with open("data/raw/cams_daily.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["date", "cams_pm25"])
            for d in sorted(cams):
                w.writerow([d.isoformat(), round(cams[d], 3)])
        print(f"saved {len(cams)} CAMS days -> data/raw/cams_daily.csv")
    except Exception as exc:                        # benchmark is optional
        print("CAMS benchmark skipped:", exc)


if __name__ == "__main__":
    main()
