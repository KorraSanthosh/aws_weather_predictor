"""Generate SYNTHETIC Delhi-like daily data so the pipeline can be tested offline.

!! This is NOT real air quality data. It only exists to test the code.
!! It writes data/daily.meta.json with source="synthetic" so the website
!! shows a DEMO DATA banner. Never present synthetic results as real.
"""
import csv
import json
import math
import random
from datetime import date, timedelta
from pathlib import Path

random.seed(7)
START, END = date(2022, 9, 1), date(2026, 10, 7)
KEYS = ["temp", "rh", "wind", "precip", "pressure", "wind_u", "wind_v"]


def main():
    rows, d = [], START
    latent = 0.0          # persistent pollution "memory"
    wind_z = 0.0
    while d <= END:
        doy = d.timetuple().tm_yday
        season = math.cos(2 * math.pi * (doy - 340) / 365.25)          # peak early Dec
        temp = 25 + 10 * math.cos(2 * math.pi * (doy - 170) / 365.25) + random.gauss(0, 2)
        monsoon = 1.0 if 180 <= doy <= 260 else 0.0
        rh = 50 + 30 * monsoon + random.gauss(0, 8)
        wind_z = 0.6 * wind_z + random.gauss(0, 0.8)
        wind = max(1.0, 8 + 3 * wind_z + 2 * monsoon)
        precip = max(0.0, random.gauss(0, 1) * (6 if monsoon else 0.8) - (0 if monsoon else 1.2))
        pressure = 990 + 8 * season + random.gauss(0, 2)
        ang = random.uniform(0, 2 * math.pi)
        wind_u, wind_v = -wind * math.sin(ang), -wind * math.cos(ang)

        base = 95 + 85 * season
        latent = 0.72 * latent + random.gauss(0, 0.20)
        logpm = math.log(max(base, 25)) + latent - 0.045 * (wind - 8) - 0.04 * min(precip, 15)
        pm = max(5.0, math.exp(logpm))
        rows.append([d.isoformat(), round(pm, 3), 4,
                     round(temp, 3), round(rh, 3), round(wind, 3), round(precip, 3),
                     round(pressure, 3), round(wind_u, 3), round(wind_v, 3),
                     round(max(5, pm * math.exp(random.gauss(-0.25, 0.30))), 3)])
        d += timedelta(days=1)

    # knock out ~4% of days to mimic real station gaps
    for r in random.sample(rows, int(len(rows) * 0.04)):
        r[1], r[2], r[-1] = "", 0, ""

    Path("data").mkdir(exist_ok=True)
    with open("data/daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "pm25", "n_stations"] + KEYS + ["cams_pm25"])
        w.writerows(rows)
    Path("data/daily.meta.json").write_text(json.dumps({
        "source": "synthetic", "first": START.isoformat(), "last": END.isoformat(),
        "note": "SYNTHETIC test data - not real measurements"}, indent=2))
    print(f"wrote data/daily.csv ({len(rows)} days, SYNTHETIC)")


if __name__ == "__main__":
    main()
