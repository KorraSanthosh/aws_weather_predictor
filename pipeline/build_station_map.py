"""Build web/data/stations.json: the latest measured PM2.5 / AQI at each Delhi station, for the dashboard map.

    python pipeline/build_station_map.py

Reads the raw hourly files in data/raw (archive, API backfill, recent top-up), averages each station's
IST day (needs >= 16 hourly readings, same rule as the dataset), and writes the most recent day that at
least 3 stations reported. Station coordinates are approximate (config/station_coords.json).

"tomorrow_est" is NOT a station-level model forecast. The model forecasts the city mean, so each
station's value is scaled by (city forecast / city today). The map labels it as an estimate.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.aqi import pm25_to_aqi, aqi_to_category  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
SOURCES = ["openaq", "openaq_api", "openaq_recent"]


def station_days(lid: int) -> dict:
    hours = defaultdict(dict)                      # IST date -> {hour-start: value}
    for src in SOURCES:
        p = ROOT / "data" / "raw" / src / f"{lid}.csv"
        if not p.exists():
            continue
        with p.open() as f:
            for row in csv.DictReader(f):
                try:
                    t = datetime.fromisoformat(row["datetime"]).astimezone(IST)
                    v = float(row["pm25"])
                except (ValueError, KeyError, TypeError):
                    continue
                if 0 <= v <= 1000:
                    hours[t.date()][t.replace(minute=0, second=0, microsecond=0)] = v
    return {d: sum(h.values()) / len(h) for d, h in hours.items() if len(h) >= 16}


def main():
    stations = json.loads((ROOT / "config" / "stations.json").read_text())
    coords = json.loads((ROOT / "config" / "station_coords.json").read_text())
    days = {s["location_id"]: station_days(s["location_id"]) for s in stations}
    count = defaultdict(int)
    for d in days.values():
        for day in d:
            count[day] += 1
    usable = [day for day, n in count.items() if n >= 3]
    if not usable:
        sys.exit("No day with >= 3 stations found in data/raw.")
    day = max(usable)

    ratio = None
    fc_path = ROOT / "web" / "data" / "forecast.json"
    if fc_path.exists():
        fc = json.loads(fc_path.read_text())
        if fc.get("today_pm25") and fc.get("pm25"):
            ratio = fc["pm25"] / fc["today_pm25"]

    out = []
    for s in stations:
        c = coords.get(str(s["location_id"]))
        if not c:
            continue
        pm = days[s["location_id"]].get(day)
        rec = {"id": s["location_id"], "name": s["name"], "short": c["short"], "lat": c["lat"], "lon": c["lon"],
               "pm25": None, "aqi": None, "category": None, "tomorrow_est": None}
        if pm is not None:
            aqi = pm25_to_aqi(pm)
            rec.update(pm25=round(pm, 1), aqi=aqi, category=aqi_to_category(aqi))
            if ratio:
                est = pm * ratio
                ea = pm25_to_aqi(est)
                rec["tomorrow_est"] = {"pm25": round(est, 1), "aqi": ea, "category": aqi_to_category(ea)}
        out.append(rec)
    dest = ROOT / "web" / "data" / "stations.json"
    dest.write_text(json.dumps({"date": day.isoformat(), "stations": out}, indent=1) + "\n")
    print(f"wrote {dest} for {day} with {sum(1 for r in out if r['pm25'] is not None)}/{len(out)} stations")


if __name__ == "__main__":
    main()
