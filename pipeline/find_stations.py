"""Step 1: list PM2.5 stations around Delhi and write config/stations.json.

Needs a free OpenAQ API key in the environment:
    export OPENAQ_API_KEY=...        (Windows PowerShell: $env:OPENAQ_API_KEY="...")

Usage:  python pipeline/find_stations.py [--bbox 76.84,28.40,77.35,28.88]
Review the printed table and edit config/stations.json if you want to
drop or add stations. Prefer stations with long history AND recent data.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.openaq import find_pm25_locations  # noqa: E402

DELHI_BBOX = "76.84,28.40,77.35,28.88"
# the bbox also catches neighbouring cities; the "city mean" must be Delhi only
NOT_DELHI = ("gurugram", "gurgaon", "faridabad", "noida", "ghaziabad", "bahadurgarh", "airnow", "stateair")


def is_delhi(loc):
    name = (loc.get("name") or "").lower()
    return not any(k in name for k in NOT_DELHI)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbox", default=DELHI_BBOX)
    ap.add_argument("--max", type=int, default=6, help="max stations to keep")
    ap.add_argument("--history", action="store_true",
                    help="instead pick stations with OLD data (started before --history-before), "
                         "even if they stopped reporting; writes config/stations_history.json for "
                         "extra TRAINING winters. Download them with download_openaq.py --stations "
                         "config/stations_history.json --start 2018-01-01")
    ap.add_argument("--history-before", default="2022-10-01")
    args = ap.parse_args()

    key = os.environ.get("OPENAQ_API_KEY")
    if not key:
        sys.exit("Set OPENAQ_API_KEY first (free: https://explore.openaq.org/register)")

    locs = [l for l in find_pm25_locations(key, args.bbox) if is_delhi(l)]
    now = datetime.now(timezone.utc)
    print(f"{len(locs)} PM2.5 locations in bbox {args.bbox}\n")
    print(f"{'id':>8}  {'first':10}  {'last':10}  {'sensor':>8}  name (provider)")
    scored = []
    for l in locs:
        first, last = l["first"], l["last"]
        years = (last - first).days / 365 if first and last else 0
        fresh = bool(last and now - last < timedelta(days=14))
        print(f"{l['location_id']:>8}  {str(first)[:10]:10}  {str(last)[:10]:10}  "
              f"{str(l['pm25_sensor_id']):>8}  {l['name']} ({l['provider']})"
              f"{'' if fresh else '   [STALE]'}")
        if fresh and l["pm25_sensor_id"] and years >= 1.5:
            scored.append((years, l))

    if args.history:
        cutoff = datetime.fromisoformat(args.history_before).replace(tzinfo=timezone.utc)
        old = sorted([l for l in locs if l["first"] and l["first"] <= cutoff and l["pm25_sensor_id"]
                      and l["last"] and (l["last"] - l["first"]).days >= 365],
                     key=lambda l: l["first"])[: args.max]
        if not old:
            sys.exit(f"\nNo station with data from before {args.history_before}. Try a wider --bbox.")
        out = [{"location_id": l["location_id"], "name": l["name"],
                "pm25_sensor_id": l["pm25_sensor_id"]} for l in old]
        Path("config").mkdir(exist_ok=True)
        Path("config/stations_history.json").write_text(json.dumps(out, indent=2))
        print(f"\nWrote config/stations_history.json with {len(out)} long-history stations (oldest first):")
        for l in old:
            print(f"   {l['location_id']}  {str(l['first'])[:10]} -> {str(l['last'])[:10]}  {l['name']}")
        return

    scored.sort(key=lambda t: -t[0])
    keep = [l for _, l in scored[: args.max]]
    if not keep:
        sys.exit("\nNo station with >=1.5 years of history and recent data. "
                 "Widen --bbox or edit config/stations.json by hand.")
    out = [{"location_id": l["location_id"], "name": l["name"],
            "pm25_sensor_id": l["pm25_sensor_id"]} for l in keep]
    Path("config").mkdir(exist_ok=True)
    Path("config/stations.json").write_text(json.dumps(out, indent=2))
    print(f"\nWrote config/stations.json with {len(out)} stations:")
    for s in out:
        print("  ", s["location_id"], s["name"])


if __name__ == "__main__":
    main()
