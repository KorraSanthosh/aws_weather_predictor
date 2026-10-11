"""Data audit for the 2025 backtest: what exists, per year, from the real files.  Writes audit.json."""
import csv, json, sys
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in (ROOT, ROOT / "model", ROOT / "pipeline", ROOT / "experiments" / "weather_source_comparison"):
    sys.path.insert(0, str(p))
import build_dataset as bd
import forecast_training_experiment as fte, option_screen as osn, winter_backtest as wb
import run_comparison as rc
from core.weather import WEATHER_KEYS
from core.daily import MIN_STATIONS

YEARS = list(range(2012, 2026))


def per_station_days():
    """{source/station: {year: days with at least one reading}}"""
    out = {}
    for sub in ("openaq", "openaq_api", "openaq_recent", "opencity_hourly", "opencity_2024_25"):
        for p in sorted((ROOT / "data/raw" / sub).glob("*.csv")):
            try:
                rd = bd.load_station(p)
            except Exception as e:                                                 # noqa: BLE001
                out[f"{sub}/{p.stem}"] = {"error": str(e)[:80]}; continue
            days = {fte.osn.__dict__.get("to_ist", lambda x: x)(ts).date() if False else ts.date() for ts, v in rd}
            c = Counter(d.year for d in days)
            if c:
                out[f"{sub}/{p.stem}"] = {str(y): n for y, n in sorted(c.items())}
    for p in sorted((ROOT / "data/raw/cpcb").glob("*.csv")):
        d = bd.load_cpcb_daily(p)
        c = Counter(k.year for k in d)
        out[f"cpcb/{p.stem}"] = {str(y): n for y, n in sorted(c.items())}
    return out


def main():
    rows = list(csv.DictReader(open(ROOT / "data/daily_opencity.csv")))
    meta = json.loads((ROOT / "data/daily_opencity.meta.json").read_text())
    byd = {date.fromisoformat(r["date"]): r for r in rows}
    wx = {date.fromisoformat(r["date"]): r for r in csv.DictReader(open(ROOT / "data/raw/weather_daily.csv"))}
    res = {"file": "data/daily_opencity.csv", "meta": meta, "min_stations_rule": MIN_STATIONS,
           "first_row": min(byd).isoformat(), "last_row": max(byd).isoformat(),
           "first_weather_day": min(wx).isoformat(), "last_weather_day": max(wx).isoformat(), "years": {}}
    excl = meta.get("excluded_ranges", [])
    for y in YEARS:
        days = [date(y, 1, 1) + timedelta(i) for i in range((date(y + 1, 1, 1) - date(y, 1, 1)).days)]
        have = [d for d in days if d in byd and byd[d]["pm25"] not in ("", None)]
        ns = [int(float(byd[d]["n_stations"])) for d in have]
        months = Counter(d.month for d in have)
        wdays = [d for d in days if d in wx and all(wx[d][k] not in ("", None) for k in WEATHER_KEYS)]
        res["years"][str(y)] = {
            "calendar_days": len(days), "days_with_city_pm25_ge3_stations": len(have), "share": round(len(have) / len(days), 3),
            "days_with_complete_weather": len(wdays),
            "stations_per_day_min_median_max": [min(ns), sorted(ns)[len(ns) // 2], max(ns)] if ns else None,
            "days_by_month": {str(m): months.get(m, 0) for m in range(1, 13)},
            "has_genuine_usable_pm25": len(have) > 0}
    res["no_pm25_before"] = min(d for d in byd if byd[d]["pm25"] not in ("", None)).isoformat()
    res["excluded_ranges_in_build"] = excl
    res["station_days_by_source_and_year"] = per_station_days()
    return res


if __name__ == "__main__":
    r = main()
    (HERE / "audit.json").write_text(json.dumps(r, indent=1))
    for y, v in r["years"].items():
        print(y, v["days_with_city_pm25_ge3_stations"], "/", v["calendar_days"], v["stations_per_day_min_median_max"], "wx", v["days_with_complete_weather"])
    print(r["first_row"], r["last_row"], r["no_pm25_before"])
