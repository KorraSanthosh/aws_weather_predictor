"""Step 4: merge raw station readings + weather into data/daily.csv and print a data-quality report.

Usage:  python pipeline/build_dataset.py [--min-stations 3]
Reads  data/raw/openaq/*.csv, data/raw/openaq_api/*.csv, data/raw/openaq_recent/*.csv, data/raw/cpcb/*.csv (CPCB portal daily files), data/raw/weather_daily.csv,
       (optional) data/raw/cams_daily.csv
Writes data/daily.csv and data/daily.meta.json

A day counts only if at least --min-stations stations have a valid 24h mean, so the "city average"
always means roughly the same thing (days covered by a single station are dropped).
"""
import argparse
import csv
import json
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from core.daily import MIN_STATIONS, city_daily_mean, city_daily_stats, readings_to_daily  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402

RAW = Path("data/raw")

# CPCB portal files (daily averages) -> the OpenAQ station they correspond to, so one physical
# station is never counted twice. Unknown stations are kept under their own "cpcb_<site>" key.
CPCB_TO_OPENAQ = {"anand_vihar": "235", "r_k_puram": "17", "rk_puram": "17", "punjabi_bagh": "50",
                  "rohini": "10831", "mandir_marg": "6358", "dwarka": "6931", "okhla": "8239",
                  "alipur": "6932", "pusa": "5404", "ito": "5613", "shadipur": "5630",
                  "najafgarh": "10488", "wazirpur": "8915", "bawana": "8472", "narela": "10485",
                  "jahangirpuri": "8235", "vivek_vihar": "6938", "patparganj": "6960",
                  "nehru_nagar": "8365", "sonia_vihar": "8475", "ashok_vihar": "8917",
                  "mundka": "10486", "lodhi_road": "5634", "dtu": "5626", "nsit": "5622"}


def cpcb_station_key(path: Path) -> str:
    """'Raw_Data_2023_site_301_anand_vihar_delhi_dpcc_1D.csv' -> '235' (mapped) or 'cpcb_301'."""
    import re
    m = re.search(r"site_(\d+)_(.*?)_1D", path.name)
    if not m:
        return "cpcb_" + path.stem
    name = m.group(2)
    for k, v in CPCB_TO_OPENAQ.items():
        if name.startswith(k):
            return v
    return "cpcb_" + m.group(1)


def load_cpcb_daily(path: Path):
    """CPCB repository file with DAILY averages (frequency 24H). Returns {date: (pm25, 24)}."""
    import pandas as pd
    df = pd.read_csv(path)
    col = next((c for c in df.columns if c.strip().lower().startswith("pm2.5")), None)
    if col is None or "Timestamp" not in df.columns:
        return {}
    out = {}
    for ts, v in zip(pd.to_datetime(df["Timestamp"], errors="coerce"), pd.to_numeric(df[col], errors="coerce")):
        if ts is pd.NaT or v != v or v < 0 or v > 1000:
            continue
        out[ts.date()] = (float(v), 24)
    return out


def load_station(path: Path):
    readings = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                ts = datetime.fromisoformat(row["datetime"].replace("Z", "+00:00"))
                readings.append((ts, float(row["pm25"])))
            except (ValueError, KeyError):
                continue
    return readings


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-stations", type=int, default=MIN_STATIONS,
                    help="minimum stations with a valid daily mean for a day to count")
    ap.add_argument("--exclude", action="append", default=[], metavar="START:END",
                    help="drop PM2.5 days in this range (YYYY-MM-DD:YYYY-MM-DD), e.g. the COVID lockdown "
                         "--exclude 2020-03-25:2020-06-30 . Can be repeated.")
    args = ap.parse_args(argv)
    excluded = []
    for r in args.exclude:
        a, b = r.split(":")
        excluded.append((date.fromisoformat(a), date.fromisoformat(b)))

    # one entry per station; archive (S3) and API-backfill files are merged
    station_files: dict[str, list[Path]] = {}
    for sub in ("openaq", "openaq_api", "openaq_recent"):
        for p in sorted((RAW / sub).glob("*.csv")):
            station_files.setdefault(p.stem, []).append(p)
    if not station_files and not list((RAW / "cpcb").glob("*.csv")):
        sys.exit("No station files in data/raw/openaq. Run download_openaq.py first.")
    if not (RAW / "weather_daily.csv").exists():
        sys.exit("Missing data/raw/weather_daily.csv. Run download_weather.py first.")

    cpcb_files: dict[str, list[Path]] = {}
    for p in sorted((RAW / "cpcb").glob("*.csv")):
        cpcb_files.setdefault(cpcb_station_key(p), []).append(p)
    for sid in cpcb_files:
        station_files.setdefault(sid, [])
    per_station, coverage = [], {}
    for sid, paths in station_files.items():
        readings = []
        for p in paths:
            readings.extend(load_station(p))
        daily = readings_to_daily(readings)   # same-hour readings are averaged, so overlaps are harmless
        for p in cpcb_files.get(sid, []):     # official daily files fill days the hourly data lacks
            for day, val in load_cpcb_daily(p).items():
                daily.setdefault(day, val)
        per_station.append(daily)
        coverage[sid] = len(daily)
    city_all = city_daily_mean(per_station)                       # every day with any station
    city = city_daily_mean(per_station, min_stations=args.min_stations)
    stats = city_daily_stats(per_station, min_stations=args.min_stations)
    dropped = len(city_all) - len(city)
    n_excluded = 0
    for d in [d for d in list(city) if any(a <= d <= b for a, b in excluded)]:
        del city[d]
        n_excluded += 1
    if not city:
        sys.exit(f"No day has {args.min_stations}+ stations with valid data. Check the station files.")

    weather = {}
    with open(RAW / "weather_daily.csv", newline="") as f:
        for row in csv.DictReader(f):
            weather[date.fromisoformat(row["date"])] = {k: float(row[k]) for k in WEATHER_KEYS}

    cams = {}
    if (RAW / "cams_daily.csv").exists():
        with open(RAW / "cams_daily.csv", newline="") as f:
            for row in csv.DictReader(f):
                cams[date.fromisoformat(row["date"])] = float(row["cams_pm25"])

    days = sorted(set(city) | set(weather))
    first = min(city) if city else None
    last = max(city) if city else None
    days = [d for d in days if first <= d <= last]

    Path("data").mkdir(exist_ok=True)
    cols = ["date", "pm25", "n_stations", "pm25_sd", "pm25_max", "pm25_min"] + WEATHER_KEYS + ["cams_pm25"]
    with open("data/daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for d in days:
            pm, n = city.get(d, (None, 0))
            wx = weather.get(d, {})
            st = stats.get(d) if pm is not None else None
            spread = ["" if (st is None or st[k] != st[k]) else round(st[k], 3)
                      for k in ("pm25_sd", "pm25_max", "pm25_min")]
            w.writerow([d.isoformat(), "" if pm is None else round(pm, 3), n] + spread
                       + [round(wx[k], 4) if k in wx else "" for k in WEATHER_KEYS]
                       + [round(cams[d], 3) if d in cams else ""])

    total = (last - first).days + 1
    have = sum(1 for d in days if d in city)
    meta = {"source": "openaq", "stations": list(coverage), "first": first.isoformat(),
            "last": last.isoformat(), "days_with_pm25": have, "calendar_days": total,
            "min_stations": args.min_stations, "excluded_ranges": args.exclude, "days_excluded": n_excluded, "days_dropped_too_few_stations": dropped}
    Path("data/daily.meta.json").write_text(json.dumps(meta, indent=2))

    valid_days = sorted(city)
    longest_gap = max(((b - a).days - 1 for a, b in zip(valid_days, valid_days[1:])), default=0)
    print("\n=== DATA QUALITY REPORT ===")
    print(f"range: {first} -> {last}  ({total} calendar days)")
    print(f"days with a valid city-mean PM2.5 (>= {args.min_stations} stations): {have} ({100 * have / total:.0f}%)")
    print(f"days dropped because fewer than {args.min_stations} stations reported: {dropped}")
    if excluded:
        print(f"days removed by --exclude {args.exclude}: {n_excluded}")
    print(f"longest gap between valid days: {longest_gap} days")
    for sid, n in coverage.items():
        print(f"  station {sid}: {n} valid days")
    print("valid days per year: " + ", ".join(
        f"{y}: {sum(1 for d in city if d.year == y)}" for y in sorted({d.year for d in city})))
    winters = sum(1 for y in sorted({d.year for d in city})
                  if sum(1 for d in city if date(y, 10, 15) <= d <= date(y + 1, 2, 15)) >= 90)
    print(f"winters (15 Oct - 15 Feb) with 90+ valid days: {winters}  "
          "(the model is evaluated on winters; 3+ is good, 1 is not enough)")
    missing_wx = sum(1 for d in days if d not in weather)
    print(f"days without weather (these days cannot be used for training): {missing_wx}")
    stale = (date.today() - last).days
    if stale > 3:
        print(f"WARNING: newest PM2.5 day is {stale} days old - live forecasts need fresh data.")
    if have < 400:
        print("WARNING: under ~400 days of data; the model will be weak. Add stations or go further back.")
    print("wrote data/daily.csv")


if __name__ == "__main__":
    main()
