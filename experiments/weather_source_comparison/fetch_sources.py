"""Probe and download archived 24 h / 48 h-ahead forecasts from several weather models (Open-Meteo Previous Runs API).

    python experiments/weather_source_comparison/fetch_sources.py --probe      # coverage first (1 variable, whole period)
    python experiments/weather_source_comparison/fetch_sources.py --download   # all 6 variables, lead 1 and 2, half-year chunks

Writes only  data/raw/weather_sources/<model>/<chunk>.json  (git-ignored raw data)  and
experiments/weather_source_comparison/coverage.json.  No API key is used or needed (free, non-commercial endpoint);
nothing here reads or writes any .env file.

`<var>_previous_dayN` = the value that was forecast N x 24 h before the valid hour (Open-Meteo wording).  The same
convention is used for every model so sources are lead-matched to each other.  Its availability caveat (a run may be
published after the 23:00 IST cutoff) is analysed in the report, not hidden here.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from core.weather import HOURLY_VARS, http_get_json  # noqa: E402

URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
LAT, LON = 28.6139, 77.2090            # backend/forecast_handler.py defaults (the live Lambda sets no LAT/LON)
START, END = date(2024, 1, 1), date(2026, 2, 16)
LEADS = (1, 2)
# label -> (Open-Meteo `models=` identifier or None for the default best_match the live Lambda uses)
MODELS = {
    "production_default": None,
    "ecmwf_ifs_hres": "ecmwf_ifs",
    "ecmwf_ifs025": "ecmwf_ifs025",
    "gfs_global": "gfs_global",
    "icon_global": "icon_global",
    "ecmwf_aifs025": "ecmwf_aifs025_single",
}
RAW = ROOT / "data" / "raw" / "weather_sources"


def variables(vars_=HOURLY_VARS, leads=LEADS):
    return [f"{v}_previous_day{n}" for n in leads for v in vars_]


def chunks(a: date, b: date, days=183):
    d = a
    while d <= b:
        e = min(b, d + timedelta(days=days - 1))
        yield d, e
        d = e + timedelta(days=1)


def request(model_id, a, b, hourly, get=http_get_json, retries=3):
    p = {"latitude": LAT, "longitude": LON, "start_date": a.isoformat(), "end_date": b.isoformat(), "hourly": ",".join(hourly), "timezone": "Asia/Kolkata"}
    if model_id:
        p["models"] = model_id
    last = None
    for k in range(retries):
        try:
            return get(URL, p, retries=1)
        except Exception as exc:                       # noqa: BLE001
            last = exc
            time.sleep(4 * (k + 1))
    raise RuntimeError(str(last)[:300])


def probe(models=MODELS, get=http_get_json):
    """Coverage of temperature_2m_previous_day1/2 for the whole period, per model."""
    out = {"point": {"latitude": LAT, "longitude": LON}, "period": [START.isoformat(), END.isoformat()], "models": {}}
    for label, mid in models.items():
        rec = {"model_id": mid or "(default: best_match)"}
        try:
            data = request(mid, START, END, ["temperature_2m_previous_day1", "temperature_2m_previous_day2"], get)
            h = data["hourly"]
            t = h["time"]
            for lead in LEADS:
                v = h[f"temperature_2m_previous_day{lead}"]
                ok = [i for i, x in enumerate(v) if x is not None]
                by_month = {}
                for i, x in enumerate(v):
                    m = t[i][:7]
                    c = by_month.setdefault(m, [0, 0])
                    c[0] += x is not None
                    c[1] += 1
                rec[f"lead{lead}"] = {"hours": len(v), "non_null_hours": len(ok), "first_valid": t[ok[0]] if ok else None, "last_valid": t[ok[-1]] if ok else None,
                                      "months_below_90pct_filled": sorted(m for m, (n, d) in by_month.items() if n / d < 0.9)}
            rec["grid_point"] = {"latitude": data.get("latitude"), "longitude": data.get("longitude"), "elevation": data.get("elevation")}
            rec["units"] = data.get("hourly_units", {}).get("temperature_2m_previous_day1")
        except Exception as exc:                       # noqa: BLE001
            rec["error"] = str(exc)[:200]
        out["models"][label] = rec
        print(label, {k: (v if k != "lead1" and k != "lead2" else {a: b for a, b in v.items() if a != "months_below_90pct_filled"}) for k, v in rec.items()})
        time.sleep(1.0)
    return out


def download(models=MODELS, get=http_get_json):
    for label, mid in models.items():
        d = RAW / label
        d.mkdir(parents=True, exist_ok=True)
        for a, b in chunks(START, END):
            f = d / f"{a.isoformat()}_{b.isoformat()}.json"
            if f.exists():
                continue
            data = request(mid, a, b, variables(), get)
            data["_request"] = {"model": mid or "best_match(default)", "start": a.isoformat(), "end": b.isoformat(), "fetched_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            f.write_text(json.dumps(data))
            print("saved", f.relative_to(ROOT), len(data["hourly"]["time"]), "hours")
            time.sleep(1.5)


if __name__ == "__main__":  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--download", action="store_true")
    a = ap.parse_args()
    if a.probe:
        (HERE / "coverage.json").write_text(json.dumps(probe(), indent=1))
    if a.download:
        download()
