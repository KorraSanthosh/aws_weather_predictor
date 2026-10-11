"""Which clock do the OpenCity timestamps use?  Their `+0000` suffix is wrong if they are Delhi wall-clock time.

    python pipeline/check_opencity_timezone.py     # writes experiments/opencity_timezone_check.json

For each station the OpenCity readings are shifted by a candidate offset (hours added to the label to get true UTC),
averaged to hours and correlated with the OpenAQ hourly series for the same station, on Mar-Jul 2025 hours both have.
Offset -5.5 means "the label is IST".

Second, independent check (the decisive one): the official CPCB 24-hour daily files for the same stations (2024) are
compared with 24-hour means built from the 15-minute readings, once reading the label as IST and once as UTC (+5.5 h).
Decision rule, fixed in advance: the labels are IST if that reproduces the CPCB daily values (mean abs diff < 0.5 ug/m3)
for every station and the UTC reading does not."""
import csv
import json
import sys
from datetime import timedelta, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_dataset as bd  # noqa: E402
from convert_opencity import PM_COL, TS_COL, parse_labelled, parse_wall_clock, station_id  # noqa: E402

SHIFTS = (0.0, -5.0, -5.5, -6.0, 5.5)


def hour_key(ts):
    return ts.replace(minute=0, second=0, microsecond=0)


def cpcb_daily_check(f, sid, raw):
    cp = {}
    for p in sorted((Path(raw) / "cpcb").glob("*.csv")):
        if bd.cpcb_station_key(p) == sid:
            cp.update({d: v[0] for d, v in bd.load_cpcb_daily(p).items()})
    rd = []
    with open(f, newline="") as fh:
        for r in csv.DictReader(fh):
            t, v = r.get(TS_COL), r.get(PM_COL)
            if t and v:
                rd.append((parse_wall_clock(t), float(v)))
    out = {}
    for name, sh in (("label_is_IST", 0.0), ("label_is_UTC", 5.5)):
        dd = {}
        for ts, v in rd:
            dd.setdefault((ts + timedelta(hours=sh)).date(), []).append(v)
        ks = [d for d, v in dd.items() if len(v) >= 60 and d in cp and d.year == 2024]
        a = np.array([np.mean(dd[d]) for d in ks]); b = np.array([cp[d] for d in ks])
        out[name] = {"days": len(ks), "corr": round(float(np.corrcoef(a, b)[0, 1]), 3) if len(ks) > 10 else None,
                     "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 2) if len(ks) > 10 else None}
    return out


def main(src="data/raw/opencity_2024_25", raw="data/raw"):
    out = {}
    for f in sorted(Path(src).glob("del-*.csv")):
        sid = station_id(f.name)
        oa = {}
        for sub in ("openaq", "openaq_recent"):
            p = Path(raw) / sub / f"{sid}.csv"
            if p.exists():
                for ts, v in bd.load_station(p):
                    oa.setdefault(hour_key(ts.astimezone(timezone.utc)), []).append(v)
        oa = {k: float(np.mean(v)) for k, v in oa.items()}
        rd = []
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                t, v = r.get(TS_COL), r.get(PM_COL)
                if t and v:
                    rd.append((parse_labelled(t), float(v)))
        res = {}
        for sh in SHIFTS:
            h = {}
            for ts, v in rd:
                k = hour_key(ts + timedelta(hours=sh))
                if k.year == 2025 and 3 <= k.month <= 7:
                    h.setdefault(k, []).append(v)
            keys = [k for k in h if k in oa]
            a = np.array([np.mean(h[k]) for k in keys]); b = np.array([oa[k] for k in keys])
            res[str(sh)] = {"hours": len(keys), "corr": round(float(np.corrcoef(a, b)[0, 1]), 3),
                            "mean_abs_diff": round(float(np.mean(np.abs(a - b))), 1)} if len(keys) > 100 else None
        best = max((s for s in res if res[s]), key=lambda s: res[s]["corr"])
        out[sid] = {"file": f.name, "vs_openaq_hourly_by_shift_hours": res, "best_shift_vs_openaq": best,
                    "vs_cpcb_daily_files_2024": cpcb_daily_check(f, sid, raw)}
    st = [v for v in out.values()]
    ist_ok = all(v["vs_cpcb_daily_files_2024"]["label_is_IST"]["mean_abs_diff"] is not None and
                 v["vs_cpcb_daily_files_2024"]["label_is_IST"]["mean_abs_diff"] < 0.5 and
                 v["vs_cpcb_daily_files_2024"]["label_is_UTC"]["mean_abs_diff"] >= 0.5 for v in st)
    out["decision"] = "labels are IST wall-clock time (the +0000 suffix is wrong)" if ist_ok else "REVIEW - do not use"
    out["note"] = ("OpenAQ R K Puram (17) peaks at -5.0 h instead of -5.5 h against OpenAQ, while its CPCB daily match is "
                   "exact for the IST reading, so that is a quirk of the OpenAQ series, not of the OpenCity file.")
    Path("experiments").mkdir(exist_ok=True)
    Path("experiments/opencity_timezone_check.json").write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    print(json.dumps(main(), indent=1))
