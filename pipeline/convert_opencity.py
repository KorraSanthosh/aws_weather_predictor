"""Convert Delhi station files from OpenCity (CPCB 15-minute data, "<station> 15 minute AQI Data for 2024-25") into the
project's station format (`datetime,pm25`), as a GAP-FILLER for the OpenAQ files.

    python pipeline/convert_opencity.py [--src data/raw/opencity_2024_25] [--out data/raw/opencity_hourly]

Rules (fixed, written down before the data was used):
  * TIME ZONE.  The Timestamp column is labelled `+0000` but is Delhi wall-clock time.  Read as IST, 24-hour means of
    these readings reproduce the official CPCB daily files for 2024 exactly at all four stations (mean abs diff 0.0
    ug/m3); read as UTC they are off by 8-11 ug/m3.  Against OpenAQ hourly data the IST reading correlates 0.95-0.997 and
    the UTC reading 0.27-0.49 (pipeline/check_opencity_timezone.py, experiments/opencity_timezone_check.json).
    Timestamps are therefore written as IST (+05:30).
  * GAP-FILL ONLY.  A reading is kept only if the station has NO OpenAQ reading at all on that IST day.  Days OpenAQ
    already covers are untouched, so nothing in the existing hourly history changes and no day is double counted.
  * PARSING.  The timestamp text is read by parse_wall_clock() below, NOT by datetime.fromisoformat: Python 3.10 cannot read
    the `+0000` form (no colon) or odd fraction lengths, so the same file converted on 3.10 and on 3.11+ must not differ.
    The wall-clock fields are kept exactly as written and the offset is discarded without shifting anything.
  * Blank PM2.5 values are dropped; nothing is interpolated.
  * Stations are mapped to the OpenAQ station ids already used by the CPCB-daily loader (pipeline/build_dataset.py).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import build_dataset as bd  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
PM_COL, TS_COL = "PM2.5 (µg/m³)", "Timestamp"


# "2024-03-01T00:15:00.000000+0000" and its relatives. Date, optional time (seconds and fraction optional), optional offset
# written Z, +HH, +HHMM or +HH:MM. Written by hand because datetime.fromisoformat differs between Python 3.10 and 3.11+.
_TS = re.compile(r"""^(\d{4})-(\d{2})-(\d{2})
                     (?:[T ](\d{2}):(\d{2})(?::(\d{2})(?:[.,](\d{1,9}))?)?
                        (?:(Z)|([+-])(\d{2})(?::?(\d{2}))?)?)?$""", re.X)


def _parse_timestamp(text: str):
    """(naive wall-clock datetime, UTC offset as a timedelta or None). Raises ValueError for anything unrecognised."""
    m = _TS.match(text.strip()) if isinstance(text, str) else None
    if not m:
        raise ValueError(f"unrecognised OpenCity timestamp {text!r}")
    y, mo, d, hh, mi, ss, frac, zulu, sign, oh, om = m.groups()
    micro = int((frac or "0")[:6].ljust(6, "0"))                     # more than 6 digits: truncate, as fromisoformat does
    try:
        naive = datetime(int(y), int(mo), int(d), int(hh or 0), int(mi or 0), int(ss or 0), micro)
    except ValueError as e:
        raise ValueError(f"unrecognised OpenCity timestamp {text!r}: {e}") from None
    offset = None
    if zulu:
        offset = timedelta(0)
    elif sign:
        if int(oh) > 23 or int(om or 0) > 59:
            raise ValueError(f"unrecognised OpenCity timestamp {text!r}: bad UTC offset")
        offset = (1 if sign == "+" else -1) * timedelta(hours=int(oh), minutes=int(om or 0))
    return naive, offset


def parse_wall_clock(text: str) -> datetime:
    """The timestamp as a NAIVE wall-clock datetime: the printed date and time unchanged, the UTC offset dropped without
    converting anything (the file's `+0000` label is wrong, the clock reading is Delhi time). Same result on every Python."""
    return _parse_timestamp(text)[0]


def parse_labelled(text: str) -> datetime:
    """The timestamp taken at face value, offset included (aware when an offset is written, naive otherwise). Only for
    pipeline/check_opencity_timezone.py, which deliberately tests what the label would mean if it were true."""
    naive, offset = _parse_timestamp(text)
    return naive if offset is None else naive.replace(tzinfo=timezone(offset) if offset else timezone.utc)


def station_id(filename: str) -> str:
    """del-r-k-puram-dpcc-2024-25.csv -> '17' (via the CPCB->OpenAQ map); unknown stations -> 'ocity_<name>'."""
    m = re.match(r"del-(.+?)-(?:dpcc|cpcb|imd|iitm)-\d{4}-\d{2}\.csv$", filename)
    name = (m.group(1) if m else Path(filename).stem).replace("-", "_")
    for k, v in bd.CPCB_TO_OPENAQ.items():
        if name.startswith(k):
            return v
    return "ocity_" + name


def openaq_days(raw: Path, sid: str) -> set:
    days = set()
    for sub in ("openaq", "openaq_api", "openaq_recent"):
        p = raw / sub / f"{sid}.csv"
        if p.exists():
            for ts, _ in bd.load_station(p):
                days.add(ts.astimezone(IST).date())
    return days


def convert(src: Path, out: Path, raw: Path):
    out.mkdir(parents=True, exist_ok=True)
    report = {}
    for f in sorted(src.glob("del-*.csv")):
        sid = station_id(f.name)
        have = openaq_days(raw, sid)
        kept, dropped_cov, blank, rows = [], 0, 0, 0
        with open(f, newline="") as fh:
            for r in csv.DictReader(fh):
                t, v = r.get(TS_COL), r.get(PM_COL)
                if not t:
                    continue
                rows += 1
                if v in (None, ""):
                    blank += 1
                    continue
                try:
                    naive = parse_wall_clock(t)                              # wall-clock label; the "+0000" is wrong
                except ValueError as e:
                    raise ValueError(f"{f.name}: {e}") from None
                ts = naive.replace(tzinfo=IST)
                if ts.date() in have:
                    dropped_cov += 1
                    continue
                kept.append((ts, float(v)))
        kept.sort()
        with open(out / f"{sid}.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["datetime", "pm25"])
            for ts, v in kept:
                w.writerow([ts.isoformat(), v])
        report[sid] = {"file": f.name, "rows": rows, "blank_pm25": blank, "dropped_day_already_in_openaq": dropped_cov,
                       "kept": len(kept), "first": kept[0][0].date().isoformat() if kept else None,
                       "last": kept[-1][0].date().isoformat() if kept else None}
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--src", default="data/raw/opencity_2024_25")
    ap.add_argument("--out", default="data/raw/opencity_hourly")
    a = ap.parse_args()
    rep = convert(Path(a.src), Path(a.out), Path(a.raw))
    (Path(a.out) / "gapfill_report.json").write_text(json.dumps(rep, indent=1))
    for sid, r in rep.items():
        print(sid, r)
