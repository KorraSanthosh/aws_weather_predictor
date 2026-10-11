"""Tests of the OpenCity gap-fill converter and the build_dataset --extra-source / --out options."""
import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT, ROOT / "pipeline"):
    sys.path.insert(0, str(p))

import build_dataset as bd  # noqa: E402
import convert_opencity as co  # noqa: E402

IST = timezone(timedelta(hours=5, minutes=30))
HEAD = ["Station ID", "State", "City", "Station Name", "Timestamp", "PM2.5 (µg/m³)"]


def write_opencity(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(HEAD)
        for ts, v in rows:
            w.writerow(["site_124", "Delhi", "Delhi", "R K Puram, Delhi - DPCC", ts, v])


def test_station_id_maps_known_stations_to_openaq_ids():
    assert co.station_id("del-r-k-puram-dpcc-2024-25.csv") == "17"
    assert co.station_id("del-punjabi-bagh-dpcc-2024-25.csv") == "50"
    assert co.station_id("del-rohini-dpcc-2024-25.csv") == "10831"
    assert co.station_id("del-anand-vihar-dpcc-2024-25.csv") == "235"
    assert co.station_id("del-some-new-place-dpcc-2024-25.csv").startswith("ocity_")


def test_timestamps_are_read_as_ist_not_utc_and_blanks_are_dropped(tmp_path):
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    src.mkdir(); raw.mkdir()
    write_opencity(src / "del-r-k-puram-dpcc-2024-25.csv", [
        ("2024-03-01T00:00:00.000000+0000", "100.0"), ("2024-03-01T00:15:00.000000+0000", ""),
        ("2024-03-01T23:45:00.000000+0000", "200.0")])
    rep = co.convert(src, out, raw)
    rows = list(csv.DictReader(open(out / "17.csv")))
    assert [r["pm25"] for r in rows] == ["100.0", "200.0"]
    assert rows[0]["datetime"] == "2024-03-01T00:00:00+05:30"           # wall-clock label kept, zone corrected
    assert rep["17"]["blank_pm25"] == 1 and rep["17"]["kept"] == 2


def test_gap_fill_never_touches_days_openaq_already_has(tmp_path):
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    (raw / "openaq").mkdir(parents=True); src.mkdir()
    with open(raw / "openaq" / "17.csv", "w", newline="") as f:           # OpenAQ has 2 Mar 2024 (IST) for station 17
        w = csv.writer(f); w.writerow(["datetime", "pm25"]); w.writerow(["2024-03-02T10:00:00+05:30", "50.0"])
    write_opencity(src / "del-r-k-puram-dpcc-2024-25.csv", [
        ("2024-03-01T12:00:00.000000+0000", "10.0"), ("2024-03-02T12:00:00.000000+0000", "999.0"),
        ("2024-03-03T12:00:00.000000+0000", "30.0")])
    rep = co.convert(src, out, raw)
    got = {r["datetime"][:10]: r["pm25"] for r in csv.DictReader(open(out / "17.csv"))}
    assert got == {"2024-03-01": "10.0", "2024-03-03": "30.0"}
    assert rep["17"]["dropped_day_already_in_openaq"] == 1


def test_build_dataset_extra_source_and_out_leave_the_default_file_alone(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw"
    for sub in ("openaq", "extra"):
        (raw / sub).mkdir(parents=True)
    d0 = date(2024, 1, 1)

    def station(path, days, val):
        with open(path, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["datetime", "pm25"])
            for k in range(days):
                for h in range(24):
                    w.writerow([datetime(d0.year, d0.month, d0.day, h, tzinfo=IST).isoformat().replace(
                        f"{d0.day:02d}T", f"{(d0 + timedelta(days=k)).day:02d}T"), val])
    for sid in ("a", "b"):
        station(raw / "openaq" / f"{sid}.csv", 5, 100.0)
    station(raw / "extra" / "c.csv", 5, 100.0)
    with open(raw / "weather_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date", "temp", "rh", "wind", "precip", "pressure", "wind_u", "wind_v"])
        for k in range(5):
            w.writerow([(d0 + timedelta(days=k)).isoformat(), 10, 50, 2, 0, 1000, 1, 1])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(bd, "RAW", Path("data/raw"))
    bd.main(["--min-stations", "3", "--out", "data/alt.csv", "--extra-source", "extra"])
    assert Path("data/alt.csv").exists() and Path("data/alt.meta.json").exists()
    assert not Path("data/daily.csv").exists()
    assert "c" in json.loads(Path("data/alt.meta.json").read_text())["stations"]
    n_with = sum(1 for r in csv.DictReader(open("data/alt.csv")) if r["pm25"])
    assert n_with == 5                                                  # 3 stations -> every day counts
    with pytest.raises(SystemExit):                                     # without the extra source only 2 stations exist
        bd.main(["--min-stations", "3", "--out", "data/alt2.csv"])
    assert not Path("data/alt2.csv").exists()
