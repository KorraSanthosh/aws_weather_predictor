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
import check_opencity_timezone as cot  # noqa: E402
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


# ---------------------------------------------------------------------------------------------------------------------
# Timestamp parsing. Python 3.10's datetime.fromisoformat cannot read "...+0000" (no colon) or odd fraction lengths, which
# broke the two tests above on 3.10. The converter now parses the text itself, so every Python gives the same answer.
# ---------------------------------------------------------------------------------------------------------------------
D = datetime
WALL_CLOCK_CASES = [
    # +0000 (the form the OpenCity files use), with and without a colon, with and without microseconds
    ("2024-03-01T00:00:00.000000+0000", D(2024, 3, 1, 0, 0)),
    ("2024-03-01T00:00:00.000000+00:00", D(2024, 3, 1, 0, 0)),
    ("2024-03-01T00:15:00+0000", D(2024, 3, 1, 0, 15)),
    ("2024-03-01T00:15:00+00:00", D(2024, 3, 1, 0, 15)),
    ("2024-03-01T23:45:00.000000+0000", D(2024, 3, 1, 23, 45)),             # late evening stays on the same day
    # other offsets: the printed clock is kept, nothing is converted to another zone
    ("2024-03-01T12:30:15.250000+05:30", D(2024, 3, 1, 12, 30, 15, 250000)),
    ("2024-03-01T12:30:15.250000+0530", D(2024, 3, 1, 12, 30, 15, 250000)),
    ("2024-03-01T12:30:15-08:00", D(2024, 3, 1, 12, 30, 15)),
    ("2024-03-01T12:30:15-0800", D(2024, 3, 1, 12, 30, 15)),
    ("2024-03-01T12:30:15+05", D(2024, 3, 1, 12, 30, 15)),
    ("2024-03-01T12:30:15Z", D(2024, 3, 1, 12, 30, 15)),
    # no offset at all, other separators / precisions
    ("2024-03-01T00:15:00", D(2024, 3, 1, 0, 15)),
    ("2024-03-01 00:15:00+0000", D(2024, 3, 1, 0, 15)),
    ("2024-03-01T00:15+0000", D(2024, 3, 1, 0, 15)),
    ("2024-03-01", D(2024, 3, 1)),
    # fractions of any length (fromisoformat on 3.10 only accepts 3 or 6 digits)
    ("2024-03-01T00:15:00.5+0000", D(2024, 3, 1, 0, 15, 0, 500000)),
    ("2024-03-01T00:15:00.123+0000", D(2024, 3, 1, 0, 15, 0, 123000)),
    ("2024-03-01T00:15:00.1234567+0000", D(2024, 3, 1, 0, 15, 0, 123456)),   # beyond microseconds: truncated
]


@pytest.mark.parametrize("text,expected", WALL_CLOCK_CASES)
def test_parse_wall_clock_keeps_the_printed_clock_and_drops_the_offset(text, expected):
    got = co.parse_wall_clock(text)
    assert got == expected and got.tzinfo is None


def test_parse_wall_clock_ignores_surrounding_whitespace():
    assert co.parse_wall_clock("  2024-03-01T00:15:00.000000+0000 ") == D(2024, 3, 1, 0, 15)


def test_parsing_does_not_depend_on_datetime_fromisoformat(monkeypatch):
    """Simulate Python 3.10 on any interpreter: fromisoformat refuses the OpenCity text, parsing must still work."""
    class Py310(datetime):
        @classmethod
        def fromisoformat(cls, text):
            raise ValueError(f"Invalid isoformat string: {text!r}")
    monkeypatch.setattr(co, "datetime", Py310)
    with pytest.raises(ValueError):
        co.datetime.fromisoformat("2024-03-01T00:00:00.000000+0000")      # the simulation itself works
    for text, expected in WALL_CLOCK_CASES:
        assert co.parse_wall_clock(text) == expected
        assert co.parse_labelled(text).replace(tzinfo=None) == expected


@pytest.mark.parametrize("bad", ["", "   ", "yesterday", "2024-13-01T00:00:00+0000", "2024-03-32T00:00:00+0000",
                                 "2024-03-01T25:00:00+0000", "2024-03-01T00:61:00+0000", "2024-03-01T00:00:00+2400",
                                 "2024-03-01T00:00:00+0060", "2024-03-01T00:00:00 +0000", "2024-03-01T00:00:00+0000 extra",
                                 "01/03/2024 00:00", None])
def test_unreadable_timestamps_raise_instead_of_guessing(bad):
    with pytest.raises(ValueError):
        co.parse_wall_clock(bad)


def test_parse_labelled_keeps_the_offset_for_the_timezone_diagnostic():
    utc = co.parse_labelled("2024-03-01T00:00:00.000000+0000")
    assert utc.utcoffset() == timedelta(0) and utc.astimezone(IST).hour == 5            # taken at face value: it IS UTC
    assert co.parse_labelled("2024-03-01T12:00:00+05:30").utcoffset() == timedelta(hours=5, minutes=30)
    assert co.parse_labelled("2024-03-01T12:00:00-0800").utcoffset() == timedelta(hours=-8)
    assert co.parse_labelled("2024-03-01T12:00:00Z").utcoffset() == timedelta(0)
    assert co.parse_labelled("2024-03-01T12:00:00").tzinfo is None


@pytest.mark.skipif(sys.version_info < (3, 11), reason="fromisoformat only reads these spellings from Python 3.11")
@pytest.mark.parametrize("text", [t for t, _ in WALL_CLOCK_CASES if not t.startswith(" ")])
def test_new_parser_matches_the_old_fromisoformat_behaviour_where_that_can_read_the_text(text):
    """The previous code was datetime.fromisoformat(t).replace(tzinfo=None). Same answer wherever it works."""
    old = datetime.fromisoformat(text)
    assert co.parse_wall_clock(text) == old.replace(tzinfo=None)
    new = co.parse_labelled(text)
    assert new == old and new.utcoffset() == old.utcoffset()


# ------------------------------------------------------------------------- the converter end to end, all spellings
SPELLINGS = ["{d}T{t}.000000+0000", "{d}T{t}.000000+00:00", "{d}T{t}+0000", "{d}T{t}+00:00", "{d}T{t}Z",
             "{d}T{t}.000000+0530", "{d}T{t}+05:30", "{d}T{t}", "{d} {t}+0000"]


@pytest.mark.parametrize("spelling", SPELLINGS)
def test_every_timestamp_spelling_gives_the_same_converted_file(tmp_path, spelling):
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    src.mkdir(); raw.mkdir()
    ts = lambda d, t: spelling.format(d=d, t=t)                                       # noqa: E731
    write_opencity(src / "del-r-k-puram-dpcc-2024-25.csv", [
        (ts("2024-03-01", "00:00:00"), "100.0"), (ts("2024-03-01", "00:15:00"), ""), (ts("2024-03-01", "23:45:00"), "200.0")])
    rep = co.convert(src, out, raw)
    rows = list(csv.DictReader(open(out / "17.csv")))
    assert [(r["datetime"], r["pm25"]) for r in rows] == [("2024-03-01T00:00:00+05:30", "100.0"),
                                                          ("2024-03-01T23:45:00+05:30", "200.0")]
    assert rep["17"]["rows"] == 3 and rep["17"]["blank_pm25"] == 1 and rep["17"]["kept"] == 2
    assert rep["17"]["first"] == rep["17"]["last"] == "2024-03-01"


def test_blank_values_missing_cells_and_blank_timestamps(tmp_path):
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    src.mkdir(); raw.mkdir()
    (src / "del-r-k-puram-dpcc-2024-25.csv").write_text("\n".join([
        ",".join(HEAD),
        "site_124,Delhi,Delhi,R K,2024-03-01T00:00:00.000000+0000,",             # blank PM2.5 -> dropped, counted
        "site_124,Delhi,Delhi,R K,2024-03-01T00:15:00.000000+0000",              # row too short: no PM2.5 cell at all
        "site_124,Delhi,Delhi,R K,,55.0",                                       # no timestamp -> skipped, not even counted
        "site_124,Delhi,Delhi,R K,2024-03-01T00:30:00.000000+0000,0.0",          # zero is a real reading, not blank
        "site_124,Delhi,Delhi,R K,2024-03-01T00:45:00+0000,42.5"]) + "\n")
    rep = co.convert(src, out, raw)
    rows = list(csv.DictReader(open(out / "17.csv")))
    assert [(r["datetime"][11:19], r["pm25"]) for r in rows] == [("00:30:00", "0.0"), ("00:45:00", "42.5")]
    assert (rep["17"]["rows"], rep["17"]["blank_pm25"], rep["17"]["kept"]) == (4, 2, 2)      # nothing was interpolated


def test_coverage_check_uses_the_wall_clock_day_not_a_shifted_one(tmp_path):
    """OpenAQ already has 1 Mar 2024 (IST). OpenCity readings labelled 23:45 on 1 Mar belong to that day under every
    spelling; converting the label from UTC to IST would push them to 2 Mar and wrongly keep them."""
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    (raw / "openaq").mkdir(parents=True); src.mkdir()
    with open(raw / "openaq" / "17.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["datetime", "pm25"]); w.writerow(["2024-03-01T10:00:00+05:30", "50.0"])
    write_opencity(src / "del-r-k-puram-dpcc-2024-25.csv", [
        ("2024-03-01T23:45:00.000000+0000", "1.0"), ("2024-03-01T23:45:00+00:00", "2.0"),
        ("2024-03-01T23:45:00+05:30", "3.0"), ("2024-03-01T23:45:00", "4.0"),
        ("2024-03-02T00:15:00.000000+0000", "5.0")])
    rep = co.convert(src, out, raw)
    got = [(r["datetime"], r["pm25"]) for r in csv.DictReader(open(out / "17.csv"))]
    assert got == [("2024-03-02T00:15:00+05:30", "5.0")]
    assert rep["17"]["dropped_day_already_in_openaq"] == 4 and rep["17"]["kept"] == 1


def test_an_unreadable_timestamp_stops_the_conversion_and_names_the_file(tmp_path):
    src, out, raw = tmp_path / "src", tmp_path / "out", tmp_path / "raw"
    src.mkdir(); raw.mkdir()
    write_opencity(src / "del-r-k-puram-dpcc-2024-25.csv", [("2024-03-01T00:00:00.000000+0000", "1.0"), ("not-a-time", "2.0")])
    with pytest.raises(ValueError, match=r"del-r-k-puram-dpcc-2024-25\.csv.*not-a-time"):
        co.convert(src, out, raw)


def test_timezone_diagnostic_script_reads_the_same_timestamps(tmp_path):
    """pipeline/check_opencity_timezone.py had the same fromisoformat call; it now uses the shared parser."""
    assert cot.parse_wall_clock is co.parse_wall_clock and cot.parse_labelled is co.parse_labelled
    f = tmp_path / "del-r-k-puram-dpcc-2024-25.csv"
    write_opencity(f, [("2024-03-01T00:00:00.000000+0000", "100.0"), ("2024-03-01T00:15:00+0000", "")])
    out = cot.cpcb_daily_check(f, "17", tmp_path / "raw")                              # no CPCB files -> no days, no crash
    assert out["label_is_IST"]["days"] == 0 and out["label_is_UTC"]["days"] == 0
