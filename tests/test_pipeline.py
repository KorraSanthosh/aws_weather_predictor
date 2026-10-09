from __future__ import annotations

import csv
import json
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
for sub in ("", "scripts", "model", "pipeline"):
    sys.path.insert(0, str(ROOT / sub))

from core import openaq  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402


# ---------- OpenAQ pagination ----------
def test_fetch_sensor_hours_follows_pages(monkeypatch):
    def row(i):
        ts = datetime(2025, 1, 1, tzinfo=timezone.utc) + timedelta(hours=i)
        return {"value": 100.0 + i, "period": {"datetimeFrom": {"utc": ts.strftime("%Y-%m-%dT%H:%M:%SZ")}}}

    pages = {1: [row(i) for i in range(1000)], 2: [row(i) for i in range(1000, 1005)]}
    calls = []

    def fake_get(path, params, api_key, **kw):
        calls.append(params["page"])
        return {"results": pages[params["page"]]}

    monkeypatch.setattr(openaq, "_get", fake_get)
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    out = openaq.fetch_sensor_hours(1, t0, t0 + timedelta(days=60), "k")
    assert len(out) == 1005 and calls == [1, 2]
    assert out[0][1] == 100.0 and out[-1][1] == 1104.0


def test_fetch_sensor_hours_skips_empty_values(monkeypatch):
    monkeypatch.setattr(openaq, "_get", lambda *a, **k: {"results": [
        {"value": None, "period": {"datetimeFrom": {"utc": "2025-01-01T00:00:00Z"}}},
        {"value": 55, "period": {"datetimeFrom": {"utc": "2025-01-01T01:00:00Z"}}}]})
    t0 = datetime(2025, 1, 1, tzinfo=timezone.utc)
    assert len(openaq.fetch_sensor_hours(1, t0, t0 + timedelta(days=1), "k")) == 1


# ---------- build_dataset merges archive + API files ----------
def _write_hours(path, start_day: date, days: int, value: float, tz_suffix="+05:30"):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["datetime", "pm25"])
        for d in range(days):
            for h in range(24):
                day = start_day + timedelta(days=d)
                w.writerow([f"{day.isoformat()}T{h:02d}:00:00{tz_suffix}", value])


def test_build_dataset_merges_both_sources(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = date(2025, 3, 1)
    _write_hours(Path("data/raw/openaq_api/7.csv"), start, 21, 90.0)               # days 0..20 (API)
    _write_hours(Path("data/raw/openaq/7.csv"), start + timedelta(days=20), 20, 110.0)  # days 20..39 (archive)
    with open("data/raw/weather_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + WEATHER_KEYS)
        for d in range(40):
            w.writerow([(start + timedelta(days=d)).isoformat()] + [1.0] * len(WEATHER_KEYS))
    import build_dataset
    build_dataset.main(["--min-stations", "1"])
    rows = list(csv.DictReader(open("data/daily.csv")))
    assert len(rows) == 40
    assert all(r["pm25"] != "" for r in rows)
    by_date = {r["date"]: float(r["pm25"]) for r in rows}
    assert by_date["2025-03-05"] == pytest.approx(90.0)
    assert by_date["2025-04-05"] == pytest.approx(110.0)
    assert by_date["2025-03-21"] == pytest.approx(100.0)     # overlap day: both sources averaged


def test_build_dataset_drops_days_with_too_few_stations(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = date(2025, 3, 1)
    _write_hours(Path("data/raw/openaq/1.csv"), start, 10, 80.0)                         # days 0..9
    _write_hours(Path("data/raw/openaq/2.csv"), start + timedelta(days=5), 5, 120.0)     # days 5..9
    with open("data/raw/weather_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + WEATHER_KEYS)
        for d in range(10):
            w.writerow([(start + timedelta(days=d)).isoformat()] + [1.0] * len(WEATHER_KEYS))
    import build_dataset
    build_dataset.main(["--min-stations", "2"])
    rows = list(csv.DictReader(open("data/daily.csv")))
    assert [r["date"] for r in rows] == [(start + timedelta(days=d)).isoformat() for d in range(5, 10)]
    assert all(r["n_stations"] == "2" and float(r["pm25"]) == pytest.approx(100.0) for r in rows)
    meta = json.loads(Path("data/daily.meta.json").read_text())
    assert meta["min_stations"] == 2 and meta["days_dropped_too_few_stations"] == 5


# ---------- training evaluation modes ----------
def _synthetic_in(tmp_path, monkeypatch, trim_from: str | None = None):
    monkeypatch.chdir(tmp_path)
    import make_synthetic_data
    make_synthetic_data.main()
    if trim_from:
        lines = Path("data/daily.csv").read_text().splitlines()
        keep = [lines[0]] + [l for l in lines[1:] if l.split(",")[0] >= trim_from]
        Path("data/daily.csv").write_text("\n".join(keep) + "\n")


def test_train_uses_rolling_winter_folds(tmp_path, monkeypatch):
    _synthetic_in(tmp_path, monkeypatch)
    import train
    train.main()
    m = json.loads(Path("model/metrics.json").read_text())
    assert m["evaluation"] == "rolling_winter_folds"
    assert len(m["folds"]) >= 2
    for f in m["folds"]:
        assert f["n_train"] >= train.MIN_TRAIN and f["window"][0][5:] == "10-15"
    assert "warning" not in m
    meta = json.loads(Path("model/model_meta.json").read_text())
    assert meta["log_resid_q10"] < 0 < meta["log_resid_q90"]
    assert Path("model/model.json").exists() and Path("web/data/forecast.json").exists()
    fc = json.loads(Path("web/data/forecast.json").read_text())
    assert fc["data_source"] == "synthetic" and fc["pm25_low"] <= fc["pm25"] <= fc["pm25_high"]


def test_train_tests_the_peak_of_a_single_winter(tmp_path, monkeypatch):
    _synthetic_in(tmp_path, monkeypatch, trim_from="2025-02-19")   # same span as the real data
    import train
    train.main()
    m = json.loads(Path("model/metrics.json").read_text())
    assert m["evaluation"] == "late_winter_holdout"
    assert len(m["folds"]) == 1 and m["folds"][0]["window"] == ["2025-12-01", "2026-02-15"]
    assert m["folds"][0]["n_train"] >= train.MIN_TRAIN_LATE
    assert "single" in m["warning"] and "peak of winter" in m["test_description"]


def test_train_falls_back_loudly_without_any_winter(tmp_path, monkeypatch, capsys):
    _synthetic_in(tmp_path, monkeypatch, trim_from="2025-09-01")   # too little before the winter
    import train
    train.main()
    m = json.loads(Path("model/metrics.json").read_text())
    assert m["evaluation"] == "last_120_days_not_a_winter"
    assert "warning" in m and "OPTIMISTIC" in capsys.readouterr().out


def test_bootstrap_interval_separates_real_gain_from_noise():
    import numpy as np
    import train
    rng = np.random.default_rng(1)
    y = rng.uniform(80, 300, 140)
    persist = y + rng.normal(0, 40, 140)
    better = y + rng.normal(0, 20, 140)                  # clearly more accurate
    same = persist + rng.normal(0, 0.5, 140)             # essentially the same errors
    assert train.bootstrap_improvement(y, better, persist)["clearly_better"]
    assert not train.bootstrap_improvement(y, same, persist)["clearly_better"]


def test_train_reports_robustness_fields(tmp_path, monkeypatch):
    _synthetic_in(tmp_path, monkeypatch)
    import train
    train.main()
    m = json.loads(Path("model/metrics.json").read_text())
    assert {"low_pct", "high_pct", "clearly_better"} <= set(m["improvement_95pct_interval"])
    assert "calm_days" in m["breakdown"] and m["breakdown"]["by_month"] and m["verdict"]


def test_build_dataset_exclude_range(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = date(2025, 3, 1)
    _write_hours(Path("data/raw/openaq/7.csv"), start, 10, 90.0)
    with open("data/raw/weather_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + WEATHER_KEYS)
        for d in range(10):
            w.writerow([(start + timedelta(days=d)).isoformat()] + [1.0] * len(WEATHER_KEYS))
    import build_dataset
    build_dataset.main(["--min-stations", "1", "--exclude", "2025-03-04:2025-03-06"])
    rows = {r["date"]: r["pm25"] for r in csv.DictReader(open("data/daily.csv"))}
    assert rows["2025-03-04"] == "" and rows["2025-03-06"] == "" and rows["2025-03-07"] != ""
    assert json.loads(Path("data/daily.meta.json").read_text())["days_excluded"] == 3


def test_build_dataset_reads_cpcb_daily_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    start = date(2025, 3, 1)
    for sid in (1, 2):                                    # two hourly stations
        _write_hours(Path(f"data/raw/openaq/{sid}.csv"), start, 10, 100.0)
    Path("data/raw/cpcb").mkdir(parents=True)
    with open("data/raw/cpcb/Raw_Data_2025_site_301_anand_vihar_delhi_dpcc_1D.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Timestamp", "PM2.5 (\u00b5g/m\u00b3)", "PM10 (\u00b5g/m\u00b3)"])
        for d in range(10):
            w.writerow([(start + timedelta(days=d)).isoformat(), 400.0 if d < 8 else "", 1])
    with open("data/raw/weather_daily.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["date"] + WEATHER_KEYS)
        for d in range(10):
            w.writerow([(start + timedelta(days=d)).isoformat()] + [1.0] * len(WEATHER_KEYS))
    import build_dataset
    build_dataset.main(["--min-stations", "3"])
    rows = {r["date"]: r for r in csv.DictReader(open("data/daily.csv"))}
    assert rows["2025-03-02"]["n_stations"] == "3" and float(rows["2025-03-02"]["pm25"]) == pytest.approx(200.0)
    assert "2025-03-10" not in rows                         # CPCB blank + only 2 stations -> day dropped
