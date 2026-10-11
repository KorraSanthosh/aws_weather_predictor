"""Guards for the weather-source experiment (no network)."""
import json
import sys
from datetime import date, timedelta
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import fetch_sources as fs  # noqa: E402
import run_comparison as rc  # noqa: E402
from core.weather import HOURLY_VARS, hourly_to_daily_weather  # noqa: E402


def _hourly(day0, n_days, fill=1.0, drop=None):
    t = [(day0 + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(24 * n_days)]
    from datetime import datetime
    t = [(datetime(day0.year, day0.month, day0.day) + timedelta(hours=i)).strftime("%Y-%m-%dT%H:%M") for i in range(24 * n_days)]
    h = {"time": t}
    for lead in (1, 2):
        for v in HOURLY_VARS:
            h[f"{v}_previous_day{lead}"] = [fill] * len(t)
    if drop:
        for v in HOURLY_VARS:
            for i in drop:
                h[f"{v}_previous_day1"][i] = None
    return h


def test_variables_are_the_production_variables():
    names = fs.variables()
    assert all(f"{v}_previous_day{n}" in names for v in HOURLY_VARS for n in (1, 2))
    assert len(names) == 2 * len(HOURLY_VARS)


def test_chunks_are_contiguous_and_cover_period():
    cs = list(fs.chunks(fs.START, fs.END))
    assert cs[0][0] == fs.START and cs[-1][1] == fs.END
    assert all(b[0] == a[1] + timedelta(days=1) for a, b in zip(cs, cs[1:]))


def test_no_credentials_in_requests(monkeypatch):
    seen = {}

    def fake(url, params, retries=1):
        seen.update(params)
        return {"hourly": {"time": []}}
    fs.request("gfs_global", date(2025, 1, 1), date(2025, 1, 2), fs.variables(), get=fake)
    assert not any("key" in k.lower() or "token" in k.lower() for k in seen)
    assert seen["models"] == "gfs_global" and seen["timezone"] == "Asia/Kolkata"


def test_default_source_sends_no_models_parameter():
    seen = {}
    fs.request(None, date(2025, 1, 1), date(2025, 1, 2), fs.variables(), get=lambda u, p, retries=1: seen.update(p) or {})
    assert "models" not in seen


def test_aggregation_matches_production_incomplete_day_dropped(tmp_path, monkeypatch):
    h = _hourly(date(2025, 1, 1), 2, drop=list(range(0, 24 - 11)))       # day 1 keeps only 11 hours (< 12)
    hh = {"time": h["time"], **{v: h[f"{v}_previous_day1"] for v in HOURLY_VARS}}
    d = hourly_to_daily_weather(hh)
    assert date(2025, 1, 1) not in d and date(2025, 1, 2) in d


def test_no_substitution_in_a_arm_missing_forecast_means_missing_day():
    complete = rc.fte.complete
    assert not complete(None)
    assert not complete({k: None for k in rc.WEATHER_KEYS})


def test_decision_rule_matches_declared_config():
    r = rc.CFG["decision_rule"]["recommend_for_forward_test_only_if_all"]
    assert any("2 percent lower" in x for x in r) and any("BOTH" in x for x in r)
    assert rc.SUDDEN == 0.30 and rc.SEEDS == (42, 7, 123, 2024, 99)


def test_windows_never_reach_untouched_period():
    for Y in rc.WINTERS:
        assert rc.wb.winter_window(Y)[1] <= rc.fte.HARD_STOP
    assert rc.fte.HARD_STOP < date(2026, 10, 15)


def test_load_source_merges_chunks_in_order(tmp_path, monkeypatch):
    monkeypatch.setattr(rc, "RAW", tmp_path)
    (tmp_path / "x").mkdir()
    for i, d0 in enumerate((date(2025, 1, 1), date(2025, 1, 3))):
        (tmp_path / "x" / f"{d0}_c.json").write_text(json.dumps({"hourly": _hourly(d0, 2, fill=float(i + 1))}))
    s = rc.load_source("x")
    assert sorted(s[1]) == [date(2025, 1, d) for d in (1, 2, 3, 4)]
    assert s[1][date(2025, 1, 1)]["temp"] == 1.0 and s[1][date(2025, 1, 4)]["temp"] == 2.0
    assert rc.load_source("missing") is None


def test_pooled_dates_keep_winters_apart():
    pr = {"w1": {"dates": [date(2025, 1, 1), date(2025, 1, 2)], "y": np.ones(2), "pm0": np.ones(2), "P": {"a": np.ones(2)}},
          "w2": {"dates": [date(2026, 1, 1)], "y": np.ones(1), "pm0": np.ones(1), "P": {"a": np.ones(1)}}}
    pdates, y, pm0, P = rc.pooled(pr)
    assert len(pdates) == 3 and (pdates[2] - pdates[1]).days >= 100 and len(P["a"]) == 3
