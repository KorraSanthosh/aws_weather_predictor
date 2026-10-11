"""Leakage and consistency guards for the 2025 backtest (use the committed metrics.json; no network)."""
import hashlib, json, sys
from datetime import date, timedelta
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import run_backtest as rb

M = json.loads((HERE / "metrics.json").read_text())


def test_constants():
    assert rb.TRAIN_END == date(2024, 12, 31) and rb.TEST_START == date(2025, 1, 1) and rb.TEST_END == date(2025, 12, 31)


def test_training_targets_end_at_cutoff():
    t = M["training"]
    assert date.fromisoformat(t["last_target"]) <= rb.TRAIN_END and t["origins"] >= 365


def test_all_test_days_in_2025_and_identical_for_every_model():
    for lead in ("lead1", "lead2"):
        b = M["test"][lead]
        ds = [date.fromisoformat(d) for d in b["daily"]["dates"]]
        assert ds == sorted(set(ds)) and ds[0] >= rb.TEST_START and ds[-1] <= rb.TEST_END
        n = len(ds)
        assert n == b["n_matched_days"] == len(b["daily"]["actual"])
        for m in rb.MODELS:
            assert len(b["daily"][m]) == n and np.isfinite(b["daily"][m]).all()      # every model scored on every matched day


def test_no_row_reads_history_after_the_origin_day():
    for lead in ("lead1", "lead2"):
        a = M["test"][lead]["leakage_audit"]
        assert a["rows_reading_dates_after_origin"] == [] and a["max_history_date_read_is_origin_day"]
        assert a["forecast_days_equal_to_observed_ERA5"] == 0           # forecast weather was not silently replaced by observations
        assert a["live_23h_pm25_used_as_today"]


def test_no_scoreable_day_dropped_silently():
    for lead in ("lead1", "lead2"):
        b = M["test"][lead]
        assert sum(b["eligibility_ladder_all_365_days"].values()) == 365
        assert b["eligibility_ladder_all_365_days"].get("scoreable") == b["n_matched_days"]


def test_spy_detects_future_reads():
    h = rb.Spy({date(2025, 1, 1): 1, date(2025, 1, 3): 3})
    h[date(2025, 1, 1)]; h.get(date(2025, 1, 3))
    assert any(k > date(2025, 1, 2) for k in h.seen)


def test_saved_models_match_manifest():
    man = json.loads((HERE / "models" / "sha256.json").read_text())
    for tag in ("v1", "v2"):
        assert hashlib.sha256((HERE / "models" / f"{tag}_seed42.json").read_bytes()).hexdigest() == man[tag]


def test_predictions_are_not_fed_actuals():
    b = M["test"]["lead1"]["daily"]
    y = np.array(b["actual"])
    for m in ("v1", "v2", "hybrid"):
        assert not np.array_equal(np.array(b[m]), y)
    assert np.mean(np.abs(np.array(b["hybrid"]) - y)) > 1.0                # a leaked target would give ~0 error


def test_month_season_splits_cover_all_days():
    b = M["test"]["lead1"]
    assert sum(v["n_days"] for v in b["by_month"].values()) == b["n_matched_days"]
    assert sum(v["n_days"] for v in b["by_season"].values()) == b["n_matched_days"]


def test_training_rows_unchanged_when_all_post_cutoff_data_is_destroyed():
    """Canary: corrupt every PM2.5/weather value dated after 2024-12-31; the training set must be bit-identical."""
    import copy
    sys.path.insert(0, str(HERE.parents[1] / "model")); sys.path.insert(0, str(HERE.parents[1] / "pipeline"))
    import forecast_training_experiment as fte, option_screen as osn
    from core.weather import WEATHER_KEYS
    root = HERE.parents[1]
    import os
    cwd = os.getcwd(); os.chdir(root)
    try:
        hist = osn.load_hist("data/daily_opencity.csv")
        live = fte.build_live_view(Path("data/raw"), extra_subs=("opencity_hourly",))
        era = lambda h: (lambda t: {k: h[t].get(k) for k in WEATHER_KEYS})
        a = {t: v for t, v in fte.build(hist, live, era(hist), view="offline").items() if t <= rb.TRAIN_END}
        h2 = copy.deepcopy(hist)
        for d in h2:
            if d > rb.TRAIN_END:
                h2[d] = {k: (99999.0 if isinstance(v, (int, float)) and v is not None else v) for k, v in h2[d].items()}
        live2 = {d: (dict(v, pm25=99999.0) if d > rb.TRAIN_END else v) for d, v in live.items()}
        b = {t: v for t, v in fte.build(h2, live2, era(h2), view="offline").items() if t <= rb.TRAIN_END}
    finally:
        os.chdir(cwd)
    assert a.keys() == b.keys() and len(a) > 1000
    assert all(a[t]["y"] == b[t]["y"] and a[t]["row"] == b[t]["row"] for t in a)
