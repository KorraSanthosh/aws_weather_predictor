"""End-to-end test of the forecast Lambda with all network calls faked.

Requires model/model.json (run `python scripts/make_synthetic_data.py && python model/train.py` first).
"""
import json
import sys
from datetime import date, datetime, timedelta, timezone
from decimal import Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

pytestmark = pytest.mark.skipif(not (ROOT / "model" / "model.json").exists(),
                                reason="train a model first")

import forecast_handler as fh  # noqa: E402
from core.daily import IST  # noqa: E402
from core.weather import WEATHER_KEYS  # noqa: E402

NOW = datetime(2026, 10, 8, 17, 30, tzinfo=timezone.utc)   # 23:00 IST on 8 Oct 2026


def fake_hours(sensor_id, dt_from, dt_to, api_key):
    out, t = [], dt_from.replace(minute=0, second=0, microsecond=0)
    while t <= dt_to:
        day_factor = 100 + 10 * (t.astimezone(IST).date().toordinal() % 5)
        out.append((t, day_factor + sensor_id))
        t += timedelta(hours=1)
    return out


def fake_weather(lat, lon, past_days=10, forecast_days=3):
    today = date(2026, 10, 8)
    return {today + timedelta(days=i): {k: 10.0 + i for k in WEATHER_KEYS}
            for i in range(-past_days, forecast_days)}


@pytest.fixture(autouse=True)
def patch_network(monkeypatch):
    monkeypatch.setattr(fh.openaq, "fetch_sensor_hours", fake_hours)
    monkeypatch.setattr(fh.wx, "fetch_forecast_weather", fake_weather)


def test_run_forecast_end_to_end():
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["target_date"] == "2026-10-09"
    assert res["pm25_low"] <= res["pm25"] <= res["pm25_high"]
    assert 0 <= res["aqi"] <= 500
    assert res["category"] in ("Good", "Satisfactory", "Moderate", "Poor", "Very Poor", "Severe")
    assert res["guidance"]["decision"] in ("GO", "CAUTION", "INDOOR")
    assert res["n_stations"] == 2


def test_run_forecast_needs_enough_stations(monkeypatch):
    # only 1 of 3 sensors reports -> the "city" definition used in training is not met
    monkeypatch.setattr(fh.openaq, "fetch_sensor_hours",
                        lambda sid, a, b, k: fake_hours(sid, a, b, k) if sid == 1 else [])
    with pytest.raises(RuntimeError):
        fh.run_forecast(NOW, "key", [1, 2, 3], 28.6, 77.2)


def test_run_forecast_refuses_stale_data(monkeypatch):
    monkeypatch.setattr(fh.openaq, "fetch_sensor_hours", lambda *a, **k: [])
    with pytest.raises(RuntimeError):
        fh.run_forecast(NOW, "key", [1], 28.6, 77.2)


def test_dynamodb_conversion_has_no_floats():
    item = fh._to_dynamo({"a": 1.5, "nested": {"b": 2.25}})
    assert isinstance(item["a"], Decimal) and isinstance(item["nested"]["b"], Decimal)


def test_api_handler_shapes_response(monkeypatch):
    import api_handler

    class FakeTable:
        def query(self, **kw):
            return {"Items": [
                {"pk": "delhi", "sk": "2026-10-09", "pm25": Decimal("150.5"), "category": "Poor"},
                {"pk": "delhi", "sk": "2026-10-08", "pm25": Decimal("140"), "actual_pm25": Decimal("138")},
            ]}

    class FakeResource:
        def Table(self, name):
            return FakeTable()

    import boto3
    monkeypatch.setattr(boto3, "resource", lambda *a, **k: FakeResource())
    monkeypatch.setenv("TABLE_NAME", "t")
    resp = api_handler.handler({}, None)
    body = json.loads(resp["body"])
    assert resp["statusCode"] == 200
    assert body["forecast"]["pm25"] == 150.5
    assert body["history"] == [{"date": "2026-10-08", "predicted": 140.0, "actual": 138.0}]


def test_lookback_covers_every_lag_the_model_uses():
    """Train/serve skew guard: the Lambda must fetch at least as many days as the largest PM2.5 lag."""
    import inspect
    import re

    from core.features import FEATURE_SETS
    allf = [n for v in FEATURE_SETS.values() for n in v]
    lags = [int(m.group(1)) for n in allf for m in [re.fullmatch(r"pm_d(\d+)", n)] if m]
    window = [int(m.group(1)) for n in allf for m in [re.fullmatch(r"pm_(?:mean|min|max)(\d+)", n)] if m]
    need = max(lags + window + [0]) + 1
    default = inspect.signature(fh.run_forecast).parameters["lookback_days"].default
    assert default >= need


def test_live_features_are_complete_for_the_deployed_model():
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["pm25"] > 0                 # NaN features would still predict; check the lag-13 value was seen
    from core.daily import city_daily_mean, readings_to_daily
    start = NOW - timedelta(days=16)
    daily = city_daily_mean([readings_to_daily(fake_hours(1, start, NOW, "k"))])
    assert len([d for d in daily if d <= date(2026, 10, 8)]) >= 14


def test_live_daypart_features_are_present(monkeypatch):
    """Train/serve guard: with hourly data the Lambda must produce every one of the candidate's 35 features (no NaN)."""
    import math
    from core.features import FEATURE_SETS
    monkeypatch.setenv("MODEL_VERSION", "v2_35_candidate")
    FEATURE_NAMES = FEATURE_SETS["v2_35_candidate"]
    seen = {}

    def spy(features):
        seen["f"] = features
        return 100.0, 80.0, 120.0

    monkeypatch.setattr(fh, "predict_pm25", spy)
    fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    missing = [n for n, v in zip(FEATURE_NAMES, seen["f"]) if math.isnan(v)]
    assert not missing, f"features missing live: {missing}"


# ---------------------------------------------------------------- model versions (v1 baseline / v2 candidate)
VERSIONS = ["v1_25_baseline", "v2_35_candidate"]


def _need(version):
    if not (ROOT / "model" / version / "model.json").exists():
        pytest.skip(f"{version} artifacts not present")


def test_default_model_is_the_25_feature_baseline(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["model_version"] == "v1_25_baseline" and res["n_features"] == 25
    assert res["missing_features"] == []


@pytest.mark.parametrize("version", VERSIONS)
def test_each_stored_version_runs_end_to_end(monkeypatch, version):
    _need(version)
    monkeypatch.setenv("MODEL_VERSION", version)
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["model_version"] == version
    assert res["n_features"] == (25 if version.startswith("v1") else 35)
    assert res["pm25_low"] <= res["pm25"] <= res["pm25_high"] and res["pm25"] > 0
    assert res["missing_features"] == []             # fake data has hourly readings every day


def test_candidate_without_hourly_data_reports_missing_not_invented(monkeypatch):
    """No usable hourly day -> the 7 day-part inputs are NaN and listed in `missing_features`; the forecast still runs."""
    _need("v2_35_candidate")
    from core.features import DAYPART_FEATURES
    monkeypatch.setenv("MODEL_VERSION", "v2_35_candidate")
    monkeypatch.setattr(fh, "daypart_stats", lambda hours, *a, **k: {})
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert sorted(res["missing_features"]) == sorted(DAYPART_FEATURES)
    assert res["pm25"] > 0


def test_baseline_ignores_missing_hourly_data(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v1_25_baseline")
    monkeypatch.setattr(fh, "daypart_stats", lambda hours, *a, **k: {})
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["missing_features"] == [] and res["n_features"] == 25


def test_unknown_model_version_fails_loudly(monkeypatch):
    monkeypatch.setenv("MODEL_VERSION", "v9_does_not_exist")
    with pytest.raises(RuntimeError):
        fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)


@pytest.mark.parametrize("version", VERSIONS)
def test_forecast_ignores_everything_after_the_forecast_origin(monkeypatch, version):
    """No look-ahead: weather for days AFTER the target day must not change the forecast. (Readings after 'now'
    cannot exist; the fake API never returns any.)"""
    _need(version)
    monkeypatch.setenv("MODEL_VERSION", version)
    base = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)

    def later_changed(lat, lon, past_days=10, forecast_days=3):
        out = fake_weather(lat, lon, past_days, forecast_days)
        for d in out:
            if d > date(2026, 10, 9):
                out[d] = {k: 999.0 for k in WEATHER_KEYS}
        return out

    monkeypatch.setattr(fh.wx, "fetch_forecast_weather", later_changed)
    again = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert again["pm25"] == base["pm25"]


# ---------------------------------------------------------------- hybrid + shadow (forward test, OFF by default)
ALL3 = ["v1_25_baseline", "v2_35_candidate", "v3_hybrid"]


def _need_all():
    _need("v1_25_baseline")
    _need("v2_35_candidate")


def _run(shadow=ALL3):
    return fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2, shadow_versions=shadow)


def test_shadow_is_off_by_default(monkeypatch):
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    assert "shadow" not in fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)


def test_shadow_versions_see_identical_inputs_and_do_not_change_the_primary(monkeypatch):
    _need_all()
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    plain = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    res = _run()
    assert res["pm25"] == plain["pm25"] and res["model_version"] == "v1_25_baseline"     # primary untouched
    sh = res["shadow"]
    assert set(sh) == set(ALL3) and not any("error" in r for r in sh.values())
    assert len({r["inputs_sha256"] for r in sh.values()}) == 1                           # same inputs for all
    assert len({(r["target_date"], r["generated_for"], r["generated_at"], r["today_pm25"]) for r in sh.values()}) == 1
    assert sh["v1_25_baseline"]["pm25"] == round(plain["pm25"], 2) or abs(sh["v1_25_baseline"]["pm25"] - plain["pm25"]) < 0.06
    assert sh["v1_25_baseline"]["n_features"] == 25 and sh["v2_35_candidate"]["n_features"] == 35
    assert sh["v3_hybrid"]["served_model"] == "v2_35_candidate" and sh["v3_hybrid"]["daypart_valid"] is True
    assert sh["v3_hybrid"]["pm25"] == sh["v2_35_candidate"]["pm25"]


def test_each_shadow_forecast_is_independent_of_which_others_run(monkeypatch):
    """One version's number must not depend on the other versions or the order they run in."""
    _need_all()
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    full = _run(ALL3)["shadow"]
    for v in ALL3:
        alone = _run([v])["shadow"][v]
        assert alone["pm25"] == full[v]["pm25"] and alone["inputs_sha256"] == full[v]["inputs_sha256"]
    rev = _run(list(reversed(ALL3)))["shadow"]
    assert all(rev[v]["pm25"] == full[v]["pm25"] for v in ALL3)


def test_shadow_does_not_modify_the_shared_history(monkeypatch):
    import copy
    _need_all()
    real = fh.shadow_forecasts
    seen = {}

    def spy(history, d0, wx1, versions):
        before = copy.deepcopy(history)
        out = real(history, d0, wx1, versions)
        seen["same"] = (before.keys() == history.keys()
                        and all(repr(before[k]) == repr(history[k]) for k in before))
        return out

    monkeypatch.setattr(fh, "shadow_forecasts", spy)
    _run()
    assert seen["same"] is True


def test_hybrid_falls_back_to_v1_when_any_daypart_input_is_missing(monkeypatch):
    _need_all()
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    monkeypatch.setattr(fh, "daypart_stats", lambda hours, *a, **k: {})
    sh = _run()["shadow"]
    assert sh["v3_hybrid"]["served_model"] == "v1_25_baseline" and sh["v3_hybrid"]["daypart_valid"] is False
    assert sh["v3_hybrid"]["pm25"] == sh["v1_25_baseline"]["pm25"]
    assert sh["v3_hybrid"]["missing_features"] == []
    assert len(sh["v2_35_candidate"]["missing_features"]) == 7        # v2 still runs, with the gaps reported


def test_one_failing_shadow_version_is_recorded_and_isolated(monkeypatch):
    _need_all()
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    real = fh._predict

    def flaky(booster, meta, features):
        if meta.get("model_version") == "v2_35_candidate":
            raise RuntimeError("boom")
        return real(booster, meta, features)

    monkeypatch.setattr(fh, "_predict", flaky)
    res = _run(["v1_25_baseline", "v2_35_candidate"])
    assert "error" in res["shadow"]["v2_35_candidate"] and "pm25" in res["shadow"]["v1_25_baseline"]
    assert res["pm25"] > 0


def test_primary_can_serve_the_hybrid(monkeypatch):
    _need_all()
    monkeypatch.setenv("MODEL_VERSION", "v3_hybrid")
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["model_version"] == "v3_hybrid" and res["served_model"] == "v2_35_candidate"
    monkeypatch.setattr(fh, "daypart_stats", lambda hours, *a, **k: {})
    res = fh.run_forecast(NOW, "key", [1, 2], 28.6, 77.2)
    assert res["served_model"] == "v1_25_baseline" and res["missing_features"] == []


def test_parse_shadow_versions_ignores_unknown_names():
    assert fh.parse_shadow_versions("") == []
    assert fh.parse_shadow_versions(" v1_25_baseline, nope ,v3_hybrid,v1_25_baseline") == ["v1_25_baseline", "v3_hybrid"]


class _Table:
    def __init__(self):
        self.items, self.updates = {}, []

    def put_item(self, Item, ConditionExpression=None):
        key = (Item["pk"], Item["sk"])
        if ConditionExpression and key in self.items:
            raise RuntimeError("ConditionalCheckFailedException")
        self.items[key] = Item

    def update_item(self, **kw):
        self.updates.append(kw)


def _handler_env(monkeypatch, table, shadow_env):
    import boto3
    real = fh.run_forecast
    monkeypatch.setattr(fh, "run_forecast", lambda now, *a, **k: real(NOW, *a, **k))
    monkeypatch.setattr(boto3, "resource", lambda *a, **k: type("R", (), {"Table": lambda self, n: table})())
    monkeypatch.setenv("SENSOR_IDS", "1,2")
    monkeypatch.setenv("OPENAQ_API_KEY", "x")
    monkeypatch.setenv("TABLE_NAME", "t")
    monkeypatch.delenv("TOPIC_ARN", raising=False)
    monkeypatch.delenv("MODEL_VERSION", raising=False)
    if shadow_env is None:
        monkeypatch.delenv("SHADOW_VERSIONS", raising=False)
    else:
        monkeypatch.setenv("SHADOW_VERSIONS", shadow_env)


def test_handler_without_shadow_env_writes_only_the_primary_row(monkeypatch):
    t = _Table()
    _handler_env(monkeypatch, t, None)
    assert fh.handler({}, None)["ok"]
    assert {pk for pk, _ in t.items} == {"delhi"}


def test_handler_logs_shadow_rows_first_write_wins(monkeypatch):
    _need_all()
    t = _Table()
    _handler_env(monkeypatch, t, "v1_25_baseline,v2_35_candidate,v3_hybrid")
    fh.handler({}, None)
    shadow_keys = sorted(sk for pk, sk in t.items if pk == "shadow")
    assert [k.split("#")[1] for k in shadow_keys] == sorted(ALL3)
    assert ("delhi", "2026-10-09") in t.items
    first = dict(t.items[("shadow", shadow_keys[0])])
    fh.handler({}, None)                                           # re-run the same night
    assert t.items[("shadow", shadow_keys[0])] == first            # nothing was overwritten
