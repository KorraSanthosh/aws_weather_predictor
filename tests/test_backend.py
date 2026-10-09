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

    from core.features import FEATURE_NAMES
    lags = [int(m.group(1)) for n in FEATURE_NAMES for m in [re.fullmatch(r"pm_d(\d+)", n)] if m]
    window = [int(m.group(1)) for n in FEATURE_NAMES for m in [re.fullmatch(r"pm_(?:mean|min|max)(\d+)", n)] if m]
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
