"""Scheduled Lambda: fetch latest PM2.5 + weather forecast, predict tomorrow, store, alert.

Runs every night at ~23:00 IST (EventBridge). Environment variables:
    TABLE_NAME, TOPIC_ARN, OPENAQ_API_KEY, SENSOR_IDS (comma separated PM2.5 sensor ids),
    LAT, LON (default Delhi)
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from core import openaq
from core import weather as wx
from core.aqi import is_alert, pm25_to_aqi, pm25_to_category, school_guidance
from core.daily import IST, MIN_STATIONS, city_daily_mean, readings_to_daily
from core.features import FEATURE_NAMES, build_features

HERE = os.path.dirname(os.path.abspath(__file__))
_MODEL = None


def _load_model():
    global _MODEL
    if _MODEL is None:
        import xgboost as xgb
        # packaged build: <here>/model ; repo checkout: <repo>/model
        model_dir = os.path.join(HERE, "model")
        if not os.path.exists(os.path.join(model_dir, "model.json")):
            model_dir = os.path.join(HERE, "..", "model")
        booster = xgb.Booster()
        booster.load_model(os.path.join(model_dir, "model.json"))
        with open(os.path.join(model_dir, "model_meta.json")) as f:
            meta = json.load(f)
        if meta["features"] != FEATURE_NAMES:
            raise RuntimeError("model features do not match core.features - retrain the model")
        _MODEL = (booster, meta)
    return _MODEL


def predict_pm25(features: list[float]) -> tuple[float, float, float]:
    import numpy as np
    import xgboost as xgb
    booster, meta = _load_model()
    dm = xgb.DMatrix(np.array([features], dtype=float), feature_names=FEATURE_NAMES)
    # the model predicts the log-change from today's PM2.5 (same as training)
    pm = float(np.expm1(math.log1p(features[FEATURE_NAMES.index("pm_d0")]) + booster.predict(dm)[0]))
    return pm, pm * math.exp(meta["log_resid_q10"]), pm * math.exp(meta["log_resid_q90"])


def _to_dynamo(obj):
    return json.loads(json.dumps(obj), parse_float=Decimal)


def run_forecast(now_utc: datetime, api_key: str, sensor_ids: list[int], lat: float, lon: float,
                 lookback_days: int = 16) -> dict:   # >= 14 days: the model uses PM2.5 lags up to 13 days back
    """Pure function (no AWS calls) so it is easy to test."""
    # 1. PM2.5 for the last days
    start = now_utc - timedelta(days=lookback_days)
    per_station = []
    for sid in sensor_ids:
        per_station.append(readings_to_daily(openaq.fetch_sensor_hours(sid, start, now_utc, api_key)))
    # same rule as training: a day only counts if enough stations reported
    city = city_daily_mean(per_station, min_stations=min(MIN_STATIONS, len(sensor_ids)))
    if not city:
        raise RuntimeError("no valid PM2.5 days with enough stations - check sensor ids / API key")

    # 2. Weather (analysis for the past days + forecast for tomorrow)
    weather = wx.fetch_forecast_weather(lat, lon, past_days=lookback_days, forecast_days=3)

    today = now_utc.astimezone(IST).date()
    d0 = max(d for d in city if d <= today)                 # latest day with a valid 24h mean
    if (today - d0).days > 2:
        raise RuntimeError(f"newest valid PM2.5 day is {d0}, too old to forecast from")
    target = d0 + timedelta(days=1)
    if target not in weather:
        raise RuntimeError(f"no weather forecast for {target}")

    history = {}
    for d in set(city) | set(weather):
        rec = dict(weather.get(d, {}))
        if d in city:
            rec["pm25"] = city[d][0]
        history[d] = rec

    feats = build_features(history, d0, weather[target])
    pm, lo, hi = predict_pm25(feats)
    category = pm25_to_category(pm)
    return {
        "target_date": target.isoformat(),
        "generated_for": d0.isoformat(),
        "generated_at": now_utc.isoformat(),
        "pm25": round(pm, 1), "pm25_low": round(lo, 1), "pm25_high": round(hi, 1),
        "aqi": pm25_to_aqi(pm), "category": category,
        "guidance": school_guidance(category),
        "today_pm25": round(city[d0][0], 1),
        "n_stations": city[d0][1],
        "actuals": {d.isoformat(): round(v[0], 1) for d, v in city.items()},
    }


def handler(event, context):
    import boto3
    now = datetime.now(timezone.utc)
    sensor_ids = [int(s) for s in os.environ["SENSOR_IDS"].split(",") if s.strip()]
    result = run_forecast(now, os.environ["OPENAQ_API_KEY"], sensor_ids,
                          float(os.environ.get("LAT", 28.6139)), float(os.environ.get("LON", 77.2090)))

    table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    actuals = result.pop("actuals")

    # Fill in 'actual' for earlier predictions now that those days are complete
    for day, value in actuals.items():
        try:
            table.update_item(
                Key={"pk": "delhi", "sk": day},
                UpdateExpression="SET actual_pm25 = :a",
                ConditionExpression="attribute_exists(sk)",
                ExpressionAttributeValues={":a": Decimal(str(value))})
        except Exception:        # no prediction stored for that day - fine
            pass

    table.put_item(Item=_to_dynamo({"pk": "delhi", "sk": result["target_date"], **result}))

    if is_alert(result["category"]) and os.environ.get("TOPIC_ARN"):
        boto3.client("sns").publish(
            TopicArn=os.environ["TOPIC_ARN"],
            Subject=f"Air quality alert for {result['target_date']}: {result['category']}",
            Message=(f"Predicted PM2.5 {result['pm25']} ug/m3 (AQI ~{result['aqi']}, {result['category']}).\n"
                     f"{result['guidance']['headline']}. {result['guidance']['advice']}"))
    return {"ok": True, "target_date": result["target_date"], "category": result["category"]}
