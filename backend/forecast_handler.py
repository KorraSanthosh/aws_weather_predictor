"""Scheduled Lambda: fetch latest PM2.5 + weather forecast, predict tomorrow, store, alert.

Runs every night at ~23:00 IST (EventBridge). Environment variables:
    TABLE_NAME, TOPIC_ARN, OPENAQ_API_KEY, SENSOR_IDS (comma separated PM2.5 sensor ids),
    LAT, LON (default Delhi)
    MODEL_VERSION (optional): which stored model to use, e.g. v1_25_baseline or v2_35_candidate
        (folders under model/, see model/MODEL_VERSIONS.md). Unset = the packaged default model/model.json.
The feature list always comes from the loaded model's own model_meta.json, so a 25- and a 35-feature
model both work without code changes; features the live data cannot supply are passed as NaN (never
invented) and reported in `missing_features`.
    MODEL_VERSION=v3_hybrid routes each night to v2 when all 7 day-part inputs exist, else v1 (core/hybrid.py).
    SHADOW_VERSIONS (optional, OFF by default): comma list of v1_25_baseline,v2_35_candidate,v3_hybrid. When set,
        each night every listed version also forecasts from the SAME inputs and is logged under pk="shadow"
        (first write wins). It never changes the primary forecast, the alert, or the dashboard; a shadow failure
        is recorded and ignored. See model/EVALUATION_PROTOCOL.md.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from core import openaq
from core import weather as wx
from core.aqi import is_alert, pm25_to_aqi, pm25_to_category, school_guidance
from core.daily import (IST, MIN_STATIONS, city_daily_mean, city_hourly, daypart_stats,
                        readings_to_daily, readings_to_hourly)
from core.features import ALL_FEATURES, FEATURE_SETS, build_features, feature_dict
from core.hybrid import HYBRID_VERSION, daypart_valid, hybrid_pick

HERE = os.path.dirname(os.path.abspath(__file__))
_MODELS: dict = {}          # version (None = packaged default) -> (booster, meta)
log = logging.getLogger(__name__)
SHADOW_PARTITION = "shadow"


def _model_dir(version):
    # packaged build: <here>/model ; repo checkout: <repo>/model
    base = os.path.join(HERE, "model")
    if not os.path.isdir(base):
        base = os.path.join(HERE, "..", "model")
    if version:
        d = os.path.join(base, version)
        if not os.path.exists(os.path.join(d, "model.json")):
            raise RuntimeError(f"MODEL_VERSION={version!r}: no model.json in {d}")
        return d
    return base


def _load_model(version=None):
    """Load a model once per process. Returns (booster, meta); meta['features'] drives feature building.
    version=None uses $MODEL_VERSION, or the packaged default model/model.json when that is unset."""
    if version is None:
        version = os.environ.get("MODEL_VERSION") or None
    if version not in _MODELS:
        import xgboost as xgb
        model_dir = _model_dir(version)
        booster = xgb.Booster()
        booster.load_model(os.path.join(model_dir, "model.json"))
        with open(os.path.join(model_dir, "model_meta.json")) as f:
            meta = json.load(f)
        unknown = [f for f in meta["features"] if f not in ALL_FEATURES]
        if unknown:
            raise RuntimeError(f"model uses features this code cannot build: {unknown} - retrain the model")
        if not meta.get("model_version"):          # artifacts stored before versioning: identify by feature list
            meta["model_version"] = next((k for k, f in FEATURE_SETS.items() if f == meta["features"]), "unversioned")
        v = meta.get("model_version")
        if v in FEATURE_SETS and meta["features"] != FEATURE_SETS[v]:
            raise RuntimeError(f"model_meta features differ from core.features.FEATURE_SETS[{v!r}]")
        if "pm_d0" not in meta["features"]:
            raise RuntimeError("model must include pm_d0 (it predicts the change from today's PM2.5)")
        _MODELS[version] = (booster, meta)
    return _MODELS[version]


def _predict(booster, meta, features):
    import numpy as np
    import xgboost as xgb
    names = meta["features"]
    dm = xgb.DMatrix(np.array([features], dtype=float), feature_names=names)   # NaN = missing, as in training
    # the model predicts the log-change from today's PM2.5 (same as training)
    pm = float(np.expm1(math.log1p(features[names.index("pm_d0")]) + booster.predict(dm)[0]))
    return pm, pm * math.exp(meta["log_resid_q10"]), pm * math.exp(meta["log_resid_q90"])


def predict_pm25(features: list[float], version=None) -> tuple[float, float, float]:
    booster, meta = _load_model(version)
    return _predict(booster, meta, features)


def _inputs_fingerprint(row: dict) -> str:
    """Hash of every candidate input for this forecast, so a shadow log can prove all versions saw the same data."""
    clean = {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in sorted(row.items())}
    return hashlib.sha256(json.dumps(clean, sort_keys=True).encode()).hexdigest()


def shadow_forecasts(history: dict, d0, wx1: dict, versions) -> dict:
    """Forecast tomorrow with each listed version from the SAME history/weather. Each version loads its own booster and
    builds its own features from a read-only history, so no version can influence another. Errors are recorded per
    version and never raised."""
    out = {}
    try:
        row = feature_dict(history, d0, wx1)
        dp_ok, fingerprint = daypart_valid(row), _inputs_fingerprint(row)
    except Exception as e:                                    # pragma: no cover - defensive
        return {v: {"error": f"{type(e).__name__}: {e}"} for v in versions}
    for v in versions:
        try:
            served = hybrid_pick(row) if v == HYBRID_VERSION else v
            booster, meta = _load_model(served)
            names = meta["features"]
            feats = [row[n] for n in names]
            pm, lo, hi = _predict(booster, meta, feats)
            out[v] = {"model_version": v, "served_model": served, "pm25": round(pm, 2),
                      "pm25_low": round(lo, 2), "pm25_high": round(hi, 2),
                      "aqi": pm25_to_aqi(pm), "category": pm25_to_category(pm),
                      "n_features": len(names), "daypart_valid": dp_ok, "inputs_sha256": fingerprint,
                      "missing_features": [n for n, x in zip(names, feats) if isinstance(x, float) and math.isnan(x)]}
        except Exception as e:
            log.exception("shadow forecast failed for %s", v)
            out[v] = {"error": f"{type(e).__name__}: {e}"}
    return out


def _to_dynamo(obj):
    return json.loads(json.dumps(obj), parse_float=Decimal)


def run_forecast(now_utc: datetime, api_key: str, sensor_ids: list[int], lat: float, lon: float,
                 lookback_days: int = 16, shadow_versions=()) -> dict:   # >= 14 days: the model uses PM2.5 lags up to 13 days back
    """Pure function (no AWS calls) so it is easy to test."""
    # 1. PM2.5 for the last days
    start = now_utc - timedelta(days=lookback_days)
    per_station, per_station_hourly = [], []
    for sid in sensor_ids:
        readings = openaq.fetch_sensor_hours(sid, start, now_utc, api_key)
        per_station.append(readings_to_daily(readings))
        per_station_hourly.append(readings_to_hourly(readings))      # for the day-part features
    hourly = city_hourly(per_station_hourly)
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
            rec.update(daypart_stats(hourly.get(d)))
        history[d] = rec

    wx1 = weather[target]
    version = os.environ.get("MODEL_VERSION") or None
    served = hybrid_pick(feature_dict(history, d0, wx1)) if version == HYBRID_VERSION else version
    _, meta = _load_model(served)
    names = meta["features"]
    feats = build_features(history, d0, wx1, names=names)
    missing = [n for n, v in zip(names, feats) if isinstance(v, float) and math.isnan(v)]
    pm, lo, hi = predict_pm25(feats) if version != HYBRID_VERSION else predict_pm25(feats, served)
    category = pm25_to_category(pm)
    result = {
        "target_date": target.isoformat(),
        "generated_for": d0.isoformat(),
        "generated_at": now_utc.isoformat(),
        "pm25": round(pm, 1), "pm25_low": round(lo, 1), "pm25_high": round(hi, 1),
        "aqi": pm25_to_aqi(pm), "category": category,
        "guidance": school_guidance(category),
        "today_pm25": round(city[d0][0], 1),
        "n_stations": city[d0][1],
        "model_version": HYBRID_VERSION if version == HYBRID_VERSION else meta.get("model_version", "unversioned"),
        "served_model": meta.get("model_version", "unversioned"),
        "n_features": len(names),
        "missing_features": missing,            # NaN inputs (e.g. day-parts when hourly data is short); never filled in
        "actuals": {d.isoformat(): round(v[0], 1) for d, v in city.items()},
    }
    if shadow_versions:
        shadow = shadow_forecasts(history, d0, wx1, list(shadow_versions))
        base = {"target_date": result["target_date"], "generated_for": result["generated_for"],
                "generated_at": result["generated_at"], "today_pm25": result["today_pm25"],
                "n_stations": result["n_stations"]}
        result["shadow"] = {v: {**base, **r} for v, r in shadow.items()}
    return result


ALLOWED_SHADOW = ("v1_25_baseline", "v2_35_candidate", HYBRID_VERSION)


def parse_shadow_versions(text: str) -> list[str]:
    out = []
    for v in (t.strip() for t in text.split(",")):
        if not v:
            continue
        if v not in ALLOWED_SHADOW:
            log.error("SHADOW_VERSIONS: ignoring unknown version %r", v)
        elif v not in out:
            out.append(v)
    return out


def store_shadow(table, shadow: dict, actuals: dict) -> None:
    """Log shadow forecasts under pk="shadow", sk="<target date>#<version>". FIRST WRITE WINS: a forecast that was
    logged is never replaced (so it cannot be edited after the actual is known). Today's actuals are attached to
    earlier shadow rows exactly like the primary rows. Nothing here can raise into the primary forecast."""
    for v, rec in shadow.items():
        try:
            table.put_item(Item=_to_dynamo({"pk": SHADOW_PARTITION, "sk": f"{rec['target_date']}#{v}", **rec}),
                           ConditionExpression="attribute_not_exists(sk)")
        except Exception as e:      # already logged (conditional check failed) or a DynamoDB hiccup
            log.warning("shadow row for %s not written: %s", v, e)
    for day, value in actuals.items():
        for v in ALLOWED_SHADOW:
            try:
                table.update_item(
                    Key={"pk": SHADOW_PARTITION, "sk": f"{day}#{v}"},
                    UpdateExpression="SET actual_pm25 = :a",
                    ConditionExpression="attribute_exists(sk) AND attribute_not_exists(actual_pm25)",
                    ExpressionAttributeValues={":a": Decimal(str(value))})
            except Exception:
                pass


def handler(event, context):
    import boto3
    now = datetime.now(timezone.utc)
    sensor_ids = [int(s) for s in os.environ["SENSOR_IDS"].split(",") if s.strip()]
    shadow_versions = parse_shadow_versions(os.environ.get("SHADOW_VERSIONS", ""))
    result = run_forecast(now, os.environ["OPENAQ_API_KEY"], sensor_ids,
                          float(os.environ.get("LAT", 28.6139)), float(os.environ.get("LON", 77.2090)),
                          shadow_versions=shadow_versions)
    shadow = result.pop("shadow", {})

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
    store_shadow(table, shadow, actuals)         # no-op unless SHADOW_VERSIONS is set; never raises

    if is_alert(result["category"]) and os.environ.get("TOPIC_ARN"):
        boto3.client("sns").publish(
            TopicArn=os.environ["TOPIC_ARN"],
            Subject=f"Air quality alert for {result['target_date']}: {result['category']}",
            Message=(f"Predicted PM2.5 {result['pm25']} ug/m3 (AQI ~{result['aqi']}, {result['category']}).\n"
                     f"{result['guidance']['headline']}. {result['guidance']['advice']}"))
    return {"ok": True, "target_date": result["target_date"], "category": result["category"]}
