"""HTTP API Lambda: GET /forecast -> latest forecast + last 30 predictions (with actuals filled in).

The latest forecast also carries `prediction_range`: the calibrated 90 % range around the stored point forecast
(core/prediction_range.py). It is computed here, when the forecast is read, from the stored `pm25` and `model_version` and the
packaged calibration file - the nightly forecast Lambda, the stored rows and the shadow log are not touched. If the range cannot
be given (unknown model version, missing calibration file, any error) the key is simply absent and the rest of the response is
exactly as before.
"""
import json
import logging
import os
from decimal import Decimal

log = logging.getLogger(__name__)


def _plain(o):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(type(o))


def _with_range(forecast):
    try:
        from core.prediction_range import attach_prediction_range
        return attach_prediction_range(forecast)
    except Exception:                      # the range is optional extra information; never break the API for it
        log.exception("prediction range unavailable")
        return forecast


def handler(event, context):
    import boto3
    from boto3.dynamodb.conditions import Key
    table = boto3.resource("dynamodb").Table(os.environ["TABLE_NAME"])
    items = table.query(KeyConditionExpression=Key("pk").eq("delhi"),
                        ScanIndexForward=False, Limit=31)["Items"]
    if not items:
        return {"statusCode": 404, "body": json.dumps({"error": "no forecast yet"})}
    latest, past = items[0], items[1:]
    history = [{"date": i["sk"], "predicted": i.get("pm25"), "actual": i.get("actual_pm25")}
               for i in reversed(past)]
    forecast = _with_range({k: v for k, v in latest.items() if k not in ("pk", "sk")})
    body = {"forecast": forecast, "history": history}
    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300"},
            "body": json.dumps(body, default=_plain)}
