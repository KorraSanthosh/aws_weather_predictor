"""HTTP API Lambda: GET /forecast -> latest forecast + last 30 predictions (with actuals filled in)."""
import json
import os
from decimal import Decimal


def _plain(o):
    if isinstance(o, Decimal):
        return float(o)
    raise TypeError(type(o))


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
    body = {"forecast": {k: v for k, v in latest.items() if k not in ("pk", "sk")}, "history": history}
    return {"statusCode": 200,
            "headers": {"Content-Type": "application/json", "Cache-Control": "max-age=300"},
            "body": json.dumps(body, default=_plain)}
