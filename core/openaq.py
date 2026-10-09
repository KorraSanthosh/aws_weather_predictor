"""OpenAQ v3 API helpers (needs a free API key: https://explore.openaq.org/register).

NOTE: written from the public v3 docs; response shapes are parsed
defensively, but verify against the live API on day one (see README).
"""
from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

BASE = "https://api.openaq.org/v3"


def _get(path: str, params: dict, api_key: str, retries: int = 4, timeout: int = 30) -> dict:
    url = f"{BASE}{path}?{urllib.parse.urlencode(params)}"
    req = urllib.request.Request(url, headers={"X-API-Key": api_key, "Accept": "application/json"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as exc:
            last = exc
            time.sleep(2.0 * (attempt + 1))  # also covers HTTP 429 (60 req/min limit)
    raise RuntimeError(f"OpenAQ GET {path} failed: {last}")


def _parse_utc(obj) -> datetime | None:
    """Accepts {'utc': '...Z', 'local': '...'} or a plain ISO string."""
    s = obj.get("utc") if isinstance(obj, dict) else obj
    if not s:
        return None
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def find_pm25_locations(api_key: str, bbox: str, limit: int = 1000) -> list[dict]:
    """bbox = 'min_lon,min_lat,max_lon,max_lat'. parameters_id=2 is PM2.5."""
    data = _get("/locations", {"bbox": bbox, "parameters_id": 2, "limit": limit}, api_key)
    out = []
    for loc in data.get("results", []):
        pm_sensor = None
        for s in loc.get("sensors", []):
            p = s.get("parameter", {}) or {}
            if str(p.get("name", "")).lower() in ("pm25", "pm2.5", "pm2_5"):
                pm_sensor = s.get("id")
        out.append({
            "location_id": loc.get("id"),
            "name": loc.get("name"),
            "provider": (loc.get("provider") or {}).get("name"),
            "first": _parse_utc(loc.get("datetimeFirst")),
            "last": _parse_utc(loc.get("datetimeLast")),
            "pm25_sensor_id": pm_sensor,
        })
    return out


PAGE_SIZE = 1000


def _parse_hour_rows(results: list[dict]) -> list[tuple[datetime, float]]:
    out = []
    for r in results:
        period = r.get("period") or {}
        ts = _parse_utc(period.get("datetimeFrom")) or _parse_utc(r.get("datetime"))
        val = r.get("value")
        if ts is not None and val is not None:
            out.append((ts, float(val)))
    return out


def fetch_sensor_hours(sensor_id: int, dt_from: datetime, dt_to: datetime,
                       api_key: str, max_pages: int = 30) -> list[tuple[datetime, float]]:
    """Hourly PM2.5 values for one sensor between two UTC datetimes (follows pagination)."""
    out: list[tuple[datetime, float]] = []
    for page in range(1, max_pages + 1):
        data = _get(f"/sensors/{sensor_id}/hours", {
            "datetime_from": dt_from.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "datetime_to": dt_to.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "limit": PAGE_SIZE,
            "page": page,
        }, api_key)
        results = data.get("results", [])
        out.extend(_parse_hour_rows(results))
        if len(results) < PAGE_SIZE:
            break
    return out
