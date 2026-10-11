"""Records whether Open-Meteo can supply ARCHIVED CAMS day-ahead forecasts (previous-run values) for any period."""
import json, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from core.weather import http_get_json
AQ = "https://air-quality-api.open-meteo.com/v1/air-quality"
PERIODS = [("2022-09-01", "2022-09-03"), ("2023-01-01", "2023-01-03"), ("2024-06-01", "2024-06-03"), ("2025-01-10", "2025-01-12"), ("2025-10-01", "2025-10-05"),
           ("2026-01-01", "2026-01-05"), ("2026-04-01", "2026-04-05"), ("2026-07-01", "2026-07-05"), ("2026-10-01", "2026-10-08")]
out = {"api": AQ, "point": [28.6139, 77.2090], "requests": [], "previous_runs_air_quality_endpoint": None}
for a, b in PERIODS:
    for attempt in range(3):
        try:
            h = http_get_json(AQ, {"latitude": 28.6139, "longitude": 77.2090, "start_date": a, "end_date": b, "timezone": "Asia/Kolkata",
                                   "hourly": "pm2_5,pm2_5_previous_day1,pm2_5_previous_day2"}, retries=1)["hourly"]
            out["requests"].append({"period": [a, b], **{k: {"non_null": sum(x is not None for x in v), "hours": len(v)} for k, v in h.items() if k != "time"}}); break
        except Exception as e:                                                       # noqa: BLE001
            err = str(e)[:90]; time.sleep(8)
    else:
        out["requests"].append({"period": [a, b], "error": err})
    time.sleep(1.5)
try:
    http_get_json("https://previous-runs-api.open-meteo.com/v1/air-quality", {"latitude": 28.6, "longitude": 77.2, "hourly": "pm2_5_previous_day1", "start_date": "2025-01-10", "end_date": "2025-01-11"}, retries=1)
    out["previous_runs_air_quality_endpoint"] = "exists"
except Exception as e:                                                               # noqa: BLE001
    out["previous_runs_air_quality_endpoint"] = str(e)[:90]
ok = [r for r in out["requests"] if "error" not in r]
out["summary"] = {"periods_answered": len(ok), "periods_failed_network": len(out["requests"]) - len(ok),
                  "pm2_5_has_values_in_all_answered": all(r["pm2_5"]["non_null"] == r["pm2_5"]["hours"] for r in ok),
                  "previous_day_values_present_anywhere": any(r["pm2_5_previous_day1"]["non_null"] or r["pm2_5_previous_day2"]["non_null"] for r in ok)}
(HERE / "probe.json").write_text(json.dumps(out, indent=1)); print(out["summary"], out["previous_runs_air_quality_endpoint"])
