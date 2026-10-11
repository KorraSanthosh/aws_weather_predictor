#!/usr/bin/env python3
"""Write the static prediction-range file the dashboards read when no live API is configured.

    python scripts/make_range_json.py [--version v1_25_baseline]

Reads web/data/forecast.json (NOT modified - it stays byte-identical to model/v1_25_baseline/web_forecast.json) and writes
web/data/range.json (plus a copy in frontend/public/data/ when that folder exists):

    {"target_date": ..., "pm25": ..., "model_version": ..., "prediction_range": {...}}

The dashboards use the range only when target_date and pm25 match the forecast they are showing, so a stale range.json can
never be attached to a newer forecast. The range comes from core/prediction_range.py and the packaged calibration; no model is
run and nothing is retrained. With a live API this file is not used (the API computes the range itself).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.prediction_range import prediction_range  # noqa: E402


def build(forecast: dict, version: str) -> dict:
    rng = prediction_range(forecast["pm25"], forecast.get("model_version") or version)
    if rng is None:
        raise SystemExit(f"no prediction range available for model version {forecast.get('model_version') or version!r}")
    return {"target_date": forecast["target_date"], "pm25": forecast["pm25"],
            "model_version": forecast.get("model_version") or version, "prediction_range": rng}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--version", default="v1_25_baseline",
                    help="model that produced web/data/forecast.json when that file carries no model_version (default: the v1 baseline)")
    args = ap.parse_args(argv)
    src = ROOT / "web" / "data" / "forecast.json"
    out = build(json.loads(src.read_text()), args.version)
    text = json.dumps(out, indent=1) + "\n"
    for d in (ROOT / "web" / "data", ROOT / "frontend" / "public" / "data"):
        if d.is_dir():
            (d / "range.json").write_text(text)
            print("wrote", (d / "range.json").relative_to(ROOT))
    r = out["prediction_range"]
    print(f"{out['target_date']}: point {out['pm25']} -> {r['category_low']} to {r['category_high']}, PM2.5 {r['pm25_low']}-{r['pm25_high']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
