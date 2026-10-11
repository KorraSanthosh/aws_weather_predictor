"""Download PM2.5 readings of the selected regional stations from the PUBLIC OpenAQ archive (no key, unsigned requests).

    python experiments/regional_cities/download_regional.py [--end 2026-02-15]

Reads stations.json, writes data/raw/regional/<location_id>.csv (datetime,pm25) + data/raw/regional/manifest.json.
Kept in its OWN folder so these stations can never leak into the Delhi city mean (which reads data/raw/openaq*/).
Nothing after --end is downloaded (default 2026-02-15, the last day any experiment reads; winter 2026-27 stays untouched).
The archive holds no Indian government-station data for 2023 and 2024 (both the legacy and the provider layout were checked); that
gap is recorded in the manifest, never filled.
"""
import argparse
import csv
import gzip
import io
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
B = "openaq-data-archive"
PM25 = {"pm25", "pm2.5", "pm2_5"}
s3 = boto3.client("s3", region_name="us-east-1", config=Config(signature_version=UNSIGNED, retries={"max_attempts": 10, "mode": "standard"}, max_pool_connections=96))


def keys(lid, start, end):
    out, token = [], None
    while True:
        kw = dict(Bucket=B, Prefix=f"records/csv.gz/locationid={lid}/", MaxKeys=1000)
        if token:
            kw["ContinuationToken"] = token
        r = s3.list_objects_v2(**kw)
        for o in r.get("Contents", []):
            m = re.search(r"-(\d{8})\.csv\.gz$", o["Key"])
            if m and start <= datetime.strptime(m.group(1), "%Y%m%d").date() <= end:
                out.append(o["Key"])
        if not r["IsTruncated"]:
            return sorted(out)
        token = r["NextContinuationToken"]


def rows(key):
    for attempt in range(6):
        try:
            text = gzip.decompress(s3.get_object(Bucket=B, Key=key)["Body"].read()).decode("utf-8", "replace")
            res = []
            for r in csv.DictReader(io.StringIO(text)):
                r = {k.strip().lower(): v for k, v in r.items()}
                if (r.get("parameter") or "").strip().lower() in PM25 and r.get("value") not in (None, ""):
                    res.append((r["datetime"], r["value"]))
            return res
        except Exception:
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"could not read {key}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2018-01-01")
    ap.add_argument("--end", default="2026-02-15")
    ap.add_argument("--workers", type=int, default=48)
    a = ap.parse_args()
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    stations = json.loads((HERE / "stations.json").read_text())["stations"]
    out = ROOT / "data" / "raw" / "regional"
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"end": a.end, "stations": {}}
    mpath = out / "manifest.json"
    if mpath.exists():
        manifest = json.loads(mpath.read_text())
    t0 = time.time()
    for i, st in enumerate(stations, 1):
        lid = str(st["id"])
        if lid in manifest["stations"] and (out / f"{lid}.csv").exists() and manifest["end"] == a.end:
            continue
        ks = keys(st["id"], start, end)
        with ThreadPoolExecutor(a.workers) as ex:
            chunks = list(ex.map(rows, ks))
        n = 0
        years = {}
        with open(out / f"{lid}.csv", "w", newline="") as f:
            w = csv.writer(f); w.writerow(["datetime", "pm25"])
            for k, rs in zip(ks, chunks):
                y = re.search(r"year=(\d{4})", k).group(1)
                years[y] = years.get(y, 0) + 1
                for ts, v in rs:
                    w.writerow([ts, v]); n += 1
        manifest["stations"][lid] = {"name": st["name"], "group": st["group"], "files": len(ks), "readings": n, "files_by_year": dict(sorted(years.items()))}
        mpath.write_text(json.dumps(manifest, indent=1))
        print(f"[{i}/{len(stations)}] {lid} {st['name'][:40]:40} {len(ks)} files, {n} readings  ({time.time() - t0:.0f}s)", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    sys.exit(main())
