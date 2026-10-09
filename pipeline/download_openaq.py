"""Step 2: download PM2.5 history from the OpenAQ open-data archive on AWS S3.

Public bucket, no AWS credentials needed (unsigned requests):
    s3://openaq-data-archive/records/csv.gz/locationid=<id>/year=<YYYY>/month=<MM>/location-<id>-<YYYYMMDD>.csv.gz

Usage:  python pipeline/download_openaq.py --start 2023-01-01
Reads config/stations.json, writes data/raw/openaq/<location_id>.csv
"""
import argparse
import csv
import gzip
import io
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from pathlib import Path

import boto3
from botocore import UNSIGNED
from botocore.config import Config

BUCKET = "openaq-data-archive"
REGION = "us-east-1"          # the bucket lives in us-east-1
PM25_NAMES = {"pm25", "pm2.5", "pm2_5"}


def list_keys(s3, location_id: int, start: date, end: date) -> list[str]:
    keys = []
    for year in range(start.year, end.year + 1):
        prefix = f"records/csv.gz/locationid={location_id}/year={year}/"
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                stamp = key.rsplit("-", 1)[-1].split(".")[0]      # YYYYMMDD
                try:
                    d = datetime.strptime(stamp, "%Y%m%d").date()
                except ValueError:
                    continue
                if start <= d <= end:
                    keys.append(key)
    return sorted(keys)


def read_pm25_rows(s3, key: str) -> list[tuple[str, str]]:
    body = s3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
    text = gzip.decompress(body).decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = []
    for r in reader:
        r = {k.strip().lower(): v for k, v in r.items()}
        if r.get("parameter", "").strip().lower() in PM25_NAMES and r.get("value") not in (None, ""):
            rows.append((r["datetime"], r["value"]))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=date.today().isoformat())
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--stations", default="config/stations.json")
    args = ap.parse_args()
    start, end = date.fromisoformat(args.start), date.fromisoformat(args.end)

    stations = json.loads(Path(args.stations).read_text())
    s3 = boto3.client("s3", region_name=REGION, config=Config(signature_version=UNSIGNED))
    outdir = Path("data/raw/openaq")
    outdir.mkdir(parents=True, exist_ok=True)

    for st in stations:
        lid = st["location_id"]
        keys = list_keys(s3, lid, start, end)
        print(f"[{lid}] {st.get('name')}: {len(keys)} daily files")
        if not keys:
            print("   nothing found - check the id, or the station has no archive data in this range")
            continue
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            chunks = list(pool.map(lambda k: read_pm25_rows(s3, k), keys))
        n = 0
        with open(outdir / f"{lid}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["datetime", "pm25"])
            for rows in chunks:
                for ts, val in rows:
                    w.writerow([ts, val])
                    n += 1
        print(f"   saved {n} PM2.5 readings -> {outdir / (str(lid) + '.csv')}")
    print("done")


if __name__ == "__main__":
    sys.exit(main())
