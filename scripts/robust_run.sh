#!/usr/bin/env bash
# One command: fetch long-history stations + newest days, rebuild the dataset, train, print validation.
#   export OPENAQ_API_KEY=<your key>
#   bash scripts/robust_run.sh
set -e
: "${OPENAQ_API_KEY:?Set OPENAQ_API_KEY first}"
python pipeline/find_stations.py --history --max 16          # old stations -> config/stations_history.json
python pipeline/download_openaq.py --stations config/stations_history.json --start 2016-01-01
python pipeline/download_openaq.py --start 2016-01-01        # current stations, full history
python pipeline/topup_recent.py                              # newest days from the live API
python pipeline/download_weather.py --start 2016-01-01
python pipeline/build_dataset.py --exclude 2020-03-25:2020-06-30   # COVID lockdown: abnormal emissions
python model/train.py
echo
echo "Paste everything from '=== DATA QUALITY REPORT ===' down to the end back to Claude."
