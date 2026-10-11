#!/usr/bin/env bash
# Package the Lambda code + trained model into build/lambda/ (used by template.yaml).
# Run AFTER `python model/train.py` on real data.
# Usage: bash scripts/build_lambda.sh [model_version]     e.g. v1_25_baseline (default: the files in model/)
#        SHADOW=1 bash scripts/build_lambda.sh             also packages both stored versions (needed for SHADOW_VERSIONS)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=model
[ -n "${1:-}" ] && SRC="model/$1"
[ -f "$SRC/model.json" ] || { echo "$SRC/model.json missing - run model/train.py first"; exit 1; }

rm -rf build/lambda && mkdir -p build/lambda/model
cp -r core build/lambda/core
cp backend/forecast_handler.py backend/api_handler.py build/lambda/
cp "$SRC/model.json" "$SRC/model_meta.json" build/lambda/model/
if [ "${SHADOW:-0}" = "1" ]; then
  for v in v1_25_baseline v2_35_candidate; do
    mkdir -p "build/lambda/model/$v"
    cp "model/$v/model.json" "model/$v/model_meta.json" "build/lambda/model/$v/"
  done
fi

# Lambda-compatible wheels (Linux x86_64, Python 3.12). xgboost-cpu is the small build without GPU libs.
pip install --quiet --target build/lambda \
  --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all: \
  numpy xgboost-cpu scipy

find build/lambda -name "__pycache__" -type d -prune -exec rm -rf {} +
du -sh build/lambda
echo "Built build/lambda. Next:  sam build --template template.yaml  &&  sam deploy --guided"
