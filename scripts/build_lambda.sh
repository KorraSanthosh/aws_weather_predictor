#!/usr/bin/env bash
# Package the Lambda code + trained model into build/lambda/ (used by template.yaml).
# Run AFTER `python model/train.py` on real data.  Usage: bash scripts/build_lambda.sh
set -euo pipefail
cd "$(dirname "$0")/.."

[ -f model/model.json ] || { echo "model/model.json missing - run model/train.py first"; exit 1; }

rm -rf build/lambda && mkdir -p build/lambda/model
cp -r core build/lambda/core
cp backend/forecast_handler.py backend/api_handler.py build/lambda/
cp model/model.json model/model_meta.json build/lambda/model/

# Lambda-compatible wheels (Linux x86_64, Python 3.12). xgboost-cpu is the small build without GPU libs.
pip install --quiet --target build/lambda \
  --platform manylinux2014_x86_64 --python-version 3.12 --only-binary=:all: \
  numpy xgboost-cpu scipy

find build/lambda -name "__pycache__" -type d -prune -exec rm -rf {} +
du -sh build/lambda
echo "Built build/lambda. Next:  sam build --template template.yaml  &&  sam deploy --guided"
