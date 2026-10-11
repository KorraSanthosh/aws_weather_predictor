#!/usr/bin/env bash
# Package the Lambda code + trained model into build/lambda/ (used by template.yaml).
# Run AFTER `python model/train.py` on real data.
# Usage: bash scripts/build_lambda.sh [model_version]     e.g. v1_25_baseline (default: the files in model/)
#        SHADOW=1 bash scripts/build_lambda.sh             also packages both stored versions (needed for SHADOW_VERSIONS)
# Test-only switches:  SKIP_DEPS=1 (do not pip-install numpy/xgboost/scipy)   BUILD_DIR=<dir> (default build/lambda)
set -euo pipefail
cd "$(dirname "$0")/.."

SRC=model
[ -n "${1:-}" ] && SRC="model/$1"
[ -f "$SRC/model.json" ] || { echo "$SRC/model.json missing - run model/train.py first"; exit 1; }

OUT="${BUILD_DIR:-build/lambda}"
rm -rf "$OUT" && mkdir -p "$OUT/model"
cp -r core "$OUT/core"
cp backend/forecast_handler.py backend/forward_logger.py backend/api_handler.py "$OUT/"
cp "$SRC/model.json" "$SRC/model_meta.json" "$OUT/model/"
if [ "${SHADOW:-0}" = "1" ]; then
  for v in v1_25_baseline v2_35_candidate; do
    mkdir -p "$OUT/model/$v"
    cp "model/$v/model.json" "model/$v/model_meta.json" "$OUT/model/$v/"
  done
fi

# Which build is this? (read by backend/forward_logger.py and stored in every provenance record; no secrets)
SRC="$SRC" SHADOW="${SHADOW:-0}" OUT="$OUT" python3 - <<'PY'
import json, os, subprocess
from datetime import datetime, timezone
def git(*a):
    try:
        return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None
info = {"built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_commit": git("rev-parse", "HEAD"), "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "git_dirty": (None if git("status", "--porcelain") is None else bool(git("status", "--porcelain"))),
        "shadow_package": os.environ["SHADOW"] == "1", "default_model_source": os.environ["SRC"]}
json.dump(info, open(os.path.join(os.environ["OUT"], "build_info.json"), "w"), indent=2)
print("build_info:", info)
PY

# Lambda-compatible wheels (Linux x86_64, Python 3.12). xgboost-cpu is the small build without GPU libs.
# xgboost MUST be >= 3.1: the stored model files were written by 3.4.1 and older versions (<3.1) misread their
# base_score, which makes every forecast about 1.65x too high.  Newer xgboost wheels are tagged manylinux_2_28
# (Amazon Linux 2023, the python3.12 runtime, supports it); with only the manylinux2014 tag pip silently falls back
# to xgboost 3.0.5.  So: allow both tags and pin the version the models were saved with.
if [ "${SKIP_DEPS:-0}" != "1" ]; then
  pip install --quiet --target "$OUT" \
    --platform manylinux_2_28_x86_64 --platform manylinux2014_x86_64 \
    --python-version 3.12 --implementation cp --only-binary=:all: \
    numpy "xgboost-cpu==3.4.1" scipy
  # Fail the build if the wrong xgboost ended up in the package.
  ls "$OUT" | grep -q '^xgboost_cpu-3\.4\.1\.dist-info$' || { echo "ERROR: xgboost-cpu 3.4.1 is not in $OUT" >&2; exit 1; }
fi

find "$OUT" -name "__pycache__" -type d -prune -exec rm -rf {} +
du -sh "$OUT"
echo "Built $OUT. Next:  sam build --template template.yaml  &&  sam deploy --guided"
