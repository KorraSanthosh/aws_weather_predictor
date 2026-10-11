#!/usr/bin/env bash
# Deploy "V1 serves, Hybrid in shadow" to AWS.  RUN THIS YOURSELF in a normal terminal on your Mac:  bash scripts/deploy_shadow.sh
#
# It needs things only you have: your AWS login and your NEW OpenAQ key (typed hidden into sam, never stored in a file).
# What it does, in order (it stops at the first problem; it is safe to re-run):
#   1. checks sam / aws / AWS login, the branch, and picks a Python >= 3.10 (creates ./venv if needed)
#   2. runs scripts/validate_working_copy.sh - the only FAIL allowed is "local commits not on origin/main"
#   3. commits the forward-logging work on feature/forward-logging (two commits; NOTHING is pushed)
#   4. SHADOW=1 build of build/lambda with xgboost-cpu 3.4.1, then checks build_info.json
#   5. sam build, then "sam deploy --guided": you answer the questions and you confirm the change set
# It never sets MODEL_VERSION (V1 stays the served model) and never edits a model file.
set -euo pipefail
cd "$(dirname "$0")/.."

step() { printf '\n== %s\n' "$*"; }
die()  { printf '\nSTOP: %b\n' "$*" >&2; exit 1; }

step "1. Tools, AWS login, branch, Python"
command -v sam >/dev/null || die "AWS SAM CLI not found.  Install it:  brew install aws-sam-cli"
command -v aws >/dev/null || die "AWS CLI not found.  Install it:  brew install awscli   then:  aws configure"
aws sts get-caller-identity >/dev/null 2>&1 || die "AWS login not working.  Run 'aws configure' (or 'aws sso login') and try again."
echo "ok: sam, aws, AWS login"
[ "$(git branch --show-current)" = "feature/forward-logging" ] || die "switch to the feature/forward-logging branch first (git switch feature/forward-logging)"
echo "ok: on feature/forward-logging"

PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  if [ -x venv/bin/python ]; then PY=venv/bin/python
  else
    BASE=""
    for c in python3.13 python3.12 python3.11 python3.10; do command -v "$c" >/dev/null 2>&1 && { BASE="$c"; break; }; done
    [ -n "$BASE" ] || die "need Python 3.10 or newer (the stored models need xgboost >= 3.1).  Install:  brew install python@3.12"
    echo "creating ./venv with $BASE (one time, a few minutes) ..."
    "$BASE" -m venv venv
    venv/bin/python -m pip install --quiet --upgrade pip
    venv/bin/python -m pip install --quiet -r requirements.txt
    PY=venv/bin/python
  fi
fi
"$PY" - <<'PYCHECK'
import sys, xgboost
v = tuple(int(p) for p in xgboost.__version__.split(".")[:2])
assert sys.version_info >= (3, 10) and v >= (3, 1), f"Python {sys.version.split()[0]} / xgboost {xgboost.__version__}: need Python >= 3.10 and xgboost >= 3.1"
print(f"ok: Python {sys.version.split()[0]}, xgboost {xgboost.__version__}")
PYCHECK
# the build script calls plain `pip` and `python3`: make them the ones from this environment
PYDIR="$(cd "$(dirname "$PY")" && pwd)"
export PATH="$PYDIR:$PATH"

step "2. Validation (tests, model checksums, frozen manifest, protected files, no secrets)"
OUT="$(PYTHON="$PY" bash scripts/validate_working_copy.sh 2>&1 || true)"
echo "$OUT" | grep -E '^(PASS|FAIL|INFO|==|RESULT)|pm25=' | tail -45
OTHER_FAIL="$(echo "$OUT" | grep '^FAIL' | grep -v 'local commits not on origin/main' || true)"
[ -z "$OTHER_FAIL" ] || die "validation failed:\n$OTHER_FAIL"
echo "$OUT" | grep -q 'v1_25_baseline   pm25=81.8 ' || die "section 5 did not give V1 pm25=81.8 on the fixed fake inputs"
echo "ok: validation clean (the 'local commits' line is expected: nothing is pushed yet)"

step "3. Commit (local only, nothing is pushed)"
git diff --cached --quiet || die "something is already staged; unstage it (git restore --staged .) and re-run"
commit_group() {   # $1 = message, rest = files
  local msg="$1"; shift
  git add -- "$@"
  if git diff --cached --quiet; then echo "already committed: $msg"; else git commit -q -m "$msg"; echo "committed: $msg"; fi
}
commit_group "Fix OpenCity timestamp parsing on Python 3.10 (own parser, same wall-clock behaviour)" \
  pipeline/convert_opencity.py pipeline/check_opencity_timezone.py tests/test_opencity.py
commit_group "Add Hybrid shadow-mode forward logging; pin xgboost-cpu 3.4.1 in the Lambda package

V1 keeps serving. forward_logger.handler wraps the frozen forecast handler and writes one provenance record per night.
The build now takes xgboost-cpu 3.4.1 (manylinux_2_28 wheel); the previous platform tag silently resolved to 3.0.5,
which reads the stored models' base_score wrongly (forecasts ~1.65x too high). Adds a known-answer test." \
  FORWARD_TEST_OPERATIONS.md backend/forward_logger.py core/forward_provenance.py scripts/check_hourly_feed.py \
  scripts/forward_ops.py scripts/build_lambda.sh scripts/deploy_shadow.sh requirements.txt template.yaml \
  tests/test_forward_logging.py tests/test_model_numeric.py tests/test_prediction_range.py
LEFT="$(git status --porcelain)"
[ -z "$LEFT" ] || die "these changes are not part of the commits, so the build would be marked dirty:\n$LEFT\nCommit or stash them and re-run."
echo "ok: working tree clean at $(git log -1 --oneline)"

step "4. Build the shadow package"
SHADOW=1 bash scripts/build_lambda.sh
"$PY" - <<'PYCHECK'
import json
i = json.load(open("build/lambda/build_info.json"))
assert i["shadow_package"] is True, i
assert i["git_dirty"] is False and i["git_commit"], i
print("ok: build_info", {k: i[k] for k in ("git_commit", "git_branch", "git_dirty", "shadow_package")})
PYCHECK
ls build/lambda | grep -q '^xgboost_cpu-3\.4\.1\.dist-info$' || die "xgboost-cpu 3.4.1 is not in build/lambda"
echo "ok: xgboost-cpu 3.4.1 is in the package"

step "5. sam build, then sam deploy --guided (you answer; you confirm the change set)"
sam build --template template.yaml || die "sam build failed.  If it only complains about python3.12, either install it (brew install python@3.12) or run 'sam build --use-container' (needs Docker), then re-run this script."
cat <<'ANSWERS'

Answers for 'sam deploy --guided' (everything else: press Enter):
  Stack Name ................ kal-ki-hawa
  AWS Region ................ your usual region (ap-south-1 / Mumbai is closest to Delhi)
  OpenAQApiKey .............. your NEW OpenAQ key (typed hidden)
  SensorIds ................. 12234787,12235610,396,12234702,14878,12235020,12235092,12235074,12234778,19935
  AlertEmail ................ your email (confirm the subscription e-mail AWS sends)
  ShadowVersions ............ v1_25_baseline,v2_35_candidate,v3_hybrid
  'may not have authorization defined, Is this okay?'  ->  y      (the forecast API is public by design)
  'Save arguments to configuration file' -> y  (samconfig.toml is git-ignored; the key is not saved)

ANSWERS
sam deploy --guided --template-file .aws-sam/build/template.yaml

step "Done. Next (see FORWARD_TEST_OPERATIONS.md section 6)"
echo "Do NOT invoke the Lambda by hand during the day.  After the 23:00 IST run:"
echo "  $PY scripts/forward_ops.py ops --table <TableName from the stack outputs> --region <region>"
echo "Runs on 11-13 Oct are rehearsal; the test window opens with the run on 14 Oct 23:00 IST (target 15 Oct)."
