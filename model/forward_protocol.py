"""Frozen constants of the winter 2026-27 forward (shadow) test. Single source of truth: model/forward_eval.py reads these,
model/freeze_manifest.py hashes this file, and model/EVALUATION_PROTOCOL.md explains them in words.

Do NOT edit after the freeze (see EVALUATION_PROTOCOL.md section 9). Changing any value here makes
`python model/freeze_manifest.py --check` fail, on purpose.
"""

PROTOCOL = {
    "protocol_version": "1.0",
    "baseline": "v1_25_baseline",
    "candidates": ["v2_35_candidate", "v3_hybrid"],
    "hybrid_rule": "v2_35_candidate if all 7 day-part inputs are finite, else v1_25_baseline",
    # a forecast counts if the DAY IT PREDICTS (target date) lies in this window, both ends included
    "window_target_dates": ["2026-10-15", "2027-02-15"],
    "analysis_not_before": "2027-02-16",          # one final analysis, after the window has closed
    "min_matched_days": 90,                       # matched forecast days in the complete eligible set
    "min_matched_daypart_days": 90,               # ... of which days with valid day-part inputs
    "primary_metric": "MAE (ug/m3)",
    "secondary_metric": "AQI-category accuracy",
    "mae_reduction_required_pct": 5.0,            # candidate MAE must be at least this much lower than v1
    "mae_ci_upper_must_be_below_pct": 0.0,        # ... and the 95% interval of the MAE change must lie below zero
    "category_accuracy_max_drop_pp": 2.0,         # no material degradation = not more than this many points lower
    "ci_level": 0.95,
    "bootstrap": {"block_days": 7, "n_boot": 2000, "seed": 0},
    "segments": ["complete_eligible", "daypart_days", "no_daypart_days"],
    "gate_segment": "complete_eligible",
}
