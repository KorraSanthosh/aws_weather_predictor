"""The predeclared hybrid serving rule ("v3_hybrid"). It is a ROUTER between the two stored models; it has no model
file of its own and nothing about it is trained or tuned.

    use v2_35_candidate when ALL seven day-part inputs are present for the forecast origin day, else v1_25_baseline.

"Present" means finite (not NaN): the origin day has at least 16 hourly readings and every day-part had enough hours
(see core.daily.daypart_stats). No other condition, threshold or variant exists - this single rule was written down
before any hybrid result was computed (model/EVALUATION_PROTOCOL.md).
"""
from __future__ import annotations

import math

from core.features import DAYPART_FEATURES

HYBRID_VERSION = "v3_hybrid"
HYBRID_WITH_DAYPARTS = "v2_35_candidate"
HYBRID_FALLBACK = "v1_25_baseline"
HYBRID_RULE = "v2_35_candidate if all 7 day-part inputs are finite, else v1_25_baseline"


def daypart_valid(row: dict) -> bool:
    """True when every day-part feature in `row` (feature name -> value) is a real number."""
    return all(isinstance(row.get(f), (int, float)) and math.isfinite(row[f]) for f in DAYPART_FEATURES)


def hybrid_pick(row: dict) -> str:
    """Name of the stored model the hybrid uses for this origin day."""
    return HYBRID_WITH_DAYPARTS if daypart_valid(row) else HYBRID_FALLBACK
