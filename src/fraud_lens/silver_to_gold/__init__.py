"""Compatibility exports for Silver-to-Gold imports."""

from fraud_lens.pipeline.gold import (
    PRIOR_CATEGORY_Z_DAMP_K,
    PRIOR_CATEGORY_Z_MIN_TX,
    PRIOR_CATEGORY_SHRINK_TAU,
    run,
    run_gold_features,
)

__all__ = [
    "PRIOR_CATEGORY_Z_DAMP_K",
    "PRIOR_CATEGORY_Z_MIN_TX",
    "PRIOR_CATEGORY_SHRINK_TAU",
    "run",
    "run_gold_features",
]
