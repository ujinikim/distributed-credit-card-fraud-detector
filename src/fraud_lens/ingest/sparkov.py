"""Compatibility wrapper for Sparkov ingest imports."""

from fraud_lens.benchmark.sparkov.config import load_sparkov_config
from fraud_lens.benchmark.sparkov.ingest import (
    normalize_sparkov_raw,
    run,
    run_sparkov_ingest,
)

__all__ = ["load_sparkov_config", "normalize_sparkov_raw", "run", "run_sparkov_ingest"]
