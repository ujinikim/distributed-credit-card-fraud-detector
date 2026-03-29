"""Compatibility exports for older ingest imports."""

from fraud_lens.benchmark.sparkov.config import load_sparkov_config
from fraud_lens.benchmark.sparkov.ingest import (
    normalize_sparkov_raw,
    run_sparkov_ingest,
)
from fraud_lens.pipeline.bronze import run_bronze_ingest
from fraud_lens.pipeline.paths import load_paths_config

run = run_bronze_ingest

__all__ = [
    "load_paths_config",
    "load_sparkov_config",
    "normalize_sparkov_raw",
    "run",
    "run_bronze_ingest",
    "run_sparkov_ingest",
]
