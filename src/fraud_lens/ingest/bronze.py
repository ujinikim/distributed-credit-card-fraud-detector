"""Compatibility wrapper for Bronze ingest imports."""

from fraud_lens.pipeline.bronze import run, run_bronze_ingest
from fraud_lens.pipeline.paths import load_paths_config

__all__ = ["load_paths_config", "run", "run_bronze_ingest"]
