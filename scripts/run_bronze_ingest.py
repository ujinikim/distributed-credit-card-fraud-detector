#!/usr/bin/env python3
"""Run Bronze ingest: read raw JSONL from data/raw, write Parquet to data/bronze."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from pyspark.sql import SparkSession

from fraud_lens.ingest import load_paths_config, run as run_bronze_ingest


def main() -> None:
    """Create Spark session, run Bronze ingest, print result path."""
    spark = (
        SparkSession.builder.appName("FraudLens-Bronze-Ingest")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    config = load_paths_config()
    bronze_path = config.get("data", {}).get("bronze", "data/bronze")
    run_bronze_ingest(spark)
    print(f"Bronze ingest complete. Output: {Path(bronze_path).resolve()}")
    spark.stop()


if __name__ == "__main__":
    main()
