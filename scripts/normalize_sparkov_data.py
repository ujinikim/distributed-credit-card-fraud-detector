#!/usr/bin/env python3
"""Normalize Sparkov benchmark CSV data into the canonical FraudLens raw schema."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def main() -> None:
    """Create Spark session, normalize Sparkov benchmark data, and print the output path."""
    from pyspark.sql import SparkSession

    from fraud_lens.benchmark.sparkov import (
        load_sparkov_config,
        normalize_sparkov_raw,
        resolve_sparkov_paths,
    )

    spark = (
        SparkSession.builder.appName("FraudLens-Normalize-Sparkov")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    config = load_sparkov_config().get("sparkov", {})
    output_path = resolve_sparkov_paths(config)["normalized_raw_path"]
    df = normalize_sparkov_raw(spark)
    print(
        f"Sparkov normalization complete. Output: {output_path} "
        f"(rows={df.count()})"
    )
    spark.stop()


if __name__ == "__main__":
    main()
