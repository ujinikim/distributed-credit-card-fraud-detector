#!/usr/bin/env python3
"""Run Bronze → Silver → Gold on normalized Sparkov benchmark data."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def main() -> None:
    """Run the medallion pipeline against the normalized Sparkov benchmark dataset."""
    from pyspark.sql import SparkSession

    from fraud_lens.benchmark.sparkov import (
        load_sparkov_config,
        resolve_sparkov_paths,
    )
    from fraud_lens.pipeline import (
        run_bronze_ingest,
        run_gold_features,
        run_silver_transform,
    )

    config = load_sparkov_config().get("sparkov", {})
    paths = resolve_sparkov_paths(config)
    spark_builder = SparkSession.builder.appName("FraudLens-Sparkov-Pipeline")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    raw_path = paths["normalized_raw_path"]
    bronze_path = paths["bronze_path"]
    silver_path = paths["silver_path"]
    gold_path = paths["gold_path"]

    run_bronze_ingest(
        spark,
        raw_path=raw_path,
        bronze_path=bronze_path,
        file_pattern="*.json",
    )
    print(f"Bronze ingest complete. Output: {bronze_path.resolve()}")

    df_silver = run_silver_transform(
        spark,
        bronze_path=bronze_path,
        silver_path=silver_path,
    )
    print(
        f"Silver transform complete. Output: {silver_path.resolve()} "
        f"(rows={df_silver.count()})"
    )

    df_gold = run_gold_features(
        spark,
        silver_path=silver_path,
        gold_path=gold_path,
    )
    print(
        f"Gold features complete. Output: {gold_path.resolve()} "
        f"(rows={df_gold.count()})"
    )

    spark.stop()


if __name__ == "__main__":
    main()
