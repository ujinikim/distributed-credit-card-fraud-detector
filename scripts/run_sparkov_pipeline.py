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

    from fraud_lens.bronze_to_silver import run as run_bronze_to_silver
    from fraud_lens.ingest import load_sparkov_config, run as run_bronze_ingest
    from fraud_lens.silver_to_gold import run as run_silver_to_gold

    config = load_sparkov_config().get("sparkov", {})
    spark_builder = SparkSession.builder.appName("FraudLens-Sparkov-Pipeline")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    raw_path = project_root / config.get("normalized_raw_path", "data/raw_sparkov")
    bronze_path = project_root / config.get("bronze_path", "data/benchmark/bronze_sparkov")
    silver_path = project_root / config.get("silver_path", "data/benchmark/silver_sparkov")
    gold_path = project_root / config.get("gold_path", "data/benchmark/gold_sparkov")

    run_bronze_ingest(
        spark,
        raw_path=raw_path,
        bronze_path=bronze_path,
        file_pattern="*.json",
    )
    print(f"Bronze ingest complete. Output: {bronze_path.resolve()}")

    df_silver = run_bronze_to_silver(
        spark,
        bronze_path=bronze_path,
        silver_path=silver_path,
    )
    print(
        f"Silver transform complete. Output: {silver_path.resolve()} "
        f"(rows={df_silver.count()})"
    )

    df_gold = run_silver_to_gold(
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
