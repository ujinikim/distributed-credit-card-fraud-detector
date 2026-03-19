#!/usr/bin/env python3
"""Entrypoint to run the FraudLens pipeline (Bronze → Silver → Gold)."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from pyspark.sql import SparkSession

from fraud_lens.ingest import load_paths_config, run as run_bronze_ingest
from fraud_lens.bronze_to_silver import run as run_bronze_to_silver
from fraud_lens.silver_to_gold import run as run_silver_to_gold


def main() -> None:
    """Run Bronze → Silver → Gold."""
    spark = (
        SparkSession.builder.appName("FraudLens-Pipeline")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    config = load_paths_config()

    # Bronze
    run_bronze_ingest(spark)
    bronze_path = config.get("data", {}).get("bronze", "data/bronze")
    print(f"Bronze ingest complete. Output: {Path(bronze_path).resolve()}")

    # Bronze → Silver
    df_silver = run_bronze_to_silver(spark)
    silver_path = config.get("data", {}).get("silver", "data/silver")
    print(
        f"Silver transform complete. Output: {Path(silver_path).resolve()} "
        f"(rows={df_silver.count()})"
    )

    # Silver → Gold
    df_gold = run_silver_to_gold(spark)
    gold_path = config.get("data", {}).get("gold", "data/gold")
    print(
        f"Gold features complete. Output: {Path(gold_path).resolve()} "
        f"(rows={df_gold.count()})"
    )

    spark.stop()


if __name__ == "__main__":
    main()
