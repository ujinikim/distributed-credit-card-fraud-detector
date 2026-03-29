"""Canonical Bronze ingest for the medallion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING
from pyspark.sql.functions import current_timestamp, input_file_name

from fraud_lens.pipeline.paths import load_paths_config

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame


def run_bronze_ingest(
    spark: "SparkSession",
    raw_path: str | Path | None = None,
    bronze_path: str | Path | None = None,
    paths_yaml: str | Path | None = None,
    file_pattern: str | None = None,
) -> "DataFrame":
    """Read canonical raw JSONL, add metadata, and write Bronze Parquet."""
    config = load_paths_config(paths_yaml)
    data = config.get("data", {})
    ingest_cfg = config.get("ingest", {})

    if raw_path is None:
        raw_path = data.get("raw", "data/raw")
    raw_path = str(Path(raw_path).resolve())
    if file_pattern is None:
        file_pattern = ingest_cfg.get("file_pattern", "*.jsonl")
    read_path = str(Path(raw_path) / file_pattern) if file_pattern else raw_path

    if bronze_path is None:
        bronze_path = data.get("bronze", "data/bronze")
    bronze_path = str(Path(bronze_path).resolve())

    df = spark.read.json(read_path)
    df = df.withColumn("ingestion_timestamp", current_timestamp()).withColumn(
        "source_path", input_file_name()
    )

    df.write.mode("overwrite").parquet(bronze_path)
    return df


run = run_bronze_ingest
