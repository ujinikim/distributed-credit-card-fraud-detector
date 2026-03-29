"""Canonical Bronze-to-Silver transform for the medallion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from fraud_lens.pipeline.paths import load_paths_config

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _required_schema() -> T.StructType:
    """Target schema for Silver, aligned with synthetic_data_spec."""
    return T.StructType(
        [
            T.StructField("transaction_id", T.StringType(), nullable=False),
            T.StructField("card_id", T.StringType(), nullable=False),
            T.StructField("event_time", T.StringType(), nullable=False),
            T.StructField("amount", T.DoubleType(), nullable=False),
            T.StructField("merchant_category", T.StringType(), nullable=False),
            T.StructField("latitude", T.DoubleType(), nullable=False),
            T.StructField("longitude", T.DoubleType(), nullable=False),
            T.StructField("anomaly_type", T.StringType(), nullable=False),
            T.StructField("ref_transaction_id", T.StringType(), nullable=True),
            # Optional benchmark passthrough columns. These stay nullable so the
            # shared Silver contract remains compatible with synthetic runs.
            T.StructField("merchant", T.StringType(), nullable=True),
            T.StructField("is_fraud", T.IntegerType(), nullable=True),
            T.StructField("customer_latitude", T.DoubleType(), nullable=True),
            T.StructField("customer_longitude", T.DoubleType(), nullable=True),
            T.StructField("event_time_unix", T.LongType(), nullable=True),
            # Metadata carried from Bronze
            T.StructField("ingestion_timestamp", T.TimestampType(), nullable=True),
            T.StructField("source_path", T.StringType(), nullable=True),
        ]
    )


def run_silver_transform(
    spark: "SparkSession",
    bronze_path: str | Path | None = None,
    silver_path: str | Path | None = None,
) -> DataFrame:
    """
    Transform Bronze data into a clean, typed Silver table.

    - Reads Bronze Parquet.
    - Enforces a single typed schema (cast columns to expected types).
    - Drops rows with required nulls or obviously invalid values.
    - Drops duplicate transactions by a natural transaction signature.
    - Writes Silver Parquet and returns the Silver DataFrame.
    """
    config = load_paths_config()
    data_cfg = config.get("data", {})

    if bronze_path is None:
        bronze_path = data_cfg.get("bronze", "data/bronze")
    if silver_path is None:
        silver_path = data_cfg.get("silver", "data/silver")

    bronze_path = str(Path(bronze_path).resolve())
    silver_path = str(Path(silver_path).resolve())
    target_write_partitions = max(spark.sparkContext.defaultParallelism, 32)

    df_bronze = spark.read.parquet(bronze_path)

    # Cast to target types; keep ingestion metadata and ref_transaction_id if present.
    ref_col = (
        F.col("ref_transaction_id").cast(T.StringType())
        if "ref_transaction_id" in df_bronze.columns
        else F.lit(None).cast(T.StringType())
    )
    merchant_col = (
        F.col("merchant").cast(T.StringType())
        if "merchant" in df_bronze.columns
        else F.lit(None).cast(T.StringType())
    )
    is_fraud_col = (
        F.col("is_fraud").cast(T.IntegerType())
        if "is_fraud" in df_bronze.columns
        else F.lit(None).cast(T.IntegerType())
    )
    customer_lat_col = (
        F.col("customer_latitude").cast(T.DoubleType())
        if "customer_latitude" in df_bronze.columns
        else F.lit(None).cast(T.DoubleType())
    )
    customer_lon_col = (
        F.col("customer_longitude").cast(T.DoubleType())
        if "customer_longitude" in df_bronze.columns
        else F.lit(None).cast(T.DoubleType())
    )
    event_time_unix_col = (
        F.col("event_time_unix").cast(T.LongType())
        if "event_time_unix" in df_bronze.columns
        else F.lit(None).cast(T.LongType())
    )
    df_typed = df_bronze.select(
        F.col("transaction_id").cast(T.StringType()).alias("transaction_id"),
        F.col("card_id").cast(T.StringType()).alias("card_id"),
        F.col("event_time").cast(T.StringType()).alias("event_time"),
        F.col("amount").cast(T.DoubleType()).alias("amount"),
        F.col("merchant_category").cast(T.StringType()).alias("merchant_category"),
        F.col("latitude").cast(T.DoubleType()).alias("latitude"),
        F.col("longitude").cast(T.DoubleType()).alias("longitude"),
        F.col("anomaly_type").cast(T.StringType()).alias("anomaly_type"),
        ref_col.alias("ref_transaction_id"),
        merchant_col.alias("merchant"),
        is_fraud_col.alias("is_fraud"),
        customer_lat_col.alias("customer_latitude"),
        customer_lon_col.alias("customer_longitude"),
        event_time_unix_col.alias("event_time_unix"),
        F.col("ingestion_timestamp").cast(T.TimestampType()).alias(
            "ingestion_timestamp"
        ),
        F.col("source_path").cast(T.StringType()).alias("source_path"),
    )

    # Drop rows with nulls in required fields.
    required_cols = [
        "transaction_id",
        "card_id",
        "event_time",
        "amount",
        "merchant_category",
        "latitude",
        "longitude",
        "anomaly_type",
    ]
    df_clean = df_typed.dropna(subset=required_cols)

    # Apply simple validity checks (amount non-negative, lat/lon in range).
    df_clean = df_clean.filter(
        (F.col("amount") >= 0.0)
        & (F.col("latitude") >= -90.0)
        & (F.col("latitude") <= 90.0)
        & (F.col("longitude") >= -180.0)
        & (F.col("longitude") <= 180.0)
    )

    # Deduplicate on transaction content to handle repeated generator runs where
    # transaction_id changes but the underlying event is identical.
    dedup_cols = [
        "card_id",
        "event_time",
        "amount",
        "merchant_category",
        "latitude",
        "longitude",
        "anomaly_type",
    ]
    df_clean = df_clean.dropDuplicates(dedup_cols)

    # Keep the canonical column order without forcing a Python-side RDD conversion.
    schema_cols = [field.name for field in _required_schema().fields]
    df_silver = df_clean.select(*schema_cols)

    (
        df_silver.repartition(target_write_partitions)
        .write.mode("overwrite")
        .option("parquet.enable.dictionary", "false")
        .parquet(silver_path)
    )
    return df_silver


run = run_silver_transform
