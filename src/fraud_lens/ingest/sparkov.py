"""Sparkov benchmark ingestion: map source CSV columns into the canonical FraudLens raw schema."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[3]


def load_sparkov_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load Sparkov benchmark ingestion config."""
    root = _project_root()
    if config_path is None:
        config_path = root / "config" / "sparkov.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Sparkov config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _normalize_event_time(df: DataFrame) -> F.Column:
    """Format Sparkov timestamp into the canonical ISO-8601 UTC-like string."""
    if "unix_time" in df.columns:
        parsed = F.to_timestamp(F.from_unixtime(F.col("unix_time").cast("long")))
    elif "trans_date_trans_time" in df.columns:
        parsed = F.to_timestamp("trans_date_trans_time", "yyyy-MM-dd HH:mm:ss")
    elif "trans_date" in df.columns and "trans_time" in df.columns:
        parsed = F.to_timestamp(
            F.concat_ws(" ", F.col("trans_date"), F.col("trans_time")),
            "yyyy-MM-dd HH:mm:ss",
        )
    else:
        raise ValueError(
            "Sparkov source is missing a supported timestamp layout. "
            "Expected either trans_date_trans_time or trans_date + trans_time."
        )
    return F.date_format(parsed, "yyyy-MM-dd'T'HH:mm:ss'Z'")


def _canonicalize(df: DataFrame, fraud_label_value: str, non_fraud_label_value: str) -> DataFrame:
    """Map Sparkov source columns into the canonical raw transaction schema."""
    return df.select(
        F.col("trans_num").cast("string").alias("transaction_id"),
        F.col("cc_num").cast("string").alias("card_id"),
        _normalize_event_time(df).alias("event_time"),
        F.col("amt").cast("double").alias("amount"),
        F.col("category").cast("string").alias("merchant_category"),
        F.col("merch_lat").cast("double").alias("latitude"),
        F.col("merch_long").cast("double").alias("longitude"),
        F.when(F.col("is_fraud") == 1, F.lit(fraud_label_value))
        .otherwise(F.lit(non_fraud_label_value))
        .alias("anomaly_type"),
        F.lit(None).cast("string").alias("ref_transaction_id"),
        # Keep a few useful benchmark-only columns in the normalized raw layer for later expansion.
        F.col("merchant").cast("string").alias("merchant"),
        F.col("is_fraud").cast("int").alias("is_fraud"),
        F.col("lat").cast("double").alias("customer_latitude"),
        F.col("long").cast("double").alias("customer_longitude"),
        F.col("unix_time").cast("long").alias("event_time_unix"),
    )


def run(
    spark: "SparkSession",
    input_path: str | Path | None = None,
    output_path: str | Path | None = None,
    config_path: str | Path | None = None,
) -> DataFrame:
    """
    Read Sparkov CSV data, map it into the canonical raw transaction schema, and
    write newline-delimited JSON files for downstream Bronze ingest.
    """
    config = load_sparkov_config(config_path).get("sparkov", {})
    if input_path is None:
        input_path = config.get("input_path", "data/benchmark/sparkov/*.csv")
    if output_path is None:
        output_path = config.get("normalized_raw_path", "data/raw_sparkov")

    fraud_label_value = str(config.get("fraud_label_value", "fraud"))
    non_fraud_label_value = str(config.get("non_fraud_label_value", "none"))

    input_path = str(Path(input_path).resolve())
    output_path = str(Path(output_path).resolve())

    df_source = (
        spark.read.option("header", "true")
        .option("inferSchema", "true")
        .csv(input_path)
    )
    df_canonical = _canonicalize(df_source, fraud_label_value, non_fraud_label_value)
    df_canonical.write.mode("overwrite").json(output_path)
    return df_canonical
