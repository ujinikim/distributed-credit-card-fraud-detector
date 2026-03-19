"""Silver → Gold: feature engineering for ML."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from fraud_lens.ingest import load_paths_config

if TYPE_CHECKING:
    from pyspark.sql import SparkSession


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[3]


def _haversine_col(lat1: Column, lon1: Column, lat2: Column, lon2: Column) -> Column:
    """Native Spark SQL great-circle distance in km (no Python UDF)."""
    dlat = F.radians(lat2 - lat1)
    dlon = F.radians(lon2 - lon1)
    a = (
        F.pow(F.sin(dlat / 2), 2)
        + F.cos(F.radians(lat1)) * F.cos(F.radians(lat2)) * F.pow(F.sin(dlon / 2), 2)
    )
    return F.when(
        lat1.isNull() | lon1.isNull() | lat2.isNull() | lon2.isNull(),
        F.lit(None).cast(T.DoubleType()),
    ).otherwise(2 * F.lit(6371.0) * F.asin(F.sqrt(F.least(F.lit(1.0), a))))


def run(
    spark: "SparkSession",
    silver_path: str | Path | None = None,
    gold_path: str | Path | None = None,
) -> DataFrame:
    """
    Build Gold features from Silver transactions.

    Features (per transaction, per card):
    - event_time_ts: parsed timestamp
    - time_since_last_tx_minutes
    - tx_count_last_1h, tx_count_last_24h
    - amount_zscore (per-card)
    - is_amount_spike (boolean flag)
    - distance_from_prev_km, speed_from_prev_kmh
    """
    config = load_paths_config()
    data_cfg = config.get("data", {})

    if silver_path is None:
        silver_path = data_cfg.get("silver", "data/silver")
    if gold_path is None:
        gold_path = data_cfg.get("gold", "data/gold")

    silver_path = str(Path(silver_path).resolve())
    gold_path = str(Path(gold_path).resolve())
    target_write_partitions = max(spark.sparkContext.defaultParallelism, 32)

    df = spark.read.parquet(silver_path)

    # Parse event_time string to timestamp and keep a seconds-since-epoch column for window ranges.
    # Sparkov may already provide event_time_unix; synthetic runs will fall back to parsing event_time.
    df = df.withColumn("event_time_ts", F.to_timestamp("event_time"))
    if "event_time_unix" in df.columns:
        df = df.withColumn(
            "event_time_unix",
            F.coalesce(F.col("event_time_unix"), F.col("event_time_ts").cast("long")),
        )
    else:
        df = df.withColumn("event_time_unix", F.col("event_time_ts").cast("long"))

    # When ref_transaction_id is set (synthetic impossible_travel), we'll compute distance/speed
    # from that ref row instead of from lag(previous row), so pairing is correct at any scale.
    if "ref_transaction_id" in df.columns:
        ref_df = df.select(
            F.col("transaction_id").alias("ref_tx_id"),
            F.col("latitude").alias("ref_lat"),
            F.col("longitude").alias("ref_lon"),
            F.col("event_time_unix").alias("ref_time_unix"),
        )
        df = df.join(ref_df, F.col("ref_transaction_id") == F.col("ref_tx_id"), "left")

    # Keep a dedicated range window (single ORDER BY key) for rangeBetween counts.
    w_by_card_time_range = Window.partitionBy("card_id").orderBy("event_time_unix")
    # Use a deterministic tiebreaker for row-based lag windows when timestamps tie.
    w_by_card_time = Window.partitionBy("card_id").orderBy(
        F.col("event_time_unix"),
        F.col("transaction_id"),
    )
    w_by_card_time_range_1h = w_by_card_time_range.rangeBetween(-3600, 0)
    w_by_card_time_range_24h = w_by_card_time_range.rangeBetween(-24 * 3600, 0)
    w_by_card_all = Window.partitionBy("card_id")

    # Time since last transaction in minutes via lag on the unix timestamp.
    prev_time_unix = F.lag("event_time_unix").over(w_by_card_time)
    df = df.withColumn(
        "time_since_last_tx_minutes",
        F.when(
            prev_time_unix.isNull(),
            F.lit(None).cast(T.DoubleType()),
        ).otherwise(
            (F.col("event_time_unix") - prev_time_unix) / 60.0
        ),
    )

    # Velocity: transaction counts in recent windows.
    df = df.withColumn(
        "tx_count_last_1h",
        F.count("*").over(w_by_card_time_range_1h),
    ).withColumn(
        "tx_count_last_24h",
        F.count("*").over(w_by_card_time_range_24h),
    )

    # Amount z-score per card and spike flag.
    avg_amount = F.avg("amount").over(w_by_card_all)
    std_amount = F.stddev_pop("amount").over(w_by_card_all)
    df = df.withColumn("amount_mean_card", avg_amount).withColumn(
        "amount_std_card", std_amount
    )
    df = df.withColumn(
        "amount_zscore",
        F.when(F.col("amount_std_card") > 0, (F.col("amount") - F.col("amount_mean_card")) / F.col("amount_std_card")).otherwise(
            F.lit(0.0)
        ),
    ).withColumn(
        "is_amount_spike",
        F.col("amount_zscore") >= F.lit(3.0),
    )

    # Distance and speed relative to previous transaction for impossible travel.
    prev_lat = F.lag("latitude").over(w_by_card_time)
    prev_lon = F.lag("longitude").over(w_by_card_time)
    prev_unix = F.lag("event_time_unix").over(w_by_card_time)

    df = df.withColumn(
        "distance_from_prev_km",
        _haversine_col(
            F.col("latitude"),
            F.col("longitude"),
            prev_lat,
            prev_lon,
        ),
    )

    df = df.withColumn(
        "hours_since_prev",
        (F.col("event_time_unix") - prev_unix) / 3600.0,
    )

    df = df.withColumn(
        "speed_from_prev_kmh",
        F.when(
            (F.col("hours_since_prev").isNull()) | (F.col("hours_since_prev") <= 0),
            F.lit(None),
        ).otherwise(F.col("distance_from_prev_km") / F.col("hours_since_prev")),
    )

    # Overwrite distance/hours/speed from ref row when ref_transaction_id is set (impossible_travel).
    if "ref_tx_id" in df.columns:
        dist_ref = _haversine_col(
            F.col("latitude"), F.col("longitude"), F.col("ref_lat"), F.col("ref_lon")
        )
        hours_ref = (F.col("event_time_unix") - F.col("ref_time_unix")) / 3600.0
        speed_ref = F.when(
            hours_ref.isNotNull() & (hours_ref > 0), dist_ref / hours_ref
        ).otherwise(F.lit(None))
        df = df.withColumn(
            "distance_from_prev_km",
            F.when(F.col("ref_tx_id").isNotNull(), dist_ref).otherwise(
                F.col("distance_from_prev_km")
            ),
        ).withColumn(
            "hours_since_prev",
            F.when(F.col("ref_tx_id").isNotNull(), hours_ref).otherwise(
                F.col("hours_since_prev")
            ),
        ).withColumn(
            "speed_from_prev_kmh",
            F.when(F.col("ref_tx_id").isNotNull(), speed_ref).otherwise(
                F.col("speed_from_prev_kmh")
            ),
        )
        df = df.drop("ref_tx_id", "ref_lat", "ref_lon", "ref_time_unix")

    # Select and order columns: keep join keys + features + labels.
    feature_cols = [
        "time_since_last_tx_minutes",
        "tx_count_last_1h",
        "tx_count_last_24h",
        "amount_zscore",
        "is_amount_spike",
        "distance_from_prev_km",
        "hours_since_prev",
        "speed_from_prev_kmh",
    ]

    df_gold = df.select(
        "transaction_id",
        "card_id",
        "event_time",
        "event_time_ts",
        "event_time_unix",
        "amount",
        "merchant_category",
        "merchant",
        "latitude",
        "longitude",
        "customer_latitude",
        "customer_longitude",
        "anomaly_type",
        "is_fraud",
        "ref_transaction_id",
        "ingestion_timestamp",
        "source_path",
        *feature_cols,
    )

    (
        df_gold.repartition(target_write_partitions)
        .write.mode("overwrite")
        .option("parquet.enable.dictionary", "false")
        .parquet(gold_path)
    )
    return df_gold
