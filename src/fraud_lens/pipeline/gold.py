"""Canonical Silver-to-Gold feature engineering for the medallion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from pyspark.sql import DataFrame, Column
from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.sql.window import Window

from fraud_lens.pipeline.paths import load_paths_config

if TYPE_CHECKING:
    from pyspark.sql import SparkSession

# Minimum prior (card × merchant_category) tx count before trusting category z-score.
PRIOR_CATEGORY_Z_MIN_TX = 5
# Shrink category prior mean toward prior card mean: (n*mu_cat + tau*mu_card) / (n + tau).
PRIOR_CATEGORY_SHRINK_TAU = 5.0
# Damp raw category z when eligible: z * sqrt(n / (n + k)).
PRIOR_CATEGORY_Z_DAMP_K = 5.0


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


def _ensure_optional_columns(df: DataFrame) -> DataFrame:
    """Add nullable optional columns when reading older shared Silver outputs."""
    optional_cols: dict[str, Column] = {
        "merchant": F.lit(None).cast(T.StringType()),
        "is_fraud": F.lit(None).cast(T.IntegerType()),
        "customer_latitude": F.lit(None).cast(T.DoubleType()),
        "customer_longitude": F.lit(None).cast(T.DoubleType()),
        "event_time_unix": F.lit(None).cast(T.LongType()),
    }
    for col_name, default_col in optional_cols.items():
        if col_name not in df.columns:
            df = df.withColumn(col_name, default_col)
    return df


def run_gold_features(
    spark: "SparkSession",
    silver_path: str | Path | None = None,
    gold_path: str | Path | None = None,
) -> DataFrame:
    """
    Build Gold features from Silver transactions.

    Features (per transaction, per card):
    - event_time_ts: parsed timestamp
    - time/calendar context
    - velocity and amount windows
    - amount_zscore and prior-only amount_zscore (per-card)
    - is_amount_spike (boolean flag)
    - customer-to-merchant distance
    - merchant history features when benchmark columns are available
    - distance_from_prev_km, speed_from_prev_kmh
    - prior-only amount stats per (card_id, merchant_category) with gated z-score,
      shrunk mean, damped z, and log1p(prior category count)
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

    df = _ensure_optional_columns(spark.read.parquet(silver_path))

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
    w_by_card_time_prior_rows = w_by_card_time.rowsBetween(
        Window.unboundedPreceding, -1
    )
    w_by_card_time_range_1h = w_by_card_time_range.rangeBetween(-3600, 0)
    w_by_card_time_range_5m = w_by_card_time_range.rangeBetween(-5 * 60, 0)
    w_by_card_time_range_15m = w_by_card_time_range.rangeBetween(-15 * 60, 0)
    w_by_card_time_range_24h = w_by_card_time_range.rangeBetween(-24 * 3600, 0)
    w_by_card_all = Window.partitionBy("card_id")
    w_by_merchant_time_range = Window.partitionBy("merchant").orderBy("event_time_unix")
    w_by_merchant_time = Window.partitionBy("merchant").orderBy(
        F.col("event_time_unix"),
        F.col("transaction_id"),
    )
    w_by_merchant_time_prior_rows = w_by_merchant_time.rowsBetween(
        Window.unboundedPreceding, -1
    )
    w_by_merchant_time_range_24h = w_by_merchant_time_range.rangeBetween(-24 * 3600, 0)
    w_by_card_merchant_time = Window.partitionBy("card_id", "merchant").orderBy(
        F.col("event_time_unix"),
        F.col("transaction_id"),
    )
    w_by_card_merchant_time_prior_rows = w_by_card_merchant_time.rowsBetween(
        Window.unboundedPreceding, -1
    )
    w_by_card_category_time = Window.partitionBy("card_id", "merchant_category").orderBy(
        F.col("event_time_unix"),
        F.col("transaction_id"),
    )
    w_by_card_category_prior_rows = w_by_card_category_time.rowsBetween(
        Window.unboundedPreceding, -1
    )

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

    # Calendar context from the canonical transaction time.
    df = (
        df.withColumn("hour_of_day", F.hour("event_time_ts"))
        .withColumn("day_of_week", F.dayofweek("event_time_ts"))
        .withColumn(
            "is_weekend",
            F.col("day_of_week").isin(1, 7),
        )
        .withColumn(
            "is_night_transaction",
            (F.col("hour_of_day") < 6) | (F.col("hour_of_day") >= 22),
        )
    )

    # Velocity: transaction counts in recent windows.
    df = (
        df.withColumn("tx_count_last_5m", F.count("*").over(w_by_card_time_range_5m))
        .withColumn(
            "tx_count_last_15m",
            F.count("*").over(w_by_card_time_range_15m),
        )
        .withColumn("tx_count_last_1h", F.count("*").over(w_by_card_time_range_1h))
        .withColumn(
            "tx_count_last_24h",
            F.count("*").over(w_by_card_time_range_24h),
        )
        .withColumn("amount_sum_last_1h", F.sum("amount").over(w_by_card_time_range_1h))
        .withColumn(
            "amount_sum_last_24h",
            F.sum("amount").over(w_by_card_time_range_24h),
        )
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

    # Prior-only amount history avoids leaking the current transaction into its own baseline.
    prior_tx_count_card = F.count("amount").over(w_by_card_time_prior_rows)
    prior_amount_sum_card = F.sum("amount").over(w_by_card_time_prior_rows)
    prior_amount_sumsq_card = F.sum(F.pow(F.col("amount"), 2.0)).over(
        w_by_card_time_prior_rows
    )
    prior_amount_mean_card = prior_amount_sum_card / prior_tx_count_card
    prior_amount_variance_card = F.greatest(
        (prior_amount_sumsq_card / prior_tx_count_card)
        - F.pow(prior_amount_mean_card, 2.0),
        F.lit(0.0),
    )
    prior_amount_std_card = F.sqrt(prior_amount_variance_card)
    df = (
        df.withColumn("prior_tx_count_card", prior_tx_count_card)
        .withColumn("prior_amount_mean_card", prior_amount_mean_card)
        .withColumn("prior_amount_std_card", prior_amount_std_card)
        .withColumn(
            "prior_amount_zscore",
            F.when(
                (F.col("prior_tx_count_card") > 1)
                & (F.col("prior_amount_std_card") > 0),
                (F.col("amount") - F.col("prior_amount_mean_card"))
                / F.col("prior_amount_std_card"),
            ).otherwise(F.lit(None).cast(T.DoubleType())),
        )
        .withColumn(
            "is_amount_spike_prior",
            F.when(
                F.col("prior_amount_zscore").isNull(),
                F.lit(False),
            ).otherwise(F.col("prior_amount_zscore") >= F.lit(3.0)),
        )
    )

    # Prior-only amount history per (card_id, merchant_category): gated z-score when n is large enough.
    prior_tx_count_cc = F.count("amount").over(w_by_card_category_prior_rows)
    prior_amount_sum_cc = F.sum("amount").over(w_by_card_category_prior_rows)
    prior_amount_sumsq_cc = F.sum(F.pow(F.col("amount"), 2.0)).over(
        w_by_card_category_prior_rows
    )
    prior_amount_mean_cc = prior_amount_sum_cc / prior_tx_count_cc
    prior_amount_variance_cc = F.greatest(
        (prior_amount_sumsq_cc / prior_tx_count_cc)
        - F.pow(prior_amount_mean_cc, 2.0),
        F.lit(0.0),
    )
    prior_amount_std_cc = F.sqrt(prior_amount_variance_cc)
    eligible_cat_z = (prior_tx_count_cc >= F.lit(PRIOR_CATEGORY_Z_MIN_TX)) & (
        prior_amount_std_cc > F.lit(0.0)
    )
    # Prior-only card mean is already on df; blend category mean toward it when n>0 (V2 shrinkage).
    prior_amount_mean_card_category_shrunk = F.when(
        prior_tx_count_cc > 0,
        (
            prior_tx_count_cc * prior_amount_mean_cc
            + F.lit(PRIOR_CATEGORY_SHRINK_TAU) * F.col("prior_amount_mean_card")
        )
        / (prior_tx_count_cc + F.lit(PRIOR_CATEGORY_SHRINK_TAU)),
    ).otherwise(F.col("prior_amount_mean_card"))
    z_cat_raw = (F.col("amount") - prior_amount_mean_cc) / prior_amount_std_cc
    z_cat_damp = F.sqrt(
        prior_tx_count_cc.cast("double")
        / (prior_tx_count_cc.cast("double") + F.lit(PRIOR_CATEGORY_Z_DAMP_K))
    )
    z_cat_shrunk = (F.col("amount") - prior_amount_mean_card_category_shrunk) / prior_amount_std_cc
    df = (
        df.withColumn("prior_tx_count_card_category", prior_tx_count_cc)
        .withColumn("prior_amount_mean_card_category", prior_amount_mean_cc)
        .withColumn("prior_amount_std_card_category", prior_amount_std_cc)
        .withColumn(
            "prior_amount_mean_card_category_shrunk",
            prior_amount_mean_card_category_shrunk,
        )
        .withColumn(
            "prior_amount_zscore_card_category",
            F.when(eligible_cat_z, z_cat_raw).otherwise(
                F.lit(None).cast(T.DoubleType())
            ),
        )
        .withColumn(
            "prior_amount_zscore_card_category_damped",
            F.when(eligible_cat_z, z_cat_raw * z_cat_damp).otherwise(
                F.lit(None).cast(T.DoubleType())
            ),
        )
        .withColumn(
            "prior_amount_zscore_card_category_shrunk",
            F.when(eligible_cat_z, z_cat_shrunk).otherwise(
                F.lit(None).cast(T.DoubleType())
            ),
        )
        .withColumn("prior_category_zscore_eligible", eligible_cat_z)
        .withColumn(
            "low_history_card_category",
            prior_tx_count_cc < F.lit(PRIOR_CATEGORY_Z_MIN_TX),
        )
        .withColumn(
            "prior_category_log_prior_n",
            F.log1p(prior_tx_count_cc.cast("double")),
        )
    )

    # Sparkov-aware geography features remain null-safe for synthetic runs.
    df = df.withColumn(
        "customer_to_merchant_distance_km",
        _haversine_col(
            F.col("customer_latitude"),
            F.col("customer_longitude"),
            F.col("latitude"),
            F.col("longitude"),
        ),
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

    # Merchant history features are benchmark-aware and should stay null for synthetic rows.
    prior_tx_count_merchant = F.count("transaction_id").over(w_by_merchant_time_prior_rows)
    prior_fraud_sum_merchant = F.sum(F.coalesce(F.col("is_fraud"), F.lit(0))).over(
        w_by_merchant_time_prior_rows
    )
    prior_repeat_merchant_count = F.count("transaction_id").over(
        w_by_card_merchant_time_prior_rows
    )
    df = (
        df.withColumn(
            "merchant_tx_count_last_24h",
            F.when(
                F.col("merchant").isNotNull(),
                F.count("*").over(w_by_merchant_time_range_24h),
            ).otherwise(F.lit(None).cast(T.LongType())),
        )
        .withColumn(
            "merchant_prior_tx_count",
            F.when(
                F.col("merchant").isNotNull(),
                prior_tx_count_merchant,
            ).otherwise(F.lit(None).cast(T.LongType())),
        )
        .withColumn(
            "merchant_prior_fraud_rate",
            F.when(
                F.col("merchant").isNotNull() & (prior_tx_count_merchant > 0),
                prior_fraud_sum_merchant / prior_tx_count_merchant,
            ).otherwise(F.lit(None).cast(T.DoubleType())),
        )
        .withColumn(
            "card_merchant_repeat_count_prior",
            F.when(
                F.col("merchant").isNotNull(),
                prior_repeat_merchant_count,
            ).otherwise(F.lit(None).cast(T.LongType())),
        )
    )

    # Select and order columns: keep join keys + features + labels.
    feature_cols = [
        "hour_of_day",
        "day_of_week",
        "is_weekend",
        "is_night_transaction",
        "time_since_last_tx_minutes",
        "tx_count_last_5m",
        "tx_count_last_15m",
        "tx_count_last_1h",
        "tx_count_last_24h",
        "amount_sum_last_1h",
        "amount_sum_last_24h",
        "amount_zscore",
        "is_amount_spike",
        "prior_tx_count_card",
        "prior_amount_mean_card",
        "prior_amount_std_card",
        "prior_amount_zscore",
        "is_amount_spike_prior",
        "prior_tx_count_card_category",
        "prior_amount_mean_card_category",
        "prior_amount_std_card_category",
        "prior_amount_mean_card_category_shrunk",
        "prior_amount_zscore_card_category",
        "prior_amount_zscore_card_category_damped",
        "prior_amount_zscore_card_category_shrunk",
        "prior_category_zscore_eligible",
        "low_history_card_category",
        "prior_category_log_prior_n",
        "customer_to_merchant_distance_km",
        "merchant_tx_count_last_24h",
        "merchant_prior_tx_count",
        "merchant_prior_fraud_rate",
        "card_merchant_repeat_count_prior",
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


run = run_gold_features
