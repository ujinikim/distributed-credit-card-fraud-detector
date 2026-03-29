"""Data preparation helpers for Sparkov evaluation."""

from fraud_lens.benchmark.sparkov.eval.constants import (
    BASE_FEATURE_COLS,
    MISSING_GOLD_COLUMN_CASTS,
    MODEL_FILL_DEFAULTS,
)


def ensure_gold_columns(df):
    """Backfill optional Gold columns for older parquet snapshots."""
    from pyspark.sql import functions as F

    for col_name, cast_type in MISSING_GOLD_COLUMN_CASTS.items():
        if col_name not in df.columns:
            if cast_type == "boolean":
                default_value = F.lit(False)
            else:
                default_value = F.lit(None).cast(cast_type)
            df = df.withColumn(col_name, default_value)
    return df


def build_model_df(df):
    """Select and normalize model input columns from Gold data."""
    from pyspark.sql import functions as F

    base = (
        df.where(F.col("is_fraud").isNotNull())
        .select(
            "transaction_id",
            "event_time_unix",
            "is_fraud",
            "amount",
            "hour_of_day",
            *BASE_FEATURE_COLS,
            "prior_amount_zscore_card_category",
            "prior_category_zscore_eligible",
            "prior_amount_zscore_card_category_damped",
            "prior_amount_zscore_card_category_shrunk",
            "prior_category_log_prior_n",
            "low_history_card_category",
            "prior_tx_count_card_category",
            "prior_amount_mean_card_category",
            "prior_amount_std_card_category",
        )
        .na.fill(MODEL_FILL_DEFAULTS)
        .withColumn("label", F.col("is_fraud").cast("double"))
    )

    prior_amount_zscore_clipped = F.greatest(
        F.lit(-10.0), F.least(F.col("prior_amount_zscore"), F.lit(10.0))
    )
    return (
        base.withColumn("prior_amount_zscore_clipped", prior_amount_zscore_clipped)
        .withColumn("amount_sum_last_1h_log1p", F.log1p(F.col("amount_sum_last_1h")))
        .withColumn(
            "amount_zscore_x_lowcat",
            F.col("prior_amount_zscore_clipped")
            * F.col("low_history_card_category").cast("double"),
        )
    )


def apply_time_split_and_sampling(model_df, train_fraction, validation_fraction, test_fraction):
    """Create train/validation/test time splits and optionally sample each."""
    from pyspark.sql import functions as F

    split_q70, split_q85 = model_df.approxQuantile("event_time_unix", [0.70, 0.85], 0.001)
    model_df = model_df.withColumn(
        "split",
        F.when(F.col("event_time_unix") <= F.lit(split_q70), F.lit("train"))
        .when(F.col("event_time_unix") <= F.lit(split_q85), F.lit("validation"))
        .otherwise(F.lit("test")),
    )
    fractions = {
        "train": train_fraction,
        "validation": validation_fraction,
        "test": test_fraction,
    }
    sampled_parts = []
    for split_name, fraction in fractions.items():
        split_part = model_df.where(F.col("split") == split_name)
        if fraction < 1.0:
            split_part = split_part.sample(
                withReplacement=False,
                fraction=fraction,
                seed=42,
            )
        sampled_parts.append(split_part)
    model_df = sampled_parts[0].unionByName(sampled_parts[1]).unionByName(sampled_parts[2])
    return model_df.cache()
