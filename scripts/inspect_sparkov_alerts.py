#!/usr/bin/env python3
"""Inspect top-ranked Sparkov alerts for amount-only or amount-plus-night baselines."""

import argparse
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect top-ranked alerts for Sparkov fraud baselines."
    )
    parser.add_argument(
        "--feature-set",
        choices=["amount_only", "amount_plus_night"],
        default="amount_plus_night",
        help="Features used to train the inspector model (default: three-feature baseline).",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "gbt"],
        required=True,
        help="Which model family to inspect.",
    )
    return parser.parse_args()


def main() -> None:
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from fraud_lens.ingest import load_sparkov_config

    args = _parse_args()
    config = load_sparkov_config().get("sparkov", {})
    spark_builder = SparkSession.builder.appName("FraudLens-Sparkov-Alert-Inspect")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    try:
        gold_path = project_root / config.get("gold_path", "data/benchmark/gold_sparkov")
        df = spark.read.parquet(str(gold_path.resolve()))

        feature_cols = (
            ["prior_amount_zscore", "amount_sum_last_1h"]
            if args.feature_set == "amount_only"
            else [
                "prior_amount_zscore",
                "amount_sum_last_1h",
                "is_night_transaction",
            ]
        )
        context_cols = [
            "transaction_id",
            "event_time_unix",
            "event_time_ts",
            "card_id",
            "merchant",
            "merchant_category",
            "amount",
            "prior_amount_zscore",
            "amount_sum_last_1h",
            "is_night_transaction",
            "merchant_prior_fraud_rate",
            "tx_count_last_1h",
            "hour_of_day",
            "is_weekend",
            "is_fraud",
        ]

        fill_map = {
            "prior_amount_zscore": 0.0,
            "amount_sum_last_1h": 0.0,
            "merchant_prior_fraud_rate": 0.0,
            "tx_count_last_1h": 0,
            "is_night_transaction": False,
        }
        model_df = (
            df.where(F.col("is_fraud").isNotNull())
            .select(*context_cols)
            .na.fill(fill_map)
            .withColumn("label", F.col("is_fraud").cast("double"))
        )

        split_q70, split_q85 = model_df.approxQuantile("event_time_unix", [0.70, 0.85], 0.001)
        model_df = model_df.withColumn(
            "split",
            F.when(F.col("event_time_unix") <= F.lit(split_q70), F.lit("train"))
            .when(F.col("event_time_unix") <= F.lit(split_q85), F.lit("validation"))
            .otherwise(F.lit("test")),
        )

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        assembled_df = assembler.transform(model_df).cache()
        _ = assembled_df.count()

        train_df = assembled_df.where(F.col("split") == "train").select("label", "features")
        test_df = assembled_df.where(F.col("split") == "test")

        if args.model_type == "logistic":
            model = LogisticRegression(
                featuresCol="features",
                labelCol="label",
                probabilityCol="probability",
                rawPredictionCol="rawPrediction",
                predictionCol="prediction",
                maxIter=50,
                regParam=0.01,
                elasticNetParam=0.0,
            ).fit(train_df)
        else:
            model = GBTClassifier(
                featuresCol="features",
                labelCol="label",
                maxIter=30,
                maxDepth=5,
                stepSize=0.1,
                seed=42,
            ).fit(train_df)

        scored = model.transform(test_df).select(
            "transaction_id",
            "event_time_unix",
            "event_time_ts",
            "card_id",
            "merchant",
            "merchant_category",
            "amount",
            "prior_amount_zscore",
            "amount_sum_last_1h",
            "is_night_transaction",
            "merchant_prior_fraud_rate",
            "tx_count_last_1h",
            "hour_of_day",
            "is_weekend",
            "label",
            vector_to_array("probability")[1].alias("score"),
        )

        for k in [100, 5000]:
            top_k = scored.orderBy(
                F.desc("score"),
                F.desc("event_time_unix"),
                F.asc("transaction_id"),
            ).limit(k).cache()
            _ = top_k.count()

            print(
                f"\nFeature set: {args.feature_set} | Model: {args.model_type} | Top {k} alerts"
            )
            top_k.groupBy("label").agg(F.count("*").alias("rows")).orderBy("label").show()

            print("Summary by actual label:")
            top_k.groupBy("label").agg(
                F.round(F.avg("score"), 4).alias("avg_score"),
                F.round(F.avg("amount"), 2).alias("avg_amount"),
                F.round(F.avg("prior_amount_zscore"), 2).alias("avg_prior_amount_zscore"),
                F.round(F.avg("amount_sum_last_1h"), 2).alias("avg_amount_sum_last_1h"),
                F.round(
                    F.avg(F.col("is_night_transaction").cast("double")), 4
                ).alias("avg_is_night_transaction"),
                F.round(F.avg("hour_of_day"), 2).alias("avg_hour_of_day"),
                F.round(F.avg("merchant_prior_fraud_rate"), 4).alias("avg_merchant_prior_fraud_rate"),
                F.round(F.avg(F.col("tx_count_last_1h").cast("double")), 2).alias("avg_tx_count_last_1h"),
            ).orderBy("label").show(truncate=False)

            print("Top false-positive merchant categories:")
            top_k.where(F.col("label") == 0.0).groupBy("merchant_category").agg(
                F.count("*").alias("rows"),
                F.round(F.avg("score"), 4).alias("avg_score"),
                F.round(F.avg("amount"), 2).alias("avg_amount"),
            ).orderBy(F.desc("rows"), F.desc("avg_score")).show(10, truncate=False)

            print("Sample false positives:")
            top_k.where(F.col("label") == 0.0).select(
                "transaction_id",
                "event_time_ts",
                "merchant_category",
                "merchant",
                F.round("score", 4).alias("score"),
                F.round("amount", 2).alias("amount"),
                F.round("prior_amount_zscore", 2).alias("prior_amount_zscore"),
                F.round("amount_sum_last_1h", 2).alias("amount_sum_last_1h"),
                "is_night_transaction",
                "hour_of_day",
                F.round("merchant_prior_fraud_rate", 4).alias("merchant_prior_fraud_rate"),
                "tx_count_last_1h",
            ).orderBy(
                F.desc("score"),
                F.desc("event_time_unix"),
                F.asc("transaction_id"),
            ).show(10, truncate=False)

            top_k.unpersist()
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
