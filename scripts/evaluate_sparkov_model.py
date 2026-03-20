#!/usr/bin/env python3
"""Train and evaluate a first Sparkov fraud model from Gold features."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def _threshold_metrics(df, threshold: float) -> dict[str, float]:
    """Compute basic threshold metrics from probability scores."""
    from pyspark.sql import functions as F

    scored = df.withColumn(
        "predicted_label",
        (F.col("score") >= F.lit(threshold)).cast("int"),
    )
    metrics = scored.agg(
        F.sum(
            ((F.col("label") == 1.0) & (F.col("predicted_label") == 1)).cast("long")
        ).alias("tp"),
        F.sum(
            ((F.col("label") == 0.0) & (F.col("predicted_label") == 1)).cast("long")
        ).alias("fp"),
        F.sum(
            ((F.col("label") == 0.0) & (F.col("predicted_label") == 0)).cast("long")
        ).alias("tn"),
        F.sum(
            ((F.col("label") == 1.0) & (F.col("predicted_label") == 0)).cast("long")
        ).alias("fn"),
    ).first()

    tp = int(metrics["tp"] or 0)
    fp = int(metrics["fp"] or 0)
    tn = int(metrics["tn"] or 0)
    fn = int(metrics["fn"] or 0)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall)
        else 0.0
    )
    return {
        "threshold": threshold,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
        "f1": f1,
    }


def main() -> None:
    """Run a first time-based Sparkov fraud model evaluation from Gold."""
    from pyspark.ml.classification import LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from fraud_lens.ingest import load_sparkov_config

    config = load_sparkov_config().get("sparkov", {})
    spark_builder = SparkSession.builder.appName("FraudLens-Sparkov-Eval")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    try:
        gold_path = project_root / config.get("gold_path", "data/benchmark/gold_sparkov")
        df = spark.read.parquet(str(gold_path.resolve()))

        feature_cols = [
            "prior_amount_zscore",
            "amount_sum_last_1h",
            "merchant_prior_fraud_rate",
            "tx_count_last_15m",
            "tx_count_last_1h",
        ]

        # Keep the first pass simple: use the strongest/most promising numeric features,
        # fill early-history nulls with 0, and evaluate using time-based splits.
        model_df = (
            df.where(F.col("is_fraud").isNotNull())
            .select("event_time_unix", "is_fraud", *feature_cols)
            .na.fill(
                {
                    "prior_amount_zscore": 0.0,
                    "amount_sum_last_1h": 0.0,
                    "merchant_prior_fraud_rate": 0.0,
                    "tx_count_last_15m": 0,
                    "tx_count_last_1h": 0,
                }
            )
            .withColumn("label", F.col("is_fraud").cast("double"))
        )

        split_q70, split_q85 = model_df.approxQuantile("event_time_unix", [0.70, 0.85], 0.001)
        model_df = model_df.withColumn(
            "split",
            F.when(F.col("event_time_unix") <= F.lit(split_q70), F.lit("train"))
            .when(F.col("event_time_unix") <= F.lit(split_q85), F.lit("validation"))
            .otherwise(F.lit("test")),
        )

        split_summary = (
            model_df.groupBy("split")
            .agg(
                F.count("*").alias("rows"),
                F.sum(F.col("label").cast("long")).alias("fraud_rows"),
                F.round(F.avg("label"), 6).alias("fraud_rate"),
            )
            .orderBy(
                F.when(F.col("split") == "train", 0)
                .when(F.col("split") == "validation", 1)
                .otherwise(2)
            )
        )

        print("Time-based split summary:")
        split_summary.show(truncate=False)

        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        assembled_df = assembler.transform(model_df)

        train_df = assembled_df.where(F.col("split") == "train").select(
            "label", "features"
        )
        validation_df = assembled_df.where(F.col("split") == "validation").select(
            "label", "features"
        )
        test_df = assembled_df.where(F.col("split") == "test").select("label", "features")

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

        validation_pred = model.transform(validation_df).select(
            "label",
            vector_to_array("probability")[1].alias("score"),
            "rawPrediction",
            "prediction",
        )
        test_pred = model.transform(test_df).select(
            "label",
            vector_to_array("probability")[1].alias("score"),
            "rawPrediction",
            "prediction",
        )

        pr_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderPR",
        )
        roc_evaluator = BinaryClassificationEvaluator(
            labelCol="label",
            rawPredictionCol="rawPrediction",
            metricName="areaUnderROC",
        )

        print("Validation ranking metrics:")
        print(f"  PR AUC:  {pr_evaluator.evaluate(validation_pred):.4f}")
        print(f"  ROC AUC: {roc_evaluator.evaluate(validation_pred):.4f}")

        threshold_candidates = [x / 100 for x in range(5, 100, 5)]
        validation_metrics = [
            _threshold_metrics(validation_pred, threshold)
            for threshold in threshold_candidates
        ]
        best_validation = max(validation_metrics, key=lambda row: row["f1"])

        print(
            "Best validation threshold by F1: "
            f"{best_validation['threshold']:.2f}"
        )
        print(
            "  precision={:.4f} recall={:.4f} f1={:.4f}".format(
                best_validation["precision"],
                best_validation["recall"],
                best_validation["f1"],
            )
        )

        test_pr_auc = pr_evaluator.evaluate(test_pred)
        test_roc_auc = roc_evaluator.evaluate(test_pred)
        test_metrics = _threshold_metrics(test_pred, best_validation["threshold"])

        print("Test ranking metrics:")
        print(f"  PR AUC:  {test_pr_auc:.4f}")
        print(f"  ROC AUC: {test_roc_auc:.4f}")
        print("Test threshold metrics:")
        print(
            "  threshold={:.2f} precision={:.4f} recall={:.4f} f1={:.4f} accuracy={:.4f}".format(
                test_metrics["threshold"],
                test_metrics["precision"],
                test_metrics["recall"],
                test_metrics["f1"],
                test_metrics["accuracy"],
            )
        )
        print(
            "  tp={tp} fp={fp} tn={tn} fn={fn}".format(
                tp=test_metrics["tp"],
                fp=test_metrics["fp"],
                tn=test_metrics["tn"],
                fn=test_metrics["fn"],
            )
        )

        coefficient_rows = list(zip(feature_cols, model.coefficients.toArray()))
        coefficient_rows.sort(key=lambda row: abs(row[1]), reverse=True)
        print("Model coefficients (sorted by absolute value):")
        for feature_name, coefficient in coefficient_rows:
            print(f"  {feature_name}: {coefficient:.6f}")
        print(f"  intercept: {model.intercept:.6f}")
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
