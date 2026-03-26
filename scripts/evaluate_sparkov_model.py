#!/usr/bin/env python3
"""Train and evaluate Sparkov fraud models from Gold features."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from sparkov_eval.cli import parse_args
from sparkov_eval.constants import FEATURE_SETS, THRESHOLD_CANDIDATES
from sparkov_eval.data_prep import (
    apply_time_split_and_sampling,
    build_model_df,
    ensure_gold_columns,
)
from sparkov_eval.k_sweep import run_category_k_sweep
from sparkov_eval.metrics import evaluate_feature_set


def _print_split_summary(model_df, train_fraction, validation_fraction, test_fraction):
    from pyspark.sql import functions as F

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
    print(
        "Sampling fractions:"
        f" train={train_fraction:.2f},"
        f" validation={validation_fraction:.2f},"
        f" test={test_fraction:.2f}"
    )


def main() -> None:
    """Run a time-based Sparkov fraud model evaluation from Gold."""
    from pyspark.sql import Row, SparkSession
    from pyspark.sql import functions as F

    from fraud_lens.ingest import load_sparkov_config

    args = parse_args()
    config = load_sparkov_config().get("sparkov", {})
    spark_builder = SparkSession.builder.appName("FraudLens-Sparkov-Eval")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    try:
        gold_path = project_root / config.get("gold_path", "data/benchmark/gold_sparkov")
        df = spark.read.parquet(str(gold_path.resolve()))
        df = ensure_gold_columns(df)
        model_df = build_model_df(df)
        model_df = apply_time_split_and_sampling(
            model_df=model_df,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )
        _ = model_df.count()

        _print_split_summary(
            model_df=model_df,
            train_fraction=args.train_fraction,
            validation_fraction=args.validation_fraction,
            test_fraction=args.test_fraction,
        )

        if args.category_k_grid:
            run_category_k_sweep(
                model_df=model_df,
                args=args,
                threshold_candidates=THRESHOLD_CANDIDATES,
            )
            return

        feature_sets = dict(FEATURE_SETS)
        if args.feature_set != "all":
            feature_sets = {args.feature_set: feature_sets[args.feature_set]}

        model_types = (
            ["logistic", "gbt"] if args.model_type == "both" else [args.model_type]
        )
        subset_results = []
        for feature_set_name, feature_cols in feature_sets.items():
            for mt in model_types:
                print(f"Evaluating feature set: {feature_set_name} | model: {mt}")
                result = evaluate_feature_set(
                    model_df=model_df,
                    feature_cols=feature_cols,
                    threshold_candidates=THRESHOLD_CANDIDATES,
                    model_type=mt,
                    logistic_class_weights=args.logistic_class_weights,
                    topk_secondary_signal=args.topk_secondary_signal,
                    topk_secondary_epsilon=args.topk_secondary_epsilon,
                )
                result["feature_set_name"] = feature_set_name
                subset_results.append(result)

        results_rows = [
            {k: v for k, v in r.items() if k != "top_k_rows"} for r in subset_results
        ]
        results_df = spark.createDataFrame([Row(**row) for row in results_rows])

        print("Feature subset comparison:")
        results_df.orderBy(F.desc("test_pr_auc")).select(
            "feature_set_name",
            "model_type",
            "feature_count",
            F.round("validation_pr_auc", 4).alias("validation_pr_auc"),
            F.round("test_pr_auc", 4).alias("test_pr_auc"),
            F.round("validation_roc_auc", 4).alias("validation_roc_auc"),
            F.round("test_roc_auc", 4).alias("test_roc_auc"),
            F.round("best_threshold", 2).alias("best_threshold"),
            F.round("test_precision", 4).alias("test_precision"),
            F.round("test_recall", 4).alias("test_recall"),
            F.round("test_f1", 4).alias("test_f1"),
        ).show(truncate=False)

        if args.feature_set != "all":
            for result in subset_results:
                label = f"{result['feature_set_name']} | {result['model_type']}"
                print(f"Top-K test metrics: {label}")
                top_k_df = spark.createDataFrame([Row(**row) for row in result["top_k_rows"]])
                top_k_df.orderBy("k").select(
                    "k",
                    F.round("precision", 4).alias("precision"),
                    F.round("recall", 4).alias("recall"),
                    "tp",
                ).show(truncate=False)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
