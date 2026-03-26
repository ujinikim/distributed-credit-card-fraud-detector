#!/usr/bin/env python3
"""Train and evaluate Sparkov fraud models from Gold features."""

import argparse
import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


BASE_FEATURE_COLS = [
    "prior_amount_zscore",
    "amount_sum_last_1h",
    "is_night_transaction",
    "merchant_prior_fraud_rate",
    "tx_count_last_15m",
    "tx_count_last_1h",
]

FEATURE_SETS = {
    "amount_only": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
    ],
    "amount_plus_night": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "is_night_transaction",
    ],
    "amount_plus_night_hour": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "is_night_transaction",
        "hour_of_day",
    ],
    "amount_plus_night_catz": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "is_night_transaction",
        "prior_amount_zscore_card_category",
        "prior_category_zscore_eligible",
    ],
    "amount_plus_night_catz_v2": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "is_night_transaction",
        "prior_amount_zscore_card_category_damped",
        "prior_category_zscore_eligible",
        "prior_category_log_prior_n",
    ],
    "amount_plus_night_catz_v3_shrunk": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "is_night_transaction",
        "prior_amount_zscore_card_category_shrunk",
        "prior_category_zscore_eligible",
        "prior_category_log_prior_n",
    ],
    "amount_plus_night_catz_v3_damped_shrunk": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "is_night_transaction",
        "prior_amount_zscore_card_category_damped",
        "prior_amount_zscore_card_category_shrunk",
        "prior_category_zscore_eligible",
        "prior_category_log_prior_n",
    ],
    "amount_plus_tx1h": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "tx_count_last_1h",
    ],
    "amount_plus_merchant": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "merchant_prior_fraud_rate",
    ],
    "amount_plus_velocity": [
        "prior_amount_zscore",
        "amount_sum_last_1h",
        "tx_count_last_15m",
        "tx_count_last_1h",
    ],
    "full_baseline": BASE_FEATURE_COLS,
}

FEATURE_SET_CHOICES = ["all", *FEATURE_SETS.keys()]

TOP_K_VALUES = [100, 500, 1000, 5000, 10000]

MISSING_GOLD_COLUMN_CASTS = {
    "prior_amount_zscore_card_category": "double",
    "prior_category_zscore_eligible": "boolean",
    "prior_amount_zscore_card_category_damped": "double",
    "prior_amount_zscore_card_category_shrunk": "double",
    "prior_category_log_prior_n": "double",
    "prior_tx_count_card_category": "long",
    "prior_amount_mean_card_category": "double",
    "prior_amount_std_card_category": "double",
}

MODEL_FILL_DEFAULTS = {
    "prior_amount_zscore": 0.0,
    "amount_sum_last_1h": 0.0,
    "is_night_transaction": False,
    "hour_of_day": 0,
    "merchant_prior_fraud_rate": 0.0,
    "tx_count_last_15m": 0,
    "tx_count_last_1h": 0,
    "prior_amount_zscore_card_category": 0.0,
    "prior_category_zscore_eligible": False,
    "prior_amount_zscore_card_category_damped": 0.0,
    "prior_amount_zscore_card_category_shrunk": 0.0,
    "prior_category_log_prior_n": 0.0,
    "prior_tx_count_card_category": 0,
    "prior_amount_mean_card_category": 0.0,
    "prior_amount_std_card_category": 0.0,
}


def _ensure_gold_columns(df):
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


def _top_k_metrics(df, k: int) -> dict[str, float]:
    """Compute top-K alerting metrics from scored rows."""
    from pyspark.sql import functions as F

    top_k = df.orderBy(
        F.desc("score"),
        F.desc("event_time_unix"),
        F.asc("transaction_id"),
    ).limit(k)
    positives_in_dataset = df.agg(
        F.sum(F.col("label").cast("long")).alias("positives")
    ).first()
    positives = int(positives_in_dataset["positives"] or 0)
    metrics = top_k.agg(
        F.count("*").alias("rows"),
        F.sum(F.col("label").cast("long")).alias("tp"),
    ).first()
    rows = int(metrics["rows"] or 0)
    tp = int(metrics["tp"] or 0)
    precision = tp / rows if rows else 0.0
    recall = tp / positives if positives else 0.0
    return {
        "k": k,
        "rows": rows,
        "tp": tp,
        "precision": precision,
        "recall": recall,
    }


def _evaluate_feature_set(
    spark,
    model_df,
    feature_cols: list[str],
    threshold_candidates: list[float],
    model_type: str,
    compute_validation_and_auc: bool = True,
    logistic_class_weights: bool = False,
) -> dict[str, float]:
    """Train and score one feature subset.

    When `compute_validation_and_auc=False`, we only compute Top-K metrics on the test split
    (fast path for hyperparameter / K sweeps).
    """
    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F

    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    assembled_df = assembler.transform(model_df)

    train_df = assembled_df.where(F.col("split") == "train").select("label", "features")
    test_df = assembled_df.where(F.col("split") == "test").select(
        "transaction_id",
        "event_time_unix",
        "label",
        "features",
    )

    if model_type == "logistic":
        # Optional: counter class imbalance so logistic training doesn't overfit to
        # the dominant negative class.
        if logistic_class_weights:
            class_counts = train_df.agg(
                F.sum((F.col("label") == F.lit(1.0)).cast("int")).alias("pos"),
                F.sum((F.col("label") == F.lit(0.0)).cast("int")).alias("neg"),
            ).first()
            pos_count = int(class_counts["pos"] or 0)
            neg_count = int(class_counts["neg"] or 0)
            pos_weight = (neg_count / pos_count) if pos_count else 1.0
            train_df = train_df.withColumn(
                "weight",
                F.when(F.col("label") == F.lit(1.0), F.lit(pos_weight)).otherwise(
                    F.lit(1.0)
                ),
            )
        else:
            train_df = train_df.withColumn("weight", F.lit(1.0))

        lr_kwargs = {
            "featuresCol": "features",
            "labelCol": "label",
            "probabilityCol": "probability",
            "rawPredictionCol": "rawPrediction",
            "predictionCol": "prediction",
            "maxIter": 50,
            "regParam": 0.01,
            "elasticNetParam": 0.0,
        }
        if logistic_class_weights:
            lr_kwargs["weightCol"] = "weight"
        model = LogisticRegression(**lr_kwargs).fit(train_df)
    else:
        model = GBTClassifier(
            featuresCol="features",
            labelCol="label",
            maxIter=30,
            maxDepth=5,
            stepSize=0.1,
            seed=42,
        ).fit(train_df)

    test_pred = model.transform(test_df).select(
        "transaction_id",
        "event_time_unix",
        "label",
        vector_to_array("probability")[1].alias("score"),
    )

    # Fast path: only compute Top-K (skip threshold scanning + AUC).
    top_k_rows = []
    for k in TOP_K_VALUES:
        top_k = _top_k_metrics(test_pred, k)
        top_k_rows.append(
            {
                "k": k,
                "precision": top_k["precision"],
                "recall": top_k["recall"],
                "tp": top_k["tp"],
            }
        )

    if not compute_validation_and_auc:
        return {
            "feature_set": "+".join(feature_cols),
            "feature_count": len(feature_cols),
            "model_type": model_type,
            "validation_pr_auc": None,
            "validation_roc_auc": None,
            "best_threshold": None,
            "validation_precision": None,
            "validation_recall": None,
            "validation_f1": None,
            "test_pr_auc": None,
            "test_roc_auc": None,
            "test_precision": None,
            "test_recall": None,
            "test_f1": None,
            "top_k_rows": top_k_rows,
        }

    validation_df = assembled_df.where(F.col("split") == "validation").select(
        "transaction_id",
        "event_time_unix",
        "label",
        "features",
    )

    validation_pred = model.transform(validation_df).select(
        "transaction_id",
        "event_time_unix",
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

    validation_pr_auc = pr_evaluator.evaluate(validation_pred)
    validation_roc_auc = roc_evaluator.evaluate(validation_pred)
    validation_metrics = [
        _threshold_metrics(validation_pred, threshold)
        for threshold in threshold_candidates
    ]
    best_validation = max(validation_metrics, key=lambda row: row["f1"])

    test_pred_full = model.transform(test_df).select(
        "transaction_id",
        "event_time_unix",
        "label",
        vector_to_array("probability")[1].alias("score"),
        "rawPrediction",
        "prediction",
    )

    test_pr_auc = pr_evaluator.evaluate(test_pred_full)
    test_roc_auc = roc_evaluator.evaluate(test_pred_full)
    test_metrics = _threshold_metrics(test_pred_full, best_validation["threshold"])

    return {
        "feature_set": "+".join(feature_cols),
        "feature_count": len(feature_cols),
        "model_type": model_type,
        "validation_pr_auc": validation_pr_auc,
        "validation_roc_auc": validation_roc_auc,
        "best_threshold": best_validation["threshold"],
        "validation_precision": best_validation["precision"],
        "validation_recall": best_validation["recall"],
        "validation_f1": best_validation["f1"],
        "test_pr_auc": test_pr_auc,
        "test_roc_auc": test_roc_auc,
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "top_k_rows": top_k_rows,
    }


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for faster experiment loops."""
    parser = argparse.ArgumentParser(
        description="Evaluate Sparkov fraud models from Gold features."
    )
    parser.add_argument(
        "--train-fraction",
        type=float,
        default=1.0,
        help="Fraction of train rows to sample after time-based splitting.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=1.0,
        help="Fraction of validation rows to sample after time-based splitting.",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=1.0,
        help="Fraction of test rows to sample after time-based splitting.",
    )
    parser.add_argument(
        "--feature-set",
        choices=FEATURE_SET_CHOICES,
        default="all",
        help="Which predefined feature subset to evaluate.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "gbt", "both"],
        default="logistic",
        help="Model family: logistic, gbt, or both (trains each for the selected feature set(s)).",
    )
    parser.add_argument(
        "--logistic-class-weights",
        action="store_true",
        help="Use inverse-frequency class weighting for logistic regression (improves PR/F1 under imbalance).",
    )
    parser.add_argument(
        "--category-k-grid",
        type=str,
        default="",
        help="Optional sweep over category z gating threshold K. Format: e.g. '2,3,5,8,10,15'.",
    )
    parser.add_argument(
        "--category-z-variant",
        choices=["raw", "damped", "both"],
        default="damped",
        help="Which category z variant(s) to use during K sweep: raw (Run 13), damped + log1p(n) (Run 14), or both.",
    )
    parser.add_argument(
        "--topk-primary",
        type=int,
        default=5000,
        help="Primary ranking metric during K sweep (precision@k). Default: 5000.",
    )
    parser.add_argument(
        "--topk-tie-break",
        type=int,
        default=10000,
        help="Tie-break metric during K sweep (precision@k). Default: 10000.",
    )
    parser.add_argument(
        "--topk-sanity",
        type=int,
        default=100,
        help="Sanity-check metric during K sweep (precision@k). Default: 100.",
    )
    return parser.parse_args()


def main() -> None:
    """Run a time-based Sparkov fraud model evaluation from Gold."""
    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F
    from pyspark.sql import Row

    from fraud_lens.ingest import load_sparkov_config

    args = _parse_args()
    config = load_sparkov_config().get("sparkov", {})
    spark_builder = SparkSession.builder.appName("FraudLens-Sparkov-Eval")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    try:
        gold_path = project_root / config.get("gold_path", "data/benchmark/gold_sparkov")
        df = spark.read.parquet(str(gold_path.resolve()))
        df = _ensure_gold_columns(df)

        # hour_of_day for amount_plus_night_hour and any future sets that need it
        model_df = (
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
                "prior_tx_count_card_category",
                "prior_amount_mean_card_category",
                "prior_amount_std_card_category",
            )
            .na.fill(MODEL_FILL_DEFAULTS)
            .withColumn("label", F.col("is_fraud").cast("double"))
        )

        split_q70, split_q85 = model_df.approxQuantile("event_time_unix", [0.70, 0.85], 0.001)
        model_df = model_df.withColumn(
            "split",
            F.when(F.col("event_time_unix") <= F.lit(split_q70), F.lit("train"))
            .when(F.col("event_time_unix") <= F.lit(split_q85), F.lit("validation"))
            .otherwise(F.lit("test")),
        )
        fractions = {
            "train": args.train_fraction,
            "validation": args.validation_fraction,
            "test": args.test_fraction,
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
        model_df = model_df.cache()
        _ = model_df.count()

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
            f" train={args.train_fraction:.2f},"
            f" validation={args.validation_fraction:.2f},"
            f" test={args.test_fraction:.2f}"
        )

        threshold_candidates = [x / 100 for x in range(5, 100, 5)]

        # Optional: category-z gating K sweep tuned for alert ranking (Top-K precision).
        if args.category_k_grid:
            from fraud_lens.silver_to_gold.transform import PRIOR_CATEGORY_Z_DAMP_K

            try:
                k_grid = [int(x.strip()) for x in args.category_k_grid.split(",") if x.strip()]
            except ValueError as e:
                raise ValueError(
                    f"Invalid --category-k-grid='{args.category_k_grid}'. Expected comma-separated ints."
                ) from e
            if not k_grid:
                raise ValueError("--category-k-grid produced an empty grid.")

            variant_list = (
                ["raw", "damped"] if args.category_z_variant == "both" else [args.category_z_variant]
            )
            model_types = ["logistic"]
            if args.model_type not in ["logistic", "both"]:
                raise ValueError(
                    "K sweep currently supports logistic only. Pass --model-type logistic (or both)."
                )

            base_z_raw = F.when(
                F.col("prior_amount_std_card_category") > F.lit(0.0),
                (F.col("amount") - F.col("prior_amount_mean_card_category"))
                / F.col("prior_amount_std_card_category"),
            ).otherwise(F.lit(None).cast("double"))
            n_d = F.col("prior_tx_count_card_category").cast("double")

            results = []
            for k in k_grid:
                eligibility_k = (F.col("prior_tx_count_card_category") >= F.lit(k)) & (
                    F.col("prior_amount_std_card_category") > F.lit(0.0)
                )
                eligible_flag_k = eligibility_k
                damp_factor = F.sqrt(n_d / (n_d + F.lit(float(PRIOR_CATEGORY_Z_DAMP_K))))
                damped_z_raw = base_z_raw * damp_factor
                # Log1p prior n (reliability feature) is independent of K; still computed per row.
                log_prior_n = F.log1p(n_d)

                # Always materialize both raw and damped category-z variants for this K.
                cat_z_raw_k = F.when(eligibility_k, base_z_raw).otherwise(
                    F.lit(None).cast("double")
                )
                cat_z_damped_k = F.when(eligibility_k, damped_z_raw).otherwise(
                    F.lit(None).cast("double")
                )

                model_df_k_base = (
                    model_df.withColumn("prior_category_zscore_eligible_k", eligible_flag_k)
                    .withColumn("prior_amount_zscore_card_category_k", cat_z_raw_k)
                    .withColumn("prior_amount_zscore_card_category_damped_k", cat_z_damped_k)
                    .withColumn("prior_category_log_prior_n_k", log_prior_n)
                    .na.fill(
                        {
                            "prior_amount_zscore_card_category_k": 0.0,
                            "prior_amount_zscore_card_category_damped_k": 0.0,
                            "prior_category_zscore_eligible_k": False,
                            "prior_category_log_prior_n_k": 0.0,
                        }
                    )
                )

                for v in variant_list:
                    if v == "raw":
                        feature_cols = [
                            "prior_amount_zscore",
                            "amount_sum_last_1h",
                            "is_night_transaction",
                            "prior_amount_zscore_card_category_k",
                            "prior_category_zscore_eligible_k",
                        ]
                    elif v == "damped":
                        feature_cols = [
                            "prior_amount_zscore",
                            "amount_sum_last_1h",
                            "is_night_transaction",
                            "prior_amount_zscore_card_category_damped_k",
                            "prior_category_zscore_eligible_k",
                            "prior_category_log_prior_n_k",
                        ]
                    else:
                        raise ValueError(f"Unknown category-z-variant: {v}")

                    for mt in model_types:
                        print(f"[K sweep] Evaluating K={k} | variant={v} | model={mt}")
                        r = _evaluate_feature_set(
                            spark=spark,
                            model_df=model_df_k_base,
                            feature_cols=feature_cols,
                            threshold_candidates=threshold_candidates,
                            model_type=mt,
                            compute_validation_and_auc=False,
                            logistic_class_weights=args.logistic_class_weights,
                        )

                        # Extract top-k precisions from the returned rows.
                        top_rows = {row["k"]: row["precision"] for row in r["top_k_rows"]}
                        results.append(
                            {
                                "k": k,
                                "variant": v,
                                "test_pr_auc": r["test_pr_auc"],
                                "test_roc_auc": r["test_roc_auc"],
                                "test_f1": r["test_f1"],
                                "topk_primary_precision": top_rows.get(
                                    args.topk_primary, 0.0
                                ),
                                "topk_tie_precision": top_rows.get(
                                    args.topk_tie_break, 0.0
                                ),
                                "topk_sanity_precision": top_rows.get(
                                    args.topk_sanity, 0.0
                                ),
                                "best_threshold": r["best_threshold"],
                            }
                        )

            # Select best K per variant.
            import pandas as pd

            results_df = pd.DataFrame(results)
            print("\n[K sweep summary] per-variant results:")
            for v in variant_list:
                sub = results_df[results_df["variant"] == v].copy()
                sub = sub.sort_values(
                    by=["topk_primary_precision", "topk_tie_precision", "k"],
                    ascending=[False, False, True],
                )
                print(f"\nVariant={v} candidates (sorted):")
                cols = [
                    "k",
                    "topk_primary_precision",
                    "topk_tie_precision",
                    "topk_sanity_precision",
                    "test_pr_auc",
                    "test_f1",
                ]
                print(sub[cols].to_string(index=False))

            # Pick best K by the selection rule and print a final decision.
            decisions = []
            for v in variant_list:
                sub = results_df[results_df["variant"] == v].copy()
                sub = sub.sort_values(
                    by=["topk_primary_precision", "topk_tie_precision", "k"],
                    ascending=[False, False, True],
                )
                best = sub.iloc[0].to_dict()
                decisions.append(best)
            decisions_df = pd.DataFrame(decisions)
            print("\n[K sweep best per variant]:")
            print(decisions_df[["variant", "k", "topk_primary_precision", "topk_tie_precision", "topk_sanity_precision"]].to_string(index=False))
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
                result = _evaluate_feature_set(
                    spark=spark,
                    model_df=model_df,
                    feature_cols=feature_cols,
                    threshold_candidates=threshold_candidates,
                    model_type=mt,
                    logistic_class_weights=args.logistic_class_weights,
                )
                result["feature_set_name"] = feature_set_name
                subset_results.append(result)

        # Drop nested top_k_rows so createDataFrame stays flat for the summary table
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
