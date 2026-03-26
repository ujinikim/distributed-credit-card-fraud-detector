"""Model training and metric computation helpers."""

from sparkov_eval.constants import TOP_K_VALUES


def threshold_metrics(df, threshold: float) -> dict[str, float]:
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


def top_k_metrics(df, k: int) -> dict[str, float]:
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


def evaluate_feature_set(
    model_df,
    feature_cols: list[str],
    threshold_candidates: list[float],
    model_type: str,
    compute_validation_and_auc: bool = True,
    logistic_class_weights: bool = False,
) -> dict[str, float]:
    """Train and score one feature subset."""
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

    top_k_rows = []
    for k in TOP_K_VALUES:
        top_k = top_k_metrics(test_pred, k)
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
        threshold_metrics(validation_pred, threshold) for threshold in threshold_candidates
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
    test_metrics = threshold_metrics(test_pred_full, best_validation["threshold"])

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

