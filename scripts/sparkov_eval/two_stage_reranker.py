"""Two-stage reranker evaluation (top-N shortlist rerank)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TopKResult:
    k: int
    precision: float
    recall: float
    tp: int
    rows: int


def _compute_topk_from_finalpos(
    df,
    *,
    k_values: list[int],
    pos_col: str,
    positives_total: int,
) -> list[TopKResult]:
    """Compute precision/recall@k from an already-ordered final queue.

    Note: recall denominator must be computed against the positives in the *full* dataset,
    not just within the final_queue subset used for ordering.
    """
    from pyspark.sql import functions as F

    positives = int(positives_total)

    # One pass conditional aggregation for all k's.
    agg_exprs = []
    for k in k_values:
        lit_k = F.lit(int(k))
        rows_alias = f"rows_{k}"
        tp_alias = f"tp_{k}"
        agg_exprs.append(
            F.sum(F.when(F.col(pos_col) <= lit_k, F.lit(1)).otherwise(F.lit(0))).alias(
                rows_alias
            )
        )
        agg_exprs.append(
            F.sum(
                F.when(F.col(pos_col) <= lit_k, F.col("label").cast("long")).otherwise(
                    F.lit(0)
                )
            ).alias(tp_alias)
        )

    row = df.agg(*agg_exprs).first()

    out: list[TopKResult] = []
    for k in k_values:
        rows = int(row[f"rows_{k}"] or 0)
        tp = int(row[f"tp_{k}"] or 0)
        precision = (tp / rows) if rows else 0.0
        recall = (tp / positives) if positives else 0.0
        out.append(
            TopKResult(k=int(k), precision=precision, recall=recall, tp=tp, rows=rows)
        )
    return out


def two_stage_rerank_topk(
    model_df,
    *,
    base_lr_feature_cols: list[str],
    reranker_gbt_feature_cols: list[str],
    shortlist_n: int,
    rerank_mode: str,
    alpha: float,
    topk_values: list[int],
    logistic_class_weights: bool = False,
    compute_validation_and_auc: bool = False,
    threshold_candidates: list[float] | None = None,
) -> dict[str, object]:
    """
    Train base LR + reranker GBT, create a final queue by reranking a top-N shortlist,
    then compute precision/recall@k on both baseline and reranked queues.

    rerank_mode:
      - 'pure': order shortlist by gbt_score desc, then lr_score desc
      - 'blended': order shortlist by (gbt_score + alpha * lr_score) desc
    """

    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.evaluation import BinaryClassificationEvaluator
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    from sparkov_eval.metrics import threshold_metrics

    if rerank_mode not in {"pure", "blended"}:
        raise ValueError("rerank_mode must be one of: 'pure', 'blended'")
    if rerank_mode == "blended" and alpha < 0:
        raise ValueError("alpha must be >= 0 for blended reranking")

    shortlist_n = int(shortlist_n)
    alpha = float(alpha)
    threshold_candidates = threshold_candidates or []

    # --- Train / score base LR ---
    lr_assembler = VectorAssembler(inputCols=base_lr_feature_cols, outputCol="lr_features")
    lr_assembled = lr_assembler.transform(model_df)
    lr_train_df = lr_assembled.where(F.col("split") == F.lit("train")).select("label", "lr_features")
    lr_model = None

    if logistic_class_weights:
        class_counts = lr_train_df.agg(
            F.sum((F.col("label") == F.lit(1.0)).cast("int")).alias("pos"),
            F.sum((F.col("label") == F.lit(0.0)).cast("int")).alias("neg"),
        ).first()
        pos_count = int(class_counts["pos"] or 0)
        neg_count = int(class_counts["neg"] or 0)
        pos_weight = (neg_count / pos_count) if pos_count else 1.0
        lr_train_df = lr_train_df.withColumn(
            "weight",
            F.when(F.col("label") == F.lit(1.0), F.lit(pos_weight)).otherwise(F.lit(1.0)),
        )

    lr_kwargs = {
        "featuresCol": "lr_features",
        "labelCol": "label",
        "probabilityCol": "lr_probability",
        "rawPredictionCol": "lr_rawPrediction",
        "predictionCol": "lr_prediction",
        "maxIter": 50,
        "regParam": 0.01,
        "elasticNetParam": 0.0,
    }
    if logistic_class_weights:
        lr_kwargs["weightCol"] = "weight"
    lr_model = LogisticRegression(**lr_kwargs).fit(lr_train_df)

    def _score_lr(split_name: str):
        split_df = lr_assembled.where(F.col("split") == F.lit(split_name)).select(
            "transaction_id", "event_time_unix", "label", "lr_features"
        )
        return lr_model.transform(split_df).select(
            "transaction_id",
            "event_time_unix",
            "label",
            vector_to_array("lr_probability")[1].alias("lr_score"),
        )

    # --- Train / score reranker GBT ---
    gbt_assembler = VectorAssembler(
        inputCols=reranker_gbt_feature_cols, outputCol="gbt_features"
    )
    gbt_assembled = gbt_assembler.transform(model_df)
    gbt_train_df = gbt_assembled.where(F.col("split") == F.lit("train")).select(
        "label", "gbt_features"
    )
    gbt_model = GBTClassifier(
        featuresCol="gbt_features",
        labelCol="label",
        maxIter=30,
        maxDepth=5,
        stepSize=0.1,
        seed=42,
    ).fit(gbt_train_df)

    def _score_gbt(split_name: str):
        split_df = gbt_assembled.where(F.col("split") == F.lit(split_name)).select(
            "transaction_id", "event_time_unix", "label", "gbt_features"
        )
        return gbt_model.transform(split_df).select(
            "transaction_id",
            vector_to_array("probability")[1].alias("gbt_score"),
        )

    lr_test = _score_lr("test")
    gbt_test = _score_gbt("test")
    scored = lr_test.join(gbt_test, on="transaction_id", how="inner")

    max_k = int(max(topk_values)) if topk_values else 0
    if max_k <= 0:
        raise ValueError("topk_values must contain positive integers")
    shortlist_n = min(shortlist_n, max_k)

    # Total positives for correct recall computation (across the full test set).
    positives_row = scored.agg(F.sum(F.col("label").cast("long")).alias("positives")).first()
    positives_total = int(positives_row["positives"] or 0)

    # Baseline order used for shortlist selection and baseline metrics:
    baseline_ordered = scored.orderBy(
        F.desc("lr_score"),
        F.desc("event_time_unix"),
        F.asc("transaction_id"),
    )

    # Only position/compute ranks for the top-max_k portion.
    baseline_top = baseline_ordered.limit(max_k)

    window_baseline_top = Window.orderBy(
        F.desc("lr_score"),
        F.desc("event_time_unix"),
        F.asc("transaction_id"),
    )
    baseline_top = baseline_top.withColumn(
        "baseline_pos_in_queue", F.row_number().over(window_baseline_top)
    )

    baseline_queue = baseline_top.select(
        "label", F.col("baseline_pos_in_queue").alias("final_pos_in_queue")
    )
    baseline_metrics = _compute_topk_from_finalpos(
        baseline_queue,
        k_values=topk_values,
        pos_col="final_pos_in_queue",
        positives_total=positives_total,
    )

    # Shortlist = first N items under baseline order.
    shortlist_df = baseline_top.where(F.col("baseline_pos_in_queue") <= F.lit(shortlist_n))
    remainder_next_df = baseline_top.where(F.col("baseline_pos_in_queue") > F.lit(shortlist_n))

    # Rerank only the shortlist.
    if rerank_mode == "pure":
        window_shortlist = Window.orderBy(
            F.desc("gbt_score"),
            F.desc("lr_score"),
            F.desc("event_time_unix"),
            F.asc("transaction_id"),
        )
        shortlist_ranked = shortlist_df.withColumn(
            "shortlist_pos", F.row_number().over(window_shortlist)
        )
    else:
        shortlist_df = shortlist_df.withColumn(
            "blend_score", F.col("gbt_score") + F.lit(alpha) * F.col("lr_score")
        )
        window_shortlist = Window.orderBy(
            F.desc("blend_score"),
            F.desc("lr_score"),
            F.desc("event_time_unix"),
            F.asc("transaction_id"),
        )
        shortlist_ranked = shortlist_df.withColumn(
            "shortlist_pos", F.row_number().over(window_shortlist)
        )

    shortlist_final = shortlist_ranked.select(
        "label", F.col("shortlist_pos").alias("final_pos_in_queue")
    )
    remainder_final = remainder_next_df.select(
        "label", F.col("baseline_pos_in_queue").alias("final_pos_in_queue")
    )

    final_queue = shortlist_final.unionByName(remainder_final)

    reranked_metrics = _compute_topk_from_finalpos(
        final_queue,
        k_values=topk_values,
        pos_col="final_pos_in_queue",
        positives_total=positives_total,
    )

    # --- Optional: compute PR AUC / ROC AUC / F1 by defining a final_score for all rows ---
    full_metrics: dict[str, object] | None = None
    if compute_validation_and_auc:
        if not threshold_candidates:
            raise ValueError("threshold_candidates must be provided when compute_validation_and_auc=True")

        from pyspark.ml.functions import array_to_vector

        def _final_scored_all(split_name: str):
            lr_split = _score_lr(split_name)
            gbt_split = _score_gbt(split_name)
            split_scored = lr_split.join(gbt_split, on="transaction_id", how="inner")

            shortlist_ids = (
                split_scored.orderBy(
                    F.desc("lr_score"),
                    F.desc("event_time_unix"),
                    F.asc("transaction_id"),
                )
                .limit(int(shortlist_n))
                .select("transaction_id")
                .withColumn("in_shortlist", F.lit(1))
            )

            split_scored = split_scored.join(shortlist_ids, on="transaction_id", how="left")
            split_scored = split_scored.withColumn(
                "in_shortlist", F.coalesce(F.col("in_shortlist"), F.lit(0))
            )

            if rerank_mode == "pure":
                shortlist_score = F.col("gbt_score")
            else:
                shortlist_score = F.col("gbt_score") + F.lit(float(alpha)) * F.col("lr_score")

            split_scored = split_scored.withColumn(
                "final_score",
                F.when(F.col("in_shortlist") == F.lit(1), shortlist_score).otherwise(
                    F.col("lr_score")
                ),
            )

            # BinaryClassificationEvaluator expects a rawPrediction column.
            split_scored = split_scored.withColumn(
                "rawPrediction",
                array_to_vector(F.array(F.lit(1.0) - F.col("final_score"), F.col("final_score"))),
            )

            return split_scored.select(
                "transaction_id",
                "event_time_unix",
                "label",
                F.col("final_score").alias("score"),
                "rawPrediction",
            )

        validation_scored = _final_scored_all("validation")
        test_scored = _final_scored_all("test")

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

        validation_pr_auc = pr_evaluator.evaluate(validation_scored)
        validation_roc_auc = roc_evaluator.evaluate(validation_scored)
        validation_rows = [threshold_metrics(validation_scored, t) for t in threshold_candidates]
        best_validation = max(validation_rows, key=lambda row: row["f1"])

        test_pr_auc = pr_evaluator.evaluate(test_scored)
        test_roc_auc = roc_evaluator.evaluate(test_scored)
        test_at_best = threshold_metrics(test_scored, best_validation["threshold"])

        full_metrics = {
            "validation_pr_auc": float(validation_pr_auc),
            "validation_roc_auc": float(validation_roc_auc),
            "best_threshold": float(best_validation["threshold"]),
            "validation_precision": float(best_validation["precision"]),
            "validation_recall": float(best_validation["recall"]),
            "validation_f1": float(best_validation["f1"]),
            "test_pr_auc": float(test_pr_auc),
            "test_roc_auc": float(test_roc_auc),
            "test_precision": float(test_at_best["precision"]),
            "test_recall": float(test_at_best["recall"]),
            "test_f1": float(test_at_best["f1"]),
        }

    def _as_simple(metrics: list[TopKResult]) -> list[dict[str, object]]:
        return [
            {
                "k": m.k,
                "precision": m.precision,
                "recall": m.recall,
                "tp": m.tp,
                "rows": m.rows,
            }
            for m in metrics
        ]

    return {
        "baseline_top_k_rows": _as_simple(baseline_metrics),
        "reranked_top_k_rows": _as_simple(reranked_metrics),
        "shortlist_n": shortlist_n,
        "rerank_mode": rerank_mode,
        "alpha": alpha if rerank_mode == "blended" else None,
        "full_metrics": full_metrics,
    }

