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
) -> dict[str, object]:
    """
    Train base LR + reranker GBT, create a final queue by reranking a top-N shortlist,
    then compute precision/recall@k on both baseline and reranked queues.

    rerank_mode:
      - 'pure': order shortlist by gbt_score desc, then lr_score desc
      - 'blended': order shortlist by (gbt_score + alpha * lr_score) desc
    """

    from pyspark.ml.classification import GBTClassifier, LogisticRegression
    from pyspark.ml.feature import VectorAssembler
    from pyspark.ml.functions import vector_to_array
    from pyspark.sql import functions as F
    from pyspark.sql.window import Window

    if rerank_mode not in {"pure", "blended"}:
        raise ValueError("rerank_mode must be one of: 'pure', 'blended'")
    if rerank_mode == "blended" and alpha < 0:
        raise ValueError("alpha must be >= 0 for blended reranking")

    shortlist_n = int(shortlist_n)
    alpha = float(alpha)

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

    lr_test_df = lr_assembled.where(F.col("split") == F.lit("test")).select(
        "transaction_id", "event_time_unix", "label", "lr_features"
    )
    lr_scored = lr_model.transform(lr_test_df).select(
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

    gbt_test_df = gbt_assembled.where(F.col("split") == F.lit("test")).select(
        "transaction_id", "event_time_unix", "label", "gbt_features"
    )
    gbt_scored = gbt_model.transform(gbt_test_df).select(
        "transaction_id",
        vector_to_array("probability")[1].alias("gbt_score"),
    )

    scored = lr_scored.join(gbt_scored, on="transaction_id", how="inner")

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
    }

