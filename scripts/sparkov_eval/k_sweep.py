"""Category-z K sweep helpers."""

from sparkov_eval.metrics import evaluate_feature_set


def run_category_k_sweep(model_df, args, threshold_candidates):
    """Run K sweep and print per-variant ranking summaries."""
    from fraud_lens.silver_to_gold.transform import PRIOR_CATEGORY_Z_DAMP_K
    from pyspark.sql import functions as F
    import pandas as pd

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
        damp_factor = F.sqrt(n_d / (n_d + F.lit(float(PRIOR_CATEGORY_Z_DAMP_K))))
        damped_z_raw = base_z_raw * damp_factor
        log_prior_n = F.log1p(n_d)

        cat_z_raw_k = F.when(eligibility_k, base_z_raw).otherwise(
            F.lit(None).cast("double")
        )
        cat_z_damped_k = F.when(eligibility_k, damped_z_raw).otherwise(
            F.lit(None).cast("double")
        )

        model_df_k_base = (
            model_df.withColumn("prior_category_zscore_eligible_k", eligibility_k)
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
                r = evaluate_feature_set(
                    model_df=model_df_k_base,
                    feature_cols=feature_cols,
                    threshold_candidates=threshold_candidates,
                    model_type=mt,
                    compute_validation_and_auc=False,
                    logistic_class_weights=args.logistic_class_weights,
                )
                top_rows = {row["k"]: row["precision"] for row in r["top_k_rows"]}
                results.append(
                    {
                        "k": k,
                        "variant": v,
                        "test_pr_auc": r["test_pr_auc"],
                        "test_roc_auc": r["test_roc_auc"],
                        "test_f1": r["test_f1"],
                        "topk_primary_precision": top_rows.get(args.topk_primary, 0.0),
                        "topk_tie_precision": top_rows.get(args.topk_tie_break, 0.0),
                        "topk_sanity_precision": top_rows.get(args.topk_sanity, 0.0),
                        "best_threshold": r["best_threshold"],
                    }
                )

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

    decisions = []
    for v in variant_list:
        sub = results_df[results_df["variant"] == v].copy()
        sub = sub.sort_values(
            by=["topk_primary_precision", "topk_tie_precision", "k"],
            ascending=[False, False, True],
        )
        decisions.append(sub.iloc[0].to_dict())
    decisions_df = pd.DataFrame(decisions)
    print("\n[K sweep best per variant]:")
    print(
        decisions_df[
            [
                "variant",
                "k",
                "topk_primary_precision",
                "topk_tie_precision",
                "topk_sanity_precision",
            ]
        ].to_string(index=False)
    )

