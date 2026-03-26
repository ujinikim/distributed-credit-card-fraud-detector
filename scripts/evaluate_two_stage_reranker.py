#!/usr/bin/env python3
"""Two-stage reranker experiment runner (Top-N shortlist rerank)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from sparkov_eval.constants import FEATURE_SETS, TOP_K_VALUES
from sparkov_eval.data_prep import apply_time_split_and_sampling, build_model_df, ensure_gold_columns
from sparkov_eval.two_stage_reranker import two_stage_rerank_topk


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Two-stage reranker evaluation for Sparkov.")
    parser.add_argument(
        "--base-lr-run",
        type=int,
        choices=[9, 17],
        default=9,
        help="Base LR run to use for stage-1 ranking (9 or 17).",
    )
    parser.add_argument(
        "--reranker-gbt-run",
        type=int,
        choices=[18],
        default=18,
        help="Reranker GBT run to use for stage-2 (18).",
    )
    parser.add_argument(
        "--shortlist-n",
        type=int,
        default=5000,
        help="Top-N shortlist size to rerank (anchored on base LR ranking).",
    )
    parser.add_argument(
        "--rerank-mode",
        choices=["pure", "blended"],
        default="pure",
        help="Within shortlist: pure rerank (GBT score) or blended rerank (GBT + alpha * LR).",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.10,
        help="Blending weight used only for rerank-mode=blended.",
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
        "--logistic-class-weights",
        action="store_true",
        help="Use inverse-frequency class weights for the base LR ranker.",
    )
    parser.add_argument(
        "--topk",
        type=str,
        default="100,500,1000,5000,10000",
        help="Comma-separated Top-K values to report.",
    )
    return parser.parse_args()


def _run_to_feature_set_name(run: int, *, is_lr: bool) -> str:
    if is_lr:
        if run == 9:
            return "amount_plus_night"
        if run == 17:
            return "amount_plus_night_catz_v3_damped_shrunk"
    else:
        if run == 18:
            return "amount_plus_night_catz_v3_shrunk"
    raise ValueError(f"Unsupported run={run} for is_lr={is_lr}")


def main() -> None:
    from pyspark.sql import SparkSession

    from fraud_lens.ingest import load_sparkov_config

    args = parse_args()
    config = load_sparkov_config().get("sparkov", {})

    spark_builder = SparkSession.builder.appName("FraudLens-TwoStageReranker")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    gold_path = project_root / config.get("gold_path", "data/benchmark/gold_sparkov")

    topk_values = [int(x.strip()) for x in args.topk.split(",") if x.strip()]

    base_feature_set = _run_to_feature_set_name(args.base_lr_run, is_lr=True)
    reranker_feature_set = _run_to_feature_set_name(args.reranker_gbt_run, is_lr=False)

    base_lr_feature_cols = list(FEATURE_SETS[base_feature_set])
    reranker_gbt_feature_cols = list(FEATURE_SETS[reranker_feature_set])

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

    result = two_stage_rerank_topk(
        model_df,
        base_lr_feature_cols=base_lr_feature_cols,
        reranker_gbt_feature_cols=reranker_gbt_feature_cols,
        shortlist_n=args.shortlist_n,
        rerank_mode=args.rerank_mode,
        alpha=args.alpha,
        topk_values=topk_values,
        logistic_class_weights=args.logistic_class_weights,
    )

    print("Two-stage reranker results")
    print(f"- Base LR: Run {args.base_lr_run} ({base_feature_set})")
    print(f"- Reranker GBT: Run {args.reranker_gbt_run} ({reranker_feature_set})")
    print(f"- Shortlist-N: {args.shortlist_n}")
    print(f"- Rerank mode: {args.rerank_mode}" + (f", alpha={args.alpha}" if args.rerank_mode == "blended" else ""))

    print("\nBaseline Top-K (stage-1 only):")
    for row in result["baseline_top_k_rows"]:
        print(
            f"  k={row['k']}: precision={row['precision']:.4f} recall={row['recall']:.4f} tp={row['tp']} rows={row['rows']}"
        )

    print("\nReranked Top-K (final queue):")
    for row in result["reranked_top_k_rows"]:
        print(
            f"  k={row['k']}: precision={row['precision']:.4f} recall={row['recall']:.4f} tp={row['tp']} rows={row['rows']}"
        )

    spark.stop()


if __name__ == "__main__":
    main()

