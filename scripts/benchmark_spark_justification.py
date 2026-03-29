#!/usr/bin/env python3
"""Spark justification benchmark: timed Spark eval phases vs sklearn on a row cap.

Does not modify sparkov_eval or eval CLIs—imports helpers only.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_paths() -> None:
    root = _project_root()
    for p in (root / "src", root / "scripts"):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _parse_scales(s: str) -> list[float]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        v = float(part)
        if v <= 0 or v > 1:
            raise argparse.ArgumentTypeError(
                f"scale must be in (0, 1], got {v} (per-split sampling fraction)"
            )
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("need at least one scale")
    return out


def _parse_row_caps(s: str) -> list[int]:
    out = []
    for part in s.split(","):
        part = part.strip()
        if not part:
            continue
        v = int(part)
        if v <= 0:
            raise argparse.ArgumentTypeError(f"row cap must be > 0, got {v}")
        out.append(v)
    if not out:
        raise argparse.ArgumentTypeError("need at least one row cap")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark Sparkov Gold → LR eval vs sklearn on a capped train sample."
    )
    parser.add_argument(
        "--scales",
        type=_parse_scales,
        default=[1.0, 0.5, 0.25],
        help="Comma-separated per-split sampling fractions (train/validation/test), "
        "e.g. 1.0,0.5,0.25. Same as evaluate_sparkov_model fractions.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Single small scale (0.02 per split) for smoke runs.",
    )
    parser.add_argument(
        "--feature-set",
        default="amount_plus_night",
        help="Feature set name from fraud_lens.benchmark.sparkov.eval.constants.FEATURE_SETS.",
    )
    parser.add_argument(
        "--model-type",
        choices=["logistic", "gbt"],
        default="logistic",
        help="Model type for Spark evaluate_feature_set (default: logistic, Run 9 style).",
    )
    parser.add_argument(
        "--sklearn-train-cap",
        type=int,
        default=500_000,
        help="Max train rows for sklearn baseline (after optional per-split scale).",
    )
    parser.add_argument(
        "--skip-sklearn",
        action="store_true",
        help="Only run Spark phases.",
    )
    parser.add_argument(
        "--sklearn-split-fraction",
        type=float,
        default=1.0,
        help="Per-split fraction for sklearn baseline only (default 1.0 = full time splits).",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write full metrics JSON to this path (default: print to stdout only).",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Append one row per Spark scale to this CSV.",
    )
    parser.add_argument(
        "--comparison-row-caps",
        type=_parse_row_caps,
        default=[],
        help="Optional fair comparison caps (same train/test row caps for Spark and sklearn), e.g. 50000,100000,250000.",
    )
    parser.add_argument(
        "--comparison-repeats",
        type=int,
        default=3,
        help="Repeat count per row cap for fair comparison medians.",
    )
    parser.add_argument(
        "--comparison-split-fraction",
        type=float,
        default=1.0,
        help="Per-split fraction used to build the comparison train/test pools.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _ensure_paths()

    from pyspark.sql import SparkSession
    from pyspark.sql import functions as F

    from fraud_lens.benchmark.sparkov import load_sparkov_config, resolve_sparkov_paths
    from fraud_lens.benchmark.sparkov.eval.constants import (
        FEATURE_SETS,
        THRESHOLD_CANDIDATES,
    )
    from fraud_lens.benchmark.sparkov.eval.data_prep import (
        apply_time_split_and_sampling,
        build_model_df,
        ensure_gold_columns,
    )
    from fraud_lens.benchmark.sparkov.eval.metrics import evaluate_feature_set
    scales: list[float] = [0.02] if args.fast else args.scales

    config = load_sparkov_config().get("sparkov", {})
    paths = resolve_sparkov_paths(config)
    spark_builder = SparkSession.builder.appName("FraudLens-SparkJustificationBenchmark")
    for key, value in config.get("spark_runtime", {}).items():
        spark_builder = spark_builder.config(key, str(value))
    spark = spark_builder.getOrCreate()

    try:
        if args.feature_set not in FEATURE_SETS:
            raise SystemExit(f"Unknown feature set: {args.feature_set!r}")
        feature_cols = list(FEATURE_SETS[args.feature_set])

        gold_path = paths["gold_path"]

        t_read_start = time.perf_counter()
        raw = spark.read.parquet(str(gold_path))
        df = ensure_gold_columns(raw)
        gold_rows = int(df.count())
        seconds_read = time.perf_counter() - t_read_start

        spark_runs: list[dict[str, Any]] = []

        for frac in scales:
            t_prep_start = time.perf_counter()
            built = build_model_df(df)
            model_df = apply_time_split_and_sampling(
                model_df=built,
                train_fraction=frac,
                validation_fraction=frac,
                test_fraction=frac,
            )
            model_partitions = model_df.rdd.getNumPartitions()
            total_rows = int(model_df.count())
            seconds_prep = time.perf_counter() - t_prep_start

            t_eval_start = time.perf_counter()
            _ = evaluate_feature_set(
                model_df=model_df,
                feature_cols=feature_cols,
                threshold_candidates=THRESHOLD_CANDIDATES,
                model_type=args.model_type,
                compute_validation_and_auc=True,
                logistic_class_weights=False,
                topk_secondary_signal="none",
                topk_secondary_epsilon=0.0,
            )
            seconds_eval = time.perf_counter() - t_eval_start

            model_df.unpersist()

            total_phase = seconds_prep + seconds_eval
            rows_per_sec = (total_rows / total_phase) if total_phase > 0 else None

            spark_runs.append(
                {
                    "per_split_fraction": frac,
                    "model_df_rows": total_rows,
                    "model_df_partitions": model_partitions,
                    "seconds_prep_including_count": seconds_prep,
                    "seconds_evaluate_feature_set": seconds_eval,
                    "seconds_total_spark_for_scale": total_phase,
                    "rows_per_second_end_to_end": rows_per_sec,
                }
            )

        sklearn_block: dict[str, Any] | None = None
        if not args.skip_sklearn:
            try:
                import numpy as np
                from sklearn.linear_model import LogisticRegression
            except ImportError as e:
                sklearn_block = {
                    "error": "scikit-learn not installed; pip install scikit-learn",
                    "detail": str(e),
                }
            else:
                skf = min(1.0, max(1e-9, float(args.sklearn_split_fraction)))
                t_sk_start = time.perf_counter()
                built = build_model_df(df)
                model_df = apply_time_split_and_sampling(
                    model_df=built,
                    train_fraction=skf,
                    validation_fraction=skf,
                    test_fraction=skf,
                )
                sklearn_block = {}
                try:
                    train_only = model_df.where(F.col("split") == F.lit("train")).select(
                        *feature_cols, F.col("label")
                    )
                    n_train = int(train_only.count())
                    cap = min(args.sklearn_train_cap, n_train) if n_train else 0
                    if cap <= 0:
                        sklearn_block.update(
                            {
                                "sklearn_train_rows": 0,
                                "seconds_sklearn_fit": None,
                                "rows_per_second_fit": None,
                                "note": "no train rows",
                            }
                        )
                    else:
                        frac_sample = cap / n_train if n_train > cap else 1.0
                        sampled = (
                            train_only.sample(
                                withReplacement=False,
                                fraction=float(frac_sample),
                                seed=42,
                            )
                            if frac_sample < 1.0
                            else train_only
                        )
                        sampled = sampled.limit(cap)
                        t_pd = time.perf_counter()
                        pdf = sampled.toPandas()
                        seconds_to_pandas = time.perf_counter() - t_pd
                        X = pdf[feature_cols].to_numpy(dtype=np.float64, copy=False)
                        y = pdf["label"].to_numpy(dtype=np.float64, copy=False)
                        t_fit = time.perf_counter()
                        LogisticRegression(
                            max_iter=50,
                            random_state=42,
                            solver="lbfgs",
                        ).fit(X, y)
                        seconds_fit = time.perf_counter() - t_fit
                        sklearn_block.update(
                            {
                                "per_split_fraction_for_sklearn": skf,
                                "sklearn_train_rows_available": n_train,
                                "sklearn_train_rows_used": int(len(pdf)),
                                "seconds_spark_sample_to_pandas": seconds_to_pandas,
                                "seconds_sklearn_fit": seconds_fit,
                                "rows_per_second_fit": len(pdf) / seconds_fit
                                if seconds_fit > 0
                                else None,
                            }
                        )
                finally:
                    model_df.unpersist()
                sklearn_block["seconds_sklearn_total_wall"] = (
                    time.perf_counter() - t_sk_start
                )

        fair_comparison: dict[str, Any] | None = None
        if args.comparison_row_caps:
            try:
                import numpy as np
                from sklearn.linear_model import LogisticRegression as SklearnLogisticRegression
            except ImportError as e:
                fair_comparison = {
                    "error": "scikit-learn not installed; pip install scikit-learn",
                    "detail": str(e),
                }
            else:
                from pyspark.ml.classification import LogisticRegression as SparkLogisticRegression
                from pyspark.ml.feature import VectorAssembler
                from pyspark.sql import functions as SF

                cfrac = min(1.0, max(1e-9, float(args.comparison_split_fraction)))
                cmp_model_df = apply_time_split_and_sampling(
                    model_df=build_model_df(df),
                    train_fraction=cfrac,
                    validation_fraction=cfrac,
                    test_fraction=cfrac,
                )
                try:
                    cmp_train_base = cmp_model_df.where(SF.col("split") == SF.lit("train")).select(
                        *feature_cols, SF.col("label")
                    )
                    cmp_test_base = cmp_model_df.where(SF.col("split") == SF.lit("test")).select(
                        *feature_cols, SF.col("label")
                    )
                    n_train_pool = int(cmp_train_base.count())
                    n_test_pool = int(cmp_test_base.count())

                    caps_rows: list[dict[str, Any]] = []
                    for cap in args.comparison_row_caps:
                        train_cap = min(int(cap), n_train_pool)
                        test_cap = min(int(cap), n_test_pool)
                        if train_cap <= 0 or test_cap <= 0:
                            caps_rows.append(
                                {
                                    "row_cap": int(cap),
                                    "note": "insufficient train/test rows in comparison pool",
                                }
                            )
                            continue

                        train_frac = min(1.0, train_cap / n_train_pool)
                        test_frac = min(1.0, test_cap / n_test_pool)
                        cmp_train = (
                            cmp_train_base.sample(False, float(train_frac), seed=42)
                            if train_frac < 1.0
                            else cmp_train_base
                        ).limit(train_cap)
                        cmp_test = (
                            cmp_test_base.sample(False, float(test_frac), seed=42)
                            if test_frac < 1.0
                            else cmp_test_base
                        ).limit(test_cap)
                        cmp_train = cmp_train.cache()
                        cmp_test = cmp_test.cache()
                        actual_train = int(cmp_train.count())
                        actual_test = int(cmp_test.count())

                        spark_times = []
                        sklearn_times = []
                        for i in range(max(1, int(args.comparison_repeats))):
                            assembler = VectorAssembler(
                                inputCols=feature_cols, outputCol="features"
                            )
                            spark_train = assembler.transform(cmp_train).select("label", "features")
                            spark_test = assembler.transform(cmp_test).select("label", "features")

                            t0 = time.perf_counter()
                            spark_model = SparkLogisticRegression(
                                featuresCol="features",
                                labelCol="label",
                                probabilityCol="probability",
                                rawPredictionCol="rawPrediction",
                                predictionCol="prediction",
                                maxIter=50,
                                regParam=0.01,
                                elasticNetParam=0.0,
                            ).fit(spark_train)
                            _ = spark_model.transform(spark_test).count()
                            spark_times.append(time.perf_counter() - t0)

                            t1 = time.perf_counter()
                            train_pdf = cmp_train.toPandas()
                            test_pdf = cmp_test.toPandas()
                            X_train = train_pdf[feature_cols].to_numpy(dtype=np.float64, copy=False)
                            y_train = train_pdf["label"].to_numpy(dtype=np.float64, copy=False)
                            X_test = test_pdf[feature_cols].to_numpy(dtype=np.float64, copy=False)
                            _ = SklearnLogisticRegression(
                                max_iter=50, random_state=42 + i, solver="lbfgs"
                            ).fit(X_train, y_train).predict_proba(X_test)
                            sklearn_times.append(time.perf_counter() - t1)

                        cmp_train.unpersist()
                        cmp_test.unpersist()

                        spark_median = float(statistics.median(spark_times))
                        sklearn_median = float(statistics.median(sklearn_times))
                        total_rows = actual_train + actual_test
                        caps_rows.append(
                            {
                                "row_cap": int(cap),
                                "actual_train_rows": actual_train,
                                "actual_test_rows": actual_test,
                                "repeats": int(max(1, int(args.comparison_repeats))),
                                "spark_seconds_runs": spark_times,
                                "spark_seconds_median": spark_median,
                                "spark_rows_per_second_train_plus_test": (
                                    total_rows / spark_median if spark_median > 0 else None
                                ),
                                "sklearn_seconds_runs": sklearn_times,
                                "sklearn_seconds_median": sklearn_median,
                                "sklearn_rows_per_second_train_plus_test": (
                                    total_rows / sklearn_median if sklearn_median > 0 else None
                                ),
                                "spark_over_sklearn_speed_ratio": (
                                    sklearn_median / spark_median
                                    if spark_median > 0 and sklearn_median > 0
                                    else None
                                ),
                            }
                        )

                    fair_comparison = {
                        "comparison_split_fraction": cfrac,
                        "train_pool_rows": n_train_pool,
                        "test_pool_rows": n_test_pool,
                        "caps": caps_rows,
                    }
                finally:
                    cmp_model_df.unpersist()

        payload: dict[str, Any] = {
            "environment": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "spark_version": spark.version,
                "gold_path": str(gold_path),
                "feature_set": args.feature_set,
                "model_type": args.model_type,
                "sampling_method": "apply_time_split_and_sampling per-split fraction (same for train/val/test)",
            },
            "seconds_read_parquet_and_ensure_gold_once": seconds_read,
            "gold_row_count": gold_rows,
            "spark_runs": spark_runs,
            "sklearn_baseline": sklearn_block,
            "fair_comparison": fair_comparison,
        }

        text = json.dumps(payload, indent=2)
        if args.output_json:
            args.output_json.parent.mkdir(parents=True, exist_ok=True)
            args.output_json.write_text(text, encoding="utf-8")
        print(text)

        if args.output_csv and spark_runs:
            path = args.output_csv
            path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not path.exists()
            fieldnames = [
                "per_split_fraction",
                "model_df_rows",
                "model_df_partitions",
                "seconds_prep_including_count",
                "seconds_evaluate_feature_set",
                "seconds_total_spark_for_scale",
                "rows_per_second_end_to_end",
            ]
            with path.open("a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
                if write_header:
                    w.writeheader()
                for row in spark_runs:
                    w.writerow(row)

    finally:
        spark.stop()


if __name__ == "__main__":
    main()
