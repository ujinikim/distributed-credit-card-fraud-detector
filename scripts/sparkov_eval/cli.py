"""CLI parsing for Sparkov evaluation."""

import argparse

from sparkov_eval.constants import FEATURE_SET_CHOICES


def parse_args() -> argparse.Namespace:
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
        "--topk-secondary-signal",
        choices=[
            "none",
            "neg_prior_category_log_prior_n",
            "prior_amount_zscore",
            "amount_sum_last_1h",
        ],
        default="none",
        help=(
            "Optional secondary signal used only for Top-K ordering. "
            "If epsilon is 0, uses lexicographic ordering (score, secondary, time, id). "
            "If epsilon > 0, uses blended ordering (score + epsilon * secondary)."
        ),
    )
    parser.add_argument(
        "--topk-secondary-epsilon",
        type=float,
        default=0.0,
        help=(
            "Optional score-blending weight for Top-K ordering. "
            "Use 0 for lexicographic ordering."
        ),
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

