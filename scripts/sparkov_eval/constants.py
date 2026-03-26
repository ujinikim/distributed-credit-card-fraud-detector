"""Shared constants for Sparkov evaluation scripts."""

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
    "amount_plus_night_catz_v4_interact_clip": [
        "prior_amount_zscore_clipped",
        "amount_sum_last_1h_log1p",
        "is_night_transaction",
        "prior_amount_zscore_card_category_damped",
        "prior_amount_zscore_card_category_shrunk",
        "prior_category_zscore_eligible",
        "prior_category_log_prior_n",
        "amount_zscore_x_lowcat",
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

THRESHOLD_CANDIDATES = [x / 100 for x in range(5, 100, 5)]

MISSING_GOLD_COLUMN_CASTS = {
    "prior_amount_zscore_card_category": "double",
    "prior_category_zscore_eligible": "boolean",
    "prior_amount_zscore_card_category_damped": "double",
    "prior_amount_zscore_card_category_shrunk": "double",
    "prior_category_log_prior_n": "double",
    "low_history_card_category": "boolean",
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
    "low_history_card_category": False,
    "prior_tx_count_card_category": 0,
    "prior_amount_mean_card_category": 0.0,
    "prior_amount_std_card_category": 0.0,
}

