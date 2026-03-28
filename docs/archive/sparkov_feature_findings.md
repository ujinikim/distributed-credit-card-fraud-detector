# Sparkov Feature Findings

What we learned from Gold feature expansion, the full Sparkov benchmark pipeline, and the first modeling rounds. Records which features are strong, weak, or not yet earning their keep.

## Current Read

- Amount-based features: strong
- Amount window features: strong
- Merchant history: mixed — raw signal exists but hurt the logistic baseline
- Customer-to-merchant distance: weak
- Travel speed: brittle

## Feature Detail

### Strong: `prior_amount_zscore`

| Slice | Median | p90 |
| --- | ---: | ---: |
| non-fraud | -0.27 | — |
| fraud | 2.70 | 8.60 |

Fraud is shifted strongly toward unusual amounts relative to prior card history. Prior-only is more trustworthy than the all-inclusive z-score.

### Strong: `amount_sum_last_1h`

| Slice | Median | Mean |
| --- | ---: | ---: |
| non-fraud | 57.32 | — |
| fraud | 848.0 | 1069.96 |

Fraud clusters inside heavy short-window spending bursts. More useful than raw transaction count alone.

### Mixed: `merchant_prior_fraud_rate`

| Slice | Median | p90 |
| --- | ---: | ---: |
| non-fraud | 0.0067 | — |
| fraud | 0.0204 | 1.0 |

Some raw signal, but it **hurt** the logistic baseline vs amount-only. Keep in Gold; do not include in the baseline model. Remains leakage-sensitive — must stay prior-only.

### Secondary: velocity windows

Non-fraud avg `tx_count_last_1h` ≈ 1.33; fraud avg ≈ 1.89. Directionally useful as supporting context, not a strong standalone signal.

### Weak: `customer_to_merchant_distance_km`

Non-fraud and fraud medians both ≈ 78.7 km; p90 both ≈ 113.3 km. Heavy histogram overlap. Keep in Gold (cheap to compute); do not headline or expand.

### Brittle: `speed_from_prev_kmh`

Extreme values in both classes; only interpretable after log-transform or clipping; sensitive to tiny time gaps. Keep as diagnostic only.

### Exploration: (card × category) prior counts

From [category_prior_exploration.ipynb](../../notebooks/category_prior_exploration.ipynb) on the Sparkov Gold-scale dataset (prior-only count in each `(card_id, merchant_category)` bucket before the current row):

- About **1.5%** of rows are **first-ever** spend in that category for the card (`n = 0` priors).
- About **7%** have **`n < 5`**; about **14%** have **`n < 10`**.
- Median prior **`n` ≈ 39** — most rows have a long category history, but the tail is fat.

**Fraud vs non-fraud:** fraud rows are much more often **sparse** in the category bucket — roughly **~40%** of fraud rows have **`n < 5`** vs **~7%** of non-fraud (exact rates in the notebook). That supports using a **gated** category z-score (do not trust z until **`n ≥ K`**) and keeping **card-level** `prior_amount_zscore` as the primary fallback when category history is thin.

Gold exposes `prior_amount_zscore_card_category`, `prior_category_zscore_eligible`, and related columns; **`amount_plus_night_catz`** results are Run 13 in [sparkov_model_eval.md](./sparkov_model_eval.md).

### Design note: low *n* and “penalizing” category z-score

A small prior count in `(card, category)` makes the **sample mean and std unstable**: one outlier prior can skew both, so a raw z-score can **lie**. That is why Gold **gates** the category z at **`n ≥ 5`** (and `std > 0`) and exposes **`low_history_card_category`** / **`prior_category_zscore_eligible`**. For rows below the gate, the eval path treats the missing z as **neutral** (`0.0` fill) while the eligibility flag marks that the category coordinate is **off** — a feature-level choice, not a hand-tuned model weight.

**V2 in Gold:** `prior_amount_mean_card_category_shrunk`, `prior_amount_zscore_card_category_damped`, `prior_amount_zscore_card_category_shrunk`, and `prior_category_log_prior_n` implement the same shrinkage ideas (blended mean, damped z, reliability). Eval preset **`amount_plus_night_catz_v2`** uses damped z + eligibility + `log1p` prior *n* (see Run 14 in [sparkov_model_eval.md](./sparkov_model_eval.md)). Implementation detail lives in [`../../src/fraud_lens/silver_to_gold/transform.py`](../../src/fraud_lens/silver_to_gold/transform.py).

## Practical Ranking

| Tier | Features |
| --- | --- |
| **Baseline (logistic)** | `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction` |
| **Mixed / reevaluation** | `merchant_prior_fraud_rate`, velocity windows |
| **Weak** | `customer_to_merchant_distance_km` |
| **Diagnostic only** | `speed_from_prev_kmh` |

## Practical Queue (Two-stage Reranker)

The “top-100 is empty” issue for logistic baselines is primarily a **ranking-resolution / tie** problem, not a lack of signal. The current best fix is a **two-stage reranker**:

- Stage 1: rank the full test set using a base logistic queue (Run 9 or Run 17).
- Stage 2: rerank **within the top-5000** using a GBT score (Run 18), leaving the membership of the top-5000 unchanged.

This preserves queue purity (`precision@5000`) while drastically improving the “first page” (`precision@100`). See Runs 21/22 in [sparkov_model_eval.md](./sparkov_model_eval.md).

## Pipeline Notes

Full benchmark rebuild: **8,580,255** rows through Bronze → Silver → Gold. Local runner uses `config/sparkov.yaml` Spark settings (higher heap, safer Parquet reader).

## Model Evaluation Summary

**Five-feature logistic** (`prior_amount_zscore`, `amount_sum_last_1h`, `merchant_prior_fraud_rate`, `tx_count_last_15m`, `tx_count_last_1h`):
test PR AUC `0.3104` · precision `0.3676` · recall `0.3204` · F1 `0.3424`

**Amount-only logistic** (`prior_amount_zscore`, `amount_sum_last_1h`):
test PR AUC `0.3807` · precision `0.4996` · recall `0.4869` · F1 `0.4931`

**Amount + night logistic** (`prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`):
test PR AUC `0.4031` · precision `0.5191` · recall `0.5033` · F1 `0.5111` — **current best general baseline**

**Follow-up (Runs 10–12, full data, same time split):** see [sparkov_model_eval.md](./sparkov_model_eval.md).

- **GBT vs logistic** on the **same three features:** GBT test PR AUC `0.4880` and top-100 precision `0.76`, but logistic still leads on test F1 (`0.5111` vs `0.5067`) and on **5k/10k** Top-K precision — same **small-queue vs large-queue** tradeoff as **amount-only** GBT (Run 6), not a replacement default.
- **`hour_of_day` added** on top of the night flag: test PR AUC `0.4019`, F1 `0.5104` — **does not beat** the three-feature baseline; keep the **boolean night** flag, not raw hour, in the default story.
- **Alert inspection** (`inspect_sparkov_alerts.py`, `--feature-set amount_plus_night`): top-100 still all **non-fraud** with **score ties** at 1.0; categories **personal_care**, **kids_pets**, **travel** dominate; at top-5k, FP vs fraud still separate on **night** and **hour** averages.

Key takeaways:

- Amount family carries the model; velocity and merchant features did not improve the logistic baseline
- `is_night_transaction` is the first third feature to beat amount-only cleanly
- GBT (amount-only **or** three-feature) sharpens **very small** Top-K; logistic stays stronger for **5k–10k**-style budgets
- Logistic false positives are dominated by ultra-extreme legitimate spend bursts (`travel` and related categories)
- `tx_count_last_1h` as a third feature did not beat amount-only cleanly
- Raw `hour_of_day` did not improve over `is_night_transaction` alone in the latest logistic run

Full experiment history: [sparkov_model_eval.md](./sparkov_model_eval.md)

Final runbook and defaults: [`../sparkov/model_eval_latest.md`](../sparkov/model_eval_latest.md).
