# Sparkov Feature Findings

This note captures what we learned after expanding the Gold feature set and rerunning the full Sparkov benchmark pipeline.

It reflects the current notebook sanity checks rather than a final modeling decision. The goal is to record which features look strong, which look weak, and where to spend less time next.

## Current Read

The current benchmark story is no longer just:

- `amount_zscore` looks useful
- `speed_from_prev_kmh` looks brittle

After the latest Gold expansion, the picture is:

- amount-based features look strong
- amount window features look strong
- merchant-history features look promising
- customer-to-merchant distance looks weak
- raw travel speed still looks too brittle to trust as a headline benchmark feature

## What Looks Strong

### 1. Prior-only amount behavior

`prior_amount_zscore` looks better than the original per-card `amount_zscore`.

Observed benchmark summary:

- non-fraud median `prior_amount_zscore`: about `-0.27`
- fraud median `prior_amount_zscore`: about `2.70`
- fraud 90th percentile `prior_amount_zscore`: about `8.60`

Interpretation:

- fraud rows are shifted strongly toward unusually large amounts relative to prior card history
- the prior-only definition is more trustworthy than a z-score that includes the current row in its own baseline
- this should stay in the benchmark feature set

### 2. Hourly amount accumulation

`amount_sum_last_1h` also shows clear fraud separation.

Observed benchmark summary:

- non-fraud median `amount_sum_last_1h`: about `57.32`
- fraud median `amount_sum_last_1h`: about `848.0`
- fraud average `amount_sum_last_1h`: about `1069.96`

Interpretation:

- fraud rows often appear inside much heavier short-window spending bursts
- recent amount accumulation looks more useful than raw transaction count alone
- amount windows are worth keeping and likely worth expanding carefully

### 3. Merchant prior fraud rate

`merchant_prior_fraud_rate` looks promising in the benchmark read.

Observed benchmark summary:

- non-fraud median `merchant_prior_fraud_rate`: about `0.0067`
- fraud median `merchant_prior_fraud_rate`: about `0.0204`
- fraud 90th percentile `merchant_prior_fraud_rate`: `1.0`

Interpretation:

- merchant history appears to contain meaningful signal
- this feature should stay under active evaluation
- we should keep treating merchant fraud-rate features as leakage-sensitive and ensure they remain prior-only

## What Looks Useful but Secondary

### Velocity features

Velocity windows show some separation, but less dramatically than the amount features.

Observed benchmark summary:

- non-fraud average `tx_count_last_1h`: about `1.33`
- fraud average `tx_count_last_1h`: about `1.89`

Interpretation:

- velocity is probably helpful as supporting context
- it does not currently look like the strongest standalone benchmark signal
- velocity work should continue, but amount behavior still appears to be the core benchmark path

## What Looks Weak

### Customer-to-merchant distance

`customer_to_merchant_distance_km` does not look like a strong benchmark fraud indicator.

Observed benchmark summary:

- non-fraud median distance: about `78.7 km`
- fraud median distance: about `78.7 km`
- non-fraud 90th percentile distance: about `113.3 km`
- fraud 90th percentile distance: about `113.3 km`

Notebook read:

- the fraud and non-fraud histograms overlap heavily
- fraud may be slightly shifted, but not enough to look like a strong standalone signal

Interpretation:

- keep the feature in Gold for now because it is already implemented and cheap to compute
- do not treat it as a headline benchmark feature
- do not spend more feature-engineering effort on distance-derived expansions right now unless later modeling evidence justifies it

## What Still Looks Brittle

### Travel speed

`speed_from_prev_kmh` still behaves like a tail-heavy diagnostic rather than a robust fraud feature.

Observed benchmark read:

- both fraud and non-fraud still show extreme values
- the plot only becomes interpretable after log transformation or percentile clipping
- the feature remains sensitive to tiny time gaps between transactions

Interpretation:

- keep travel speed as a diagnostic for now
- do not rely on it as a primary benchmark discriminator
- if we revisit it, the next work should be around minimum elapsed-time floors or safer derived travel flags

## Practical Feature Ranking

Current working ranking from the notebook sanity checks:

- **Strong:** `prior_amount_zscore`, `amount_sum_last_1h`
- **Promising:** `merchant_prior_fraud_rate`, velocity windows
- **Weak:** `customer_to_merchant_distance_km`
- **Brittle diagnostic only:** `speed_from_prev_kmh`

## Pipeline and Runtime Notes

The expanded Sparkov Gold pipeline now runs end to end successfully with the benchmark runner after adding Sparkov-specific local Spark settings.

Current successful benchmark rebuild:

- Bronze rows: `8,580,255`
- Silver rows: `8,580,255`
- Gold rows: `8,580,255`

Operational note:

- the default local runner needed higher heap and safer Parquet settings to avoid memory failures
- those settings are now carried in `config/sparkov.yaml` and used by `scripts/run_sparkov_pipeline.py`

## Recommended Next Step

The next step should not be another large feature expansion.

The best next move is:

1. keep the strong amount and merchant-history features in Gold
2. use the notebook to continue lightweight sanity checks
3. de-emphasize customer-distance and raw travel-speed features
4. move toward simple model-oriented evaluation to see which features still matter in combination
