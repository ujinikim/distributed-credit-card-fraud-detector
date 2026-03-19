# Sparkov Feature Roadmap

This document lists the most useful next Gold-layer features for Sparkov.

For the current layer contract and Silver schema, see [medallion_layers.md](medallion_layers.md). For source mapping, see [sparkov_benchmark_plan.md](sparkov_benchmark_plan.md).

The goal is to prioritize feature families that are:

- interpretable
- realistic for fraud detection
- implementable in Spark
- useful enough to justify the added pipeline complexity

## Design Principle

- keep Bronze and Silver shared
- let Gold expand with Sparkov-aware feature families
- preserve synthetic compatibility by keeping synthetic-specific logic isolated

## Feature Families

### 1. Customer-relative amount behavior

These are the most natural next features because Sparkov already behaves well for amount signals.

Candidate features:

- rolling mean spend per `card_id`
- rolling standard deviation per `card_id`
- prior-only z-score per `card_id`
- amount percentile rank within card history
- deviation from recent 1-day, 7-day, and 30-day card averages

Why this matters:

- fraud often shows up as amount behavior that is unusual for the specific customer
- this fits the benchmark better than purely global thresholds

Priority: **High**

## 2. Merchant behavior and merchant risk

Sparkov includes merchant identity, which the current Gold pipeline barely uses.

Candidate features:

- transaction count per merchant over recent windows
- merchant fraud rate over recent windows
- merchant/category fraud rate
- distinct cards seen by merchant over recent windows
- customer repeat-merchant frequency

Why this matters:

- fraud is often merchant-pattern driven, not just customer-pattern driven
- merchant recurrence and merchant fraud history are likely higher-value than the current raw speed feature

Priority: **High**

## 3. Customer-to-merchant distance

This is more grounded than the current merchant-to-previous-merchant speed ratio.

Sparkov gives:

- customer home coordinates
- merchant coordinates

Candidate features:

- `customer_to_merchant_distance_km`
- log-distance bucket
- unusually far-from-home flag
- rolling average customer-to-merchant distance
- customer-specific distance z-score

Why this matters:

- distance from a customer’s normal geography is easier to interpret than raw speed from one merchant to the next
- this may be a better fraud signal than the current benchmark travel-speed ratio

Priority: **High**

## 4. Time-of-day and calendar behavior

Sparkov includes enough temporal structure to make timing features useful.

Candidate features:

- hour-of-day
- day-of-week
- weekend flag
- night-transaction flag
- customer-normal-hour deviation
- merchant-normal-hour deviation

Why this matters:

- fraud often clusters at unusual hours relative to a customer’s normal behavior
- these features are inexpensive and interpretable

Priority: **High**

## 5. Velocity and burst behavior

The current velocity features are a good starting point, but Sparkov can support richer variants.

Candidate features:

- tx count last 5m, 15m, 1h, 24h
- total amount spent last 1h and 24h
- distinct merchants last 1h and 24h
- distinct categories last 1h and 24h
- merchant-switch count over recent windows

Why this matters:

- short-term bursts are a common fraud pattern
- distinct-merchant and distinct-category counts may be more useful than raw count alone

Priority: **High**

## 6. Category behavior

Sparkov category should be used more directly.

Candidate features:

- per-card category frequency
- unusual category flag for the card
- category fraud rate
- recent category repetition count
- category novelty score

Why this matters:

- many fraud cases are suspicious because the merchant category is unusual for the customer

Priority: **Medium**

## 7. Profile and demographic context

Sparkov includes profile-like metadata, but this should be used carefully.

Candidate features:

- customer profile segment
- city population bucket
- age bucket from `dob`
- urban vs rural proxy

Why this matters:

- these can improve segmentation and baseline expectations
- but they also increase modeling and fairness complexity

Priority: **Medium**

## 8. Existing enriched Sparkov rolling fields

This Sparkov mirror already includes many rolling features.

Examples:

- `customer_num_trans_1_day`
- `customer_num_trans_7_day`
- `customer_num_trans_30_day`
- `customer_avg_amount_1_day`
- `customer_avg_amount_7_day`
- `customer_avg_amount_30_day`
- `merchant_num_trans_1_day`
- `merchant_num_trans_7_day`
- `merchant_num_trans_30_day`
- `merchant_risk_1_day`
- `merchant_risk_7_day`
- `merchant_risk_30_day`
- `merchant_risk_90_day`

Why this matters:

- these may already outperform some of the handcrafted baseline Gold features
- they are also a strong reference point for validating our own implementations

Risk:

- we should not blindly trust benchmark-provided engineered features without documenting them
- if we use them, we should clearly label them as source-provided benchmark features

Priority: **Medium to High**

## Recommended First Implementation Batch

The first Sparkov-aware Gold expansion should focus on the highest-value, lowest-ambiguity features:

1. prior-only customer amount z-score
2. customer-to-merchant distance
3. time-of-day and weekend features
4. richer velocity windows
5. merchant frequency and merchant fraud-rate windows

This batch gives us:

- customer behavior
- merchant behavior
- geography
- temporal context

without overcommitting to lower-signal or higher-risk fields.

## Pipeline Separation Recommendation

Do not split the entire medallion pipeline into two unrelated systems.

Use a layered separation instead:

### Shared layers

Keep shared:

- Bronze canonicalization contract
- Silver cleaning and typing
- common Gold baseline features

These should work for both Sparkov and synthetic data.

### Sparkov-specific layer

Add a Sparkov-specific Gold extension path for benchmark-aware features.

Examples:

- customer-to-merchant distance
- merchant risk windows
- benchmark-only enriched rolling fields

This can be implemented as either:

- an additional Sparkov Gold transform module, or
- a feature-flagged extension step after the shared Gold baseline

### Synthetic-specific layer

Keep synthetic-only anomaly validation logic isolated.

Examples:

- `ref_transaction_id`
- explicit impossible-travel pairing
- synthetic-only sanity checks

## Recommended Architecture

Best option:

- one shared Bronze/Silver pipeline
- one shared baseline Gold transform
- one Sparkov-specific Gold extension module
- one synthetic-specific validation path

Why:

- shared ingestion and cleaning stay stable
- Sparkov can grow richer without breaking synthetic
- synthetic remains reusable for controlled tests later
- benchmark logic does not pollute the synthetic feature path

## What Not to Do

Avoid these extremes:

- do not keep all feature logic fully shared if the dataset semantics are diverging
- do not fork the entire pipeline into two totally separate codebases

The right middle ground is:

- shared canonical layers
- dataset-specific Gold extensions
