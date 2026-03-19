# FraudLens Medallion Layers

This document is the main contract for layer responsibilities and the current Silver schema.

Current direction:

- Sparkov is the default analysis dataset.
- Synthetic data remains a small controlled fixture for regression and anomaly validation.
- Feature extraction belongs in Gold, not Silver.

## Layer Responsibilities

### Bronze

Purpose: preserve raw transaction records with lineage metadata and minimal transformation.

Allowed:

- schema-on-read ingestion
- ingestion metadata such as `ingestion_timestamp` and `source_path`
- format normalization needed to land raw records in Parquet

Not allowed:

- business-rule cleaning
- deduplication
- feature engineering

### Silver

Purpose: create one clean, typed, trustworthy transaction table for downstream feature engineering.

Allowed:

- type casting
- required-field null filtering
- validity checks such as non-negative amounts and coordinate ranges
- deduplication
- nullable passthrough columns needed by downstream Gold feature logic

Not allowed:

- ML feature engineering
- model scoring
- source-specific heuristics that change transaction meaning

### Gold

Purpose: build ML-oriented feature tables from Silver.

Allowed:

- window functions and historical aggregations
- temporal, customer, merchant, and geography feature engineering
- source-aware feature extensions that preserve the shared baseline contract

Not allowed:

- model training
- final alerting logic
- LLM-generated reporting

## Silver Contract

Silver is the handoff point between cleaning and feature extraction. The schema below is what Gold can rely on.

| Column | Type | Required | Scope | Why it exists |
|---|---|---:|---|---|
| `transaction_id` | string | yes | shared | Row identity and downstream joins |
| `card_id` | string | yes | shared | Customer/card history windows |
| `event_time` | string | yes | shared | Canonical transaction time |
| `amount` | double | yes | shared | Amount-based features |
| `merchant_category` | string | yes | shared | Category behavior features |
| `latitude` | double | yes | shared | Merchant location features |
| `longitude` | double | yes | shared | Merchant location features |
| `anomaly_type` | string | yes | shared | Synthetic validation label or benchmark compatibility label |
| `ref_transaction_id` | string | no | synthetic-oriented | Explicit impossible-travel pairing |
| `merchant` | string | no | Sparkov passthrough | Merchant identity windows and repeat-merchant features |
| `is_fraud` | int | no | Sparkov passthrough | Benchmark fraud-rate windows and evaluation |
| `customer_latitude` | double | no | Sparkov passthrough | Customer-to-merchant distance features |
| `customer_longitude` | double | no | Sparkov passthrough | Customer-to-merchant distance features |
| `event_time_unix` | long | no | Sparkov passthrough | Efficient time-window calculations |
| `ingestion_timestamp` | timestamp | no | metadata | Lineage and debugging |
| `source_path` | string | no | metadata | Lineage and debugging |

## Why These Silver Columns Are Enough

The current Silver contract supports the next Gold feature batch:

- prior-only customer amount z-score
- customer-to-merchant distance
- time-of-day and weekend features
- richer velocity windows
- merchant frequency and merchant fraud-rate windows

That means we do not need to add profile-heavy fields such as `city`, `job`, or `dob` yet.

## Current Gold Baseline

The shared Gold transform currently computes:

- parsed event timestamp
- time since previous transaction
- transaction counts over recent windows
- per-card amount z-score and spike flag
- previous-location distance, elapsed hours, and speed

Sparkov-specific Gold work should extend this baseline rather than replace it.

## Quality Checks

Silver should satisfy:

- required shared columns are non-null
- `amount >= 0`
- `latitude` is between `-90` and `90`
- `longitude` is between `-180` and `180`
- duplicate transaction content is removed

Gold should satisfy:

- features are reproducible from Silver inputs
- transaction-level join keys remain available
- Sparkov-only nullable fields do not break synthetic runs

## Related Docs

- [README.md](../README.md)
- [sparkov_benchmark_plan.md](sparkov_benchmark_plan.md)
- [sparkov_feature_findings.md](sparkov_feature_findings.md)
- [sparkov_feature_roadmap.md](sparkov_feature_roadmap.md)
- [synthetic_data_spec.md](synthetic_data_spec.md)
