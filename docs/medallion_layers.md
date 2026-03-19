# FraudLens Medallion Layers

This document defines what each medallion layer means in FraudLens so the pipeline stays consistent as the project grows.

## Bronze

Purpose: preserve raw transaction data with minimal transformation.

- Input: raw JSON Lines files from the synthetic generator or normalized benchmark data such as Sparkov in `data/raw`
- Output: Parquet files in `data/bronze`
- Current implementation: [`src/fraud_lens/ingest/bronze.py`]

Expected schema:

- Raw transaction fields from the canonical transaction schema:
  - `transaction_id`
  - `card_id`
  - `event_time`
  - `amount`
  - `merchant_category`
  - `latitude`
  - `longitude`
  - `anomaly_type`
  - `ref_transaction_id` (nullable; synthetic-only for now)
- Ingestion metadata added in Bronze:
  - `ingestion_timestamp`
  - `source_path`

Allowed transformations:

- Schema-on-read from raw JSONL
- Add ingestion metadata such as timestamp and source file path
- Write data to Parquet for downstream processing

Not allowed in Bronze:

- Business-rule cleaning
- Deduplication
- Invalid-value filtering
- Feature engineering

Quality checks:

- Files can be read from the configured raw path
- Bronze output is written successfully to `data/bronze`
- Ingestion metadata columns are present

## Silver

Purpose: create a clean, typed, trustworthy transaction table for downstream feature engineering.

- Input: Bronze Parquet from `data/bronze`
- Output: cleaned Parquet in `data/silver`
- Current implementation: [`src/fraud_lens/bronze_to_silver/transform.py`]
Expected schema:

- `transaction_id` as string
- `card_id` as string
- `event_time` as string
- `amount` as double
- `merchant_category` as string
- `latitude` as double
- `longitude` as double
- `anomaly_type` as string
- `ref_transaction_id` as nullable string
- `ingestion_timestamp` as timestamp
- `source_path` as string

Allowed transformations:

- Cast columns to the required schema
- Drop rows with nulls in required transaction fields
- Filter invalid values such as negative amounts or out-of-range coordinates
- Drop duplicate rows by `transaction_id`

Not allowed in Silver:

- ML feature engineering
- Model scoring
- LLM investigation logic

Quality checks:

- Required columns are non-null
- `amount >= 0`
- `latitude` is between `-90` and `90`
- `longitude` is between `-180` and `180`
- `transaction_id` is unique after deduplication

## Gold

Purpose: create ML-ready fraud features from clean Silver transactions.

- Input: Silver Parquet from `data/silver`
- Output: feature tables in `data/gold`
- Current implementation: [`src/fraud_lens/silver_to_gold/transform.py`]

Gold should work against the canonical transaction schema regardless of whether the source data came from the synthetic generator or a benchmark dataset such as Sparkov.

Implemented feature areas (per transaction, per card):

- **Time-based context**
  - `event_time_ts`: parsed timestamp from `event_time`
  - `time_since_last_tx_minutes`: minutes since the previous transaction for the same `card_id`
- **Transaction velocity**
  - `tx_count_last_1h`: number of transactions for the same `card_id` in the last 1 hour
  - `tx_count_last_24h`: number of transactions for the same `card_id` in the last 24 hours
- **Amount-based spike features**
  - `amount_zscore`: z-score of `amount` within the card’s history
  - `is_amount_spike`: boolean flag for large positive z-scores (currently `amount_zscore >= 3.0`)
- **Impossible-travel features**
  - `distance_from_prev_km`: great-circle distance (Haversine) from the previous transaction’s location for the same `card_id`
  - `hours_since_prev`: hours between the current and previous transaction for the same `card_id`
  - `speed_from_prev_kmh`: `distance_from_prev_km / hours_since_prev` when `hours_since_prev > 0`, otherwise `NULL`

Allowed transformations:

- Window functions and aggregations over Silver data
- Feature engineering needed for downstream anomaly detection
- Preserve keys needed to join features back to Silver or raw records

Not allowed in Gold:

- Final model training logic
- LLM-generated reports
- Mutating historical raw meaning of the source transaction

Quality checks:

- Features are reproducible from Silver inputs
- Output keeps transaction-level join keys
- Feature columns are documented and interpretable
- Source-specific nullable fields such as `ref_transaction_id` do not break feature generation

## Layer Boundaries

Use these rules to keep responsibilities clear:

- Bronze is for raw ingestion and lineage metadata
- Silver is for cleaning, typing, and validation
- Gold is for feature engineering only

If a change does not clearly fit one of those responsibilities, document the decision before implementing it.
