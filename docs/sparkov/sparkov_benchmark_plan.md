# Sparkov Benchmark Plan

This document defines how FraudLens uses Sparkov as the default benchmark dataset.

For layer responsibilities and the authoritative Silver contract, see [../architecture/medallion_layers.md](../architecture/medallion_layers.md).

## Role in the Project

Use Sparkov as the primary dataset for:

- feature design
- scaling and pipeline validation
- notebook exploration
- benchmark evaluation of fraud vs non-fraud separation

Keep synthetic data for:

- smoke tests
- regression checks
- explicit impossible-travel and spending-spike validation

## Why Sparkov Fits

Sparkov already provides the core fields needed by FraudLens:

- transaction identifier
- card identifier
- transaction time
- amount
- merchant category
- merchant coordinates
- fraud label

It also provides useful benchmark-only fields such as merchant identity, customer coordinates, and source unix timestamps.

## Source to Canonical Mapping

| FraudLens column | Sparkov source | Notes |
|---|---|---|
| `transaction_id` | `trans_num` | Preserve original transaction identifier. |
| `card_id` | `cc_num` | Cast to string for consistency. |
| `event_time` | `unix_time` | Default canonical timestamp source; formatted to ISO-8601. |
| `amount` | `amt` | Direct mapping. |
| `merchant_category` | `category` | Direct mapping. |
| `latitude` | `merch_lat` | Merchant latitude. |
| `longitude` | `merch_long` | Merchant longitude. |
| `anomaly_type` | derived from `is_fraud` | Current benchmark compatibility label: `fraud` or `none`. |
| `ref_transaction_id` | not present | Keep null for benchmark runs. |

Current benchmark passthrough fields retained for downstream Gold:

- `merchant`
- `is_fraud`
- `customer_latitude`
- `customer_longitude`
- `event_time_unix`

## Design Notes

### Merchant coordinates remain the shared location fields

The canonical `latitude` and `longitude` columns represent merchant coordinates for Sparkov-based runs.

This keeps shared Gold location logic stable while allowing benchmark-specific geography features to use:

- merchant coordinates from `latitude`, `longitude`
- customer home coordinates from `customer_latitude`, `customer_longitude`

### Fraud labels are benchmark labels, not synthetic anomaly labels

Sparkov does not encode explicit impossible-travel or spending-spike labels.

That means:

- `anomaly_type = "fraud"` is only a generic fraud compatibility label
- `is_fraud` is the more precise benchmark evaluation field
- synthetic anomaly checks stay separate

### Keep one shared pipeline shape

FraudLens should keep:

- shared Bronze ingestion contract
- shared Silver cleaning contract
- shared baseline Gold features
- Sparkov-specific Gold extensions for benchmark-aware features

This avoids forking the whole medallion system.

## Current Implementation State

Implemented:

- Sparkov download helper
- Sparkov normalization into canonical raw records
- Sparkov Bronze, Silver, and Gold pipeline run
- benchmark passthrough columns preserved through Silver and Gold

Observed benchmark state:

- normalized raw rows: 8,580,255
- Gold rows: 8,580,255
- fraud rows: 94,806
- non-fraud rows: 8,485,449

## Gold features covered by this contract

The Sparkov Silver contract supports the Gold baseline that was built for this project, including:

- prior-only customer amount z-score
- customer-to-merchant distance
- time-of-day and weekend features
- richer velocity windows
- merchant frequency and merchant fraud-rate windows

Profile or demographic fields were out of scope for this benchmark path.
