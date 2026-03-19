# Sparkov Benchmark Plan

This document defines how FraudLens will use the Sparkov credit card fraud dataset as the primary benchmark for evaluating the medallion pipeline and fraud features.

Synthetic data still stays in the repo, but it becomes a supporting fixture rather than the main source of truth for feature quality.

## Why Sparkov

Sparkov is the best near-term fit for FraudLens because it already contains the fields our pipeline expects or can derive cleanly:

- transaction timestamp
- card identifier
- amount
- merchant category
- merchant latitude and longitude
- fraud label

Compared with the current synthetic generator, Sparkov gives us a richer and more realistic transaction table without forcing a redesign of the Bronze, Silver, and Gold layers.

## Role in the Project

Use Sparkov as the primary benchmark for:

- evaluating whether Gold features separate suspicious behavior from normal behavior
- validating row counts and schema handling across Bronze, Silver, and Gold
- grounding notebook diagnostics in a more realistic transaction distribution

Keep synthetic data for:

- end-to-end pipeline smoke tests
- controlled anomaly injection
- regression checks on impossible-travel and spending-spike logic

## Planned Dataset Positioning

- **Primary benchmark:** Sparkov
- **Secondary fixture:** internal synthetic generator output

This means future feature decisions should be justified on Sparkov first, then checked against synthetic data for controlled edge cases.

## Sparkov to FraudLens Schema Mapping

Sparkov columns map into the current transaction model as follows.

| FraudLens column | Sparkov source | Notes |
|---|---|---|
| `transaction_id` | `trans_num` | Use the original transaction identifier directly. |
| `card_id` | `cc_num` | Cast to string in Silver for consistency with existing pipeline behavior. |
| `event_time` | `trans_date_trans_time` | Normalize to ISO-8601 string format. |
| `amount` | `amt` | Direct mapping. |
| `merchant_category` | `category` | Direct mapping. |
| `latitude` | `merch_lat` | Use merchant location for travel features. |
| `longitude` | `merch_long` | Use merchant location for travel features. |
| `anomaly_type` | derived from `is_fraud` | Initial plan: map `1 -> "fraud"` and `0 -> "none"` for benchmark runs. |
| `ref_transaction_id` | not present | Set to `NULL` for Sparkov-based Bronze/Silver/Gold runs. |

Additional Sparkov fields should be preserved in Bronze where useful, then either:

- carried through Silver if we decide they support future features, or
- documented as optional benchmark-only columns if they are not yet used downstream

Examples of useful optional Sparkov fields:

- `merchant`
- `unix_time`
- `lat` and `long` for customer location
- `city`, `state`, `zip`
- `city_pop`
- `job`
- `is_fraud`

## Design Decisions

### Merchant coordinates drive travel features

For Sparkov, Gold should compute `distance_from_prev_km` and `speed_from_prev_kmh` using merchant coordinates:

- `latitude <- merch_lat`
- `longitude <- merch_long`

This keeps Gold behavior aligned with how travel is interpreted in FraudLens.

### Fraud labels are not impossible-travel labels

Sparkov does not provide a specific impossible-travel label. Because of that:

- `anomaly_type = "fraud"` should be treated as a generic fraud benchmark label
- impossible-travel notebook checks should remain available, but we should not assume Sparkov fraud labels are travel fraud labels
- synthetic data remains the controlled dataset for explicit impossible-travel evaluation

### `ref_transaction_id` remains synthetic-only for now

Sparkov has no direct equivalent to `ref_transaction_id`, so:

- Bronze and Silver should allow the column to be null
- Gold should preserve it as null on benchmark runs
- notebook checks should treat non-null `ref_transaction_id` as synthetic-only coverage, not as a universal expectation

## Planned Ingestion Approach

Add a dedicated Sparkov ingestion path rather than forcing benchmark data through the synthetic generator flow.

Recommended shape:

1. Raw benchmark files live in a separate configured input location.
2. A Sparkov-specific Bronze ingestion step maps the source columns into the FraudLens raw transaction schema.
3. Silver and Gold continue to work from the canonical schema, with only small adjustments for benchmark-specific nullability and labels.

This keeps the medallion contract stable while making room for richer upstream sources.

## Implementation Checklist

1. Add config for Sparkov raw input and normalized Bronze target paths.
2. Implement a Sparkov ingestion module that maps source columns into the FraudLens canonical transaction schema.
3. Decide which optional Sparkov columns should remain available beyond Bronze.
4. Confirm Silver validation rules still make sense when `anomaly_type` is generic fraud instead of synthetic anomaly subtype.
5. Run Bronze, Silver, and Gold on Sparkov-derived data.
6. Update the notebook so benchmark-mode checks emphasize:
   - row counts
   - schema consistency
   - fraud vs non-fraud feature separation
   - travel-feature distributions without assuming fraud equals impossible travel
7. Keep synthetic notebook checks for explicit impossible-travel and spending-spike validation.

## Open Questions

- Do we want benchmark fraud labels represented as `"fraud"` or as a separate `label` column while leaving `anomaly_type` synthetic-only?
- Which Sparkov columns should be preserved for future models without bloating the canonical transaction schema?
- Should we support mixed benchmark and synthetic runs in the same repo at the same time, or keep them as separate configured workflows?
