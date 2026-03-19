# FraudLens data sanity rules

This doc captures simple, concrete rules to sanity‑check the synthetic dataset and downstream medallion layers. The goal is to make it easy to verify that data generation, feature engineering, and labels behave as expected, especially for **spending spikes** and **impossible travel**.

These rules are implemented and explored in `notebooks/explore_silver_gold.ipynb`.

## 1. Global expectations

- **Schema alignment**
  - Bronze and Silver follow the schema in `docs/synthetic_data_spec.md` and `docs/medallion_layers.md`.
  - Gold keeps all join keys (`transaction_id`, `card_id`, `event_time`, `anomaly_type`, `ingestion_timestamp`, `source_path`) and adds feature columns from `src/fraud_lens/silver_to_gold/transform.py`.
- **Anomaly prevalence**
  - The fraction of rows where `anomaly_type != "none"` in Silver/Gold should roughly match the configured `anomaly_ratio` in `config/synthetic.yaml` (subject to filtering/deduplication in Silver).

## 2. Spending spike sanity rules

Applies on the **Gold** table.

- **Feature rule**
  - Let `amount_zscore` be the per‑card z‑score of `amount`.
  - Define a spike by rule as:
    - `is_spike_by_rule = amount_zscore >= 3.0`
- **Label consistency checks**
  - For rows with `anomaly_type = "spending_spike"`:
    - A large majority should satisfy `is_spike_by_rule` (high recall of labeled spikes).
  - For rows with `anomaly_type = "none"`:
    - Only a small fraction should satisfy `is_spike_by_rule` (few synthetic false positives).
- **Notebook metrics**
  - The exploration notebook computes, per `anomaly_type`:
    - `rows`
    - `spike_by_rule` = count of rows where `amount_zscore >= 3.0`
    - `spike_by_rule_frac` = `spike_by_rule / rows`

## 3. Impossible‑travel sanity rules

Applies on the **Gold** table.

- **Feature rule**
  - `distance_from_prev_km`: great‑circle distance from previous transaction location for the same `card_id`.
  - `hours_since_prev`: time difference in hours to the previous transaction for the same `card_id`.
  - Define impossible travel by rule as:
    - `is_impossible_by_rule = speed_from_prev_kmh > 1000.0`
    - where `speed_from_prev_kmh = distance_from_prev_km / hours_since_prev` when `hours_since_prev > 0`.
- **Label consistency checks**
  - For rows with `anomaly_type = "impossible_travel"`:
    - A large majority should satisfy `is_impossible_by_rule`.
  - For rows with `anomaly_type = "none"`:
    - Only a small fraction should satisfy `is_impossible_by_rule`.
- **Notebook metrics**
  - The exploration notebook computes, per `anomaly_type`:
    - `rows`
    - `impossible_by_rule` = count of rows where `speed_from_prev_kmh > 1000.0`
    - `impossible_by_rule_frac` = `impossible_by_rule / rows`

## 4. Visual sanity checks

The notebook includes plots (built from sampled Gold data) to visually validate the above rules:

- **Amount distributions** by `anomaly_type` (histograms of `amount`).
- **Amount z‑score distributions** by `anomaly_type` (histograms of `amount_zscore`).
- **Travel speed distributions** by `anomaly_type` (histograms of `speed_from_prev_kmh`, optionally clipped to a reasonable range).
- **Distance vs. time between transactions** (e.g., hexbin of `hours_since_prev` vs `distance_from_prev_km`) to see impossible‑travel points in the upper‑left corner.

These visualizations are for human intuition and should broadly agree with the rule‑based metrics above. If the plots and metrics disagree (e.g. many `anomaly_type = "none"` rows look like spikes or impossible travel), that is a sign to revisit the synthetic generator or feature logic.

