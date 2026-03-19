# Synthetic data spec — FraudLens

Detailed specification for the synthetic transaction generator. Used by the **Synthetic data generator** workstream (see [AGENTS.md](AGENTS.md#1-synthetic-data-generator)). Bronze ingest and Silver schema must align with this spec.

---

## 1. Purpose

Generate synthetic credit card transaction data for development and testing. Normal behavior is produced with **Gaussian (normal) distributions** so the downstream ML pipeline has realistic clusters to learn. A small fraction of records are **anomalies** (Flash Fraud: impossible travel or spending spikes) for ML validation. Output is raw JSON Lines written to a configured path (e.g. `data/raw/`) for Bronze ingest.

---

## 2. Schema

All generated records share the same flat schema. Types and names must match what Bronze/Silver expect.

| Column            | Type    | Description |
|-------------------|---------|-------------|
| `transaction_id`  | string  | Unique ID (e.g. UUID or prefixed counter). |
| `card_id`         | string  | Card (or user) identifier; used for velocity and impossible-travel logic. |
| `event_time`      | string  | ISO-8601 timestamp of the transaction. |
| `amount`          | float   | Transaction amount; non-negative. Normal: Gaussian. Anomaly: can be spike. |
| `merchant_category` | string | Category code (e.g. retail, travel, food). Normal: sample from distribution. |
| `latitude`        | float   | Merchant latitude. Normal: Gaussian per card “region.” Anomaly: impossible travel uses distant point. |
| `longitude`       | float   | Merchant longitude. Same as latitude. |
| `anomaly_type`    | string  | `"none"` for normal; `"impossible_travel"` or `"spending_spike"` for anomalies (for evaluation only; do not use at train time). |
| `ref_transaction_id` | string/null | Reference transaction for impossible-travel pairs; null for normal and spending-spike rows. |

Optional columns (can be added later without breaking ingest if they are optional in Bronze): e.g. `merchant_id`, `currency`.

---

## 3. Normal behavior

- **transaction_id:** Unique per record (e.g. `tx_<uuid>` or `tx_<run_id>_<counter>`).
- **card_id:** Sample uniformly from a fixed set of card IDs (size from config, e.g. `num_cards`). Same card can appear in many transactions.
- **event_time:** Spread over the configured date range. Use a Gaussian or uniform spread so events are not all at the same second. Optionally skew toward “business hours” (e.g. 08:00–20:00 local).
- **amount:** Gaussian with configurable `amount_mean` and `amount_std`. Clip to non-negative. Typical: mean ~50–100, std ~30–80 so most amounts are positive and moderate.
- **merchant_category:** Categorical. Sample from a discrete distribution (e.g. retail 40%, food 25%, travel 15%, other 20%). Categories must be a fixed list in config.
- **latitude / longitude:** Per-card “home” region: draw one (lat, lon) per card from a Gaussian (e.g. one “city” center). For each transaction of that card, sample (lat, lon) from a Gaussian around that center so normal transactions are geographically clustered. Use same μ, σ in config for all cards or per-region if you extend later.

Normal records must have `anomaly_type = "none"`.

---

## 4. Anomaly injection

- **Ratio:** Config parameter `anomaly_ratio` (e.g. 0.001–0.02). A fraction of generated rows are anomalies; the rest are normal.
- **Which rows:** Random selection (after generating a normal row, with probability `anomaly_ratio` replace with or overwrite to an anomaly row).

**Impossible travel**

- Same `card_id`, two transactions: first at (lat1, lon1), second at (lat2, lon2) with a time delta too small for the geographic distance (e.g. speed > max plausible speed).
- Implementation: either (a) generate a normal transaction then append a second transaction for the same card with (lat2, lon2) far away and `event_time` a few minutes later, or (b) in a second pass, pick a subset of cards and insert a “far” transaction shortly after a recent one. Mark both or the anomalous one with `anomaly_type = "impossible_travel"`.
- Use a simple distance approximation (e.g. Haversine or Euclidean in deg scaled) and a configurable max speed (km/h) or min minutes between two locations.

**Spending spike**

- Transaction amount far above the card’s usual (e.g. > K standard deviations above that card’s mean amount, or above a global threshold).
- Implementation: when generating an anomaly, set `amount` to a high value (e.g. global mean + 5*global_std, or sample from a separate “high amount” distribution). Set `anomaly_type = "spending_spike"`.

Anomaly records must have `anomaly_type` set to `"impossible_travel"` or `"spending_spike"`; do not use this field for model training, only for evaluation.

---

## 5. Output format

- **Format:** JSON Lines (`.jsonl`): one JSON object per line, UTF-8 encoded. Each line is a single transaction; no outer array.
- **Path:** From config (e.g. `data.raw` in `config/paths.yaml`). Generator writes under that directory.
- **Naming:** Default mode is a clean overwrite to `transactions.jsonl` so each run produces one self-contained raw dataset. Optional append mode can keep `transactions_<run_id>.jsonl` history when explicitly enabled.
- **Order:** No strict requirement; can be by event_time or by generation order. Downstream Bronze can sort if needed.

---

## 6. Config

Parameters the generator reads (from `config/synthetic.yaml` or a dedicated section in `config/paths.yaml` / main config):

| Parameter         | Type   | Description |
|-------------------|--------|-------------|
| `num_transactions`| int    | Total number of records to generate. |
| `start_date`      | string | Start of event_time range (ISO date or datetime). |
| `end_date`        | string | End of event_time range. |
| `seed`            | int    | Random seed for reproducibility. |
| `anomaly_ratio`   | float  | Fraction of rows that are anomalies (0.0 to 1.0). |
| `num_cards`       | int    | Size of card_id pool. |
| `amount_mean`     | float  | Mean of Gaussian for normal amount. |
| `amount_std`      | float  | Std of Gaussian for normal amount. |
| `output_path`     | string | Directory to write JSONL files (can override from paths.yaml). |
| `raw_write_mode`  | string | `"overwrite"` for a clean raw dataset per run, or `"append"` to keep timestamped raw files. |

Optional later: `merchant_categories` (list and weights), `geo_center` and `geo_std` for lat/long, `max_speed_kmh` for impossible travel, `spike_amount_multiplier` for spending spikes.

For label quality, normal same-card events should also respect a configurable minimum time gap so unlabeled rows do not routinely create impossible-looking speeds on their own.

---

## 7. Example records

**Normal:**

```json
{"transaction_id": "tx_abc123_1", "card_id": "card_0042", "event_time": "2013-09-15T14:32:00Z", "amount": 47.23, "merchant_category": "retail", "latitude": 48.8566, "longitude": 2.3522, "anomaly_type": "none"}
```

**Anomaly (spending spike):**

```json
{"transaction_id": "tx_abc123_2", "card_id": "card_0007", "event_time": "2013-09-15T11:20:00Z", "amount": 2500.0, "merchant_category": "travel", "latitude": 40.7128, "longitude": -74.0060, "anomaly_type": "spending_spike"}
```

**Anomaly (impossible travel):** Same card as a recent transaction but far away and minutes apart.

```json
{"transaction_id": "tx_abc123_3", "card_id": "card_0011", "event_time": "2013-09-15T09:05:00Z", "amount": 32.10, "merchant_category": "food", "latitude": -33.8688, "longitude": 151.2093, "anomaly_type": "impossible_travel"}
```

(Previous transaction for `card_0011` might be at 09:00 in Paris; this one at 09:05 in Sydney — physically impossible.)

---

## 8. Alignment with downstream

- **Bronze:** Ingest reads these JSONL files; can add `ingestion_timestamp`, `source_path`. Schema-on-read: same column names and types as above.
- **Silver:** Clean and type; enforce the same column names and types; handle nulls/duplicates. No new columns required for v1; optional `ingestion_timestamp` from Bronze.
- **Gold:** Will add features (e.g. velocity, time since last tx, distance) from this schema; `event_time`, `card_id`, `latitude`, `longitude`, `amount` are the inputs for that.

Implement the generator in `src/fraud_lens/synthetic/` and drive it via config and a script or CLI entrypoint.
