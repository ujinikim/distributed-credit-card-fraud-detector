# FraudLens — Agent briefs

This doc defines what each "agent" (or workstream) is responsible for. Use it when starting a task so scope and constraints are clear. Project rules live in [.cursorrules](../.cursorrules).

---

## 1. Synthetic data generator

- **Scope:** Generate synthetic credit card transaction data for development and testing.
- **Spec:** Full schema, normal behavior, anomaly rules, output format, and examples: [docs/synthetic_data_spec.md](synthetic_data_spec.md).
- **Constraints:**
  - Use Gaussian (normal) distributions for "normal" behavior so the ML pipeline has realistic patterns to learn.
  - Output raw JSON Lines (`.jsonl`) that matches the Bronze ingest schema in the spec.
  - Inject a configurable fraction of anomaly records (impossible travel, spending spikes) per spec.
- **Inputs:** Config (e.g. `config/synthetic.yaml`): row count, date range, seed, anomaly ratio, num_cards, amount distribution, output path.
- **Outputs:** JSONL file(s) under the configured raw path (e.g. `data/raw/`) for Bronze ingest.
- **Key files:** `src/fraud_lens/synthetic/` (generator module), `config/` for parameters.
- **Out of scope:** No Spark in the generator itself; no ML; no LLM.

---

## 2. Spark environment

- **Scope:** Set up and document the Spark runtime (local first; path to off‑prem later).
- **Constraints:**
  - PySpark 3.x, DataFrame API only (no RDDs / MapReduce).
  - Reproducible: dependencies in `requirements.txt` or `pyproject.toml`, config in `config/`.
- **Inputs:** Project layout, dependency list, optional cluster/config for off‑prem.
- **Outputs:** Working local (and optionally cloud) environment, README or runbook for "how to run."
- **Key files:** `requirements.txt` or `pyproject.toml`, `config/`, `README.md`, `scripts/` runners.
- **Out of scope:** No application logic in this brief; only env and runnability.

---

## 3. Ingest / Bronze

- **Scope:** Read raw transaction data (e.g. JSON from the synthetic generator) and write to the Bronze layer.
- **Constraints:**
  - DataFrame API only; avoid unnecessary shuffles.
  - Bronze = "raw" copy: schema-on-read, minimal transformation (e.g. add ingestion timestamp, source path).
- **Near-term direction:** Support both synthetic input and a benchmark ingestion path for Sparkov while keeping one canonical transaction schema for downstream layers.
- **Inputs:** Raw JSON path(s) or stream; config for paths and format.
- **Outputs:** Bronze tables/files (e.g. Parquet) with consistent schema and metadata.
- **Key files:** `src/fraud_lens/ingest/`, `config/` paths.
- **Out of scope:** No cleaning or typing (that's Silver); no feature engineering.

---

## 4. Bronze → Silver

- **Scope:** Clean, type, and validate Bronze data; enforce a single typed schema.
- **Constraints:**
  - DataFrame API only; efficient transformations (minimize shuffles).
  - Handle nulls, duplicates, and invalid values per project rules (e.g. drop or flag).
- **Inputs:** Bronze table(s) or paths.
- **Outputs:** Silver table(s) (e.g. Parquet) with clean, typed columns ready for Gold.
- **Key files:** `src/fraud_lens/bronze_to_silver/`, schema definitions, `config/`.
- **Out of scope:** No feature engineering (that's Gold); no ML.

---

## 5. Silver → Gold (feature engineering)

- **Scope:** Build ML-ready features from Silver for fraud detection (Flash Fraud: impossible travel, spending spikes).
- **Constraints:**
  - DataFrame API only; use window/aggregations efficiently.
  - Features should support unsupervised anomaly detection (e.g. K-Means later).
- **Suggested features:** Velocity (tx count in last N hours), amount (raw or binned), time since last transaction, distance (if geo available) for impossible-travel logic.
- **Near-term direction:** Validate feature behavior on Sparkov first, then use synthetic data for controlled anomaly-specific checks.
- **Inputs:** Silver table(s).
- **Outputs:** Gold table(s) with feature columns and any keys needed for joining back to raw/Silver.
- **Key files:** `src/fraud_lens/silver_to_gold/`, feature list in this doc or `config/`.
- **Out of scope:** No model training or scoring in this brief; only feature tables.

---

## 6. ML pipeline (later)

- **Scope:** Train and apply an unsupervised model (e.g. K-Means) on Gold features to score anomalies.
- **Constraints:**
  - Use `pyspark.ml.Pipeline` (e.g. VectorAssembler, StandardScaler, K-Means).
  - Anomaly = e.g. distance from cluster center above a threshold.
- **Inputs:** Gold feature table(s).
- **Outputs:** Anomaly scores and/or labels per transaction (or per card/time window); model artifact and config.
- **Key files:** `src/fraud_lens/ml/` (or similar), `config/` for model params.
- **Out of scope:** No LLM in this brief; only ML scoring.

---

## 7. Auto-Investigator / LLM (later)

- **Scope:** For each anomaly (or batch), call an LLM to produce a human-readable incident report.
- **Constraints:** TBD (provider, API, rate limits, output format).
- **Inputs:** Anomaly records (IDs, features, context).
- **Outputs:** Reports (e.g. Markdown or JSON) and storage location.
- **Key files:** TBD (e.g. `src/fraud_lens/llm/` or `scripts/`).
- **Out of scope:** No model training; no Bronze/Silver/Gold logic.

---

## Cross-cutting

- **Code style:** PEP 8; Black or Ruff; type hints on public APIs; short docstrings.
- **Project name:** FraudLens. Package name: `fraud_lens`.
- **v1 focus:** Dataset (synthetic), Spark environment, and data processing (Bronze → Silver → Gold). ML and LLM are later phases.
- **Current data strategy:** Sparkov becomes the primary benchmark dataset; synthetic data remains a controlled secondary fixture.
