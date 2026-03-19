# FraudLens

Distributed credit card fraud detection: PySpark, medallion architecture (Bronze → Silver → Gold), and ML for "Flash Fraud" (impossible travel, spending spikes).

Current focus:

- Synthetic dataset support for pipeline development and controlled anomaly tests
- Sparkov benchmark integration for more realistic feature evaluation
- Spark environment and data processing pipeline

ML and the LLM Auto-Investigator come later.

## Repo layout

- `config/` — Paths and pipeline config
- `data/` — Bronze, Silver, Gold layers
- `src/fraud_lens/` — Ingest, Bronze→Silver, Silver→Gold, synthetic generator
- `scripts/` — Pipeline runners
- `docs/AGENTS.md` — Task briefs per workstream
- `docs/sparkov_benchmark_plan.md` — Sparkov benchmark dataset plan and schema mapping

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run (when implemented): `python scripts/run_pipeline.py`

## Data Strategy

FraudLens now uses two complementary data paths:

- **Synthetic data** for controlled pipeline validation, anomaly injection, and regression testing
- **Sparkov benchmark data** as the primary dataset for evaluating whether Bronze, Silver, Gold, and the engineered fraud features behave credibly on richer transaction records

The Sparkov consolidation plan lives in [docs/sparkov_benchmark_plan.md](docs/sparkov_benchmark_plan.md).

## Benchmark Workflow

Download the Sparkov source CSV with:

```bash
python scripts/download_sparkov_data.py
```

Then normalize Sparkov CSV files into the canonical raw transaction schema with:

```bash
python scripts/normalize_sparkov_data.py
```

Run the benchmark medallion pipeline with:

```bash
python scripts/run_sparkov_pipeline.py
```

By default these scripts read and write the paths configured in [config/sparkov.yaml](config/sparkov.yaml), downloading the source CSV to `data/benchmark/sparkov/data.csv`, writing canonical JSON records to `data/raw_sparkov`, and writing benchmark Bronze/Silver/Gold outputs under `data/benchmark/`.
