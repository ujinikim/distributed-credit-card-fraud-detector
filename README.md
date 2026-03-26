# FraudLens

Distributed credit card fraud detection: PySpark, medallion architecture (Bronze → Silver → Gold), and ML-oriented transaction feature engineering.

Current focus:

- Sparkov benchmark analysis as the default dataset path
- Synthetic dataset support as a smaller controlled test fixture
- Spark environment and data processing pipeline

ML and the LLM Auto-Investigator come later.

## Current Direction

- Sparkov benchmark analysis is the default dataset path.
- Synthetic data remains a smaller controlled fixture for regression tests, smoke tests, and explicit anomaly validation.
- Feature extraction belongs primarily in the Gold layer, after Bronze ingestion and Silver cleaning.

## Repo layout

- `config/` — Paths and pipeline config
- `data/` — Bronze, Silver, Gold layers
- `src/fraud_lens/` — Ingest, Bronze→Silver, Silver→Gold, synthetic generator
- `scripts/` — Pipeline runners
- `docs/AGENTS.md` — Short contributor orientation
- `docs/PROJECT_GUIDE.md` — Quick project navigation and command map
- `docs/medallion_layers.md` — Layer responsibilities and the current Silver contract
- `docs/model_eval_latest.md` — Current model recommendation and key metrics
- `docs/sparkov_model_eval.md` — Full experiment history archive
- `docs/sparkov_benchmark_plan.md` — Sparkov dataset role and source-to-canonical mapping
- `docs/sparkov_feature_roadmap.md` — Next Sparkov-aware Gold feature families

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run (when implemented): `python scripts/run_pipeline.py`

## Data Strategy

FraudLens now uses two complementary data paths:

- **Sparkov benchmark data** as the default dataset for feature design, scaling checks, and exploratory analysis
- **Synthetic data** as a smaller controlled fixture for pipeline validation, regression tests, and explicitly constructed anomaly cases

The project contract and recommended reading order are:

1. [docs/medallion_layers.md](docs/medallion_layers.md)
2. [docs/sparkov_benchmark_plan.md](docs/sparkov_benchmark_plan.md)
3. [docs/sparkov_feature_findings.md](docs/sparkov_feature_findings.md)
4. [docs/model_eval_latest.md](docs/model_eval_latest.md)
5. [docs/sparkov_model_eval.md](docs/sparkov_model_eval.md)
6. [docs/sparkov_feature_roadmap.md](docs/sparkov_feature_roadmap.md)

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

## Synthetic Workflow

Synthetic data is still useful, but it is no longer the primary analysis path.

Use it when you need:

- deterministic small-scale regression checks
- controlled impossible-travel or spending-spike examples
- fast pipeline smoke tests

The synthetic generator spec lives in [docs/synthetic_data_spec.md](docs/synthetic_data_spec.md).
