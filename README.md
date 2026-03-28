# FraudLens

Distributed credit card fraud detection: PySpark, medallion architecture (Bronze → Silver → Gold), and ML-oriented transaction feature engineering. The default path uses **Sparkov** benchmark data at multi-million-row scale; **synthetic** data remains a small fixture for smoke tests and regression checks. Feature extraction lives in **Gold** after Bronze ingestion and Silver cleaning.

## Documentation

**Start here:** [docs/README.md](docs/README.md) — reading order and map of all docs.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Default benchmark pipeline: `python scripts/run_sparkov_pipeline.py` (paths in [config/sparkov.yaml](config/sparkov.yaml)).

## Repo layout

| Path | Role |
| --- | --- |
| `config/` | Paths and pipeline config |
| `data/` | Bronze, Silver, Gold layer outputs |
| `src/fraud_lens/` | Ingest, Bronze→Silver, Silver→Gold, synthetic generator |
| `scripts/` | Pipeline runners and evaluation CLIs |
| `docs/` | `architecture/`, `sparkov/`, `reference/`, `archive/` — map: [docs/README.md](docs/README.md) |

## Data strategy

- **Sparkov:** default dataset for feature design, scaling checks, and model evaluation.
- **Synthetic:** controlled fixture for pipeline validation and explicit anomaly cases ([docs/reference/synthetic_data_spec.md](docs/reference/synthetic_data_spec.md)).

## Benchmark workflow (Sparkov)

```bash
python scripts/download_sparkov_data.py
python scripts/normalize_sparkov_data.py
python scripts/run_sparkov_pipeline.py
```

Paths and Spark settings: [config/sparkov.yaml](config/sparkov.yaml) (e.g. `data/benchmark/sparkov/data.csv`, `data/benchmark/` for Bronze/Silver/Gold).

Optional two-stage reranker (queue UX, improves top-100 without changing top-5000 membership):

```bash
python scripts/evaluate_two_stage_reranker.py --base-lr-run 9 --reranker-gbt-run 18 --shortlist-n 5000 --rerank-mode pure --topk 100,500,1000,5000,10000
```

## Synthetic workflow

Use synthetic data for deterministic regression checks, impossible-travel or spending-spike examples, and fast smoke tests. Spec: [docs/reference/synthetic_data_spec.md](docs/reference/synthetic_data_spec.md).
