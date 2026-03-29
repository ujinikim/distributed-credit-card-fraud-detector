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

## Final Thoughts

This project started as a way to learn distributed data engineering through a problem that felt concrete: credit card fraud detection. It grew into a full Bronze, Silver, Gold pipeline with a benchmark evaluation path on Sparkov, and that made it a much better engineering exercise than a model-only project would have been. What made the work interesting was not just whether a model could separate fraud from non-fraud, but whether the data contracts, transformations, and repo structure stayed understandable as the project got bigger.

The biggest lesson was that clean pipeline boundaries mattered more than clever ideas. A lot of the engineering value came from keeping each layer honest: Bronze for ingestion, Silver for cleaning and a stable schema, and Gold for feature work. That separation made it easier to reason about mistakes, easier to test the pipeline in smaller pieces, and easier to evolve the project without turning every change into a full rewrite. I also learned that keeping reusable implementation in `src/fraud_lens/` and runnable commands in `scripts/` was worth the extra structure because it made the repo easier to navigate once the number of entry points started to grow.

A few concrete numbers helped anchor the project for me. The full benchmark rebuild runs through `8,580,255` rows, including `94,806` fraud rows. On the modeling side, the strongest baseline logistic setup reached a PR AUC of `0.4031` and a `precision@5000` of `0.6200`. Those are not the whole story, but they were enough to show that the pipeline was doing real work at meaningful scale.

Another useful lesson was that simple features often carried more value than more exotic ones. Prior amount behavior, short-window spending, and a small amount of time context consistently mattered more than some of the more interesting geography or merchant-history ideas. Synthetic data still mattered, but mostly as a sanity tool: it was the safe place to check assumptions and catch regressions. Sparkov was what made the system feel realistic.

I also became more precise about why PySpark made sense here. The best justification was not that Spark wins every small benchmark, because on capped `50k-100k` row comparisons a single-node sklearn path can still be faster. The real case for Spark was that this project runs an integrated Bronze, Silver, Gold pipeline plus repeated feature preparation and evaluation on `8,580,255` Gold rows, which is where a DataFrame-based pipeline becomes much more practical than a local in-memory workflow.

If I were starting over, I would explain the repo structure earlier, formalize packaging sooner, and be more disciplined about separating “current guidance” docs from historical notes. The main takeaway I’m leaving with is that building a data system is as much about clarity and restraint as it is about throughput or model quality. The code matters, but the real work is in making the whole pipeline understandable enough to trust.

Default benchmark pipeline: `python scripts/run_sparkov_pipeline.py` (paths in [config/sparkov.yaml](config/sparkov.yaml)).

Repo mental model: `src/fraud_lens/` contains the reusable package code; `scripts/` contains thin CLI runners that import from it.

## Repo layout

| Path | Role |
| --- | --- |
| `config/` | Paths and pipeline config |
| `data/` | Bronze, Silver, Gold layer outputs |
| `src/fraud_lens/` | Package code: medallion pipeline stages, Sparkov helpers, synthetic fixture support |
| `scripts/` | Entry points: runnable pipeline/evaluation commands that call into `fraud_lens` |
| `docs/` | `architecture/`, `sparkov/`, `reference/`, `archive/` — map: [docs/README.md](docs/README.md) |

Common entry points:

- `python scripts/run_sparkov_pipeline.py` — default Bronze → Silver → Gold benchmark pipeline
- `python scripts/run_pipeline.py` — generic/local Bronze → Silver → Gold runner
- `python scripts/normalize_sparkov_data.py` — normalize Sparkov source data into canonical raw records

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

Use synthetic data only for deterministic regression checks and fast smoke tests. Spec: [docs/reference/synthetic_data_spec.md](docs/reference/synthetic_data_spec.md).
