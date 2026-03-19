# FraudLens

Distributed credit card fraud detection: PySpark, medallion architecture (Bronze → Silver → Gold), and ML for "Flash Fraud" (impossible travel, spending spikes).

**v1:** Synthetic dataset, Spark environment, and data processing pipeline. ML and LLM Auto-Investigator come later.

## Repo layout

- `config/` — Paths and pipeline config
- `data/` — Bronze, Silver, Gold layers
- `src/fraud_lens/` — Ingest, Bronze→Silver, Silver→Gold, synthetic generator
- `scripts/` — Pipeline runners
- `docs/AGENTS.md` — Task briefs per workstream

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run (when implemented): `python scripts/run_pipeline.py`
