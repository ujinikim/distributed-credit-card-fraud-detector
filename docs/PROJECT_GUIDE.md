# Project Guide

Quick navigation for the FraudLens Sparkov workflow.

## Start Here

1. Read `README.md` for setup and pipeline entrypoints.
2. Read `docs/medallion_layers.md` for Bronze/Silver/Gold responsibilities.
3. Read `docs/model_eval_latest.md` for the current modeling decision.

## Where Things Live

- `src/fraud_lens/silver_to_gold/transform.py`  
  Gold feature engineering, including card x category features and low-history handling.

- `scripts/evaluate_sparkov_model.py`  
  Model training/evaluation CLI, feature-set wiring, and Top-K metrics.

- `scripts/inspect_sparkov_alerts.py`  
  Error analysis for top-ranked alerts.

- `docs/sparkov_model_eval.md`  
  Full run-by-run experiment archive.

- `docs/model_eval_latest.md`  
  Current recommendation and key run comparison.

## Common Tasks

- Rebuild Sparkov medallion layers:
  - `python scripts/run_sparkov_pipeline.py`

- Evaluate one feature set:
  - `python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night --model-type logistic`

- Compare LR vs GBT:
  - `python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night --model-type both`

- Inspect top alerts:
  - `python scripts/inspect_sparkov_alerts.py --feature-set amount_plus_night --model-type logistic`

## Terminology

- **Gating K:** minimum prior count needed before category z-score is considered eligible.
- **Top-K:** evaluation budget (`precision@100`, `precision@5000`, `precision@10000`), unrelated to gating K.

## Doc Reading Order (Modeling)

1. `docs/sparkov_feature_findings.md` (what worked)
2. `docs/model_eval_latest.md` (what to use now)
3. `docs/sparkov_model_eval.md` (full run history)
4. `docs/sparkov_feature_roadmap.md` (what to try next)
