# Project guide

Quick navigation and commands for the FraudLens Sparkov workflow. **Docs index:** [README.md](README.md).

## Where things live

- `src/fraud_lens/silver_to_gold/transform.py` — Gold feature engineering (card × category, low-history handling)
- `scripts/evaluate_sparkov_model.py` — Model training/evaluation CLI and Top-K metrics
- `scripts/inspect_sparkov_alerts.py` — Error analysis for top-ranked alerts
- `docs/archive/sparkov_model_eval.md` — Full run-by-run experiment archive
- `docs/sparkov/model_eval_latest.md` — Current recommendation and key run comparison

## Common tasks

- Rebuild Sparkov medallion layers:
  - `python scripts/run_sparkov_pipeline.py`

- Evaluate one feature set:
  - `python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night --model-type logistic`

- Compare LR vs GBT:
  - `python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night --model-type both`

- Two-stage reranker (best “first page” of alerts):
  - Base Run 9 LR → rerank top-5000 using Run 18 GBT:
    - `python scripts/evaluate_two_stage_reranker.py --base-lr-run 9 --reranker-gbt-run 18 --shortlist-n 5000 --rerank-mode pure --topk 100,500,1000,5000,10000`
  - Base Run 17 LR → same:
    - `python scripts/evaluate_two_stage_reranker.py --base-lr-run 17 --reranker-gbt-run 18 --shortlist-n 5000 --rerank-mode pure --topk 100,500,1000,5000,10000`

- Inspect top alerts:
  - `python scripts/inspect_sparkov_alerts.py --feature-set amount_plus_night --model-type logistic`

## Terminology

- **Gating K:** minimum prior count before category z-score is eligible.
- **Top-K:** evaluation budget (`precision@100`, `precision@5000`, `precision@10000`), unrelated to gating K.
