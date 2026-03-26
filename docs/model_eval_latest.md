# Model Evaluation (Latest)

Current, decision-oriented model snapshot for Sparkov.  
For full experiment history, see `docs/sparkov_model_eval.md`.

## Current Default

- **Primary baseline (general quality):** Run 9 logistic
- **Queue-optimized logistic:** Run 17 logistic (best LR `precision@5000`)
- **Small-queue specialist:** Run 16/18 GBT (strong top-100, weaker top-5000)

## Selection Criteria

- Primary objective: maximize `precision@5000`
- Tie-break: `precision@10000` when `precision@5000` is within `+-0.005`
- Guardrail: reject if `precision@100 < 0.05`
- Baseline floor (Run 9): `precision@5000 >= 0.62`, `precision@10000 >= 0.52`, `PR AUC >= 0.38`, `F1 >= 0.49`

## Key Runs (Test)

| Run | Model | Feature Set | PR AUC | F1 | P@100 | P@5000 | P@10000 | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 9 | Logistic | `amount_plus_night` | 0.4031 | 0.5111 | 0.0000 | 0.6200 | 0.5226 | Best global balance baseline |
| 13 | Logistic | `amount_plus_night_catz` | 0.3413 | 0.4582 | 0.0100 | 0.6446 | 0.5286 | Better queue purity, worse global metrics |
| 14 | Logistic | `amount_plus_night_catz_v2` | 0.3170 | 0.4343 | 0.0100 | 0.6394 | 0.5326 | Damped V2 under baseline globally |
| 17 | Logistic | `amount_plus_night_catz_v3_damped_shrunk` | 0.3178 | 0.4345 | 0.0100 | 0.6450 | 0.5344 | Best LR queue purity among category variants |
| 16 | GBT | `amount_plus_night_catz_v3_shrunk` | 0.4673 | 0.5130 | 0.9900 | 0.6076 | 0.5216 | Strong very-top ranking, weak 5k queue purity |
| 18 | GBT | `amount_plus_night_catz_v3_damped_shrunk` | 0.4665 | 0.5130 | 0.9900 | 0.6070 | 0.5204 | Similar to Run 16 |
| 19 | Logistic + class weights | `amount_plus_night_catz_v3_damped_shrunk` | 0.3014 | 0.4891 | 0.0000 | 0.4464 | 0.5024 | Rejected: large queue-quality drop |

## Current Recommendation

- Keep **Run 9** as the default for balanced detection quality.
- Use **Run 17** only when queue purity at `k=5000` is the primary objective.
- Do not use class-weighted logistic in this setup (Run 19 regression).

## Next Practical Focus

1. Improve ranking resolution at the top of the queue (tie-break signal).
2. Reduce extreme-spend false positives (clip/log transforms).
3. Add one interaction feature for linear models (`amount z-score x low-history`).
