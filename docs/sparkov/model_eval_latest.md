# Model Evaluation (Latest)

Current, decision-oriented model snapshot for Sparkov.  
For full experiment history, see [`../archive/sparkov_model_eval.md`](../archive/sparkov_model_eval.md).

For **throughput, scaling, and Spark vs single-node sklearn** notes (benchmark script and tables to fill in), see [`spark_performance_report.md`](spark_performance_report.md).

## Current Default

- **Primary baseline (general quality):** Run 9 logistic
- **Queue-optimized logistic:** Run 17 logistic (best LR `precision@5000`)
- **Small-queue specialist:** Run 16/18 GBT (strong top-100, weaker top-5000)
- **Best practical queue UX (two-stage):** Rerank top-5000 from Run 9/17 using Run 18 GBT (improves top-100 without changing `precision@5000`)

## Selection Criteria

- Primary objective: maximize `precision@5000`
- Tie-break: `precision@10000` when `precision@5000` is within `+-0.005`
- Guardrail: reject if `precision@100 < 0.05`
- Baseline floor (Run 9): `precision@5000 >= 0.62`, `precision@10000 >= 0.52`, `PR AUC >= 0.38`, `F1 >= 0.49`

## Key Runs (Test)

<!-- markdownlint-disable MD060 -->
| Run | Model                    | Feature Set                               | PR AUC | F1     | P@100  | P@5000 | P@10000 | Notes                                         |
| --- | ------------------------ | ----------------------------------------- | ------ | ------ | ------ | ------ | ------- | --------------------------------------------- |
| 9   | Logistic                 | `amount_plus_night`                       | 0.4031 | 0.5111 | 0.0000 | 0.6200 | 0.5226  | Best global balance baseline                  |
| 13  | Logistic                 | `amount_plus_night_catz`                  | 0.3413 | 0.4582 | 0.0100 | 0.6446 | 0.5286  | Better queue purity, worse global metrics     |
| 14  | Logistic                 | `amount_plus_night_catz_v2`               | 0.3170 | 0.4343 | 0.0100 | 0.6394 | 0.5326  | Damped V2 under baseline globally             |
| 16  | GBT                      | `amount_plus_night_catz_v3_shrunk`        | 0.4673 | 0.5130 | 0.9900 | 0.6076 | 0.5216  | Strong very-top ranking, weak 5k queue purity |
| 17  | Logistic                 | `amount_plus_night_catz_v3_damped_shrunk` | 0.3178 | 0.4345 | 0.0100 | 0.6450 | 0.5344  | Best LR queue purity among category variants  |
| 18  | GBT                      | `amount_plus_night_catz_v3_damped_shrunk` | 0.4665 | 0.5130 | 0.9900 | 0.6070 | 0.5204  | Similar to Run 16                             |
| 19  | Logistic + class weights | `amount_plus_night_catz_v3_damped_shrunk` | 0.3014 | 0.4891 | 0.0000 | 0.4464 | 0.5024  | Rejected: large queue-quality drop            |
| 20L | Logistic                 | `amount_plus_night_catz_v4_interact_clip` | 0.2854 | 0.4053 | 0.2500 | 0.4660 | 0.4951  | Rejected: queue quality regressed vs baseline |
| 20G | GBT                      | `amount_plus_night_catz_v4_interact_clip` | 0.4684 | 0.5158 | 0.9900 | 0.6056 | 0.5248  | Better global metrics, still below P@5k floor |
| 21  | Two-stage                | `amount_plus_night` + `amount_plus_night_catz_v3_shrunk` | 0.4436 | 0.5153 | 1.0000 | 0.6200 | 0.5226 | Run 9 LR + 18 GBT pure rerank; improves P@100, preserves P@5k |
| 22  | Two-stage                | `amount_plus_night_catz_v3_damped_shrunk` + `amount_plus_night_catz_v3_shrunk` | 0.3594 | 0.4380 | 1.0000 | 0.6450 | 0.5344 | Run 17 LR + 18 GBT pure rerank; improves P@100, preserves P@5k |

<!-- markdownlint-enable MD060 -->

## Current Recommendation

- Keep **Run 9** as the default for balanced detection quality.
- Use **Run 17** when queue purity at `k=5000` is the primary objective.
- If you want a better “first page” without changing the base queue purity at 5k: use the **two-stage reranker** (Run 21 or Run 22).
- Do not use class-weighted logistic in this setup (Run 19 regression).
- Keep Run 20 as an archived experiment (no promotion): it does not pass the `precision@5000` floor.

This repository is maintained as a **complete** FraudLens benchmark snapshot; no further modeling experiments are planned here.
