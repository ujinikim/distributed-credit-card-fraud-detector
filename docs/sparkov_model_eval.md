# Sparkov Model Evaluation Log (History)

This file is the full run-by-run experiment archive.

- For the current recommendation and quick comparison, see `docs/model_eval_latest.md`.
- For project navigation, see `docs/PROJECT_GUIDE.md`.

Each entry below records what changed, the metrics, and what we learned.

## Current Baseline

- **Model:** Spark ML logistic regression
- **Split:** time-based on `event_time_unix`
- **Label:** `is_fraud`
- **Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`
- **Test:** PR AUC `0.4031` · precision `0.5191` · recall `0.5033` · F1 `0.5111`

Rationale: cleanly beats the amount-only baseline on all main metrics and larger Top-K budgets (Run 9).

## Dataset Split

| Partition | Rows | Fraud rows | Fraud rate |
|---|---:|---:|---:|
| Train | 6,003,677 | 61,348 | 1.02% |
| Validation | 1,285,755 | 19,015 | 1.48% |
| Test | 1,290,823 | 14,443 | 1.12% |

Time-based split avoids future-behavior leakage and keeps card history in realistic temporal order.

## Run History

### Run 1: Five-feature logistic baseline

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `merchant_prior_fraud_rate`, `tx_count_last_15m`, `tx_count_last_1h`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4216 | 0.9148 | 0.5220 | 0.3425 | 0.4136 |
| Test (t=0.05) | 0.3104 | 0.9017 | 0.3676 | 0.3204 | 0.3424 |

Confusion: tp=4628 fp=7961 tn=1,268,419 fn=9815

**Takeaway:** Real ranking signal (strong ROC AUC) but weak practical precision. Justified threshold analysis and feature subset comparison rather than a model-family switch.

### Run 2: Threshold sweep on Run 1

| Threshold | Precision | Recall | F1 |
|---:|---:|---:|---:|
| 0.05 | 0.3676 | 0.3204 | 0.3424 |
| 0.10 | 0.4492 | 0.2743 | — |
| 0.20 | 0.5159 | 0.2128 | — |
| 0.50 | 0.6162 | 0.1219 | — |

Top-K: top-1000 `0.533`, top-5000 `0.5232`, top-10000 `0.415`.

**Takeaway:** Model works better as a ranked alerting queue than a binary classifier. Next question: is low precision from thresholds or from noisy features?

### Run 3: Sampled feature subset comparison

Sampled split (train 10%, val 20%, test 20%) for fast directional comparison.

| Subset | PR AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|
| amount_only | 0.3916 | 0.5045 | 0.4948 | 0.4996 |
| amount + velocity | 0.3886 | 0.4993 | 0.4794 | 0.4891 |
| amount + merchant | 0.3141 | 0.3636 | 0.3224 | 0.3417 |
| full_baseline | 0.3167 | 0.3685 | 0.3238 | 0.3447 |

**Takeaway:** Amount family carries the model. `merchant_prior_fraud_rate` is the main precision drag despite looking promising in raw histograms. Velocity neutral-to-negative. Warranted full-data confirmation.

### Run 4: Full-data amount-only confirmation

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4262 | 0.8937 | — | — | — |
| Test (t=0.05) | 0.3807 | 0.8945 | 0.4996 | 0.4869 | 0.4931 |

**Takeaway:** Sampled result held on full data. Amount-only materially outperformed the five-feature baseline. Became the new baseline.

### Run 5: Amount-only Top-K read

| Top-K | Precision | Recall |
|---:|---:|---:|
| 100 | 0.0000 | 0.0000 |
| 500 | 0.2880 | 0.0100 |
| 1,000 | 0.4710 | 0.0326 |
| 5,000 | 0.5976 | 0.2069 |
| 10,000 | 0.4982 | 0.3449 |

**Takeaway:** Strong for larger budgets (5k–10k range), but the very top of the queue is empty — logistic ties all pile up. Motivated a tree-based comparison.

### Run 6: GBT on the same amount-only features

**Model:** `GBTClassifier`, same two features.

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.3911 | 0.9580 | — | — | — |
| Test (t=0.10) | 0.3307 | 0.9567 | 0.3029 | 0.7180 | 0.4260 |

Top-K: top-100 `0.7700`, top-500 `0.3880`, top-5000 `0.2640`, top-10000 `0.3141`.

**Takeaway:** GBT is much sharper at the very top (top-100 precision 0.77) but loses queue quality quickly. Logistic still stronger on PR AUC and larger budgets. Different model families prioritize different transaction profiles.

### Run 7: Alert inspection — logistic vs GBT

Inspected top-100 and top-5000 test alerts from both models.

**Logistic top-100:** all false positives, dominated by `travel` / `personal_care` / `kids_pets` with extreme amounts (avg amount ~10,919, avg `prior_amount_zscore` ~80, avg `amount_sum_last_1h` ~12,945). By top-5000: 2,988 TP at 0.5976 precision.

**GBT top-100:** 77 TP / 23 FP. FP still travel-dominated but far less extreme (avg amount ~373, avg zscore ~1.75). By top-5000: only 1,320 TP at 0.2640 precision.

**Takeaway:** Logistic over-ranks ultra-extreme legitimate spend bursts. GBT finds a fraud-rich pocket at the top but spreads too thin. Next useful step: find a feature that helps logistic stop over-ranking giant legitimate spends.

### Run 8: Third feature — `tx_count_last_1h`

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `tx_count_last_1h`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4223 | 0.8958 | — | — | — |
| Test (t=0.05) | 0.3785 | 0.8965 | 0.4949 | 0.4761 | 0.4853 |

Top-K: top-100 `0.0100`, top-5000 `0.6020`, top-10000 `0.4943`.

**vs amount-only:** Slightly better top-of-queue (+0.01 at top-100, +0.004 at top-5000) but slightly worse on all main metrics (PR AUC -0.002, F1 -0.008). Does not beat the baseline cleanly.

### Run 9: Third feature — `is_night_transaction`

**Motivation:** Alert inspection showed top-5000 false positives averaged `hour_of_day` ~6.5 with `is_night_transaction` ~0.42, while true frauds averaged hour ~11.6 with night flag ~0.006.

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4659 | 0.9403 | — | — | — |
| Test (t=0.05) | 0.4031 | 0.9241 | 0.5191 | 0.5033 | 0.5111 |

Top-K: top-100 `0.0000`, top-500 `0.3100`, top-1000 `0.4900`, top-5000 `0.6200`, top-10000 `0.5226`.

**vs amount-only:** PR AUC +0.022, precision +0.020, recall +0.016, F1 +0.018, top-5000 +0.022, top-10000 +0.024. Clean win on all main metrics. Top-100 still at zero.

**Takeaway:** Confirms the alert-inspection hypothesis — many giant-spend false positives were night/early-morning legitimate bursts. New best general baseline.

### Run 13: Logistic — gated (card × category) amount z-score on top of amount + night

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`, `prior_amount_zscore_card_category`, `prior_category_zscore_eligible`

**Gold / gating:** category z is null unless `prior_tx_count_card_category >= K` and `prior_amount_std_card_category > 0`; eval fills null z with `0.0`.

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4230 | 0.9395 | — | — | — |
| Test (best val threshold) | 0.3413 | 0.9139 | 0.5005 | 0.4225 | 0.4582 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.0100 | 1.0E-4 | 1 |
| 500 | 0.3880 | 0.0134 | 194 |
| 1,000 | 0.5510 | 0.0382 | 551 |
| 5,000 | 0.6446 | 0.2232 | 3223 |
| 10,000 | 0.5286 | 0.3660 | 5286 |

**Takeaway:** Worse PR/ROC and F1 than Run 9, but improved top-5k/10k precision.

### Run 14: Logistic — damped category z + `log1p(n)` reliability (V2) on top of amount + night

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`, `prior_amount_zscore_card_category_damped`, `prior_category_zscore_eligible`, `prior_category_log_prior_n`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4006 | 0.9288 | — | — | — |
| Test (best val threshold) | 0.3170 | 0.8933 | 0.4708 | 0.4029 | 0.4343 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.0100 | 1.0E-4 | 1 |
| 500 | 0.3820 | 0.0132 | 191 |
| 1,000 | 0.5480 | 0.0379 | 548 |
| 5,000 | 0.6394 | 0.2214 | 3197 |
| 10,000 | 0.5326 | 0.3688 | 5326 |

**Takeaway:** Slight top-5k/10k precision bump vs Run 9, but weaker overall PR/ROC and F1.

### Category-z gating `K` sweep (V1 raw vs V2 damped)

Fast Top-K-only sweep over gating `K ∈ {2,3,5,8,10}` on a sampled split (`train/val/test = 0.5/0.5/0.5`) showed:

| Variant | Best `K` (sampled) | Precision@5k | Precision@10k |
|---|---:|---:|---:|
| V1 raw gated (`Run 13` behavior) | 8 | 0.5334 | 0.4089 |
| V2 damped (`Run 14` behavior) | 5 | 0.5346 | 0.3216 |

Full evaluation of the top candidates `K ∈ {5,8}` on the full split (`train/val/test = 1.0/1.0/1.0`) confirmed `K=5` is best for both:

| Variant | `K` | Precision@5k | Precision@10k |
|---|---:|---:|---:|
| V1 raw gated | 5 | 0.6446 | 0.5286 |
| V1 raw gated | 8 | 0.6352 | 0.5313 |
| V2 damped | 5 | 0.6394 | 0.5326 |
| V2 damped | 8 | 0.6272 | 0.5218 |

**Takeaway:** For alert-queue purity, prefer `K=5` (card×category eligibility) over the tested alternatives.

### Run 15: Logistic — V3 shrunk category z on top of amount + night

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`, `prior_amount_zscore_card_category_shrunk`, `prior_category_zscore_eligible`, `prior_category_log_prior_n`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4013 | 0.9287 | — | — | — |
| Test (best val threshold) | 0.3177 | 0.8932 | 0.4717 | 0.4028 | 0.4345 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.0100 | 1.0E-4 | 1 |
| 500 | 0.3840 | 0.0133 | 192 |
| 1,000 | 0.5480 | 0.0379 | 548 |
| 5,000 | 0.6432 | 0.2227 | 3216 |
| 10,000 | 0.5336 | 0.3695 | 5336 |

**Takeaway:** PR/ROC and F1 essentially unchanged vs Run 14, with a small Top-5000 precision bump.

### Run 16: GBT — V3 shrunk category z on top of amount + night

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`, `prior_amount_zscore_card_category_shrunk`, `prior_category_zscore_eligible`, `prior_category_log_prior_n`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.6017 | 0.9680 | — | — | — |
| Test (best val threshold) | 0.4673 | 0.9575 | 0.4089 | 0.6882 | 0.5130 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.9900 | 0.0069 | 99 |
| 500 | 0.8340 | 0.0289 | 417 |
| 1,000 | 0.8350 | 0.0578 | 835 |
| 5,000 | 0.6076 | 0.2104 | 3038 |
| 10,000 | 0.5216 | 0.3612 | 5216 |

**Takeaway:** Much higher recall/F1 than LR, but lower Top-5000 precision than the LR variants.

### Run 17: Logistic — V3 damped+shrunk category z on top of amount + night

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`, `prior_amount_zscore_card_category_damped`, `prior_amount_zscore_card_category_shrunk`, `prior_category_zscore_eligible`, `prior_category_log_prior_n`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.4011 | 0.9286 | — | — | — |
| Test (best val threshold) | 0.3178 | 0.8929 | 0.4724 | 0.4023 | 0.4345 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.0100 | 1.0E-4 | 1 |
| 500 | 0.3880 | 0.0134 | 194 |
| 1,000 | 0.5530 | 0.0383 | 553 |
| 5,000 | 0.6450 | 0.2233 | 3225 |
| 10,000 | 0.5344 | 0.3701 | 5344 |

**Takeaway:** Best LR Top-5000 precision in the V3 set (while keeping PR/ROC and F1 basically flat vs Run 14).

### Run 18: GBT — V3 damped+shrunk category z on top of amount + night

**Features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`, `prior_amount_zscore_card_category_damped`, `prior_amount_zscore_card_category_shrunk`, `prior_category_zscore_eligible`, `prior_category_log_prior_n`

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.6013 | 0.9710 | — | — | — |
| Test (best val threshold) | 0.4665 | 0.9602 | 0.4082 | 0.6901 | 0.5130 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.9900 | 0.0069 | 99 |
| 500 | 0.8300 | 0.0287 | 415 |
| 1,000 | 0.8350 | 0.0578 | 835 |
| 5,000 | 0.6070 | 0.2102 | 3035 |
| 10,000 | 0.5204 | 0.3604 | 5204 |

**Takeaway:** Similar recall/F1 to Run 16, with Top-5000 precision slightly below LR.

### Run 19: Logistic — V3 damped+shrunk with inverse-frequency class weights

**Features:** same as Run 17, plus `--logistic-class-weights`.

| Set | PR AUC | ROC AUC | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|
| Val | 0.3946 | 0.9400 | — | — | — |
| Test (best val threshold) | 0.3014 | 0.9111 | 0.4237 | 0.5783 | 0.4891 |

Top-K (test):
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.0000 | 0.0000 | 0 |
| 500 | 0.0020 | 1.0E-4 | 1 |
| 1,000 | 0.0650 | 0.0045 | 65 |
| 5,000 | 0.4464 | 0.1546 | 2232 |
| 10,000 | 0.5024 | 0.3479 | 5024 |

**Takeaway:** Class weighting significantly hurts both main PR AUC/F1 and queue purity (Top-5000 precision).

### Run 20: V4 (tie-break) — Logistic + GBT on derived amount transforms + interaction

**Feature set:** `amount_plus_night_catz_v4_interact_clip`  
**Top-K tie-break:** `--topk-secondary-signal neg_prior_category_log_prior_n`

**Features:** `prior_amount_zscore_clipped`, `amount_sum_last_1h_log1p`, `is_night_transaction`, `prior_amount_zscore_card_category_damped`, `prior_amount_zscore_card_category_shrunk`, `prior_category_zscore_eligible`, `prior_category_log_prior_n`, `amount_zscore_x_lowcat`

| Model | PR AUC (Val) | ROC AUC (Val) | PR AUC (Test) | ROC AUC (Test) | Precision (Test) | Recall (Test) | F1 (Test) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Logistic | 0.3873 | 0.9167 | 0.2854 | 0.8952 | 0.3395 | 0.5029 | 0.4053 |
| GBT | 0.6023 | 0.9708 | 0.4684 | 0.9598 | 0.4122 | 0.6890 | 0.5158 |

Top-K (test) — Logistic:
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.2500 | 0.0017 | 25 |
| 500 | 0.4220 | 0.0146 | 211 |
| 1,000 | 0.3730 | 0.0258 | 373 |
| 5,000 | 0.4660 | 0.1613 | 2330 |
| 10,000 | 0.4951 | 0.3428 | 4951 |

Top-K (test) — GBT:
| k | precision | recall | tp |
|---|---:|---:|---:|
| 100 | 0.9900 | 0.0069 | 99 |
| 500 | 0.9080 | 0.0314 | 454 |
| 1,000 | 0.8350 | 0.0578 | 835 |
| 5,000 | 0.6056 | 0.2097 | 3028 |
| 10,000 | 0.5248 | 0.3634 | 5248 |

**Pass criteria outcome:** Rejected.  
GBT improves PR AUC/F1 and keeps strong top-100, but does not meet the baseline floor on queue purity (`precision@5000 >= 0.62`). Logistic regresses sharply on `precision@5000`.

## Conclusions

- **Baseline features:** `prior_amount_zscore`, `amount_sum_last_1h`, `is_night_transaction`
- **Category z variants:** Run 13-17 improve top-5k/10k precision but reduce main PR/ROC and F1 vs Run 9; V3 mainly shifts queue purity (best LR Top-5000: Run 17 @0.645).
- **Gating threshold:** category-z eligibility `K=5` is optimal (confirmed by Top-K-only sweep + full recheck).
- **Class-weighted logistic:** Run 19 significantly hurts both queue purity (Top-5000 precision) and main PR AUC/F1; keep unweighted.
- Velocity: not yet earning its keep in logistic
- Merchant fraud rate: keep in Gold, not in baseline
- Logistic: best general baseline; GBT: sharper at very small queues
- Future experiments compare against the amount + night baseline, not the older five-feature set

## Recommended Next Step

1. Inspect top-5k false positives for Run 17 by `merchant_category` to see what remains (and why Top-100 precision stays near-zero)
2. Add a tie-break / reranking strategy so the very top of the queue is not all score-tied
3. Keep the V3 unweighted setup; avoid inverse-frequency class weights (they tank queue purity)

### Commands (scripts)

If Spark workers pick a different Python than the project venv (driver), set both to the same interpreter, for example:

`export PYSPARK_PYTHON="$PWD/.venv/bin/python" PYSPARK_DRIVER_PYTHON="$PWD/.venv/bin/python"`

```bash
# 1) Three-feature baseline alert inspection (default --feature-set amount_plus_night)
python scripts/inspect_sparkov_alerts.py --feature-set amount_plus_night --model-type logistic

# 2) GBT vs same three features on one run
python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night --model-type both

# 3) LR: night baseline vs night + hour_of_day
python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night --model-type logistic
python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night_hour --model-type logistic

# 4) LR: gated card×category amount z-score (Run 13)
python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night_catz --model-type logistic

# 5) LR: damped category z + log1p(n) reliability (Run 14)
python scripts/evaluate_sparkov_model.py --feature-set amount_plus_night_catz_v2 --model-type logistic
```

Use `--train-fraction` / `--validation-fraction` / `--test-fraction` for faster smoke tests.
