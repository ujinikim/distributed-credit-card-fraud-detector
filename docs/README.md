# FraudLens documentation

## Layout

| Folder | Contents |
| --- | --- |
| [architecture/](architecture/) | Medallion contracts, data sanity |
| [sparkov/](sparkov/) | Benchmark plan, feature roadmap, current model metrics, performance benchmarks |
| [reference/](reference/) | Synthetic data spec |
| [archive/](archive/) | Historical experiment write-ups (full run log, feature findings) |

Code layout:

- [`../src/fraud_lens/`](../src/fraud_lens/) — reusable package code
- [`../scripts/`](../scripts/) — runnable CLI entry points

## Reading order

1. [architecture/medallion_layers.md](architecture/medallion_layers.md) — Bronze, Silver, Gold responsibilities and Silver contract
2. [sparkov/sparkov_benchmark_plan.md](sparkov/sparkov_benchmark_plan.md) — Sparkov dataset role and source mapping
3. [archive/sparkov_feature_findings.md](archive/sparkov_feature_findings.md) — What worked in feature experiments (historical)
4. [sparkov/model_eval_latest.md](sparkov/model_eval_latest.md) — Current model recommendation and metrics
5. [archive/sparkov_model_eval.md](archive/sparkov_model_eval.md) — Full experiment history (archive)
6. [sparkov/spark_performance_report.md](sparkov/spark_performance_report.md) — Throughput and Spark justification benchmarks

Supporting references:

- [reference/synthetic_data_spec.md](reference/synthetic_data_spec.md) — Synthetic fixture behavior
- [architecture/data_sanity_rules.md](architecture/data_sanity_rules.md) — Data quality expectations
- [AGENTS.md](AGENTS.md) — Contributor conventions
- [PROJECT_GUIDE.md](PROJECT_GUIDE.md) — Command cheat sheet and key file paths

## By task

| Goal | Doc |
| --- | --- |
| Understand layers | [architecture/medallion_layers.md](architecture/medallion_layers.md) |
| Run Sparkov pipeline | [../README.md](../README.md) + [PROJECT_GUIDE.md](PROJECT_GUIDE.md) |
| Pick a model / metrics | [sparkov/model_eval_latest.md](sparkov/model_eval_latest.md) |
| Spark performance story | [sparkov/spark_performance_report.md](sparkov/spark_performance_report.md) |
| Contribute code | [AGENTS.md](AGENTS.md) |
