# FraudLens contributor guide

Conventions for working in this repo. **Layer contracts and architecture** are in [architecture/medallion_layers.md](architecture/medallion_layers.md); **commands and file map** are in [PROJECT_GUIDE.md](PROJECT_GUIDE.md) and [README.md](../README.md).

## Doc map

- [README.md](../README.md) — setup and benchmark entrypoints
- [architecture/medallion_layers.md](architecture/medallion_layers.md) — Bronze, Silver, Gold responsibilities
- [sparkov/sparkov_benchmark_plan.md](sparkov/sparkov_benchmark_plan.md) — Sparkov mapping and benchmark usage
- [reference/synthetic_data_spec.md](reference/synthetic_data_spec.md) — synthetic fixture behavior

## Conventions

- Use the **PySpark DataFrame API**, not RDD-based logic.
- Prefer shared layer contracts over source-specific forks.
- Add new fields intentionally and document them in [architecture/medallion_layers.md](architecture/medallion_layers.md) when they affect Silver or Gold.
- Evaluate models on **time-based splits**, not random transaction splits.
