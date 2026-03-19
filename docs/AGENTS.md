# FraudLens Contributor Guide

This is a short orientation doc for working in the repo.

Use the other docs as the main sources of truth:

- [../README.md](../README.md) for project direction and run flows
- [medallion_layers.md](medallion_layers.md) for Bronze, Silver, and Gold responsibilities
- [sparkov_benchmark_plan.md](sparkov_benchmark_plan.md) for Sparkov mapping and benchmark usage
- [sparkov_feature_roadmap.md](sparkov_feature_roadmap.md) for next Gold feature work
- [synthetic_data_spec.md](synthetic_data_spec.md) for synthetic fixture behavior

## Current Project Focus

- Sparkov is the default analysis dataset.
- Synthetic data is a controlled fixture for regression, smoke tests, and explicit anomaly cases.
- Feature engineering belongs in Gold after shared Bronze and Silver processing.

## Working Areas

### Ingest and Bronze

- Keep the canonical raw transaction contract stable.
- Allow source-specific normalization upstream of Bronze when needed.
- Limit Bronze to ingestion and lineage metadata.

### Bronze to Silver

- Keep Silver clean, typed, and trustworthy.
- Preserve only the nullable passthrough columns needed for Gold features.
- Avoid adding feature logic here.

### Silver to Gold

- Treat Gold as the main feature-extraction layer.
- Prefer interpretable window and aggregation features first.
- Keep the shared baseline Gold path intact, then add Sparkov-specific extensions on top.

### Synthetic Data

- Keep synthetic data small and intentional.
- Use it for regression checks and explicit anomaly validation, not as the main feature-quality benchmark.

### ML and LLM

- These come after the Gold feature layer is stable.
- Evaluate models on time-based splits rather than random transaction splits.

## Repo Conventions

- Use the PySpark DataFrame API, not RDD-based logic.
- Prefer shared layer contracts over source-specific forks.
- Add new fields intentionally and document them in [medallion_layers.md](medallion_layers.md) when they affect Silver or Gold.
