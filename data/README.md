# Data directory

Medallion layers:

- **bronze/** — Raw ingested data (e.g. JSON → Parquet).
- **silver/** — Cleaned, typed data.
- **gold/** — Feature-engineered data for ML.

Generated and processed data live here. Large files are gitignored; add sample outputs if needed for tests.
