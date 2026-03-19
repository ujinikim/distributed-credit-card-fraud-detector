"""Bronze ingest: read raw JSONL, add metadata, write Parquet."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml
from pyspark.sql.functions import current_timestamp, input_file_name

if TYPE_CHECKING:
    from pyspark.sql import SparkSession
    from pyspark.sql import DataFrame


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[3]


def load_paths_config(paths_yaml: str | Path | None = None) -> dict[str, Any]:
    """Load data and ingest paths from config/paths.yaml."""
    root = _project_root()
    if paths_yaml is None:
        paths_yaml = root / "config" / "paths.yaml"
    path = Path(paths_yaml)
    if not path.exists():
        raise FileNotFoundError(f"Paths config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run(
    spark: "SparkSession",
    raw_path: str | Path | None = None,
    bronze_path: str | Path | None = None,
    paths_yaml: str | Path | None = None,
    file_pattern: str | None = None,
) -> "DataFrame":
    """
    Read raw JSONL from raw_path, add ingestion_timestamp and source_path, write Parquet to bronze_path.
    Paths default from config/paths.yaml if not provided.
    Returns the Bronze DataFrame (before write) for tests or chaining.
    """
    config = load_paths_config(paths_yaml)
    data = config.get("data", {})
    ingest_cfg = config.get("ingest", {})

    if raw_path is None:
        raw_path = data.get("raw", "data/raw")
    raw_path = str(Path(raw_path).resolve())
    if file_pattern is None:
        file_pattern = ingest_cfg.get("file_pattern", "*.jsonl")
    read_path = str(Path(raw_path) / file_pattern) if file_pattern else raw_path

    if bronze_path is None:
        bronze_path = data.get("bronze", "data/bronze")
    bronze_path = str(Path(bronze_path).resolve())

    df = spark.read.json(read_path)
    df = df.withColumn("ingestion_timestamp", current_timestamp()).withColumn(
        "source_path", input_file_name()
    )

    df.write.mode("overwrite").parquet(bronze_path)
    return df
