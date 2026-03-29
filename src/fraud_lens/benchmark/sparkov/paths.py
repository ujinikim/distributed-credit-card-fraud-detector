"""Resolved Sparkov benchmark paths."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[4]


def resolve_sparkov_paths(
    config: dict[str, Any] | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Path]:
    """Resolve frequently used Sparkov paths from config into absolute paths."""
    root = Path(project_root).resolve() if project_root is not None else _project_root()
    sparkov_cfg = (config or {}).get("sparkov", config or {})
    return {
        "download_path": (root / sparkov_cfg.get("download_path", "data/benchmark/sparkov/data.csv")).resolve(),
        "input_path": (root / sparkov_cfg.get("input_path", "data/benchmark/sparkov/data.csv")).resolve(),
        "normalized_raw_path": (root / sparkov_cfg.get("normalized_raw_path", "data/raw_sparkov")).resolve(),
        "bronze_path": (root / sparkov_cfg.get("bronze_path", "data/benchmark/bronze_sparkov")).resolve(),
        "silver_path": (root / sparkov_cfg.get("silver_path", "data/benchmark/silver_sparkov")).resolve(),
        "gold_path": (root / sparkov_cfg.get("gold_path", "data/benchmark/gold_sparkov")).resolve(),
    }
