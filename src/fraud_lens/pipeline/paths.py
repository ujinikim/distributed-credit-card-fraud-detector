"""Path helpers for the canonical medallion pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[3]


def load_paths_config(paths_yaml: str | Path | None = None) -> dict[str, Any]:
    """Load data and ingest paths from ``config/paths.yaml``."""
    root = _project_root()
    if paths_yaml is None:
        paths_yaml = root / "config" / "paths.yaml"
    path = Path(paths_yaml)
    if not path.exists():
        raise FileNotFoundError(f"Paths config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
