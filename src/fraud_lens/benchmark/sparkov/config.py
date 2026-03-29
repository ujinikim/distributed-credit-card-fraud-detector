"""Config helpers for the Sparkov benchmark workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def _project_root() -> Path:
    """Project root (repo root): parent of src/."""
    return Path(__file__).resolve().parents[4]


def load_sparkov_config(config_path: str | Path | None = None) -> dict[str, Any]:
    """Load Sparkov benchmark config from ``config/sparkov.yaml``."""
    root = _project_root()
    if config_path is None:
        config_path = root / "config" / "sparkov.yaml"
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Sparkov config not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
