"""Test package bootstrap for local, non-installed imports."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)
