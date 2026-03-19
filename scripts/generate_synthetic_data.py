#!/usr/bin/env python3
"""Generate synthetic transaction data (JSONL) for FraudLens Bronze ingest."""

import sys
from pathlib import Path

# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))

from fraud_lens.synthetic import run


def main() -> None:
    """Load config, generate data, write to data/raw/ (or config path)."""
    out_path = run()
    print(f"Generated synthetic transactions: {out_path}")


if __name__ == "__main__":
    main()
