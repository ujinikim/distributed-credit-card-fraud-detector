#!/usr/bin/env python3
"""Download the Sparkov benchmark dataset to the local benchmark data directory."""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.request import urlopen


# Add src so we can import fraud_lens
project_root = Path(__file__).resolve().parents[1]
src = project_root / "src"
if str(src) not in sys.path:
    sys.path.insert(0, str(src))


def main() -> None:
    """Download the configured Sparkov dataset file if it is not already present."""
    from fraud_lens.ingest import load_sparkov_config

    config = load_sparkov_config().get("sparkov", {})
    url = str(config["download_url"])
    output_path = (project_root / config["download_path"]).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists() and output_path.stat().st_size > 0:
        print(f"Sparkov dataset already present: {output_path}")
        return

    print(f"Downloading Sparkov dataset from {url}")
    with urlopen(url) as response, open(output_path, "wb") as f:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            f.write(chunk)

    print(f"Downloaded Sparkov dataset to: {output_path}")


if __name__ == "__main__":
    main()
