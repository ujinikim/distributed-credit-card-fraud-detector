"""Synthetic fixture tests."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from fraud_lens.synthetic import GeneratorConfig, generate, run, write_jsonl


class SyntheticFixtureTests(unittest.TestCase):
    def test_generate_returns_records(self) -> None:
        config = GeneratorConfig(
            num_transactions=12,
            start_date="2013-09-01T00:00:00",
            end_date="2013-09-01T23:59:59",
            seed=42,
            anomaly_ratio=0.2,
            num_cards=4,
            amount_mean=75.0,
            amount_std=15.0,
            output_path="unused",
            merchant_categories=[{"name": "retail", "weight": 1.0}],
            geo_center_lat=40.0,
            geo_center_lon=-74.0,
            geo_std=0.1,
            min_minutes_between_locations=120,
            min_distance_km_impossible_travel=500.0,
            spike_amount_multiplier=5.0,
            impossible_travel_fraction=0.5,
            raw_write_mode="overwrite",
            normal_min_minutes_between_transactions=30,
        )

        transactions = generate(config, run_id="test_run")

        self.assertTrue(transactions)
        self.assertTrue(all(tx.transaction_id for tx in transactions))
        self.assertTrue(all(tx.event_time for tx in transactions))

    def test_write_jsonl_and_run_produce_fixture_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config_path = tmp_path / "synthetic.yaml"
            paths_path = tmp_path / "paths.yaml"

            config_path.write_text(
                "\n".join(
                    [
                        "num_transactions: 4",
                        'start_date: "2013-09-01T00:00:00"',
                        'end_date: "2013-09-01T23:59:59"',
                        "seed: 7",
                        "anomaly_ratio: 0.0",
                        "num_cards: 2",
                        "amount_mean: 50.0",
                        "amount_std: 5.0",
                        'output_path: "ignored"',
                        'raw_write_mode: "overwrite"',
                        "merchant_categories:",
                        '  - { name: "retail", weight: 1.0 }',
                        "geo_center_lat: 40.0",
                        "geo_center_lon: -74.0",
                        "geo_std: 0.1",
                        "min_minutes_between_locations: 120",
                        "min_distance_km_impossible_travel: 500.0",
                        "spike_amount_multiplier: 5.0",
                        "impossible_travel_fraction: 0.0",
                        "normal_min_minutes_between_transactions: 30",
                    ]
                ),
                encoding="utf-8",
            )
            paths_path.write_text(
                'data:\n  raw: "' + str(tmp_path / "raw_output") + '"\n',
                encoding="utf-8",
            )

            output_path = run(config_path=config_path, paths_yaml=paths_path, run_id="fixture")

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.name, "transactions.jsonl")
            self.assertGreater(output_path.stat().st_size, 0)

            lines = output_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 4)


if __name__ == "__main__":
    unittest.main()
