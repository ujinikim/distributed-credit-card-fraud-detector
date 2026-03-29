"""Small end-to-end smoke test for the medallion pipeline."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from pyspark.sql import SparkSession

from fraud_lens.pipeline import run_bronze_ingest, run_gold_features, run_silver_transform


class PipelineSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spark = (
            SparkSession.builder.master("local[1]")
            .appName("FraudLens-Pipeline-Smoke")
            .config("spark.ui.enabled", "false")
            .getOrCreate()
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls.spark.stop()

    def test_local_raw_to_gold_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            raw_dir = root / "raw"
            bronze_dir = root / "bronze"
            silver_dir = root / "silver"
            gold_dir = root / "gold"
            raw_dir.mkdir()

            records = [
                {
                    "transaction_id": "tx_1",
                    "card_id": "card_1",
                    "event_time": "2013-09-01T01:00:00Z",
                    "amount": 10.0,
                    "merchant_category": "retail",
                    "latitude": 40.0,
                    "longitude": -74.0,
                    "anomaly_type": "none",
                },
                {
                    "transaction_id": "tx_2",
                    "card_id": "card_1",
                    "event_time": "2013-09-01T02:00:00Z",
                    "amount": 25.0,
                    "merchant_category": "retail",
                    "latitude": 40.1,
                    "longitude": -74.1,
                    "anomaly_type": "none",
                },
            ]
            raw_file = raw_dir / "transactions.jsonl"
            raw_file.write_text(
                "\n".join(json.dumps(record) for record in records) + "\n",
                encoding="utf-8",
            )

            bronze_df = run_bronze_ingest(
                self.spark,
                raw_path=raw_dir,
                bronze_path=bronze_dir,
            )
            silver_df = run_silver_transform(
                self.spark,
                bronze_path=bronze_dir,
                silver_path=silver_dir,
            )
            gold_df = run_gold_features(
                self.spark,
                silver_path=silver_dir,
                gold_path=gold_dir,
            )

            self.assertEqual(bronze_df.count(), 2)
            self.assertEqual(silver_df.count(), 2)
            self.assertEqual(gold_df.count(), 2)
            self.assertIn("amount_sum_last_1h", gold_df.columns)
            self.assertIn("prior_amount_zscore", gold_df.columns)


if __name__ == "__main__":
    unittest.main()
