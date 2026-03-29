"""Compatibility import tests for the FraudLens package layout."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT / "src", ROOT / "scripts"):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)


class ImportCompatibilityTests(unittest.TestCase):
    def test_new_pipeline_imports_exist(self) -> None:
        from fraud_lens.pipeline import (
            load_paths_config,
            run_bronze_ingest,
            run_gold_features,
            run_silver_transform,
        )

        self.assertTrue(callable(load_paths_config))
        self.assertTrue(callable(run_bronze_ingest))
        self.assertTrue(callable(run_silver_transform))
        self.assertTrue(callable(run_gold_features))

    def test_legacy_pipeline_imports_still_resolve(self) -> None:
        from fraud_lens.bronze_to_silver import run as run_b2s
        from fraud_lens.ingest import run as run_bronze
        from fraud_lens.silver_to_gold import run as run_s2g

        self.assertTrue(callable(run_bronze))
        self.assertTrue(callable(run_b2s))
        self.assertTrue(callable(run_s2g))

    def test_new_sparkov_namespace_exists(self) -> None:
        from fraud_lens.benchmark.sparkov import (
            load_sparkov_config,
            normalize_sparkov_raw,
            resolve_sparkov_paths,
        )

        self.assertTrue(callable(load_sparkov_config))
        self.assertTrue(callable(normalize_sparkov_raw))
        self.assertTrue(callable(resolve_sparkov_paths))

    def test_legacy_sparkov_eval_wrapper_still_exports(self) -> None:
        from sparkov_eval.constants import FEATURE_SETS
        from sparkov_eval.metrics import evaluate_feature_set

        self.assertIn("amount_plus_night", FEATURE_SETS)
        self.assertTrue(callable(evaluate_feature_set))


if __name__ == "__main__":
    unittest.main()
