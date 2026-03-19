"""Synthetic transaction data generator (Gaussian + optional anomalies)."""

from fraud_lens.synthetic.generator import (
    GeneratorConfig,
    Transaction,
    load_config,
    generate,
    write_jsonl,
    run,
)

__all__ = [
    "GeneratorConfig",
    "Transaction",
    "load_config",
    "generate",
    "write_jsonl",
    "run",
]
