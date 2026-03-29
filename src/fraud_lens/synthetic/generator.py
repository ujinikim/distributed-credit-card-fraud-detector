"""Compatibility wrapper for the synthetic fixture module."""

from fraud_lens.synthetic.fixture import (
    GeneratorConfig,
    Transaction,
    generate,
    load_config,
    run,
    write_jsonl,
)

__all__ = [
    "GeneratorConfig",
    "Transaction",
    "generate",
    "load_config",
    "run",
    "write_jsonl",
]
