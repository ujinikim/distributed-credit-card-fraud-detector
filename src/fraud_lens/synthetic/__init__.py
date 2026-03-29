"""Synthetic fixture support for smoke tests and lightweight validation."""

from fraud_lens.synthetic.fixture import (
    GeneratorConfig,
    Transaction,
    generate,
    load_config,
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
