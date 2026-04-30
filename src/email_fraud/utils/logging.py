"""Logging helpers — stdlib + wandb integration.

Call setup_logging() once at the top of each script.  All other modules use
the stdlib logging.getLogger(__name__) pattern; this module configures the
root handler so you don't need wandb imported everywhere.
"""

from __future__ import annotations

import logging
import sys
from typing import Any


def setup_logging(level: str = "INFO") -> None:
    """Configure root logger to stdout with a compact format."""
    numeric = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        stream=sys.stdout,
        level=numeric,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def log_config(config: Any, logger: logging.Logger | None = None) -> None:
    """Pretty-print an ExperimentConfig (or any Pydantic model) to the logger."""
    lg = logger or logging.getLogger(__name__)
    try:
        import json

        lg.info("Experiment config:\n%s", json.dumps(config.model_dump(), indent=2))
    except Exception:
        lg.info("Experiment config: %s", config)


def wandb_watch(model: Any, log_freq: int = 100) -> None:
    """Call wandb.watch() if wandb is initialised, otherwise no-op."""
    try:
        import wandb

        if wandb.run is not None:
            wandb.watch(model, log="gradients", log_freq=log_freq)
    except ImportError:
        pass
