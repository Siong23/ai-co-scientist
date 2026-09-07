"""Bounded, redacted runtime logs for diagnosing local research sessions."""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

from .utils import redact_secrets


class RedactingFormatter(logging.Formatter):
    def format(self, record):
        # Redact after formatting so arguments and exception tracebacks are covered.
        return redact_secrets(super().format(record))


def configure_runtime_logging(directory: str | Path = "results") -> Path:
    path = Path(directory).resolve() / f"runtime-{os.getpid()}.log"
    root = logging.getLogger()
    if any(getattr(handler, "baseFilename", None) == str(path) for handler in root.handlers):
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(path, maxBytes=5_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(RedactingFormatter("%(asctime)s %(levelname)s %(threadName)s %(name)s: %(message)s"))
    root.addHandler(handler)
    return path
