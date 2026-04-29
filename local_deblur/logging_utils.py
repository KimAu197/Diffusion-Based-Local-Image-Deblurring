"""Logging setup for scripts."""

from __future__ import annotations

import logging
from pathlib import Path


def configure_logging(
    name: str = "local_deblur",
    log_file: str | Path | None = None,
    level: int = logging.INFO,
    file_mode: str = "w",
) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(path, mode=file_mode)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
