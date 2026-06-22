# -*- coding: utf-8 -*-
"""Unified logging configuration for InsureRAG."""

from __future__ import annotations

import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def configure_logging(
    log_level: str | None = None,
    log_dir: str | Path = "logs",
    log_file: str = "insurerag.log",
) -> None:
    """Configure console and file logging once for the whole process."""
    level_name = (log_level or os.getenv("INSURERAG_LOG_LEVEL") or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger()
    if getattr(root, "_insurerag_logging_configured", False):
        root.setLevel(level)
        return

    project_root = Path(__file__).resolve().parents[1]
    output_dir = Path(log_dir)
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(level)

    # rotate to avoid unbounded log growth: 5 MB per file, keep 3 backups.
    file_handler = RotatingFileHandler(
        output_dir / log_file, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    root.handlers.clear()
    root.setLevel(level)
    root.addHandler(console_handler)
    root.addHandler(file_handler)
    root._insurerag_logging_configured = True


def get_logger(name: str) -> logging.Logger:
    configure_logging()
    return logging.getLogger(name)
