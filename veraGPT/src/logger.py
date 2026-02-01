"""
logger.py
─────────────────────────────────────────────────────────────────────
Centralized logging setup.  Every module pulls its logger from here:
    from logger import get_logger
    logger = get_logger(__name__)
─────────────────────────────────────────────────────────────────────
"""

import logging
import sys
from typing import Optional

from config import LoggingConfig

# ── Global flag so we only configure the root logger once ───────
_configured: bool = False


def _build_formatter() -> logging.Formatter:
    """Return a human-readable formatter with timestamp and module name."""
    fmt = "[%(asctime)s] %(levelname)-8s %(name)s: %(message)s"
    return logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")


def setup_logging(cfg: Optional[LoggingConfig] = None) -> None:
    """
    Configure the root logger once.
    Call this early in main(); subsequent calls are no-ops.
    """
    global _configured
    if _configured:
        return

    if cfg is None:
        cfg = LoggingConfig()

    level = getattr(logging, cfg.log_level, logging.INFO)
    root = logging.getLogger()
    root.setLevel(level)

    formatter = _build_formatter()

    # ── Console handler (stdout) ─────────────────────────────────
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    # ── File handler (optional) ──────────────────────────────────
    if cfg.log_file:
        file_handler = logging.FileHandler(cfg.log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a named logger (lazily sets up root if needed)."""
    if not _configured:
        setup_logging()
    return logging.getLogger(name)
