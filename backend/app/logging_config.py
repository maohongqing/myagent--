from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]


def configure_logging() -> None:
    log_dir = BACKEND_DIR / "data" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("competitor_agent")
    logger.setLevel(logging.INFO)
    if any(isinstance(handler, RotatingFileHandler) for handler in logger.handlers):
        return

    handler = RotatingFileHandler(
        log_dir / "app.log",
        maxBytes=2_000_000,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s")
    )
    logger.addHandler(handler)
