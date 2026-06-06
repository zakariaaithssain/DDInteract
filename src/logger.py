"""Logging configuration for the DDI pipeline.

Provides a pre-configured ``logger`` singleton with colored console
output (INFO-level) and daily rotating file handler (DEBUG-level).
"""

import logging
import sys
from pathlib import Path

from src.config import LOG_PATH, LOGS_DIR

Path(LOGS_DIR).mkdir(exist_ok=True)

logger: logging.Logger = logging.getLogger("ddi")
logger.setLevel(logging.DEBUG)

fmt: logging.Formatter = logging.Formatter(
    "%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s",
    datefmt="%H:%M:%S",
)

sh: logging.StreamHandler = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(fmt)
logger.addHandler(sh)

fh: logging.FileHandler = logging.FileHandler(LOG_PATH)
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

__all__: list[str] = ["logger"]
