import logging
import sys
from pathlib import Path

Path("logs").mkdir(exist_ok=True)

logger = logging.getLogger("ddi")
logger.setLevel(logging.DEBUG)

fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(lineno)d - %(message)s", datefmt="%H:%M:%S")

sh = logging.StreamHandler(sys.stdout)
sh.setLevel(logging.INFO)
sh.setFormatter(fmt)
logger.addHandler(sh)

fh = logging.FileHandler("logs/pipeline.log")
fh.setLevel(logging.DEBUG)
fh.setFormatter(fmt)
logger.addHandler(fh)

__all__ = ["logger"]
