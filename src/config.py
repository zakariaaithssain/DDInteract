"""Centralized configuration, paths, and logger for the DDI pipeline."""

import logging
import sys
from pathlib import Path

# --- Directory roots ---
MODELS_DIR = "models"
LOGS_DIR = "logs"
DATA_DIR = "data"

# --- Data files ---
RAW_DATA_PATH = f"{DATA_DIR}/raw_ddi.csv"
DATA_PATH = f"{DATA_DIR}/chemical_ddi.csv"
FEATURE_CACHE = f"{DATA_DIR}/features.npy"
LABEL_CACHE = f"{DATA_DIR}/labels.npy"

# --- Model / preprocessing artifacts ---
MODEL_PATH = f"{MODELS_DIR}/model.joblib"
BEST_MODEL_PATH = f"{MODELS_DIR}/best_model.joblib"
BEST_PARAMS_PATH = f"{MODELS_DIR}/best_params.json"

# --- Drift ---
DRIFT_REFERENCE_PATH = f"{MODELS_DIR}/drift_reference.json"
DRIFT_REPORT_PATH = "drift_report.json"

# --- Logging ---
LOG_PATH = f"{LOGS_DIR}/pipeline.log"

# --- Results ---
RESULTS_PATH = "results.json"

# --- Training metadata ---
EXPERIMENT_NAME = "DDI_Structural_Severity"
TEST_SIZE = 0.2
CLASS_NAMES = ["Minor", "Moderate", "Major"]
REGISTRY_NAME = "DDI-Severity"

# --- Logger singleton ---
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
