"""Centralized configuration for paths and training constants."""

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
SCALER_PATH = f"{MODELS_DIR}/scaler.joblib"
PCA_PATH = f"{MODELS_DIR}/pca.joblib"
MODEL_PATH = f"{MODELS_DIR}/model.joblib"
BEST_MODEL_PATH = f"{MODELS_DIR}/best_model.joblib"

# --- Drift ---
DRIFT_REFERENCE_PATH = f"{MODELS_DIR}/drift_reference.json"
DRIFT_REPORT_PATH = "drift_report.json"

# --- Logging ---
LOG_PATH = f"{LOGS_DIR}/pipeline.log"

# --- Results ---
RESULTS_PATH = "results.json"

# --- Training hyperparameters ---
EXPERIMENT_NAME = "DDI_Structural_Severity"
TEST_SIZE = 0.2
N_PCA = 50
CLASS_NAMES = ["Minor", "Moderate", "Major"]
REGISTRY_NAME = "DDI-Severity"
