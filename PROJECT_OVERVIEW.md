# DDI Severity Predictor — Project Overview

An end-to-end MLOps pipeline that predicts drug-drug interaction severity
(Major / Moderate / Minor) from chemical structure using Morgan fingerprints
and machine learning.

---

## Problem

Drug-drug interactions (DDIs) are a major clinical concern. Existing databases
list interactions but require a drug name lookup — they cannot generalize to
novel compounds. This project shows that chemical structure alone (SMILES →
molecular fingerprints) is sufficient to predict interaction severity,
enabling predictions for any molecule.

---

## Source Modules (`src/`)

| File | Purpose |
|------|---------|
| `logger.py` | Configures Python `logging` with coloured console output and daily rotating file handler to `logs/`. Single shared logger. |
| `features.py` | RDKit feature engineering. Converts SMILES pairs into a fixed-length 1045-dim feature matrix: 256-bit Morgan fingerprints (×4 operations), Tanimoto similarity, molecular descriptors (×2 diff/sum). |
| `models.py` | Model definitions and hyperparameter grids. 5 families (LR, SVC, RF, KNN, XGBoost) with 3 parameter sets each. |
| `train.py` | Main training pipeline. Loads data, builds/caches features, loops 15 model/param combos, logs metrics to MLflow, registers best model, exports scaler+PCA. |
| `export_model.py` | Queries MLflow for the highest macro-F1 run, rebuilds scaler+PCA from training data, saves model + scaler + PCA as joblib to `models/`. |
| `api.py` | FastAPI server: `GET /` (frontend), `GET /health`, `POST /predict`. Loads model/scaler/PCA on startup. |
| `fetch_smiles.py` | PubChem SMILES resolution. Looks up drug names via pubchempy (5 workers), falls back to alternate names, outputs `chemical_ddi.csv`. |
| `drift.py` | Data drift detection. Computes fingerprint density on training set, runs KS test at inference to detect distribution shift. |

## Config (`config/`)

| File | Purpose |
|------|---------|
| `config.yaml` | Root Hydra config composing features, training, and models sub-configs. |

## Tests (`tests/`) — 77 tests, 98% coverage

| File | What it covers |
|------|----------------|
| `test_api.py` | FastAPI health, predict, frontend routes |
| `test_chemistry.py` | Morgan fingerprint shape, values, known molecules |
| `test_drift.py` | KS test, threshold logic, edge cases |
| `test_export_model.py` | Cache/no-cache paths, model export, failed search |
| `test_features.py` | Feature matrix shape, edge cases (same drug, single row) |
| `test_fetch_smiles.py` | PubChem API mocking, caching, fallback, error handling |
| `test_logger.py` | Console + file handler config, rotation |
| `test_model.py` | Feature dimension assertion (1045 columns) |
| `test_models.py` | Model configs, param grids, RANDOM_STATE |
| `test_train.py` | Ordinal MAE, evaluate_and_log, load_or_build_features, main pipeline, register_best_model |

## Infrastructure

| File | Purpose |
|------|---------|
| `Makefile` | `install`, `train`, `test`, `lint`, `format`, `typecheck`, `export`, `api`, `clean`, `hooks` |
| `Dockerfile` | `python:3.14-slim` with uv, copies code + models, runs uvicorn |
| `.pre-commit-config.yaml` | ruff check (--fix), ruff format, mypy, pytest |
| `.github/workflows/ci-cd.yaml` | GitHub Actions: lint → format-check → mypy → test |
| `dvc.yaml` | DVC pipeline: train stage with deps and outputs |
| `pyproject.toml` | Metadata, deps, ruff/mypy/pytest config |
| `.gitignore` | Caches, venv, MLflow, models/, DVC tmp/cache |

## Key Design Decisions

**SMILES-only input.** Model receives only chemical structure. Drug names are
resolved at data-prep time, never at inference. Works for any molecule.

**Morgan fingerprints + molecular descriptors.** 256-bit fingerprints capture
substructure presence; descriptors capture bulk properties (MolWt, LogP,
TPSA, etc.). Together: 1045 features per pair.

**PCA + StandardScaler.** 1045 → 50 components (~95% variance). Scaler+PCA
fitted on training split, saved alongside model.

**Best model by macro-F1.** With class imbalance (5% Minor, 75% Moderate,
20% Major), macro-F1 treats all classes equally.

**uv instead of pip.** Faster deterministic installs. Used in Docker
(`uv sync --no-dev --frozen`) and CI (`astral-sh/setup-uv@v3`).

## Metrics

| Metric | What it measures |
|--------|------------------|
| Per-class F1 | How well each severity is predicted |
| Macro-F1 | Unweighted average across classes — primary target |
| Weighted-F1 | Weighted by class support |
| Cohen's Kappa | Agreement beyond chance |
| MAE | Ordinal error (predicted vs true severity) |
