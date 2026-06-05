# DDI Severity Predictor — Project Overview

An end-to-end MLOps pipeline that predicts drug-drug interaction severity (Major / Moderate / Minor) directly from chemical structures using Morgan fingerprints and machine learning.

---

## Problem

Drug-drug interactions (DDIs) are a major clinical concern. Existing databases list interactions but require a drug name lookup — they can't generalize to novel compounds. This project shows that chemical structure alone (SMILES → molecular fingerprints) is sufficient to predict interaction severity, enabling predictions for any molecule.

---

## File-by-File Reference

### Source Code (`src/`)

| File | Purpose |
|---|---|
| `logger.py` | Configures Python `logging` with coloured console output and daily rotating file handler to `logs/`. Single shared logger used across all modules. |
| `features.py` | All RDKit feature engineering. Converts a DataFrame of SMILES pairs into a fixed-length numeric feature matrix: 256-bit Morgan fingerprints (×4: A, B, diff, product), Tanimoto similarity (1), molecular descriptors (10 × 2: diff + sum). Total: 1045 features per pair. |
| `models.py` | Model definitions and hyperparameter grids. 5 model families (LogisticRegression, LinearSVC, RandomForest, KNN, XGBoost) with 3 parameter sets each. Centralised so train.py stays clean. |
| `train.py` | Main training pipeline. Loads data, builds or caches features, loops over 15 model/param combos, logs all metrics to MLflow, registers the best model to the Model Registry, and exports scaler+PCA for serving. |
| `export_model.py` | One-time export script. Queries MLflow for the run with the highest macro-F1, downloads the model, rebuilds the scaler and PCA from training data (or loads cached features), and saves all three as joblib files to `models/`. |
| `api.py` | FastAPI server with three routes: `GET /` (frontend), `GET /health`, `POST /predict` (SMILES → severity + probabilities). Loads model/scaler/PCA on startup. |
| `fetch_smiles.py` | PubChem SMILES resolution. Reads `raw_ddi.csv`, looks up each unique drug name via pubchempy (5 concurrent workers), falls back to alternate names for problematic drugs, outputs `chemical_ddi.csv`. |
| `drift.py` | Data drift detection. Computes fingerprint density statistics on the training set. At inference time, runs a two-sample KS test against new inputs to detect distribution shift. |

### Static Files (`src/static/`)

| File | Purpose |
|---|---|
| `index.html` | Minimal vanilla JS frontend. Two SMILES inputs, predict button, severity badge with colour, animated probability bars. No frameworks, no build step. |

### Configuration (`config/`)

| File | Purpose |
|---|---|
| `config.yaml` | Root Hydra config — composes features, training, and models sub-configs. |
| `features/default.yaml` | Feature engineering parameters (n_bits, radius, descriptor list). |
| `training/default.yaml` | Training parameters (CV folds, cache paths, export toggle). |
| `models/default.yaml` | Hyperparameter grids as YAML lists for each model family. |

### Data (`data/`)

| File | Purpose |
|---|---|
| `chemical_ddi.csv` | Training data. 109,441 drug pairs from DDInter 2.0 with SMILES for both drugs and a severity label (0=Minor, 1=Moderate, 2=Major). This is the only file `train.py` needs. |

### Tests (`tests/`)

| File | Purpose |
|---|---|
| `test_chemistry.py` | Validates RDKit fingerprint generation: correct shape (1024 bits), binary values, works for common molecules (water, aspirin). |
| `test_model.py` | Validates feature engineering dimensions. Builds features for a 2-row dummy DataFrame and asserts the output column count is exactly 4×N_BITS + 1 + 2×10 = 1045. |

### Infrastructure

| File | Purpose |
|---|---|
| `Makefile` | Common commands: `make train`, `make api`, `make test`, `make lint`, `make format`, `make clean`, `make export`. |
| `Dockerfile` | `python:3.14-slim` image, installs deps, copies code and models, runs uvicorn. |
| `.pre-commit-config.yaml` | Pre-commit hooks: ruff lint+fix, ruff format, mypy type checking, pytest. |
| `.github/workflows/ci-cd.yaml` | GitHub Actions: 3-job pipeline (lint → test → train + validate with macro-F1 ≥ 0.76 gate). |
| `dvc.yaml` | DVC pipeline definition: training stage with deps and outputs. |
| `pyproject.toml` | Project metadata, dependencies, ruff config, pytest config. |
| `requirements.txt` | Pip dependencies for reproducibility. |

---

## Key Design Decisions

**SMILES-only input.** The model receives only chemical structure. Drug names are resolved at data-prep time, never at inference. This means the model works for any molecule, not just known drugs.

**Morgan fingerprints + molecular descriptors.** Fingerprints capture substructure presence/absence; descriptors capture bulk properties (molecular weight, LogP, TPSA, etc.). Together they cover both local and global molecular properties.

**Interaction features.** For each pair we compute `|fp_A − fp_B|`, `fp_A × fp_B`, Tanimoto similarity, and per-descriptor diff/sum. These symmetric features ensure order invariance (A+B = B+A).

**PCA + StandardScaler.** Raw features are 1045-dimensional. PCA reduces to 50 components while retaining ~95% variance. The scaler+PCA transformer is fitted on the training split and saved alongside the model.

**Best model by macro-F1.** With class imbalance (5% Minor, 75% Moderate, 20% Major), macro-F1 treats all classes equally. Accuracy alone would be misleading.

---

## Metrics

| Metric | What it measures |
|---|---|
| Per-class F1 | How well each severity is predicted (especially the rare Minor class) |
| Macro F1 | Unweighted average across classes — primary optimization target |
| Weighted F1 | Weighted by class support — reflects overall performance |
| Cohen's Kappa | Agreement beyond chance; adjusts for class imbalance |
| MAE | Ordinal error: predicting Major when truth is Moderate (error=1) vs Minor (error=2) |
