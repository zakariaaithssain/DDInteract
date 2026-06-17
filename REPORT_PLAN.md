# Project Report Plan — MLOps Pipeline

## 1. Introduction
- Problem statement: DDI prediction from chemical structure
- Why MLOps matters for reproducibility, deployment, and monitoring
- Project goals and scope

## 2. Data Pipeline
- **Source**: DDInter 2.0 (109K drug pairs, 3 severity classes)
- **SMILES resolution**: PubChem API lookup with fallback names (concurrent, 5 workers)
- **Train/validation split**: 80/20 stratified by severity
- **DVC**: Data versioning, pipeline reproducibility, local remote; `dvc.yaml` uses `uv run` for environment consistency

## 3. Feature Engineering
- **Molecular fingerprints**: 256-bit Morgan fingerprints (radius=2)
- **Interaction features**: fingerprint diff, product, Tanimoto similarity
- **Molecular descriptors**: MolWt, LogP, TPSA, HBA/HBD, rotatable bonds, ring counts, CSP3 fraction, heteroatom count
- **Dimensionality reduction**: PCA (50 components, ~95% variance)
- **Feature cache**: numpy cache to skip RDKit rebuild on re-run

## 4. Modeling
- **Models**: LogisticRegression, LinearSVC, RandomForest, KNN, XGBoost
- **Hyperparameter search**: grid search (3 configs each, 15 total runs)
- **Class imbalance**: balanced class weights, macro-F1 as primary metric
- **Best model**: RandomForest (macro-F1 = 0.765)
- **Autolog removed**: `mlflow.sklearn.autolog` / `mlflow.xgboost.autolog` were removed in favor of manual logging to avoid conflicts with custom metric tracking
- **Local model checkpoint**: best model saved to `models/best_model.joblib` for immediate use without MLflow query

## 5. Evaluation
- **Per-class**: Precision, Recall, F1 (Minor/Moderate/Major)
- **Aggregate**: Macro-F1, Weighted-F1, Accuracy
- **Ordinal**: Cohen's Kappa, MAE
- **Cross-validation**: 3-fold on training set
- **Visualization**: Confusion matrix plots logged to MLflow

## 6. Experiment Tracking (MLflow)
- **Local tracking**: MLflow UI at `mlflow ui`
- **Logged per run**: all hyperparameters, metrics, model artifact, confusion matrix
- **Model Registry**: best model registered as "DDI-Severity" with `production` alias

## 7. Model Serving
- **Export pipeline**: best model (`models/best_model.joblib` from training, or best MLflow run) + fitted scaler + PCA → joblib files; `export_model.py` handles XGBoost and sklearn models
- **API**: FastAPI with `/predict` endpoint (SMILES → severity + probabilities) and `/drift` endpoint (monitoring status)
- **Drift monitoring in API**: feature vectors accumulated per-prediction, drift check every 100 predictions against training reference
- **Frontend**: Minimal HTML/JS interface
- **Containerization**: Docker with python:3.14-slim, uv-based

## 8. CI/CD
- **GitHub Actions**: 2 jobs (lint → test)
- **Quality gates**: ruff lint/format, mypy typecheck, pytest (103 tests, 96% coverage)
- **Pre-commit hooks**: ruff check (--fix), ruff format, mypy, pytest

## 9. Monitoring
- **Data drift**: Evidently `DataDriftPreset` comparing 5 aggregate features (fp_a_density, fp_b_density, fp_diff_mean, fp_product_mean, Tanimoto) instead of a single fingerprint-density KS test
- **Reference stats**: `compute_reference_stats` stores full reference feature distributions from training data
- **Drift detection**: per-column KS p-values parsed from Evidently report; drift flagged when >50% of columns drift beyond threshold
- **API integration**: predictions buffer feature vectors in memory, run detection every 100 requests, exposed via `GET /drift`
- **Logging**: structured logging to console + rotating files
- **Drift report**: JSON output saved when drift is detected

## 10. Infrastructure & Best Practices
- **uv** for dependency management (instead of pip)
- **Pre-commit hooks**: ruff format/lint, mypy, pytest
- **Makefile**: standardized commands
- **Configuration management**: centralized `src/config.py` replacing scattered path constants (Hydra no longer used)
- **Dependencies**: pyproject.toml (uv.lock locked)
- **DVC pipeline**: uses `uv run` for stage commands to match project's Python environment setup

## 11. Results & Discussion
- Best model performance (macro-F1, per-class breakdown)
- Comparison across model families
- Impact of feature choices (fingerprint size, PCA, descriptors)
- Limitations: Minor class still challenging, no mechanism/effect context

## 12. Future Work
- Ensemble of best models (RandomForest + LinearSVC)
- Deep learning (GNNs on molecular graphs)
- Active learning for rare classes
- Real-time monitoring dashboard
- A/B testing for model updates
