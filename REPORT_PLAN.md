# Project Report Plan — MLOps Pipeline

## 1. Introduction
- Problem statement: DDI prediction from chemical structure
- Why MLOps matters for reproducibility, deployment, and monitoring
- Project goals and scope

## 2. Data Pipeline
- **Source**: DDInter 2.0 (109K drug pairs, 3 severity classes)
- **SMILES resolution**: PubChem API lookup with fallback names (concurrent, 5 workers)
- **Train/validation split**: 80/20 stratified by severity
- **DVC**: Data versioning and pipeline reproducibility

## 3. Feature Engineering
- **Molecular fingerprints**: 256-bit Morgan fingerprints (radius=2)
- **Interaction features**: fingerprint diff, product, Tanimoto similarity
- **Molecular descriptors**: MolWt, LogP, TPSA, HBA/HBD, rotatable bonds, ring counts, CSP3 fraction, heteroatom count
- **Dimensionality reduction**: PCA (50 components, ~95% variance)
- **Feature cache**: numpy cache to skip 8-min RDKit rebuild on re-run

## 4. Modeling
- **Models**: LogisticRegression, LinearSVC, RandomForest, KNN, XGBoost
- **Hyperparameter search**: grid search (3 configs each, 15 total runs)
- **Class imbalance**: balanced class weights, macro-F1 as primary metric
- **Best model**: RandomForest (macro-F1 = 0.765)

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
- **Export pipeline**: best model + fitted scaler + PCA → joblib files
- **API**: FastAPI with `/predict` endpoint (SMILES → severity + probabilities)
- **Frontend**: Minimal HTML/JS interface
- **Containerization**: Docker with python:3.14-slim

## 8. CI/CD
- **GitHub Actions**: 3 jobs (lint → test → train + validate)
- **Quality gates**: ruff linting, pytest, macro-F1 ≥ 0.76 threshold
- **DVC repro**: full pipeline reproducibility in CI

## 9. Monitoring
- **Data drift**: KS test on fingerprint density distribution
- **Logging**: structured logging to console + rotating files
- **Drift report**: JSON output with KS statistic and p-value

## 10. Infrastructure & Best Practices
- **Pre-commit hooks**: ruff format/lint, mypy, pytest
- **Makefile**: standardized commands
- **Configuration management**: Hydra YAML configs
- **Dependencies**: pyproject.toml + requirements.txt

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
