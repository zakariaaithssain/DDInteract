# DDI Severity Predictor

MLOps pipeline for predicting Drug-Drug Interaction severity from SMILES pairs. Trains ensemble classifiers with hyperparameter tuning via Optuna, exposes a FastAPI inference API with drift monitoring, and tracks experiments through MLflow.

## setup

### 1. install dependencies

```bash
uv sync
```

### 2. install pre-commit hooks (optional)

```bash
uv run pre-commit install
```

### 3. prepare environment

This project uses [DVC](https://dvc.org) for data and pipeline versioning. The raw dataset is tracked via DVC.

```bash
dvc pull
```

## how to run

### dev mode (using uv)
check the Makefile for full commands. 

### run using Docker

```bash
docker build -t ddi-predictor .
docker run -p 8000:8000 ddi-predictor
```

## structure and architecture

```
├── README.md
├── Dockerfile
├── Makefile
├── pyproject.toml
├── dvc.yaml
├── data/
│   ├── raw_ddi.csv
│   └── chemical_ddi.csv
├── models/
├── logs/
├── mlruns/
└── src/
    ├── api.py                    # FastAPI inference server
    ├── build_features.py         # DVC stage: feature computation
    ├── clinical_metrics.py       # Clinical evaluation metrics
    ├── config.py                 # Paths, constants, logger
    ├── drift.py                  # Drift detection
    ├── evaluate.py               # Model evaluation
    ├── export_model.py           # Model export
    ├── features.py               # Molecular feature functions
    ├── fetch_smiles.py           # SMILES fetching from PubChem
    ├── search_hyperparams.py     # Optuna hyperparameter search
    ├── threshold_optimizer.py    # Prediction threshold tuning
    ├── train.py                  # Model training
    ├── static/
    │   ├── index.html            # API frontend
    │   └── drift.html            # Drift monitoring dashboard
    └── tests/
        └── ...
```
