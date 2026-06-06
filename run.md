# DDI Severity Predictor — Quick Reference

## Prerequisites

- Python 3.14
- uv (installed automatically by `make install`)

## Setup

```bash
make install      # install uv + sync dependencies
make hooks        # enable pre-commit hooks
```

## Training

```bash
make train
# or
python src/train.py
```

Trains 5 models (LR, SVC, RF, KNN, XGBoost) × 3 hyperparameter configs.
Logs all metrics to MLflow. Best model registered to Model Registry as `DDI-Severity`.

## Testing

```bash
make test
# or
pytest -v tests/
```

77 tests, 98% coverage across all modules.

## Quality Checks

```bash
make lint        # ruff check src/ tests/
make format      # ruff format src/ tests/
make typecheck   # mypy src/
```

All three pass with zero issues.

## Export Best Model

```bash
make export
# or
python src/export_model.py
```

Downloads the run with highest macro-F1 from MLflow, rebuilds scaler + PCA,
saves `models/model.joblib`, `models/scaler.joblib`, `models/pca.joblib`.

## API

```bash
make api
# or
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `GET /` — frontend (vanilla JS)
- `GET /health` — model version + timestamp
- `POST /predict` — SMILES pair → severity + probabilities

## DVC Pipeline

```bash
make dvc-repro
# or
dvc repro
```

Reproduces the training pipeline. Remote storage: `/home/zeco/Work/dvc_storage`.

## Docker

```bash
docker build -t ddi-predictor .
docker run -p 8000:8000 ddi-predictor
```

Based on `python:3.14-slim`, uses `uv sync --no-dev --frozen`.

## MLflow UI

```bash
mlflow ui
```

## Cleanup

```bash
make clean
```

Removes `models/`, cached features, MLflow artifacts, `results.json`.
