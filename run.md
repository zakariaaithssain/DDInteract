# DDI Severity Predictor — MLOps Pipeline

Predict drug-drug interaction severity (Minor / Moderate / Major) from chemical structure (SMILES → Morgan fingerprints).

---

## Quick Start

```bash
make install
make train       # trains 5 models × 3 hyperparameter sets, logs to MLflow
make test        # runs pytest
make api         # starts FastAPI at :8000
```

---

## Project Structure

```
src/
  logger.py       — Logging setup (console + file rotation)
  features.py     — RDKit feature engineering (fingerprints, descriptors, Tanimoto)
  models.py       — Model definitions + hyperparameter grids (5 models × 3 configs)
  train.py        — Training loop with MLflow tracking, caching, model registry
  export_model.py — Export best model + scaler + PCA as local joblib files
  api.py          — FastAPI server for inference
  drift.py        — Data drift detection via KS test on fingerprint density
  fetch_smiles.py — PubChem SMILES resolution (drug name → SMILES)
  static/
    index.html    — Minimal frontend (vanilla JS)
data/
  chemical_ddi.csv — SMILES-enriched training data (109K pairs)
config/
  config.yaml      — Hydra experiment config
  features/default.yaml
  training/default.yaml
  models/default.yaml
tests/
  test_chemistry.py — RDKit fingerprint validation
  test_model.py     — Feature dimension assertions
```

---

## Commands

| What | Command |
|---|---|
| Train all models | `make train` or `python src/train.py` |
| Run tests | `make test` or `pytest -v tests/` |
| Lint | `make lint` or `ruff check src/ tests/` |
| Format | `make format` or `ruff format src/ tests/` |
| Export best model | `make export` or `python src/export_model.py` |
| Start API | `make api` or `uvicorn src.api:app` |
| Reproduce pipeline | `make dvc-repro` or `dvc repro` |
| Clean artifacts | `make clean` |
| MLflow UI | `mlflow ui` |

---

## Pipeline

```
data/chemical_ddi.csv
       │
       ▼
src/train.py
  ├── features.py   (RDKit → fingerprints, descriptors, Tanimoto)
  ├── models.py     (LR, SVC, RF, KNN, XGBoost × 3 param sets each)
  ├── caching       (skips RDKit if features.npy exists)
  └── MLflow        (per-class metrics, macro-F1, Kappa, MAE, CM plots)
       │
       ├── Model Registry  →  "DDI-Severity" (production alias)
       │
       ▼
src/export_model.py  →  models/model.joblib + scaler + PCA
       │
       ▼
src/api.py           →  FastAPI :8000  (+ static/index.html frontend)
```

---

## MLOps Features

| Feature | Status | Details |
|---|---|---|
| **DVC pipeline** | ✅ | `dvc.yaml` with training stage, `dvc repro` one-command reproduction |
| **MLflow tracking** | ✅ | Per-class metrics, macro-F1, Kappa, MAE, confusion matrix plots |
| **MLflow Model Registry** | ✅ | Best model auto-registered as `DDI-Severity`, promoted to `production` |
| **Hyperparameter search** | ✅ | Grid search over 3 configs × 5 model families (15 runs) |
| **Feature caching** | ✅ | `data/features.npy` — avoids 8-min RDKit rebuild on re-run |
| **Model export** | ✅ | `make export` — dumps model + scaler + PCA as local joblib |
| **FastAPI serving** | ✅ | `make api` — inference endpoint with probability output |
| **Frontend** | ✅ | Minimal vanilla JS UI at `/` |
| **Structured logging** | ✅ | `src/logger.py` — console + rotating file handler |
| **Pre-commit hooks** | ✅ | `.pre-commit-config.yaml` — ruff lint/format, mypy, pytest |
| **CI/CD** | ✅ | GitHub Actions: lint → test → train + model validation gate (≥0.76) |
| **Docker** | ✅ | `Dockerfile` — slim Python image, ready for deployment |
| **Config management** | ✅ | Hydra configs in `config/` — override without touching code |
| **Data drift monitoring** | ✅ | `src/drift.py` — KS test on fingerprint density vs training distribution |
| **Makefile** | ✅ | `make train/test/api/lint/format/export/clean` |

---

## Metrics Tracked (per run)

- **Per-class**: Precision, Recall, F1 (Minor / Moderate / Major)
- **Aggregate**: Macro F1, Weighted F1, Accuracy
- **Ordinal**: Cohen's Kappa, MAE
- **CV**: 3-fold mean + std
- **Artifacts**: Confusion matrix PNG, model pickle
