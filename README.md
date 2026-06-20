# DDI Severity Predictor — MLOps Pipeline

Predict drug-drug interaction severity (Minor / Moderate / Major) from chemical structure (SMILES → Morgan fingerprints).

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **Language** | Python 3.14 | Core development |
| **ML** | scikit-learn, XGBoost | Classification (Random Forest, XGBoost) |
| **Chemistry** | RDKit | Morgan fingerprints, molecular descriptors |
| **Data** | Pandas, NumPy | Data processing |
| **Dependency mgmt** | uv | Fast, deterministic dependency resolution |
| **Data Versioning** | DVC | Pipeline reproducibility & data versioning |
| **Experiment Tracking** | MLflow | Log metrics, params, artifacts, model registry |
| **Testing** | pytest, pytest-cov | Unit tests + coverage (98%) |
| **Linting** | ruff | Lint + format (line length 120, py314 target) |
| **Type Checking** | mypy | Static type checking with per-module overrides |
| **Pre-commit** | ruff, mypy, pytest | Quality gates before every commit |
| **CI/CD** | GitHub Actions | Lint → format-check → mypy → pytest on push |
| **Serving** | FastAPI, Uvicorn | REST API for predictions |
| **Containerization** | Docker (uv-based, slim image) | Reproducible deployment |
| **Monitoring** | Drift detection (Wasserstein, chi-squared, PSI) | Covariate shift, label shift, concept drift |
| **HPO** | Optuna + TPE | Bayesian hyperparameter search with pruning |

---

## Pipeline

```
data/chemical_ddi.csv  (109K pairs, DDInter 2.0)
        │
        ▼
src/build_features.py
  ├── src/features.py   (RDKit → 533-dim features)
  └── caching           (data/features.npy — skips RDKit rebuild)
        │
        ▼
src/search_hyperparams.py
  ├── Optuna TPE search (50 trials, 20% subsample)
  ├── Dual pruning (MedianPruner for RF, XGBoostPruningCallback for XGBoost)
  └── models/best_params.json
        │
        ▼
src/train.py
  ├── Full-data retraining with best params
  ├── MLflow best_overall run (metrics, model w/ signature, confusion matrix, report)
  ├── Model Registry → "DDI-Severity" (production alias)
  └── models/best_model.joblib  (local fallback)
        │
        ▼
src/export_model.py  →  models/model.joblib
        │
        ▼
src/api.py           →  FastAPI :8000  (+ static/index.html frontend)
        │
        ▼
src/drift.py         →  Covariate / label / concept drift detection
```

---

## Getting Started

```bash
git clone <repo>
make install      # installs uv + uv sync
make hooks        # enable pre-commit hooks
make train        # full pipeline: features → search → train
make test         # run pytest (104 tests, 98% coverage)
make export       # export best model to models/
make api          # start FastAPI at localhost:8000
```

---

## Commands

| What | Command |
|------|---------|
| Install deps | `make install` |
| Enable pre-commit | `make hooks` |
| Train all models | `make train` or `dvc repro` |
| Run tests | `make test` or `pytest -v tests/` |
| Lint | `make lint` or `ruff check src/ tests/` |
| Format | `make format` or `ruff format src/ tests/` |
| Type check | `make typecheck` or `mypy src/` |
| Export best model | `make export` or `python src/export_model.py` |
| Start API | `make api` or `uvicorn src.api:app` |
| DVC repro | `make dvc-repro` or `dvc repro` |
| Clean artifacts | `make clean` |
| MLflow UI | `mlflow ui` |

---

## Project Structure

```
├── .github/workflows/ci-cd.yaml   — GitHub Actions (lint → typecheck → test)
├── data/
│   ├── chemical_ddi.csv            — 109K drug pairs (DVC input)
│   ├── features.npy                — Cached feature matrix (gitignored)
│   └── labels.npy                  — Cached labels (gitignored)
├── models/                         — Exported joblib artifacts (gitignored)
├── src/
│   ├── api.py                      — FastAPI inference server
│   ├── build_features.py           — DVC stage 1: feature engineering + caching
│   ├── chemistry.py                — RDKit fingerprint generation
│   ├── search_hyperparams.py       — DVC stage 2: Optuna HPO (50 trials)
│   ├── train.py                    — DVC stage 3: full-data training + MLflow
│   ├── evaluate.py                 — Metrics, confusion matrix, classification report
│   ├── export_model.py             — Best model export + MLflow Registry
│   ├── features.py                 — Feature engineering (533-dim)
│   ├── fetch_smiles.py             — PubChem SMILES resolution
│   ├── config.py                   — Centralized paths + logger
│   ├── drift.py                    — Data drift detection
│   ├── logger.py                   — Structured logging to console + file
│   └── static/index.html           — Vanilla JS frontend
├── tests/
│   ├── test_api.py                 — FastAPI endpoint tests
│   ├── test_chemistry.py           — RDKit fingerprint tests
│   ├── test_drift.py               — Drift detection tests
│   ├── test_drift_pipeline.py      — Drift pipeline integration tests
│   ├── test_export_model.py        — Export pipeline tests
│   ├── test_features.py            — Feature engineering tests
│   ├── test_fetch_smiles.py        — PubChem resolution tests
│   ├── test_logger.py              — Logging tests
│   ├── test_model.py               — Model dimension tests
│   ├── test_models.py              — Model config tests
│   ├── test_features_model_pipeline.py — End-to-end feature → model tests
│   └── test_train.py               — Search + train pipeline tests
├── report/                         — LaTeX report
│   └── Chapters/chapter2.tex       — Training methodology chapter
├── .pre-commit-config.yaml         — ruff lint/fix, ruff-format, mypy, pytest
├── .gitignore                      — Caches, venv, MLflow, DVC artifacts
├── dvc.yaml                        — DVC pipeline (3 stages)
├── Makefile                        — Standardized commands
├── pyproject.toml                  — Project metadata + tool config
├── Dockerfile                      — uv-based slim image for serving
└── README.md
```

---

## MLOps Features

| Feature | Status | Details |
|---------|--------|---------|
| **DVC pipeline** | ✅ | `dvc.yaml` with 3 stages (build_features → search_hyperparams → train), local remote at `dvc_storage` |
| **MLflow tracking** | ✅ | Per-class metrics, macro-F1, Kappa, MAE, confusion matrix plots, classification report |
| **MLflow Model Registry** | ✅ | Best model registered as `DDI-Severity`, promoted to `production` with version tags |
| **Hyperparameter search** | ✅ | Optuna TPE (50 trials, 20% subsample, dual pruning) |
| **Feature caching** | ✅ | `data/features.npy` — avoids RDKit rebuild on re-run |
| **Model export** | ✅ | `make export` — dumps model as joblib |
| **FastAPI serving** | ✅ | `make api` — inference with probability output, graceful 503 if model missing |
| **Frontend** | ✅ | Minimal vanilla JS UI at `/` |
| **Structured logging** | ✅ | `src/logger.py` — console + rotating file handler |
| **Pre-commit hooks** | ✅ | `ruff check --fix`, `ruff format`, `mypy`, `pytest` |
| **CI/CD** | ✅ | GitHub Actions: lint → format-check → mypy → pytest |
| **Docker** | ✅ | `python:3.14-slim` with `uv sync --no-dev --frozen` |
| **Data drift monitoring** | ✅ | `src/drift.py` — covariate shift (Wasserstein), label shift (chi-squared), concept drift (PSI) |
| **Type checking** | ✅ | `mypy src/` — all modules clean, rdkit stubs pinned |
| **Coverage** | ✅ | 98% (104 tests) |

---

## Metrics Tracked (per run)

- **Per-class**: Precision, Recall, F1 (Minor / Moderate / Major)
- **Aggregate**: Macro-F1, Weighted-F1, Accuracy
- **Ordinal**: Cohen's Kappa, MAE
- **CV**: 3-fold mean + std
- **Artifacts**: Confusion matrix PNG, classification report, model with signature (MLflow), hyperparams JSON

---

## Evaluation Strategy

| Metric | Why |
|--------|-----|
| **Macro-F1** | Primary target — treats all classes equally despite imbalance |
| **Weighted-F1** | Reflects overall performance weighted by class support |
| **Cohen's Kappa** | Agreement beyond chance; adjusts for class imbalance |
| **MAE** | Ordinal error: predicting Major when truth is Moderate (error=1) vs Minor (error=2) |

---

## Design Decisions

- **SMILES-only input**: Model receives only chemical structure. Drug names are resolved at data-prep time, never at inference.
- **Morgan fingerprints + molecular descriptors**: 256-bit fingerprints (×2 operations: difference + product) + Tanimoto + 10 descriptors diff/sum = 533 features.
- **No PCA or scaling**: Tree-based models are invariant to monotonic transformations; dimensionality reduction adds complexity without benefit.
- **Best model by macro-F1**: With class imbalance (5% Minor, 75% Moderate, 20% Major), macro-F1 treats all classes equally.
- **RDKit 2026.3.2 pinned**: Later versions ship broken stub files that crash mypy.
- **uv over pip**: Faster deterministic installs via `uv sync --frozen` in Docker and CI.
- **Separate lint/format/typecheck/test stages**: CI mirrors pre-commit hooks exactly.
