# DDI Severity Predictor — MLOps Pipeline

Predict drug-drug interaction severity (Minor / Moderate / Major) from chemical structure (SMILES → Morgan fingerprints).

---

## Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **Language** | Python 3.14 | Core development |
| **ML** | scikit-learn, XGBoost | Classification (5 model families) |
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
| **Monitoring** | Drift detection (KS test) | Data drift on fingerprint density |
| **Config** | Hydra | YAML-based experiment configuration |

---

## Pipeline

```
data/chemical_ddi.csv  (109K pairs, DDInter 2.0)
        │
        ▼
src/train.py
  ├── src/features.py   (RDKit → 1045-dim features)
  ├── src/models.py     (LR, SVC, RF, KNN, XGBoost × 3 configs)
  ├── caching           (data/features.npy — skips RDKit rebuild)
  └── MLflow            (per-class metrics, macro-F1, Kappa, MAE, CM plots)
        │
        ├── Model Registry  →  "DDI-Severity" (production alias)
        │
        ▼
src/export_model.py  →  models/model.joblib + scaler + PCA
        │
        ▼
src/api.py           →  FastAPI :8000  (+ static/index.html frontend)
        │
        ▼
src/drift.py         →  KS test on fingerprint density
```

---

## Getting Started

```bash
git clone <repo>
make install      # installs uv + uv sync
make hooks        # enable pre-commit hooks
make train        # train 5 models × 3 configs, log to MLflow
make test         # run pytest (77 tests, 98% coverage)
make export       # export best model + scaler + PCA to models/
make api          # start FastAPI at localhost:8000
```

---

## Commands

| What | Command |
|------|---------|
| Install deps | `make install` |
| Enable pre-commit | `make hooks` |
| Train all models | `make train` or `python src/train.py` |
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
│   ├── chemistry.py                — RDKit fingerprint generation
│   ├── drift.py                    — Data drift detection (KS test)
│   ├── export_model.py             — Best model export pipeline
│   ├── features.py                 — Feature engineering (1045-dim)
│   ├── fetch_smiles.py             — PubChem SMILES resolution
│   ├── logger.py                   — Structured logging to console + file
│   ├── models.py                   — Model defs + hyperparameter grids
│   ├── train.py                    — Main training pipeline
│   └── static/index.html           — Vanilla JS frontend
├── tests/
│   ├── test_api.py                 — FastAPI endpoint tests
│   ├── test_chemistry.py           — RDKit fingerprint tests
│   ├── test_drift.py               — Drift detection tests
│   ├── test_export_model.py        — Export pipeline tests
│   ├── test_features.py            — Feature engineering tests
│   ├── test_fetch_smiles.py        — PubChem resolution tests
│   ├── test_logger.py              — Logging tests
│   ├── test_model.py               — Model dimension tests
│   ├── test_models.py              — Model config tests
│   └── test_train.py               — Training pipeline tests (15 tests)
├── .pre-commit-config.yaml         — ruff lint/fix, ruff-format, mypy, pytest
├── .gitignore                      — Caches, venv, MLflow, DVC artifacts
├── dvc.yaml                        — DVC pipeline (train stage)
├── Makefile                        — Standardized commands
├── pyproject.toml                  — Project metadata + tool config
├── Dockerfile                      — uv-based slim image for serving
└── README.md
```

---

## MLOps Features

| Feature | Status | Details |
|---------|--------|---------|
| **DVC pipeline** | ✅ | `dvc.yaml` with training stage, local remote at `dvc_storage` |
| **MLflow tracking** | ✅ | Per-class metrics, macro-F1, Kappa, MAE, confusion matrix plots |
| **MLflow Model Registry** | ✅ | Best model registered as `DDI-Severity`, promoted to `production` |
| **Hyperparameter search** | ✅ | Grid search over 3 configs × 5 model families (15 runs) |
| **Feature caching** | ✅ | `data/features.npy` — avoids RDKit rebuild on re-run |
| **Model export** | ✅ | `make export` — dumps model + scaler + PCA as joblib |
| **FastAPI serving** | ✅ | `make api` — inference with probability output |
| **Frontend** | ✅ | Minimal vanilla JS UI at `/` |
| **Structured logging** | ✅ | `src/logger.py` — console + rotating file handler |
| **Pre-commit hooks** | ✅ | `ruff check --fix`, `ruff format`, `mypy`, `pytest` |
| **CI/CD** | ✅ | GitHub Actions: lint → format-check → mypy → pytest |
| **Docker** | ✅ | `python:3.14-slim` with `uv sync --no-dev --frozen` |
| **Data drift monitoring** | ✅ | `src/drift.py` — KS test on fingerprint density |
| **Type checking** | ✅ | `mypy src/` — all modules clean |
| **Coverage** | ✅ | 98% (77 tests) |

---

## Metrics Tracked (per run)

- **Per-class**: Precision, Recall, F1 (Minor / Moderate / Major)
- **Aggregate**: Macro-F1, Weighted-F1, Accuracy
- **Ordinal**: Cohen's Kappa, MAE
- **CV**: 3-fold mean + std
- **Artifacts**: Confusion matrix PNG, model pickle

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
- **Morgan fingerprints + molecular descriptors**: 256-bit fingerprints (×4 operations) + Tanimoto + 10 descriptors diff/sum = 1045 features.
- **PCA + StandardScaler**: 1045 → 50 components (~95% variance). Fit on train split, saved alongside model.
- **Best model by macro-F1**: With class imbalance (5% Minor, 75% Moderate, 20% Major), macro-F1 treats all classes equally.
- **uv over pip**: Faster deterministic installs via `uv sync --frozen` in Docker and CI.
- **Separate lint/format/typecheck/test stages**: CI mirrors pre-commit hooks exactly.
