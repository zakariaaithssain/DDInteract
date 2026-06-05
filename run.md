# Running the DDI Pipeline

## Prerequisites

Ensure your Python environment has all dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

---

## What Still Needs Setup

### 1. Initialize DVC

DVC is not yet initialized in this repo. Run:

```bash
dvc init
git add .dvc/config .dvcignore
git commit -m "init DVC"
```

Then track the raw input data (DVC will hash it and keep it out of Git):

```bash
dvc add data/raw_ddi.csv
git add data/raw_ddi.csv.dvc data/.gitignore
git commit -m "track raw DDI data with DVC"
```

### 2. Configure MLflow Tracking (Optional)

By default MLflow logs to the local `mlruns/` directory and `mlflow.db`. This works out of the box. If you want to use a remote tracking server, set:

```bash
export MLFLOW_TRACKING_URI=<your-server-uri>
```

### 3. DVC Remote (for sharing data)

To share or version data across machines, add a remote (S3, GCS, SSH, etc.):

```bash
dvc remote add myremote s3://my-bucket/pfa-data
dvc push
```

---

## How to Run

### Ad-hoc (step-by-step)

```bash
source .venv/bin/activate

# Step 1 — Fetch SMILES from PubChem
python src/fetch_smiles.py

# Step 2 — Train the XGBoost model
python src/train.py
```

### Orchestrated via DVC

```bash
source .venv/bin/activate
dvc repro
```

DVC will skip the `fetch_smiles` stage if `data/chemical_ddi.csv` is already up to date, and skip `train` if nothing changed. Use `dvc repro --force` to rerun everything.

### Run Tests

```bash
source .venv/bin/activate
pytest -v tests/
```

---

## Pipeline Overview

```
data/raw_ddi.csv
       │
       ▼
src/fetch_smiles.py    ←── pubchempy (PubChem API)
       │
       ▼
data/chemical_ddi.csv  ←── SMILES + numeric labels
       │
       ▼
src/train.py           ←── RDKit fingerprints + XGBoost
       │
       ▼
    MLflow run         ←── metrics, model artifacts
```

## CI/CD (GitHub Actions)

The workflow at `.github/workflows/ci-cd.yaml` runs automatically on push/PR:

1. Installs Python + cached dependencies
2. Runs `pytest`
3. Executes `dvc repro`

It expects DVC to already be initialized in the repo so that `dvc repro` can resolve the pipeline stages. Push the `.dvc/` directory and any `.dvc` files to GitHub for CI to work.
