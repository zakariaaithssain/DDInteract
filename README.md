# End-to-End MLOps for Drug-Drug Interaction Prediction

## 1. Project Title & Summary

**Title:** *PharmaGraph: A Production-Ready MLOps Pipeline for Predicting Adverse Drug-Drug Interactions via Knowledge Graph Neural Networks*

**Summary:** build a system that predicts the severity of interactions between drug pairs using a Graph Neural Network (GNN) trained on biomedical knowledge graphs. The core deliverable is not just the model, but a reproducible MLOps pipeline that automates data versioning, experiment tracking, model deployment, and performance monitoring—with explicit handling of cold-start drugs that enter the market after training.

---

## 2. Problem Statement

**Clinical Context:** When a patient takes multiple medications, unexpected interactions can alter efficacy or trigger adverse events. Existing rule-based databases (e.g., DrugBank) are manually curated and struggle to keep pace with new approvals.

**Technical Challenge:** a **link prediction** problem on a heterogeneous knowledge graph. Nodes include drugs, proteins, diseases, and side-effects. Edges represent relationships (treats, targets, causes, interacts-with). Given a pair of drugs, the model predicts the probability and severity of an adverse interaction.

**MLOps Angle:** New drugs, targets, and clinical evidence arrive continuously. A static model degrades. the pipeline must detect this degradation and trigger retraining.

---

## 3. Technical Architecture

### Model: Knowledge Graph Neural Network (KGNN)
- **Embedding Layer:** Initialize drug nodes using molecular fingerprints (Morgan/ECFP) or pre-trained bio-encoders (e.g., MolBERT).
- **Message Passing:** Use a Relational Graph Convolutional Network (R-GCN) or GraphSAGE to propagate information across the knowledge graph.
- **Decoder:** A DistMult or RotatE scoring function takes the embeddings of two drug nodes and predicts interaction type (e.g., 0: None, 1: Minor, 2: Moderate, 3: Major, 4: Contraindicated).
- **Loss:** Cross-entropy with class weights to handle severe imbalance.

### Cold-Start Handling
For drugs unseen during training, the pipeline falls back to molecular structure embeddings, ensuring the model is never completely blind to new entities.

---

## 4. The MLOps Pipeline (End-to-End)

### Stage A: Data Engineering & Versioning
1. **Ingestion:** Download DrugBank, BioSNAP, and TWOSIDES snapshots.
2. **Graph Construction:** Build a heterogeneous graph (Neo4j or NetworkX → PyTorch Geometric ).
3. **Negative Sampling:** Generate non-interacting drug pairs. Use a time-aware split to prevent leakage.
4. **Versioning:** Snapshot the processed graph with DVC. Git tracks code; DVC tracks the  and  artifacts.

### Stage B: Experimentation & Training
1. **Feature Store:** Store pre-computed molecular embeddings and node features in a local feature registry (Feast or a simple Parquet store).
2. **Experiment Tracking:** Every training run logs hyperparameters, graph statistics, and metrics (AUROC, AUPRC, F1 per class) to MLflow.
3. **Hyperparameter Search:** Run Optuna to optimize learning rate, embedding dimension, number of GNN layers, and negative sampling ratio.
4. **Artifact Storage:** Best models are logged to the MLflow Model Registry with tags (, ).

### Stage C: Continuous Integration
1. **Data Validation:** On every commit, run Great Expectations to assert that the knowledge graph has the expected schema, no isolated drug nodes, and valid SMILES strings.
2. **Model Testing:** A CI job (GitHub Actions) trains a smoke test model for 1 epoch on a tiny subgraph to catch code regressions.
3. **Containerization:** Dockerfile builds a reproducible training environment.

### Stage D: Deployment & Serving
1. **Model Packaging:** Export the production model to TorchScript or ONNX for faster inference.
2. **API:** FastAPI service exposes two endpoints:
   -  → Accepts two DrugBank IDs, returns interaction probability and severity.
   -  → Returns model version and last training timestamp.
3. **Container Orchestration:** Docker Compose runs the API, a Redis cache for frequent queries, and a monitoring sidecar.

### Stage E: Monitoring & Observability
1. **Data Drift:** Weekly job compares the distribution of new drug approvals (molecular weight, target classes) against the training set using Kolmogorov-Smirnov tests. If drift exceeds a threshold, alert.
2. **Performance Drift:** Track prediction confidence histograms. A sudden drop in mean confidence for new drug pairs signals potential decay.
3. **Concept Drift:** Compare model predictions against newly published interaction evidence (e.g., from PubMed abstracts mined monthly). If disagreement rate rises, trigger retraining.
4. **Dashboard:** A Streamlit or Grafana dashboard visualizes:
   - Daily prediction volume
   - Drift metrics over time
   - Top-k most uncertain drug pairs for human review

### Stage F: Automated Retraining
1. **Trigger:** A Prefect/Airflow DAG runs monthly. It checks drift monitors and new data availability.
2. **Pipeline Execution:** If triggered, the DAG pulls the latest data, runs the training pipeline, evaluates against a holdout set of recent interactions, and promotes the new model to staging.
3. **Human-in-the-Loop:** Promotion to production requires a manual approval step in MLflow (simulating clinical governance).

---

## 5. Tech Stack

| Layer | Tool | Purpose |
|-------|------|---------|
| **Language** | Python 3.10+ | Core development |
| **ML/DL** | PyTorch, PyTorch Geometric | GNN implementation |
| **Data** | Pandas, NetworkX, Neo4j (optional) | Graph construction & querying |
| **Data Versioning** | DVC | Version control for datasets & models |
| **Experiment Tracking** | MLflow | Log metrics, params, artifacts |
| **Orchestration** | Prefect 2.x (or Apache Airflow) | Pipeline DAGs & scheduling |
| **Feature Store** | Feast (lightweight local mode) | Centralized feature serving |
| **Data Quality** | Great Expectations | Schema & distribution validation |
| **Serving** | FastAPI, Uvicorn | REST API for predictions |
| **Containerization** | Docker, Docker Compose | Environment reproducibility |
| **CI/CD** | GitHub Actions | Automated testing & builds |
| **Monitoring** | Evidently AI, custom scripts | Drift detection |
| **Dashboard** | Streamlit | Visualization layer |
| **Storage** | MinIO (S3-compatible) or local filesystem | Artifact store for DVC/MLflow |

---

## 6. Recommended Project Structure

```bash

pharmagraph-mlops/
├── .github/
│   └── workflows/
│       ├── ci.yml              # Lint, test, smoke-train
│       └── drift-check.yml     # Weekly drift detection
├── data/
│   ├── raw/                    # DrugBank, TWOSIDES (gitignored, DVC tracked)
│   ├── processed/              # HeteroData objects (DVC tracked)
│   └── expectations/           # Great Expectations suites
├── src/
│   ├── data/
│   │   ├── build_graph.py
│   │   ├── negative_sampling.py
│   │   └── validation.py
│   ├── models/
│   │   ├── rgcn.py
│   │   ├── decoder.py
│   │   └── train.py
│   ├── features/
│   │   └── molecular_embeddings.py
│   ├── pipeline/
│   │   ├── train_pipeline.py   # Prefect flow
│   │   └── deploy_pipeline.py
│   ├── api/
│   │   ├── main.py
│   │   └── schemas.py
│   └── monitoring/
│       ├── drift_detector.py
│       └── evidently_reports.py
├── notebooks/
│   └── eda.ipynb
├── tests/
│   ├── unit/
│   └── integration/
├── docker/
│   ├── Dockerfile.api
│   └── Dockerfile.train
├── docker-compose.yml
├── dvc.yaml                    # DVC pipeline definition
├── params.yaml                 # Hyperparameters & config
└── README.md
```

---

## 7. Evaluation Strategy

| Metric | Why |
|--------|-----|
| **Macro-F1 / Weighted-F1** | Class imbalance makes accuracy misleading |
| **AUPRC (PR-AUC)** | More informative than AUROC for rare positive interactions |
| **Hits@K** | Standard for knowledge graph link prediction |
| **Per-Class Recall** | Ensure we are not missing Contraindicated interactions |

**Validation Scheme:** Split by *time* and by *drug*. Hold out all interactions involving drugs approved after 2022 as the true generalization test.

---

## 8. Deliverables Checklist

For the end-of-year submission, we aim to present:

- [ ] **Git Repository:** Clean, documented, with DVC remotes configured.
- [ ] **Reproducible Training:** One command ( or ) executes the full training pipeline.
- [ ] **Live API Demo:** A Dockerized FastAPI instance running locally that accepts drug pairs and returns predictions with model versioning metadata.
- [ ] **Monitoring Dashboard:** Screenshot or live demo showing drift metrics and prediction logs.
- [ ] **Technical Report:** 8–12 pages covering clinical motivation, architecture, MLOps design decisions, and an analysis of a simulated model failure (e.g., performance drop on 2023 cold-start drugs).

---

## 9. First Steps to Start This Week

1. **Data Acquisition:** Download the [BioSNAP DDI dataset](https://snap.stanford.edu/biodata/datasets/10001/10001-ChG-Miner.html) and a subset of [DrugBank](https://go.drugbank.com/) (open data for academic use).
2. **Proof of Concept:** Build a simple 2-layer GraphSAGE model in PyTorch Geometric that overfits a small graph. Verify that we can predict links.
3. **MLOps Skeleton:** Initialize the Git repo, set up DVC, and log the first experiment to MLflow locally.
4. **Baseline:** Implement a non-graph baseline (e.g., XGBoost on concatenated Morgan fingerprints) to beat.
