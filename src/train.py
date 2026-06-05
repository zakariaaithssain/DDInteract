import os
import json
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    cohen_kappa_score,
    confusion_matrix,
)
from features import build_features
from models import MODEL_GRIDS, RANDOM_STATE
from logger import logger

DATA_PATH = "data/chemical_ddi.csv"
EXPERIMENT_NAME = "DDI_Structural_Severity"
TEST_SIZE = 0.2
N_BITS = 256
N_PCA = 50
CLASS_NAMES = ["Minor", "Moderate", "Major"]
FEATURE_CACHE = "data/features.npy"
LABEL_CACHE = "data/labels.npy"
REGISTRY_NAME = "DDI-Severity"


def ordinal_mae(y_true, y_pred):
    return np.abs(y_true - y_pred).mean()


def log_confusion_matrix(cm, run_name, params_str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Confusion Matrix — {run_name}\n{params_str}", fontsize=9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticklabels(CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        fig.savefig(f.name, bbox_inches="tight")
        mlflow.log_artifact(f.name, artifact_path="confusion_matrices")
    plt.close(fig)
    os.unlink(f.name)


def evaluate_and_log(model, X_test, y_test, run_name, params):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, labels=[0, 1, 2])
    macro_f1 = np.mean(f1)
    _, _, weighted_f1, _ = precision_recall_fscore_support(
        y_test, preds, labels=[0, 1, 2], average="weighted"
    )
    kappa = cohen_kappa_score(y_test, preds)
    mae = ordinal_mae(y_test, preds)
    cm = confusion_matrix(y_test, preds, labels=[0, 1, 2])

    for i, cls in enumerate(CLASS_NAMES):
        mlflow.log_metrics({
            f"{cls}_precision": prec[i],
            f"{cls}_recall": rec[i],
            f"{cls}_f1": f1[i],
        })
    mlflow.log_metrics({
        "test_accuracy": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "cohen_kappa": kappa,
        "mae": mae,
    })

    log_confusion_matrix(cm, run_name, str(params))

    return {"accuracy": acc, "macro_f1": macro_f1, "weighted_f1": weighted_f1, "kappa": kappa, "mae": mae}


def load_or_build_features(df):
    if Path(FEATURE_CACHE).exists() and Path(LABEL_CACHE).exists():
        logger.info("Loading cached features from %s", FEATURE_CACHE)
        X = np.load(FEATURE_CACHE)
        y = np.load(LABEL_CACHE)
        if len(X) == len(df):
            return X, y
        logger.warning("Cache size mismatch, rebuilding features")
    logger.info("Building features from SMILES (this may take a few minutes)")
    y = df["severity_label"].values
    X = build_features(df)
    np.save(FEATURE_CACHE, X)
    np.save(LABEL_CACHE, y)
    logger.info("Features cached to %s and %s", FEATURE_CACHE, LABEL_CACHE)
    return X, y


def register_best_model(run_id, family, macro_f1):
    model_uri = f"runs:/{run_id}/model"
    try:
        result = mlflow.register_model(model_uri, REGISTRY_NAME)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTRY_NAME, "production", result.version)
        logger.info("Registered %s as version %s of '%s' (macro_f1=%.4f)", family, result.version, REGISTRY_NAME, macro_f1)
    except Exception as e:
        logger.warning("Model registration failed (MLflow registry may be local-only): %s", e)


def main():
    os.makedirs("models", exist_ok=True)
    os.makedirs("logs", exist_ok=True)

    mlflow.set_experiment(EXPERIMENT_NAME)

    df = pd.read_csv(DATA_PATH)
    logger.info("Loaded %d rows from %s", len(df), DATA_PATH)

    X, y = load_or_build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info("Train/val split: %d train, %d test", len(X_train), len(X_test))

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_s)
    X_test_pca = pca.transform(X_test_s)

    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(pca, "models/pca.joblib")
    logger.info("Scaler and PCA saved to models/")

    all_results = []
    best_across_all = {"macro_f1": -1, "name": None, "family": None, "params": None, "model": None, "run_id": None}

    for family, cfg in MODEL_GRIDS.items():
        family_best = {"macro_f1": -1, "params": None, "model": None, "run_id": None}

        for params in cfg["params"]:
            param_suffix = "_".join(f"{k}-{v}" for k, v in params.items() if k not in ("objective", "num_class"))
            run_name = f"{family}_{param_suffix}"
            logger.info("Training %s", run_name)

            with mlflow.start_run(run_name=run_name) as run:
                mlflow.set_tag("model_family", family)

                if family == "XGBoost":
                    mlflow.xgboost.autolog(silent=True)
                else:
                    mlflow.sklearn.autolog(silent=True)

                mlflow.log_params({
                    "model_family": family,
                    "n_bits": N_BITS,
                    "n_pca": N_PCA,
                    "n_features_raw": X.shape[1],
                    "pca_explained_var": pca.explained_variance_ratio_.sum(),
                })
                for k, v in params.items():
                    mlflow.log_param(k, v)

                if family == "KNN":
                    model = cfg["model"](**params)
                else:
                    model = cfg["model"](**params, random_state=RANDOM_STATE)
                model.fit(X_train_pca, y_train)

                mlflow.log_artifact("models/scaler.joblib", artifact_path="preprocessing")
                mlflow.log_artifact("models/pca.joblib", artifact_path="preprocessing")

                cv_scores = cross_val_score(model, X_train_pca, y_train, cv=3)
                mlflow.log_metric("cv_mean", cv_scores.mean())
                mlflow.log_metric("cv_std", cv_scores.std())

                metrics = evaluate_and_log(model, X_test_pca, y_test, family, params)

                signature = mlflow.models.infer_signature(X_test_pca, model.predict(X_test_pca[:5]))
                if family == "XGBoost":
                    mlflow.xgboost.log_model(model, artifact_path="model", signature=signature)
                else:
                    mlflow.sklearn.log_model(model, artifact_path="model", signature=signature)

                all_results.append((run_name, family, params, metrics, model))

                logger.info("  %s macro_f1=%.4f kappa=%.4f mae=%.4f", run_name, metrics["macro_f1"], metrics["kappa"], metrics["mae"])

                if metrics["macro_f1"] > family_best["macro_f1"]:
                    family_best = {"macro_f1": metrics["macro_f1"], "params": params, "model": model, "run_id": run.info.run_id}

                if metrics["macro_f1"] > best_across_all["macro_f1"]:
                    best_across_all = {
                        "macro_f1": metrics["macro_f1"],
                        "name": run_name,
                        "family": family,
                        "params": params,
                        "model": model,
                        "run_id": run.info.run_id,
                    }

        with mlflow.start_run(run_name=f"best_{family}"):
            mlflow.set_tag("model_family", family)
            mlflow.set_tag("best_of_family", "true")
            mlflow.log_params(family_best["params"])
            mlflow.log_metric("best_macro_f1", family_best["macro_f1"])
            if family == "XGBoost":
                mlflow.xgboost.log_model(family_best["model"], artifact_path="best_model")
            else:
                mlflow.sklearn.log_model(family_best["model"], artifact_path="best_model")

    if best_across_all["model"] is not None:
        logger.info("Best overall: %s (macro_f1=%.4f)", best_across_all["name"], best_across_all["macro_f1"])
        with mlflow.start_run(run_name="best_overall"):
            mlflow.set_tag("best_overall", "true")
            mlflow.log_params(best_across_all["params"])
            mlflow.log_metric("best_macro_f1", best_across_all["macro_f1"])
            mlflow.log_param("best_model_name", best_across_all["name"])
            if best_across_all["family"] == "XGBoost":
                mlflow.xgboost.log_model(best_across_all["model"], artifact_path="best_model")
            else:
                mlflow.sklearn.log_model(best_across_all["model"], artifact_path="best_model")

        register_best_model(best_across_all["run_id"], best_across_all["family"], best_across_all["macro_f1"])

    results_summary = sorted(
        [
            {"run_name": r[0], "family": r[1], **r[3]}
            for r in all_results
        ],
        key=lambda x: -x["macro_f1"],
    )

    logger.info("--- Results (sorted by macro F1) ---")
    for r in results_summary:
        logger.info("%-45s  acc=%.4f  macro_f1=%.4f  kappa=%.4f  mae=%.4f", r["run_name"], r["accuracy"], r["macro_f1"], r["kappa"], r["mae"])

    with open("results.json", "w") as f:
        json.dump(results_summary, f, indent=2)
    logger.info("Results saved to results.json")


if __name__ == "__main__":
    main()
