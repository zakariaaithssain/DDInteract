import mlflow

from src.config import REGISTRY_NAME
from src.logger import logger


def register_best_model(run_id: str, family: str, macro_f1: float) -> None:
    model_uri = f"runs:/{run_id}/model"
    try:
        result = mlflow.register_model(model_uri, REGISTRY_NAME)
        client = mlflow.MlflowClient()
        client.set_registered_model_alias(REGISTRY_NAME, "production", result.version)
        logger.info(
            "Registered %s as version %s of '%s' (macro_f1=%.4f)", family, result.version, REGISTRY_NAME, macro_f1
        )
    except Exception as e:
        logger.warning("Model registration failed (MLflow registry may be local-only): %s", e)
