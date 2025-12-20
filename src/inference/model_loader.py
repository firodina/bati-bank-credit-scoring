# ========================================================
# src/inference/model_loader.py
# ========================================================

# import mlflow.sklearn
# import logging

# logger = logging.getLogger(__name__)

# # Global variable to hold the loaded model
# model_pipeline = None


# def load_model():
#     """
#     Load the latest registered version of the MLflow model.
#     Uses MLflow Model Registry to fetch the model automatically.
#     """
#     global model_pipeline
#     if model_pipeline is None:
#         try:
#             # Load the latest version from the MLflow registry
#             model_pipeline = mlflow.sklearn.load_model(
#                 "models:/credit_risk_pipeline/latest")
#             logger.info("✅ Model loaded successfully")
#         except Exception as e:
#             logger.error(f"Failed to load model: {e}")
#             raise e


# def get_model():
#     """
#     Returns the loaded model. Load it first if not already loaded.
#     """
#     global model_pipeline
#     if model_pipeline is None:
#         load_model()
#     return model_pipeline


# ========================================================
# src/inference/model_loader.py
# ========================================================

import mlflow.sklearn
import logging
import os
import joblib

logger = logging.getLogger(__name__)

# Global variable to hold the loaded model
model_pipeline = None


def load_model():
    global model_pipeline
    if model_pipeline is None:
        try:
            model_path = os.path.join(
                "models", "credit_risk_pipeline_20251218_150604.pkl")
            model_pipeline = joblib.load(model_path)
            logger.info("Model loaded successfully from local .pkl")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise e


def get_model():
    """
    Returns the loaded model. Load it first if not already loaded.
    """
    global model_pipeline
    if model_pipeline is None:
        load_model()
    return model_pipeline
