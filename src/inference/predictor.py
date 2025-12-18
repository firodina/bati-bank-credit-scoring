import pandas as pd
from src.inference.model_loader import model_pipeline


def predict(features: dict):
    if model_pipeline is None:
        raise ValueError("Model not loaded")

    df = pd.DataFrame([features])
    proba = model_pipeline.predict_proba(df)[:, 1][0]
    label = "high" if proba > 0.5 else "low"
    return {"default_probability": float(proba), "risk_label": label}
