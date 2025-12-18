from fastapi import FastAPI
from src.inference.model_loader import get_model
from src.api.pydantic_models import CreditFeatures, PredictionResponse
import pandas as pd

app = FastAPI(title="Credit Risk Prediction API", version="1.0.0")

# Endpoint for health check


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict_risk(features: CreditFeatures):
    model = get_model()

    # Convert to DataFrame so the pipeline can process
    input_df = pd.DataFrame([features.model_dump()])

    # Use full pipeline
    proba = model.predict_proba(input_df)[:, 1][0]
    label = "high" if proba > 0.5 else "low"

    return PredictionResponse(default_probability=proba, risk_label=label)
