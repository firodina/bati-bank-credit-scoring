from pydantic import BaseModel, Field, ConfigDict


class CreditFeatures(BaseModel):
    CountryCode: int
    Amount: float
    Value: float
    PricingStrategy: int
    CurrencyCode_UGX: int

    # # ---------------------------
    # # Categorical (RAW)
    # # ---------------------------
    # ProviderId: int
    # ProductId: int
    # ChannelId: int

    # # ---------------------------
    # # Allowed leakage column (used in training)
    # # ---------------------------
    # FraudResult: int


class PredictionResponse(BaseModel):
    default_probability: float = Field(..., example=0.73)
    risk_label: str = Field(..., example="high")


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_loaded: bool = Field(..., example=True)
