"""Pydantic request / response models for the churn prediction API."""

from pydantic import BaseModel, Field


class CustomerFeatures(BaseModel):
    gender: str = Field(..., examples=["Male"])
    SeniorCitizen: int = Field(..., ge=0, le=1, examples=[0])
    Partner: str = Field(..., examples=["Yes"])
    Dependents: str = Field(..., examples=["No"])
    tenure: int = Field(..., ge=0, examples=[12])
    PhoneService: str = Field(..., examples=["Yes"])
    MultipleLines: str = Field(..., examples=["No"])
    InternetService: str = Field(..., examples=["DSL"])
    OnlineSecurity: str = Field(..., examples=["No"])
    OnlineBackup: str = Field(..., examples=["Yes"])
    DeviceProtection: str = Field(..., examples=["No"])
    TechSupport: str = Field(..., examples=["No"])
    StreamingTV: str = Field(..., examples=["No"])
    StreamingMovies: str = Field(..., examples=["No"])
    Contract: str = Field(..., examples=["Month-to-month"])
    PaperlessBilling: str = Field(..., examples=["Yes"])
    PaymentMethod: str = Field(..., examples=["Electronic check"])
    MonthlyCharges: float = Field(..., ge=0, examples=[65.50])
    TotalCharges: float = Field(..., ge=0, examples=[786.0])

    model_config = {"json_schema_extra": {"example": {
        "gender": "Male", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
        "tenure": 12, "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "Fiber optic", "OnlineSecurity": "No",
        "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "Yes", "StreamingMovies": "Yes",
        "Contract": "Month-to-month", "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 89.10, "TotalCharges": 1069.2,
    }}}


class BatchRequest(BaseModel):
    customers: list[CustomerFeatures] = Field(
        ...,
        min_length=1,
        max_length=500,
        description="Customers to score in one request (1-500).",
    )


class PredictionResponse(BaseModel):
    churn_probability: float = Field(..., description="P(churn) from 0–1")
    churn_prediction: bool   = Field(..., description="True if probability >= threshold")
    threshold: float         = Field(..., description="Decision threshold used")
    top_factors: list[dict]  = Field(..., description="Top SHAP features driving this prediction")


class ExplanationResponse(BaseModel):
    churn_probability: float = Field(..., description="P(churn) from 0-1")
    threshold: float = Field(..., description="Decision threshold used")
    shap_values: list[dict] = Field(..., description="Per-feature SHAP values sorted by impact")


class BatchResponse(BaseModel):
    predictions: list[PredictionResponse] = Field(
        ...,
        description="Predictions returned in the same order as the request.",
    )


class HealthResponse(BaseModel):
    status: str
    model_name: str
    n_features: int
