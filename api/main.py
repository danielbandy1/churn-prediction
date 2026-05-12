"""
FastAPI churn prediction service.

Endpoints:
  GET  /health   — liveness + model info
  POST /predict  — single-customer churn probability with SHAP explanation
"""

import pathlib
import sys
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException

# Allow running from repo root or api/ directory
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.features import build_features
from src.train import load_model
from src.explain import local_explanation
from api.schema import CustomerFeatures, PredictionResponse, HealthResponse

MODEL_PATH = pathlib.Path(__file__).parent.parent / "models" / "churn_model.joblib"
THRESHOLD = 0.40   # lower threshold → catch more churners at cost of precision

_model = None
_feature_names: list[str] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _feature_names
    if not MODEL_PATH.exists():
        raise RuntimeError(f"Model not found at {MODEL_PATH}. Run train_pipeline.py first.")
    _model, _feature_names = load_model(MODEL_PATH)
    yield
    _model = None
    _feature_names = []


app = FastAPI(
    title="Churn Prediction API",
    description="Real-time customer churn scoring with SHAP explanations.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_input_df(customer: CustomerFeatures) -> pd.DataFrame:
    row = customer.model_dump()
    df_raw = pd.DataFrame([row])
    df_feat = build_features(df_raw, fit=False)
    # Align columns to training schema
    for col in _feature_names:
        if col not in df_feat.columns:
            df_feat[col] = 0
    return df_feat[_feature_names]


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return HealthResponse(
        status="ok",
        model_name=type(_model).__name__,
        n_features=len(_feature_names),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(customer: CustomerFeatures):
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    X = _build_input_df(customer)

    prob = float(_model.predict_proba(X)[0, 1])
    prediction = prob >= THRESHOLD

    try:
        explanation = local_explanation(_model, X, _feature_names)
        top_factors = [
            {"feature": row["feature"], "value": float(row["value"]), "shap": round(float(row["shap_value"]), 4)}
            for _, row in explanation.head(5).iterrows()
        ]
    except Exception:
        top_factors = []

    return PredictionResponse(
        churn_probability=round(prob, 4),
        churn_prediction=prediction,
        threshold=THRESHOLD,
        top_factors=top_factors,
    )
