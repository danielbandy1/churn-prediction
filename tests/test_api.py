"""Tests for the FastAPI prediction service."""

import pathlib
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

# Train and save a tiny model before the app loads
from src.features import build_features, get_X_y
from src.train import train_best_model, save_model

_MODEL_PATH = pathlib.Path(tempfile.mkdtemp()) / "churn_model.joblib"


def _make_correlated_dataset():
    """
    Generate synthetic churn data WITH real feature-churn correlations so
    that a Logistic Regression can reliably learn the right direction.
    Mirrors the logistic model from generate_data.py.
    """
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
    import generate_data as gd

    # Temporarily redirect output to a temp file
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        # Use the generate_data module's internals directly
        rng = np.random.default_rng(99)
        n = 1_000

        gender         = rng.choice(["Male", "Female"], n, p=[0.505, 0.495])
        senior_citizen = rng.binomial(1, 0.162, n)
        partner        = rng.choice(["Yes", "No"], n, p=[0.483, 0.517])
        dependents     = rng.choice(["Yes", "No"], n, p=[0.299, 0.701])
        tenure         = rng.integers(1, 73, n)
        phone_service  = rng.choice(["Yes", "No"], n, p=[0.903, 0.097])
        multiple_lines = np.where(phone_service == "No", "No phone service",
                                  rng.choice(["Yes", "No"], n, p=[0.535, 0.465]))
        internet_service = rng.choice(["DSL", "Fiber optic", "No"], n,
                                      p=[0.344, 0.439, 0.217])

        def addon(has_internet, p_yes=0.50):
            return np.where(has_internet == "No", "No internet service",
                            rng.choice(["Yes", "No"], n, p=[p_yes, 1 - p_yes]))

        online_security   = addon(internet_service, 0.287)
        online_backup     = addon(internet_service, 0.344)
        device_protection = addon(internet_service, 0.343)
        tech_support      = addon(internet_service, 0.289)
        streaming_tv      = addon(internet_service, 0.384)
        streaming_movies  = addon(internet_service, 0.389)
        contract          = rng.choice(["Month-to-month", "One year", "Two year"],
                                       n, p=[0.550, 0.210, 0.240])
        paperless         = rng.choice(["Yes", "No"], n, p=[0.592, 0.408])
        payment           = rng.choice(
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.335, 0.228, 0.219, 0.218])
        monthly_charges = np.round(np.where(
            internet_service == "No", rng.uniform(18, 30, n),
            np.where(internet_service == "DSL",
                     rng.uniform(25, 75, n), rng.uniform(50, 120, n))), 2)
        total_charges = np.round(monthly_charges * tenure * rng.uniform(0.95, 1.05, n), 2)

        log_odds = (
            -1.8
            + 1.4 * (contract == "Month-to-month").astype(float)
            + 0.5 * (contract == "One year").astype(float)
            - 0.04 * tenure
            + 0.02 * (monthly_charges - 65)
            + 0.5 * (internet_service == "Fiber optic").astype(float)
            + 0.4 * (online_security == "No").astype(float)
            + 0.3 * (tech_support == "No").astype(float)
            + 0.35 * (payment == "Electronic check").astype(float)
            + 0.15 * (paperless == "Yes").astype(float)
            + rng.normal(0, 0.5, n)
        )
        churn_prob = 1 / (1 + np.exp(-log_odds))
        churn = np.where(rng.uniform(0, 1, n) < churn_prob, "Yes", "No")

        df_raw = pd.DataFrame({
            "customerID":       [f"T{i:04d}" for i in range(n)],
            "gender":           gender, "SeniorCitizen": senior_citizen,
            "Partner":          partner, "Dependents": dependents,
            "tenure":           tenure,
            "PhoneService":     phone_service, "MultipleLines": multiple_lines,
            "InternetService":  internet_service,
            "OnlineSecurity":   online_security, "OnlineBackup": online_backup,
            "DeviceProtection": device_protection, "TechSupport": tech_support,
            "StreamingTV":      streaming_tv, "StreamingMovies": streaming_movies,
            "Contract":         contract, "PaperlessBilling": paperless,
            "PaymentMethod":    payment,
            "MonthlyCharges":   monthly_charges, "TotalCharges": total_charges,
            "Churn":            churn,
        })

    df = build_features(df_raw, fit=True)
    X, y = get_X_y(df)
    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# Build and save model once at collection time with correlated training data
X_tr, X_te, y_tr, y_te = _make_correlated_dataset()
_result = train_best_model(X_tr, y_tr, X_te, y_te, model_name="Logistic Regression")
save_model(_result["model"], list(X_tr.columns), path=_MODEL_PATH)

# Patch MODEL_PATH before the app is imported so lifespan uses our temp model
import api.main as _main_module
_main_module.MODEL_PATH = _MODEL_PATH

from api.main import app


@pytest.fixture(scope="module")
def client():
    """Session-scoped TestClient that triggers the lifespan startup."""
    with TestClient(app) as c:
        yield c


# ── Fixtures ──────────────────────────────────────────────────────────────────

VALID_PAYLOAD = {
    "gender": "Male", "SeniorCitizen": 0,
    "Partner": "No", "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No",
    "StreamingTV": "Yes", "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 89.10,
    "TotalCharges": 1069.2,
}


# ── /health ───────────────────────────────────────────────────────────────────

def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["n_features"] > 0


def test_health_returns_model_name(client):
    resp = client.get("/health")
    assert "model_name" in resp.json()


# ── /predict ──────────────────────────────────────────────────────────────────

def test_predict_returns_200(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    assert resp.status_code == 200


def test_predict_response_fields(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    data = resp.json()
    assert "churn_probability" in data
    assert "churn_prediction" in data
    assert "threshold" in data
    assert "top_factors" in data


def test_predict_probability_in_range(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    prob = resp.json()["churn_probability"]
    assert 0.0 <= prob <= 1.0


def test_predict_prediction_consistent_with_threshold(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    data = resp.json()
    expected = data["churn_probability"] >= data["threshold"]
    assert data["churn_prediction"] == expected


def test_predict_top_factors_not_empty(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    factors = resp.json()["top_factors"]
    assert isinstance(factors, list)


def test_predict_top_factors_have_keys(client):
    resp = client.post("/predict", json=VALID_PAYLOAD)
    factors = resp.json()["top_factors"]
    if factors:
        for f in factors:
            assert "feature" in f
            assert "shap" in f


def test_predict_low_churn_customer(client):
    """Extreme low-risk vs extreme high-risk should differ in direction."""
    low_risk = {
        **VALID_PAYLOAD,
        "tenure": 70, "Contract": "Two year",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes", "OnlineBackup": "Yes",
        "DeviceProtection": "Yes", "TechSupport": "Yes",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 30.0, "TotalCharges": 2100.0,
    }
    high_risk = {
        **VALID_PAYLOAD,
        "tenure": 1, "Contract": "Month-to-month",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No", "OnlineBackup": "No",
        "DeviceProtection": "No", "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 105.0, "TotalCharges": 105.0,
    }
    low_resp  = client.post("/predict", json=low_risk).json()
    high_resp = client.post("/predict", json=high_risk).json()
    assert low_resp["churn_probability"] < high_resp["churn_probability"]


def test_predict_missing_field_returns_422(client):
    bad = {k: v for k, v in VALID_PAYLOAD.items() if k != "tenure"}
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422


def test_predict_invalid_senior_citizen_returns_422(client):
    bad = VALID_PAYLOAD | {"SeniorCitizen": 5}   # must be 0 or 1
    resp = client.post("/predict", json=bad)
    assert resp.status_code == 422
