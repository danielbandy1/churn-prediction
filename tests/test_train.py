"""Tests for src/train.py"""

import pathlib
import tempfile

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, get_X_y
from src.train import compare_models, train_best_model, save_model, load_model


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def small_dataset():
    """200-row synthetic dataset with real feature pipeline."""
    rng = np.random.default_rng(0)
    n = 200
    rows = []
    for i in range(n):
        tenure = int(rng.integers(1, 72))
        monthly = float(rng.uniform(20, 110))
        rows.append({
            "customerID": f"T{i:04d}",
            "gender": rng.choice(["Male", "Female"]),
            "SeniorCitizen": int(rng.integers(0, 2)),
            "Partner": rng.choice(["Yes", "No"]),
            "Dependents": rng.choice(["Yes", "No"]),
            "tenure": tenure,
            "PhoneService": "Yes",
            "MultipleLines": rng.choice(["Yes", "No"]),
            "InternetService": rng.choice(["DSL", "Fiber optic", "No"]),
            "OnlineSecurity": rng.choice(["Yes", "No"]),
            "OnlineBackup": rng.choice(["Yes", "No"]),
            "DeviceProtection": rng.choice(["Yes", "No"]),
            "TechSupport": rng.choice(["Yes", "No"]),
            "StreamingTV": rng.choice(["Yes", "No"]),
            "StreamingMovies": rng.choice(["Yes", "No"]),
            "Contract": rng.choice(["Month-to-month", "One year", "Two year"]),
            "PaperlessBilling": rng.choice(["Yes", "No"]),
            "PaymentMethod": rng.choice([
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ]),
            "MonthlyCharges": monthly,
            "TotalCharges": str(round(monthly * tenure, 2)),
            "Churn": rng.choice(["Yes", "No"], p=[0.27, 0.73]),
        })
    df_raw = pd.DataFrame(rows)
    df = build_features(df_raw, fit=True)
    X, y = get_X_y(df)
    # Train/test split (manual, no sklearn import needed for fixture)
    split = int(len(X) * 0.8)
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


# ── compare_models ─────────────────────────────────────────────────────────────

def test_compare_models_returns_dataframe(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    result = compare_models(X, y, cv_folds=2)
    assert isinstance(result, pd.DataFrame)
    assert "Model" in result.columns
    assert "AUC (mean)" in result.columns
    assert len(result) == 4


def test_compare_models_sorted_by_auc(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    X = pd.concat([X_train, X_test])
    y = pd.concat([y_train, y_test])
    result = compare_models(X, y, cv_folds=2)
    aucs = result["AUC (mean)"].tolist()
    assert aucs == sorted(aucs, reverse=True)


# ── train_best_model ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("model_name", ["Logistic Regression", "Random Forest", "XGBoost"])
def test_train_best_model_runs(small_dataset, model_name):
    X_train, X_test, y_train, y_test = small_dataset
    result = train_best_model(X_train, y_train, X_test, y_test, model_name=model_name)
    assert "model" in result
    assert "auc" in result
    assert 0.0 <= result["auc"] <= 1.0


def test_train_best_model_metrics_present(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    result = train_best_model(X_train, y_train, X_test, y_test, model_name="XGBoost")
    for key in ("auc", "f1", "precision", "recall", "roc_curve", "pr_curve", "confusion_matrix"):
        assert key in result


def test_train_best_model_predictions_shape(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    result = train_best_model(X_train, y_train, X_test, y_test, model_name="XGBoost")
    assert len(result["y_prob"]) == len(X_test)
    assert len(result["y_pred"]) == len(X_test)


def test_train_best_model_roc_curve_shape(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    result = train_best_model(X_train, y_train, X_test, y_test, model_name="XGBoost")
    fpr, tpr = result["roc_curve"]
    assert len(fpr) == len(tpr)
    assert len(fpr) >= 2


def test_train_best_model_confusion_matrix_shape(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    result = train_best_model(X_train, y_train, X_test, y_test, model_name="XGBoost")
    cm = result["confusion_matrix"]
    assert cm.shape == (2, 2)
    assert cm.sum() == len(X_test)


# ── save / load model ─────────────────────────────────────────────────────────

def test_save_load_roundtrip(small_dataset):
    X_train, X_test, y_train, y_test = small_dataset
    result = train_best_model(X_train, y_train, X_test, y_test, model_name="XGBoost")
    feature_names = list(X_train.columns)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = pathlib.Path(tmpdir) / "model.joblib"
        saved_path = save_model(result["model"], feature_names, path=path)
        assert saved_path.exists()

        model_loaded, names_loaded = load_model(path)
        assert names_loaded == feature_names

        probs = model_loaded.predict_proba(X_test)[:, 1]
        assert len(probs) == len(X_test)
        assert all(0.0 <= p <= 1.0 for p in probs)
