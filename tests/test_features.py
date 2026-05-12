"""Tests for src/features.py"""

import numpy as np
import pandas as pd
import pytest

from src.features import build_features, get_X_y, get_feature_names, ONEHOT_COLUMNS, _encode_binary


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_row(**overrides):
    base = {
        "customerID": "TID_0000001",
        "gender": "Male", "SeniorCitizen": 0,
        "Partner": "Yes", "Dependents": "No",
        "tenure": 12,
        "PhoneService": "Yes", "MultipleLines": "No",
        "InternetService": "DSL",
        "OnlineSecurity": "No", "OnlineBackup": "Yes",
        "DeviceProtection": "No", "TechSupport": "No",
        "StreamingTV": "No", "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 55.0, "TotalCharges": "660.0",
        "Churn": "No",
    }
    base.update(overrides)
    return base


@pytest.fixture()
def single_row_df():
    return pd.DataFrame([_make_row()])


@pytest.fixture()
def multi_row_df():
    rows = [
        _make_row(InternetService="DSL",         Contract="Month-to-month", tenure=1),
        _make_row(InternetService="Fiber optic",  Contract="One year",       tenure=24),
        _make_row(InternetService="No",           Contract="Two year",       tenure=60, Churn="Yes"),
    ]
    return pd.DataFrame(rows)


# ── _encode_binary ─────────────────────────────────────────────────────────────

def test_encode_binary_yes():
    s = pd.Series(["Yes", "No", "No phone service", "No internet service"])
    result = _encode_binary(s)
    assert list(result) == [1, 0, 0, 0]


def test_encode_binary_fillna():
    s = pd.Series(["Yes", None, "No"])
    result = _encode_binary(s)
    assert result.isna().sum() == 0


# ── build_features ─────────────────────────────────────────────────────────────

def test_drops_customer_id(single_row_df):
    df = build_features(single_row_df, fit=True)
    assert "customerID" not in df.columns


def test_total_charges_coercion():
    row = _make_row(TotalCharges=" ")   # new customer blank value
    df = build_features(pd.DataFrame([row]), fit=True)
    assert df["TotalCharges"].iloc[0] == 0.0


def test_churn_encoded(single_row_df):
    df = build_features(single_row_df, fit=True)
    assert df["Churn"].dtype in (int, np.int64, np.int32)
    assert df["Churn"].iloc[0] == 0


def test_churn_yes_encoded():
    row = _make_row(Churn="Yes")
    df = build_features(pd.DataFrame([row]), fit=True)
    assert df["Churn"].iloc[0] == 1


def test_gender_encoded(single_row_df):
    df = build_features(single_row_df, fit=True)
    assert df["gender"].iloc[0] == 1   # Male → 1


def test_gender_female():
    row = _make_row(gender="Female")
    df = build_features(pd.DataFrame([row]), fit=True)
    assert df["gender"].iloc[0] == 0


def test_onehot_contract_columns(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    assert any(c.startswith("Contract_") for c in df.columns)
    assert "Contract" not in df.columns


def test_onehot_internet_service(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    assert any(c.startswith("InternetService_") for c in df.columns)


def test_onehot_payment_method(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    assert any(c.startswith("PaymentMethod_") for c in df.columns)


def test_num_services(single_row_df):
    df = build_features(single_row_df, fit=True)
    assert "num_services" in df.columns
    assert df["num_services"].iloc[0] >= 0


def test_avg_monthly_charges(single_row_df):
    df = build_features(single_row_df, fit=True)
    assert "avg_monthly_charges" in df.columns
    assert df["avg_monthly_charges"].iloc[0] > 0


def test_avg_monthly_charges_zero_tenure():
    row = _make_row(tenure=0, MonthlyCharges=70.0, TotalCharges="0.0")
    df = build_features(pd.DataFrame([row]), fit=True)
    # np.where: tenure==0 → use MonthlyCharges
    assert df["avg_monthly_charges"].iloc[0] == pytest.approx(70.0)


def test_is_new_customer_flag(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    assert df["is_new_customer"].iloc[0] == 1   # tenure=1 < 6
    assert df["is_new_customer"].iloc[1] == 0   # tenure=24


def test_is_long_term_flag(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    assert df["is_long_term"].iloc[2] == 1   # tenure=60 >= 24
    assert df["is_long_term"].iloc[0] == 0   # tenure=1


def test_no_nans(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    assert df.isna().sum().sum() == 0


def test_fit_records_onehot_columns(multi_row_df):
    import src.features as feat_module
    build_features(multi_row_df, fit=True)
    assert len(feat_module.ONEHOT_COLUMNS) > 0


def test_inference_alignment(multi_row_df):
    """Inference row missing a dummy column should get it added as 0."""
    build_features(multi_row_df, fit=True)
    # Row with only DSL — Fiber optic column will be absent before alignment
    row = _make_row(InternetService="DSL", Contract="Month-to-month",
                    PaymentMethod="Mailed check")
    df_infer = build_features(pd.DataFrame([row]), fit=False)
    # All known one-hot columns should be present
    import src.features as feat_module
    for col in feat_module.ONEHOT_COLUMNS:
        assert col in df_infer.columns


# ── get_X_y ───────────────────────────────────────────────────────────────────

def test_get_X_y(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    X, y = get_X_y(df)
    assert "Churn" not in X.columns
    assert y.name == "Churn"
    assert len(X) == len(y)


def test_get_feature_names(multi_row_df):
    df = build_features(multi_row_df, fit=True)
    names = get_feature_names(df)
    assert "Churn" not in names
    assert len(names) == len(df.columns) - 1
