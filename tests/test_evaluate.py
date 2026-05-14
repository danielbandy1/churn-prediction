"""Tests for src/evaluate.py"""

import numpy as np
import pandas as pd
import pytest
from src.evaluate import (
    calibration_analysis,
    plot_calibration,
    tenure_bucket,
    cohort_metrics,
    contract_cohort_analysis,
    tenure_cohort_analysis,
    plot_cohort_analysis,
    _decode_onehot,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

RNG = np.random.default_rng(11)
N   = 300

Y_TRUE = RNG.integers(0, 2, N)
Y_PROB = np.clip(Y_TRUE * 0.6 + RNG.normal(0, 0.25, N), 0, 1)
Y_PRED = (Y_PROB >= 0.5).astype(int)

# Simulate encoded features with one-hot contract columns
CONTRACT_VALUES = RNG.choice(
    ["Month-to-month", "One year", "Two year"], N, p=[0.55, 0.21, 0.24]
)
TENURE_VALUES = pd.Series(RNG.integers(1, 73, N))

_OH_CONTRACTS = pd.get_dummies(
    pd.Series(CONTRACT_VALUES), prefix="Contract"
).astype(int)

X_FEATURES = pd.DataFrame({
    "tenure": TENURE_VALUES,
    **_OH_CONTRACTS,
})


# ── calibration_analysis ──────────────────────────────────────────────────────

def test_calibration_analysis_required_keys():
    cal = calibration_analysis(Y_TRUE, Y_PROB)
    for key in ("brier_score", "fraction_of_positives", "mean_predicted_value",
                "calibration_slope", "calibration_intercept"):
        assert key in cal


def test_calibration_brier_in_range():
    cal = calibration_analysis(Y_TRUE, Y_PROB)
    assert 0.0 <= cal["brier_score"] <= 1.0


def test_calibration_arrays_same_length():
    cal = calibration_analysis(Y_TRUE, Y_PROB)
    assert len(cal["fraction_of_positives"]) == len(cal["mean_predicted_value"])


def test_calibration_perfect_model_low_brier():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.02, 0.03, 0.02, 0.97, 0.96, 0.98])
    cal = calibration_analysis(y, p, n_bins=4)
    assert cal["brier_score"] < 0.05


def test_calibration_random_worse_than_good():
    rng = np.random.default_rng(7)
    y = np.tile([0, 1], 50)
    p_good   = np.where(y == 1, 0.85, 0.15).astype(float)
    p_random = np.clip(rng.normal(0.5, 0.3, len(y)), 0, 1)
    assert calibration_analysis(y, p_good)["brier_score"] < calibration_analysis(y, p_random)["brier_score"]


def test_calibration_slope_near_one_for_good_model():
    y = np.tile([0, 1], 100)
    p = np.where(y == 1, 0.80, 0.20).astype(float)
    cal = calibration_analysis(y, p, n_bins=5)
    assert 0.5 < cal["calibration_slope"] < 2.0


def test_plot_calibration_returns_figure():
    fig = plot_calibration(Y_TRUE, Y_PROB)
    assert fig is not None


def test_plot_calibration_has_two_axes():
    import matplotlib.pyplot as plt
    fig = plot_calibration(Y_TRUE, Y_PROB)
    assert len(fig.axes) == 2


# ── tenure_bucket ─────────────────────────────────────────────────────────────

def test_tenure_bucket_new():
    s = pd.Series([1, 3, 6])
    result = tenure_bucket(s)
    assert str(result.iloc[0]) == "New (0–6 mo)"
    assert str(result.iloc[2]) == "New (0–6 mo)"


def test_tenure_bucket_loyal():
    s = pd.Series([60, 72])
    result = tenure_bucket(s)
    assert all(str(v) == "Loyal (48+ mo)" for v in result)


def test_tenure_bucket_no_nulls():
    s = pd.Series(range(1, 73))
    assert tenure_bucket(s).isna().sum() == 0


def test_tenure_bucket_four_categories():
    s = pd.Series([1, 12, 36, 60])
    cats = tenure_bucket(s).unique()
    assert len(cats) == 4


# ── cohort_metrics ────────────────────────────────────────────────────────────

def test_cohort_metrics_returns_dataframe():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    assert isinstance(result, pd.DataFrame)


def test_cohort_metrics_required_columns():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    for col in ("cohort", "n", "actual_churn_rate", "precision", "recall", "f1"):
        assert col in result.columns


def test_cohort_metrics_one_row_per_group():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    assert len(result) == 3


def test_cohort_metrics_precision_in_range():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    assert (result["precision"] >= 0).all() and (result["precision"] <= 1).all()


def test_cohort_metrics_recall_in_range():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    assert (result["recall"] >= 0).all() and (result["recall"] <= 1).all()


def test_cohort_metrics_n_sums_to_total():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    assert result["n"].sum() == N


def test_cohort_metrics_sorted_by_recall_desc():
    groups = pd.Series(CONTRACT_VALUES)
    result = cohort_metrics(Y_TRUE, Y_PRED, Y_PROB, groups)
    recalls = result["recall"].tolist()
    assert recalls == sorted(recalls, reverse=True)


# ── _decode_onehot ────────────────────────────────────────────────────────────

def test_decode_onehot_returns_series():
    result = _decode_onehot(X_FEATURES, "Contract")
    assert isinstance(result, pd.Series)


def test_decode_onehot_correct_values():
    result = _decode_onehot(X_FEATURES, "Contract")
    unique_vals = set(result.unique())
    assert unique_vals.issubset({"Month-to-month", "One year", "Two year"})


def test_decode_onehot_missing_prefix_returns_unknown():
    result = _decode_onehot(X_FEATURES, "PaymentMethod")
    assert (result == "Unknown").all()


# ── contract / tenure analysis ────────────────────────────────────────────────

def test_contract_cohort_analysis_runs():
    result = contract_cohort_analysis(Y_TRUE, Y_PRED, Y_PROB, X_FEATURES)
    assert len(result) > 0


def test_tenure_cohort_analysis_runs():
    result = tenure_cohort_analysis(Y_TRUE, Y_PRED, Y_PROB, X_FEATURES)
    assert len(result) > 0


def test_tenure_cohort_rows_cover_population():
    result = tenure_cohort_analysis(Y_TRUE, Y_PRED, Y_PROB, X_FEATURES)
    assert result["n"].sum() == N


# ── plot (smoke test) ─────────────────────────────────────────────────────────

def test_plot_cohort_analysis_runs():
    contract_df = contract_cohort_analysis(Y_TRUE, Y_PRED, Y_PROB, X_FEATURES)
    tenure_df   = tenure_cohort_analysis(Y_TRUE, Y_PRED, Y_PROB, X_FEATURES)
    fig = plot_cohort_analysis(contract_df, tenure_df)
    assert fig is not None
