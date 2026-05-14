"""Tests for monitor.py"""

import numpy as np
import pandas as pd
import pytest
from monitor import (
    population_stability_index,
    detect_feature_drift,
    check_prediction_drift,
)


# ── population_stability_index ────────────────────────────────────────────────

def test_psi_identical_distributions_near_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 1000)
    psi = population_stability_index(x, x.copy())
    assert psi < 0.01


def test_psi_very_different_distributions_large():
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, 1000)
    x2 = rng.normal(5, 1, 1000)   # shifted mean
    psi = population_stability_index(x1, x2)
    assert psi > 0.25


def test_psi_moderate_shift():
    rng = np.random.default_rng(0)
    x1 = rng.normal(0, 1, 1000)
    x2 = rng.normal(0.5, 1, 1000)
    psi = population_stability_index(x1, x2)
    assert 0.0 < psi < 1.0


def test_psi_non_negative():
    rng = np.random.default_rng(3)
    for _ in range(10):
        x1 = rng.normal(0, 1, 500)
        x2 = rng.normal(rng.uniform(-2, 2), 1, 500)
        assert population_stability_index(x1, x2) >= 0.0


def test_psi_constant_feature_returns_zero():
    x1 = np.ones(100)
    x2 = np.ones(100)
    assert population_stability_index(x1, x2) == 0.0


def test_psi_binary_identical():
    rng = np.random.default_rng(5)
    x = rng.integers(0, 2, 500).astype(float)
    psi = population_stability_index(x, x.copy())
    assert psi < 0.01


# ── detect_feature_drift ──────────────────────────────────────────────────────

def _make_df(rng, means, n=200):
    cols = {f"f{i}": rng.normal(m, 1, n) for i, m in enumerate(means)}
    return pd.DataFrame(cols)


def test_detect_drift_returns_dataframe():
    rng = np.random.default_rng(10)
    X_train = _make_df(rng, [0, 0, 0])
    X_new   = _make_df(rng, [0, 0, 0])
    result  = detect_feature_drift(X_train, X_new)
    assert isinstance(result, pd.DataFrame)


def test_detect_drift_required_columns():
    rng = np.random.default_rng(10)
    X_train = _make_df(rng, [0, 0])
    X_new   = _make_df(rng, [0, 0])
    result  = detect_feature_drift(X_train, X_new)
    assert "feature" in result.columns
    assert "psi" in result.columns
    assert "status" in result.columns


def test_detect_drift_stable_when_same_distribution():
    rng = np.random.default_rng(10)
    X_train = _make_df(rng, [0, 0, 0], n=1000)
    X_new   = _make_df(rng, [0, 0, 0], n=500)
    result  = detect_feature_drift(X_train, X_new)
    assert (result["status"] == "stable").all()


def test_detect_drift_flags_shifted_feature():
    rng = np.random.default_rng(10)
    X_train = pd.DataFrame({"normal": rng.normal(0, 1, 1000),
                             "shifted": rng.normal(0, 1, 1000)})
    X_new   = pd.DataFrame({"normal": rng.normal(0, 1, 500),
                             "shifted": rng.normal(8, 1, 500)})  # large shift
    result  = detect_feature_drift(X_train, X_new)
    shifted_row = result[result["feature"] == "shifted"]
    assert shifted_row["status"].iloc[0] == "retrain"


def test_detect_drift_sorted_by_psi_desc():
    rng = np.random.default_rng(10)
    X_train = _make_df(rng, [0, 0, 0])
    X_new   = _make_df(rng, [0, 1, 5])
    result  = detect_feature_drift(X_train, X_new)
    psi_vals = result["psi"].tolist()
    assert psi_vals == sorted(psi_vals, reverse=True)


def test_detect_drift_status_values_valid():
    rng = np.random.default_rng(10)
    X_train = _make_df(rng, [0, 0, 0])
    X_new   = _make_df(rng, [0, 1, 5])
    result  = detect_feature_drift(X_train, X_new)
    valid = {"stable", "monitor", "retrain"}
    assert set(result["status"].unique()).issubset(valid)


# ── check_prediction_drift ────────────────────────────────────────────────────

class _DummyModel:
    def __init__(self, pred_prob):
        self._prob = pred_prob
    def predict_proba(self, X):
        return np.column_stack([
            1 - np.full(len(X), self._prob),
            np.full(len(X), self._prob),
        ])


def test_prediction_drift_stable_when_same():
    model = _DummyModel(0.22)
    X = pd.DataFrame({"f": np.zeros(100)})
    result = check_prediction_drift(model, X, expected_churn_rate=0.22, tolerance=0.05)
    assert result["prediction_drift"] is False
    assert result["status"] == "stable"


def test_prediction_drift_flagged_when_large():
    model = _DummyModel(0.50)
    X = pd.DataFrame({"f": np.zeros(100)})
    result = check_prediction_drift(model, X, expected_churn_rate=0.22, tolerance=0.05)
    assert result["prediction_drift"] is True
    assert result["status"] == "DRIFT DETECTED"


def test_prediction_drift_returns_required_keys():
    model = _DummyModel(0.22)
    X = pd.DataFrame({"f": np.zeros(100)})
    result = check_prediction_drift(model, X, 0.22)
    for k in ("expected_churn_rate", "observed_churn_rate", "delta",
              "tolerance", "prediction_drift", "status"):
        assert k in result


def test_prediction_drift_delta_calculation():
    model = _DummyModel(0.30)
    X = pd.DataFrame({"f": np.zeros(50)})
    result = check_prediction_drift(model, X, expected_churn_rate=0.20, tolerance=0.05)
    assert result["delta"] == pytest.approx(0.10, abs=0.001)
