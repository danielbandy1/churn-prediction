"""Tests for src/business.py"""

import numpy as np
import pytest
from src.business import (
    expected_value_at_threshold,
    threshold_sweep,
    top_n_simulation,
    plot_expected_value_curve,
    plot_cumulative_gains,
)

RNG = np.random.default_rng(7)
N   = 500
Y_PROB  = RNG.uniform(0, 1, N)
Y_TRUE  = (Y_PROB + RNG.normal(0, 0.25, N) > 0.6).astype(int)
LTV     = 600.0
SAVE    = 0.30
COST    = 15.0


# ── expected_value_at_threshold ───────────────────────────────────────────────

def test_ev_returns_required_keys():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, SAVE, COST)
    for k in ("threshold", "n_contacted", "true_positives", "false_positives",
              "precision", "recall", "revenue_saved", "total_cost", "net_value"):
        assert k in result


def test_ev_precision_in_range():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, SAVE, COST)
    assert 0.0 <= result["precision"] <= 1.0


def test_ev_recall_in_range():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, SAVE, COST)
    assert 0.0 <= result["recall"] <= 1.0


def test_ev_high_threshold_contacts_few():
    r1 = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.9, LTV, SAVE, COST)
    r2 = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.1, LTV, SAVE, COST)
    assert r1["n_contacted"] < r2["n_contacted"]


def test_ev_zero_threshold_contacts_all():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.0, LTV, SAVE, COST)
    assert result["n_contacted"] == N


def test_ev_one_threshold_contacts_none():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 1.0, LTV, SAVE, COST)
    assert result["n_contacted"] == 0


def test_ev_net_value_subtracts_cost():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, SAVE, COST)
    expected_net = result["revenue_saved"] - result["total_cost"]
    assert result["net_value"] == pytest.approx(expected_net, rel=1e-4)


def test_ev_no_contact_cost():
    result = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, SAVE, 0.0)
    assert result["total_cost"] == 0.0
    assert result["net_value"] == result["revenue_saved"]


def test_ev_higher_save_rate_higher_revenue():
    r1 = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, 0.10, COST)
    r2 = expected_value_at_threshold(Y_TRUE, Y_PROB, 0.5, LTV, 0.50, COST)
    assert r1["revenue_saved"] < r2["revenue_saved"]


# ── threshold_sweep ───────────────────────────────────────────────────────────

def test_sweep_returns_dataframe():
    import pandas as pd
    df = threshold_sweep(Y_TRUE, Y_PROB, LTV, SAVE, COST, n_steps=20)
    assert isinstance(df, pd.DataFrame)


def test_sweep_has_correct_columns():
    df = threshold_sweep(Y_TRUE, Y_PROB, LTV, SAVE, COST, n_steps=20)
    assert "threshold" in df.columns
    assert "net_value" in df.columns
    assert "precision" in df.columns
    assert "recall" in df.columns


def test_sweep_threshold_increases():
    df = threshold_sweep(Y_TRUE, Y_PROB, LTV, SAVE, COST, n_steps=20)
    thresholds = df["threshold"].tolist()
    assert thresholds == sorted(thresholds)


def test_sweep_has_n_steps_rows():
    df = threshold_sweep(Y_TRUE, Y_PROB, LTV, SAVE, COST, n_steps=50)
    assert len(df) == 50


# ── top_n_simulation ──────────────────────────────────────────────────────────

def test_top_n_returns_required_keys():
    result = top_n_simulation(Y_TRUE, Y_PROB, 100, LTV, SAVE, COST)
    for k in ("n_contacted", "true_positives", "false_positives",
              "precision_at_n", "recall_at_n", "revenue_saved",
              "contact_cost", "net_value", "random_baseline_net", "lift_over_random"):
        assert k in result


def test_top_n_contacts_exactly_n():
    result = top_n_simulation(Y_TRUE, Y_PROB, 100, LTV, SAVE, COST)
    assert result["n_contacted"] == 100


def test_top_n_lift_positive():
    result = top_n_simulation(Y_TRUE, Y_PROB, 50, LTV, SAVE, COST)
    assert result["lift_over_random"] >= 1.0


def test_top_n_clamps_to_dataset_size():
    result = top_n_simulation(Y_TRUE, Y_PROB, 99_999, LTV, SAVE, COST)
    assert result["n_contacted"] == N


def test_top_n_precision_at_n_in_range():
    result = top_n_simulation(Y_TRUE, Y_PROB, 100, LTV, SAVE, COST)
    assert 0.0 <= result["precision_at_n"] <= 1.0


def test_top_n_recall_at_n_in_range():
    result = top_n_simulation(Y_TRUE, Y_PROB, 100, LTV, SAVE, COST)
    assert 0.0 <= result["recall_at_n"] <= 1.0


# ── plots (smoke tests) ───────────────────────────────────────────────────────

def test_plot_expected_value_curve_runs():
    df = threshold_sweep(Y_TRUE, Y_PROB, LTV, SAVE, COST, n_steps=20)
    fig = plot_expected_value_curve(df)
    assert fig is not None


def test_plot_cumulative_gains_runs():
    fig = plot_cumulative_gains(Y_TRUE, Y_PROB)
    assert fig is not None
