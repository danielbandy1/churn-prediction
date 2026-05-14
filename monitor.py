#!/usr/bin/env python3
"""
Data and prediction drift monitoring for the churn model.

Computes Population Stability Index (PSI) for each feature and flags
distributions that have shifted significantly since training. Use this
before deploying a new customer batch to determine whether the existing
model is still reliable or needs retraining.

PSI interpretation:
  < 0.10  — stable, no action needed
  0.10–0.25 — moderate shift, monitor closely
  > 0.25  — significant shift, retrain recommended

Usage:
    python monitor.py --new-data data/raw/new_customers.csv
    python monitor.py --new-data data/raw/new_customers.csv --model models/churn_model.joblib
    python monitor.py --new-data data/raw/new_customers.csv --output reports/drift_report.csv
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.features import build_features, get_feature_names
from src.train import load_model

TRAINING_DATA  = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
MODEL_PATH     = pathlib.Path("models/churn_model.joblib")
FIGURES_DIR    = pathlib.Path("figures")

# PSI thresholds
_PSI_STABLE   = 0.10
_PSI_MODERATE = 0.25


def population_stability_index(
    expected: np.ndarray,
    actual: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute PSI between a reference (training) and new distribution.
    Works for both numeric features (quantile bins) and low-cardinality
    integers (value-based bins).
    """
    expected = np.asarray(expected, dtype=float)
    actual   = np.asarray(actual,   dtype=float)

    # Use quantiles from expected to define bins
    quantiles = np.linspace(0, 100, n_bins + 1)
    breakpoints = np.unique(np.percentile(expected, quantiles))

    if len(breakpoints) < 2:
        # Degenerate feature (constant) — PSI is 0
        return 0.0

    e_counts, _ = np.histogram(expected, bins=breakpoints)
    a_counts, _ = np.histogram(actual,   bins=breakpoints)

    # Avoid zero counts (add small epsilon)
    e_pct = (e_counts + 1e-8) / (expected.size + 1e-8 * len(e_counts))
    a_pct = (a_counts + 1e-8) / (actual.size   + 1e-8 * len(a_counts))

    psi = float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))
    return round(psi, 6)


def detect_feature_drift(
    X_train: pd.DataFrame,
    X_new: pd.DataFrame,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Compute PSI for every feature shared between training and new data.

    Returns a DataFrame sorted by PSI descending with a `status` column:
        stable | monitor | retrain
    """
    shared = [c for c in X_train.columns if c in X_new.columns]
    rows = []
    for col in shared:
        psi = population_stability_index(
            X_train[col].dropna().values,
            X_new[col].dropna().values,
            n_bins=n_bins,
        )
        if psi < _PSI_STABLE:
            status = "stable"
        elif psi < _PSI_MODERATE:
            status = "monitor"
        else:
            status = "retrain"

        rows.append({"feature": col, "psi": psi, "status": status})

    return (
        pd.DataFrame(rows)
        .sort_values("psi", ascending=False)
        .reset_index(drop=True)
    )


def check_prediction_drift(
    model,
    X_new: pd.DataFrame,
    expected_churn_rate: float,
    tolerance: float = 0.05,
) -> dict:
    """
    Check whether the model's predicted churn rate on new data has shifted
    significantly compared to the training period.
    """
    probs      = model.predict_proba(X_new)[:, 1]
    new_rate   = float(probs.mean())
    delta      = abs(new_rate - expected_churn_rate)
    drifted    = delta > tolerance

    return {
        "expected_churn_rate": round(expected_churn_rate, 4),
        "observed_churn_rate": round(new_rate, 4),
        "delta":               round(delta, 4),
        "tolerance":           tolerance,
        "prediction_drift":    drifted,
        "status":              "DRIFT DETECTED" if drifted else "stable",
    }


def plot_drift_report(
    drift_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """Horizontal bar chart of PSI values, colour-coded by severity."""
    top = drift_df.head(20).iloc[::-1]   # show worst 20, ascending order for barh

    colors = top["status"].map(
        {"stable": "steelblue", "monitor": "goldenrod", "retrain": "crimson"}
    )

    fig, ax = plt.subplots(figsize=(9, max(4, len(top) * 0.38)))
    bars = ax.barh(top["feature"], top["psi"], color=colors, alpha=0.85)

    ax.axvline(_PSI_STABLE,   color="goldenrod", ls="--", lw=1.2, label="Monitor (0.10)")
    ax.axvline(_PSI_MODERATE, color="crimson",   ls="--", lw=1.2, label="Retrain (0.25)")

    ax.set_xlabel("Population Stability Index (PSI)")
    ax.set_title("Feature Drift Report")
    ax.legend(loc="lower right")

    # Add value labels
    for bar, val in zip(bars, top["psi"]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def run_monitoring_report(
    new_data_path: str | pathlib.Path,
    model_path: str | pathlib.Path | None = None,
    output_path: str | pathlib.Path | None = None,
    plot: bool = True,
) -> dict:
    """
    Full drift monitoring pipeline:
      1. Load training reference + new batch
      2. Apply feature pipeline to both
      3. Compute PSI for all features
      4. Check prediction distribution drift
      5. Print report and optionally save CSV + figure

    Returns a dict with keys: feature_drift (DataFrame), prediction_drift (dict).
    """
    model_path = model_path or MODEL_PATH

    # ── Load data ──────────────────────────────────────────────────────────────
    if not TRAINING_DATA.exists():
        print("Training reference data not found — generating synthetic data...")
        import generate_data
        generate_data.main()
    df_train_raw = pd.read_csv(TRAINING_DATA)

    df_new_raw = pd.read_csv(new_data_path)

    # ── Feature pipeline ───────────────────────────────────────────────────────
    df_train_feat = build_features(df_train_raw, fit=True)
    df_new_feat   = build_features(df_new_raw,   fit=False)

    feature_names = get_feature_names(df_train_feat)
    X_train = df_train_feat[feature_names]
    X_new   = df_new_feat[[c for c in feature_names if c in df_new_feat.columns]]

    # ── PSI per feature ────────────────────────────────────────────────────────
    drift_df = detect_feature_drift(X_train, X_new)

    # ── Prediction drift ───────────────────────────────────────────────────────
    model, _ = load_model(model_path)
    expected_churn_rate = float(
        (df_train_raw["Churn"] == "Yes").mean()
        if "Churn" in df_train_raw.columns else 0.226
    )
    pred_drift = check_prediction_drift(model, X_new, expected_churn_rate)

    # ── Report ─────────────────────────────────────────────────────────────────
    n_retrain = (drift_df["status"] == "retrain").sum()
    n_monitor = (drift_df["status"] == "monitor").sum()

    print("=" * 56)
    print("  CHURN MODEL DRIFT MONITORING REPORT")
    print("=" * 56)
    print(f"  Training reference : {len(X_train):,} customers")
    print(f"  New batch          : {len(X_new):,} customers")
    print(f"  Features checked   : {len(drift_df)}")
    print(f"  Retrain signals    : {n_retrain}  (PSI > {_PSI_MODERATE})")
    print(f"  Monitor signals    : {n_monitor}  (PSI {_PSI_STABLE}–{_PSI_MODERATE})")
    print()
    print("  Prediction drift check:")
    print(f"    Expected churn rate : {pred_drift['expected_churn_rate']:.1%}")
    print(f"    Observed churn rate : {pred_drift['observed_churn_rate']:.1%}")
    print(f"    Status              : {pred_drift['status']}")
    print()

    if n_retrain > 0:
        print("  Top drifted features:")
        top_drift = drift_df[drift_df["status"] == "retrain"].head(5)
        for _, row in top_drift.iterrows():
            print(f"    {row['feature']:<35} PSI = {row['psi']:.4f}  [{row['status']}]")
        print()
        print("  *** RECOMMENDATION: Consider retraining the model. ***")
    elif n_monitor > 0:
        print("  Some features show moderate drift — monitor before next batch.")
    else:
        print("  All features stable. No action needed.")
    print("=" * 56)

    if output_path:
        drift_df.to_csv(output_path, index=False)
        print(f"\nFull report saved → {output_path}")

    if plot:
        FIGURES_DIR.mkdir(exist_ok=True)
        plot_path = FIGURES_DIR / "drift_report.png"
        plot_drift_report(drift_df, save_path=str(plot_path))
        print(f"Drift chart saved → {plot_path}")

    return {"feature_drift": drift_df, "prediction_drift": pred_drift}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Churn model drift monitor")
    parser.add_argument("--new-data", required=True,
                        help="Path to CSV with new customer batch")
    parser.add_argument("--model",   default=None,
                        help="Path to model joblib (default: models/churn_model.joblib)")
    parser.add_argument("--output",  default=None,
                        help="Save drift report CSV to this path")
    parser.add_argument("--no-plot", action="store_true",
                        help="Skip saving the drift chart")
    args = parser.parse_args()

    run_monitoring_report(
        new_data_path=args.new_data,
        model_path=args.model,
        output_path=args.output,
        plot=not args.no_plot,
    )
