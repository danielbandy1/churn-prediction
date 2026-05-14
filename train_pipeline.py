#!/usr/bin/env python3
"""
End-to-end training pipeline.

Usage:
    python train_pipeline.py [--model XGBoost] [--figs] [--compare]
"""

import argparse
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.features import build_features, get_X_y, get_feature_names, ONEHOT_COLUMNS
from src.train import compare_models, train_best_model, save_model
from src.explain import global_importance
from src.business import threshold_sweep, top_n_simulation, plot_expected_value_curve, plot_cumulative_gains
from src.evaluate import (
    calibration_analysis, plot_calibration,
    contract_cohort_analysis, tenure_cohort_analysis, plot_cohort_analysis,
)

DATA_PATH = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
FIGURES_DIR = pathlib.Path("figures")
DECISION_THRESHOLD = 0.40   # tuned to maximise expected net value (see business.py)


def load_raw() -> pd.DataFrame:
    if not DATA_PATH.exists():
        print("Data not found — generating synthetic dataset...")
        import generate_data
        generate_data.main()
    return pd.read_csv(DATA_PATH)


def save_figure(fig, name: str):
    FIGURES_DIR.mkdir(exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path}")


def plot_roc(result: dict):
    fpr, tpr = result["roc_curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {result['auc']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve — {result['model_name']}")
    ax.legend()
    save_figure(fig, "roc_curve.png")


def plot_pr(result: dict):
    prec, rec = result["pr_curve"]
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(rec, prec, lw=2, color="darkorange")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve — {result['model_name']}")
    save_figure(fig, "pr_curve.png")


def plot_confusion(result: dict):
    import numpy as np
    cm = result["confusion_matrix"]
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    labels = ["No Churn", "Churn"]
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_yticks([0, 1]); ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    save_figure(fig, "confusion_matrix.png")


def plot_shap(model, X_test: pd.DataFrame, feature_names: list[str]):
    imp = global_importance(model, X_test.head(500), feature_names)
    top = imp.head(15)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(top["feature"][::-1], top["mean_abs_shap"][::-1], color="steelblue")
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance (SHAP)")
    plt.tight_layout()
    save_figure(fig, "shap_importance.png")


def main():
    parser = argparse.ArgumentParser(description="Train churn prediction model")
    parser.add_argument("--model",   default="XGBoost", help="Model name from CLASSIFIERS")
    parser.add_argument("--figs",    action="store_true", help="Save all diagnostic figures")
    parser.add_argument("--compare", action="store_true", help="Run cross-validated model comparison first")
    args = parser.parse_args()

    print("Loading data...")
    df_raw = load_raw()
    print(f"  {len(df_raw):,} rows loaded")

    print("Building features...")
    df = build_features(df_raw, fit=True)
    X, y = get_X_y(df)
    feature_names = get_feature_names(df)
    print(f"  {X.shape[1]} features  |  churn rate = {y.mean():.1%}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    if args.compare:
        print("\nComparing models (5-fold CV) ...")
        comparison = compare_models(X_train, y_train)
        print(comparison.to_string(index=False))
        print()

    mlflow.set_experiment("churn-prediction")
    with mlflow.start_run(run_name=args.model):
        mlflow.log_params({
            "model_name":         args.model,
            "test_size":          0.20,
            "random_state":       42,
            "decision_threshold": DECISION_THRESHOLD,
        })

        print(f"Training {args.model} on full training set...")
        result = train_best_model(X_train, y_train, X_test, y_test, model_name=args.model)

        # Log model hyperparams from the fitted estimator
        try:
            fitted = result["model"]
            clf    = getattr(fitted, "named_steps", {}).get("clf", fitted)
            keep   = ("n_estimators", "max_depth", "learning_rate", "subsample",
                      "colsample_bytree", "scale_pos_weight", "class_weight", "C", "max_iter")
            mlflow.log_params({k: v for k, v in clf.get_params().items() if k in keep})
        except Exception:
            pass

        # ── Calibration ───────────────────────────────────────────────────────
        cal = calibration_analysis(y_test.values, result["y_prob"])

        # ── Metrics ───────────────────────────────────────────────────────────
        mlflow.log_metrics({
            "auc":                 result["auc"],
            "f1":                  result["f1"],
            "precision":           result["precision"],
            "recall":              result["recall"],
            "brier_score":         cal["brier_score"],
            "calibration_slope":   cal["calibration_slope"],
        })

        print(f"\n  AUC              : {result['auc']}")
        print(f"  F1               : {result['f1']}")
        print(f"  Precision        : {result['precision']}")
        print(f"  Recall           : {result['recall']}")
        print(f"  Brier score      : {cal['brier_score']}  (lower is better; ~0.20 = random)")
        print(f"  Calibration slope: {cal['calibration_slope']}  (1.0 = perfectly calibrated)")

        model_path = save_model(result["model"], feature_names)
        mlflow.log_artifact(str(model_path), artifact_path="model")
        mlflow.sklearn.log_model(result["model"], artifact_path="sklearn-model")
        print(f"\nModel saved → {model_path}")

        if args.figs:
            print("Generating figures...")
            plot_roc(result)
            plot_pr(result)
            plot_confusion(result)
            try:
                plot_shap(result["model"], X_test, feature_names)
            except Exception as e:
                print(f"  SHAP figure skipped: {e}")

            # Calibration figure
            print("  Generating calibration figure...")
            fig_cal = plot_calibration(y_test.values, result["y_prob"], model_name=args.model)
            save_figure(fig_cal, "calibration.png")

            # Business impact figures
            print("  Generating business impact figures...")
            y_test_arr = y_test.values
            y_prob_arr = result["y_prob"]

            LTV, SAVE_RATE, CONTACT_COST = 600.0, 0.30, 15.0
            df_sweep = threshold_sweep(y_test_arr, y_prob_arr, LTV, SAVE_RATE, CONTACT_COST)
            fig_ev = plot_expected_value_curve(df_sweep, contact_cost=CONTACT_COST)
            save_figure(fig_ev, "expected_value_curve.png")

            fig_cg = plot_cumulative_gains(y_test_arr, y_prob_arr)
            save_figure(fig_cg, "cumulative_gains.png")

            # Cohort error analysis
            print("  Generating cohort analysis figures...")
            y_pred_arr = result["y_pred"]
            contract_df = contract_cohort_analysis(y_test_arr, y_pred_arr, y_prob_arr, X_test)
            tenure_df   = tenure_cohort_analysis(y_test_arr, y_pred_arr, y_prob_arr, X_test)
            fig_cohort  = plot_cohort_analysis(contract_df, tenure_df)
            save_figure(fig_cohort, "cohort_analysis.png")

            # Log all figures to MLflow
            for fig_file in sorted(FIGURES_DIR.glob("*.png")):
                mlflow.log_artifact(str(fig_file), artifact_path="figures")

            # Business summary
            sim = top_n_simulation(y_test_arr, y_prob_arr, 200, LTV, SAVE_RATE, CONTACT_COST)
            mlflow.log_metrics({
                "business_precision_at_200": round(sim["precision_at_n"], 4),
                "business_revenue_saved":    round(sim["revenue_saved"], 2),
                "business_lift":             round(sim["lift_over_random"], 4),
            })
            print(f"\n  Business Impact (top 200 at-risk customers):")
            print(f"    Churners captured : {sim['true_positives']} / {int(y_test_arr.sum())}")
            print(f"    Precision@200     : {sim['precision_at_n']:.0%}")
            print(f"    Expected revenue  : ${sim['revenue_saved']:,.0f}")
            print(f"    Lift over random  : {sim['lift_over_random']:.1f}×")

        run_id = mlflow.active_run().info.run_id
        print(f"\nMLflow run ID : {run_id}")
        print("View runs     : mlflow ui")


if __name__ == "__main__":
    main()
