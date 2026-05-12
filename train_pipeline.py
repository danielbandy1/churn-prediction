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
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from src.features import build_features, get_X_y, get_feature_names, ONEHOT_COLUMNS
from src.train import compare_models, train_best_model, save_model
from src.explain import global_importance

DATA_PATH = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
FIGURES_DIR = pathlib.Path("figures")


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

    print(f"Training {args.model} on full training set...")
    result = train_best_model(X_train, y_train, X_test, y_test, model_name=args.model)

    print(f"\n  AUC       : {result['auc']}")
    print(f"  F1        : {result['f1']}")
    print(f"  Precision : {result['precision']}")
    print(f"  Recall    : {result['recall']}")

    model_path = save_model(result["model"], feature_names)
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


if __name__ == "__main__":
    main()
