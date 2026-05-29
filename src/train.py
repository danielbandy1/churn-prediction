"""
Model training, comparison, and serialisation.

Trains four classifiers with 5-fold stratified cross-validation,
picks the best by ROC-AUC, and saves it alongside the feature column list.
"""

import joblib
import pathlib
import numpy as np
import pandas as pd
from src import features as features_module
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score,
    roc_curve, precision_recall_curve, confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

MODEL_DIR = pathlib.Path("models")

CLASSIFIERS = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1_000, random_state=42, class_weight="balanced")),
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=300, max_depth=8, class_weight="balanced",
        random_state=42, n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, random_state=42,
    ),
    "XGBoost": XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=3,   # handles class imbalance
        random_state=42, eval_metric="logloss", verbosity=0,
    ),
}


def compare_models(
    X: pd.DataFrame,
    y: pd.Series,
    cv_folds: int = 5,
) -> pd.DataFrame:
    """
    Cross-validate all classifiers. Returns a DataFrame with mean ± std AUC and F1.
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    rows = []
    for name, clf in CLASSIFIERS.items():
        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc",   n_jobs=-1)
        f1s  = cross_val_score(clf, X, y, cv=cv, scoring="f1",        n_jobs=-1)
        rows.append({
            "Model":       name,
            "AUC (mean)":  round(aucs.mean(), 4),
            "AUC (std)":   round(aucs.std(),  4),
            "F1 (mean)":   round(f1s.mean(),  4),
            "F1 (std)":    round(f1s.std(),   4),
        })
    return pd.DataFrame(rows).sort_values("AUC (mean)", ascending=False).reset_index(drop=True)


def train_best_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test:  pd.DataFrame,
    y_test:  pd.Series,
    model_name: str = "XGBoost",
) -> dict:
    """
    Train the chosen model on the full training set, evaluate on the held-out
    test set, and return a dict with the fitted model and evaluation metrics.
    """
    clf = CLASSIFIERS[model_name]
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    fpr, tpr, roc_thresholds = roc_curve(y_test, y_prob)
    prec, rec, pr_thresholds  = precision_recall_curve(y_test, y_prob)
    cm                         = confusion_matrix(y_test, y_pred)

    return {
        "model":       clf,
        "model_name":  model_name,
        "y_prob":      y_prob,
        "y_pred":      y_pred,
        "auc":         round(roc_auc_score(y_test, y_prob), 4),
        "f1":          round(f1_score(y_test, y_pred),       4),
        "precision":   round(precision_score(y_test, y_pred),4),
        "recall":      round(recall_score(y_test, y_pred),   4),
        "roc_curve":   (fpr, tpr),
        "pr_curve":    (prec, rec),
        "confusion_matrix": cm,
    }


def save_model(model, feature_names: list[str], path: pathlib.Path | None = None) -> pathlib.Path:
    MODEL_DIR.mkdir(exist_ok=True)
    path = path or MODEL_DIR / "churn_model.joblib"
    # Persist the trained model alongside the feature list and the
    # one-hot column metadata so the API can align inference inputs.
    joblib.dump({
        "model": model,
        "feature_names": feature_names,
        # ONEHOT_COLUMNS is only populated after engineer_features() runs in
        # this process. Always call engineer_features() before save_model() or
        # this will persist as an empty list, breaking API inference alignment.
        "onehot_columns": getattr(features_module, "ONEHOT_COLUMNS", []),
    }, path)
    return path


def load_model(path: pathlib.Path | None = None) -> tuple:
    """Backward-compatible loader returning (model, feature_names).

    Use `load_model_with_metadata` if you need the persisted `onehot_columns`.
    """
    path = path or MODEL_DIR / "churn_model.joblib"
    obj = joblib.load(path)
    return obj["model"], obj["feature_names"]


def load_model_with_metadata(path: pathlib.Path | None = None) -> tuple:
    """Load the model and return (model, feature_names, onehot_columns)."""
    path = path or MODEL_DIR / "churn_model.joblib"
    obj = joblib.load(path)
    return obj["model"], obj["feature_names"], obj.get("onehot_columns", [])
