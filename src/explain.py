"""
SHAP-based model explanations — global feature importance and local per-customer breakdowns.
"""

import numpy as np
import pandas as pd
import shap


def _get_explainer(model, X_background: pd.DataFrame):
    """Return the right SHAP explainer for the given model type."""
    name = type(model).__name__

    # Pipeline (e.g. Logistic Regression wrapped in StandardScaler)
    if name == "Pipeline":
        inner = model.named_steps.get("clf") or list(model.named_steps.values())[-1]
        X_bg_transformed = model[:-1].transform(X_background)
        return shap.LinearExplainer(inner, X_bg_transformed), True

    if name in ("XGBClassifier",):
        return shap.TreeExplainer(model), False

    if name in ("RandomForestClassifier", "GradientBoostingClassifier"):
        return shap.TreeExplainer(model), False

    # Generic fallback
    return shap.KernelExplainer(model.predict_proba, shap.sample(X_background, 100)), False


def compute_shap_values(model, X: pd.DataFrame) -> np.ndarray:
    """
    Return SHAP values for the positive class (churn=1) as a (n_samples, n_features) array.
    X should be the raw feature DataFrame (not scaled — the model handles scaling internally).
    """
    explainer, is_linear = _get_explainer(model, X)

    if is_linear:
        inner = model.named_steps.get("clf") or list(model.named_steps.values())[-1]
        X_transformed = model[:-1].transform(X)
        sv = explainer.shap_values(X_transformed)
    else:
        sv = explainer.shap_values(X)

    # TreeExplainer on a classifier returns a list [class0, class1]; take class 1
    if isinstance(sv, list):
        sv = sv[1]
    return sv


def global_importance(model, X: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature, sorted descending."""
    sv = compute_shap_values(model, X)
    importance = np.abs(sv).mean(axis=0)
    return (
        pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def local_explanation(model, X_row: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """
    SHAP breakdown for a single customer row.
    Returns a DataFrame with feature, value, and shap_value columns.
    """
    sv = compute_shap_values(model, X_row)
    row_sv = sv[0] if sv.ndim > 1 else sv
    return (
        pd.DataFrame({
            "feature":    feature_names,
            "value":      X_row.iloc[0].values,
            "shap_value": row_sv,
        })
        .sort_values("shap_value", key=abs, ascending=False)
        .reset_index(drop=True)
    )
