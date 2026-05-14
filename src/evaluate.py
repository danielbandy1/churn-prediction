"""
Cohort-level error analysis: breaks model predictions down by customer segment
to surface systematic failure modes.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score, recall_score, f1_score


def tenure_bucket(tenure: pd.Series) -> pd.Series:
    """Bin raw tenure (months) into interpretable labels."""
    bins   = [0, 6, 24, 48, float("inf")]
    labels = ["New (0–6 mo)", "Growing (6–24 mo)",
              "Established (24–48 mo)", "Loyal (48+ mo)"]
    return pd.cut(tenure, bins=bins, labels=labels, right=True)


def cohort_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: pd.Series,
) -> pd.DataFrame:
    """
    Compute classification metrics for each group.

    Returns a DataFrame with columns:
        group, n, churn_rate, precision, recall, f1,
        false_positive_rate, avg_predicted_prob
    sorted by recall descending (surface the highest-miss-rate cohorts first).
    """
    rows = []
    for grp in groups.unique():
        mask = (groups == grp).values
        yt   = y_true[mask]
        yp   = y_pred[mask]
        ypr  = y_prob[mask]

        n_pos = int(yt.sum())
        n_neg = int((1 - yt).sum())
        n     = len(yt)
        if n == 0:
            continue

        prec = float(precision_score(yt, yp, zero_division=0))
        rec  = float(recall_score(yt, yp,    zero_division=0))
        f1   = float(f1_score(yt, yp,        zero_division=0))
        fp   = int(((yp == 1) & (yt == 0)).sum())
        fpr  = fp / n_neg if n_neg > 0 else 0.0

        rows.append({
            "cohort":              str(grp),
            "n":                   n,
            "actual_churn_rate":   round(yt.mean(), 4),
            "avg_predicted_prob":  round(float(ypr.mean()), 4),
            "precision":           round(prec, 4),
            "recall":              round(rec, 4),
            "f1":                  round(f1, 4),
            "false_positive_rate": round(fpr, 4),
        })

    return (
        pd.DataFrame(rows)
        .sort_values("recall", ascending=False)
        .reset_index(drop=True)
    )


def contract_cohort_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    X_features: pd.DataFrame,
) -> pd.DataFrame:
    """Cohort breakdown by Contract type (decoded from one-hot columns)."""
    contract_col = _decode_onehot(X_features, "Contract")
    return cohort_metrics(y_true, y_pred, y_prob, contract_col)


def tenure_cohort_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    X_features: pd.DataFrame,
) -> pd.DataFrame:
    """Cohort breakdown by tenure bucket."""
    buckets = tenure_bucket(X_features["tenure"])
    return cohort_metrics(y_true, y_pred, y_prob, buckets)


def _decode_onehot(X: pd.DataFrame, prefix: str) -> pd.Series:
    """Reconstruct a categorical column from its one-hot dummies."""
    dummy_cols = [c for c in X.columns if c.startswith(prefix + "_")]
    if not dummy_cols:
        return pd.Series(["Unknown"] * len(X), index=X.index)
    # Argmax over dummies gives the active category
    categories = pd.Series(
        [c.replace(prefix + "_", "") for c in dummy_cols], index=dummy_cols
    )
    active = X[dummy_cols].idxmax(axis=1)
    return active.map(categories)


def plot_cohort_analysis(
    contract_df: pd.DataFrame,
    tenure_df: pd.DataFrame,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Two-panel bar chart: precision and recall by contract type and tenure bucket.
    Highlights where the model under-serves specific customer segments.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, df, title in [
        (axes[0], contract_df, "by Contract Type"),
        (axes[1], tenure_df,   "by Tenure Bucket"),
    ]:
        cohorts = df["cohort"].tolist()
        x = np.arange(len(cohorts))
        w = 0.30

        bars_p = ax.bar(x - w/2, df["precision"], w,
                        label="Precision", color="steelblue", alpha=0.85)
        bars_r = ax.bar(x + w/2, df["recall"],    w,
                        label="Recall",    color="darkorange", alpha=0.85)

        # Annotate churn rate
        for i, row in df.iterrows():
            ax.text(i, 0.02, f"{row['actual_churn_rate']:.0%}\nchurn",
                    ha="center", va="bottom", fontsize=7.5, color="dimgray")

        ax.set_xticks(x)
        ax.set_xticklabels(cohorts, rotation=15, ha="right", fontsize=9)
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Score")
        ax.set_title(f"Model Performance {title}")
        ax.legend()
        ax.axhline(0.5, color="gray", ls=":", lw=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
