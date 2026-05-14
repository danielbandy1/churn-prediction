"""
Cohort-level error analysis and probability calibration diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, precision_score, recall_score, f1_score


def calibration_analysis(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
) -> dict:
    """
    Compute calibration metrics for the model.

    Returns:
        brier_score:          lower is better; 0.0 = perfect, ~0.20 = random on this dataset
        fraction_of_positives: actual churn rate per predicted-probability bin
        mean_predicted_value:  mean predicted probability per bin
        calibration_slope:    linear fit slope — 1.0 means probabilities are on-target
        calibration_intercept: linear fit intercept — 0.0 is ideal
    """
    brier = float(brier_score_loss(y_true, y_prob))
    fop, mpv = calibration_curve(y_true, y_prob, n_bins=n_bins, strategy="uniform")
    slope, intercept = np.polyfit(mpv, fop, 1)
    return {
        "brier_score":           round(brier, 4),
        "fraction_of_positives": fop,
        "mean_predicted_value":  mpv,
        "calibration_slope":     round(float(slope), 4),
        "calibration_intercept": round(float(intercept), 4),
    }


def plot_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    model_name: str = "XGBoost",
    n_bins: int = 10,
) -> plt.Figure:
    """
    Two-panel calibration figure:
      left  — reliability diagram (predicted probability vs actual churn rate)
      right — histogram of predicted probabilities with decision threshold marked
    """
    cal = calibration_analysis(y_true, y_prob, n_bins=n_bins)
    fop = cal["fraction_of_positives"]
    mpv = cal["mean_predicted_value"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    ax1.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax1.plot(mpv, fop, "s-", color="darkorange", lw=2, ms=7, label=model_name)
    ax1.set_xlabel("Mean predicted probability")
    ax1.set_ylabel("Fraction of positives (actual churn rate)")
    ax1.set_title(
        f"Reliability Diagram — {model_name}\n"
        f"Brier score = {cal['brier_score']:.4f}  |  "
        f"Calibration slope = {cal['calibration_slope']:.3f}"
    )
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)

    ax2.hist(y_prob, bins=30, color="steelblue", alpha=0.75, edgecolor="white")
    ax2.axvline(0.40, color="red", ls="--", lw=1.5, label="Decision threshold (0.40)")
    ax2.set_xlabel("Predicted churn probability")
    ax2.set_ylabel("Count")
    ax2.set_title("Distribution of Predicted Probabilities")
    ax2.legend()

    plt.tight_layout()
    return fig


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
