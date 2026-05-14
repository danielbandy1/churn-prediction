"""
Business impact simulation for the churn model.

Translates probability predictions into expected revenue impact, answering:
"If we use this model to prioritise retention outreach, what's the ROI?"
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def expected_value_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float,
    ltv: float,
    save_rate: float,
    contact_cost: float = 0.0,
) -> dict:
    """
    At a given probability threshold, compute expected net business value.

    Parameters
    ----------
    y_true       : true churn labels (1 = churned)
    y_prob       : predicted churn probabilities
    threshold    : flag customer as at-risk if P(churn) >= threshold
    ltv          : average revenue saved per successfully retained customer
    save_rate    : fraction of contacted true churners who are retained
    contact_cost : cost per outreach attempt (call, email, discount offer)
    """
    flagged    = y_prob >= threshold
    tp = int(( flagged & (y_true == 1)).sum())
    fp = int(( flagged & (y_true == 0)).sum())
    fn = int((~flagged & (y_true == 1)).sum())
    n_contacted = tp + fp

    revenue_saved = tp * ltv * save_rate
    total_cost    = n_contacted * contact_cost
    net_value     = revenue_saved - total_cost
    precision     = tp / n_contacted if n_contacted > 0 else 0.0
    recall        = tp / (tp + fn)   if (tp + fn) > 0   else 0.0

    return {
        "threshold":     round(threshold, 4),
        "n_contacted":   n_contacted,
        "true_positives":  tp,
        "false_positives": fp,
        "precision":     round(precision, 4),
        "recall":        round(recall, 4),
        "revenue_saved": round(revenue_saved, 2),
        "total_cost":    round(total_cost, 2),
        "net_value":     round(net_value, 2),
    }


def threshold_sweep(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    ltv: float,
    save_rate: float,
    contact_cost: float = 0.0,
    n_steps: int = 200,
) -> pd.DataFrame:
    """
    Sweep probability thresholds and return a DataFrame of business metrics.
    Use this to find the threshold that maximises expected net value.
    """
    thresholds = np.linspace(0.01, 0.99, n_steps)
    rows = [
        expected_value_at_threshold(y_true, y_prob, t, ltv, save_rate, contact_cost)
        for t in thresholds
    ]
    return pd.DataFrame(rows)


def top_n_simulation(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_contact: int,
    ltv: float,
    save_rate: float,
    contact_cost: float = 0.0,
) -> dict:
    """
    Simulate contacting exactly the top-N highest-risk customers.
    Returns expected revenue saved, ROI, and comparison to random targeting.
    """
    n_contact = min(n_contact, len(y_true))
    top_idx   = np.argsort(y_prob)[::-1][:n_contact]

    tp = int(y_true[top_idx].sum())
    fp = n_contact - tp
    revenue_saved = tp * ltv * save_rate
    total_cost    = n_contact * contact_cost
    net_value     = revenue_saved - total_cost

    # Random baseline: expected TP if we picked n_contact at random
    churn_rate   = y_true.mean()
    random_tp    = n_contact * churn_rate
    random_rev   = random_tp * ltv * save_rate - total_cost
    lift         = (revenue_saved / random_rev) if random_rev > 0 else float("inf")

    precision_at_n = tp / n_contact if n_contact > 0 else 0.0
    recall_at_n    = tp / int(y_true.sum()) if y_true.sum() > 0 else 0.0

    return {
        "n_contacted":      n_contact,
        "true_positives":   tp,
        "false_positives":  fp,
        "precision_at_n":   round(precision_at_n, 4),
        "recall_at_n":      round(recall_at_n, 4),
        "revenue_saved":    round(revenue_saved, 2),
        "contact_cost":     round(total_cost, 2),
        "net_value":        round(net_value, 2),
        "random_baseline_net": round(random_rev, 2),
        "lift_over_random": round(lift, 2),
    }


def plot_expected_value_curve(
    df_sweep: pd.DataFrame,
    save_path: str | None = None,
    contact_cost: float = 0.0,
) -> plt.Figure:
    """Plot net value vs threshold with the optimal threshold marked."""
    best_idx = df_sweep["net_value"].idxmax()
    best     = df_sweep.loc[best_idx]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: net value curve
    ax = axes[0]
    ax.plot(df_sweep["threshold"], df_sweep["net_value"],
            color="steelblue", lw=2, label="Net value")
    ax.axvline(best["threshold"], color="crimson", ls="--", lw=1.5,
               label=f"Optimal threshold = {best['threshold']:.2f}")
    ax.scatter([best["threshold"]], [best["net_value"]],
               color="crimson", zorder=5, s=60)
    ax.set_xlabel("Probability Threshold")
    ax.set_ylabel("Expected Net Value ($)")
    ax.set_title("Expected Net Value vs. Decision Threshold")
    ax.legend()
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))

    # Right: precision / recall trade-off
    ax2 = axes[1]
    ax2.plot(df_sweep["threshold"], df_sweep["precision"],
             color="darkorange", lw=2, label="Precision")
    ax2.plot(df_sweep["threshold"], df_sweep["recall"],
             color="teal", lw=2, label="Recall")
    ax2.axvline(best["threshold"], color="crimson", ls="--", lw=1.5,
                label=f"Optimal = {best['threshold']:.2f}")
    ax2.set_xlabel("Probability Threshold")
    ax2.set_ylabel("Score")
    ax2.set_title("Precision / Recall vs. Threshold")
    ax2.legend()
    ax2.set_ylim(0, 1.05)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_cumulative_gains(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Cumulative gains chart: % of churners captured vs % of customers contacted.
    The steeper the curve, the better the model at prioritising retention calls.
    """
    order      = np.argsort(y_prob)[::-1]
    y_sorted   = y_true[order]
    n          = len(y_true)
    n_churners = y_true.sum()

    pct_contacted = np.arange(1, n + 1) / n * 100
    pct_captured  = np.cumsum(y_sorted) / n_churners * 100

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(pct_contacted, pct_captured,
            color="steelblue", lw=2.5, label="Model")
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="Random baseline")
    ax.plot([0, n_churners / n * 100, 100],
            [0, 100, 100], color="forestgreen", lw=1.5, ls=":",
            label="Perfect model")

    # Annotate 20% mark
    idx_20 = int(0.20 * n)
    cap_20 = float(np.cumsum(y_sorted)[idx_20] / n_churners * 100)
    ax.annotate(
        f"Top 20%\ncaptures {cap_20:.0f}%\nof churners",
        xy=(20, cap_20), xytext=(35, cap_20 - 15),
        arrowprops=dict(arrowstyle="->", color="gray"),
        fontsize=9, color="steelblue",
    )

    ax.set_xlabel("% of Customers Contacted (sorted by risk score)")
    ax.set_ylabel("% of Actual Churners Captured")
    ax.set_title("Cumulative Gains Chart")
    ax.legend(loc="lower right")
    ax.set_xlim(0, 100); ax.set_ylim(0, 105)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
