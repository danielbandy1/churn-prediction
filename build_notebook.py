#!/usr/bin/env python3
"""Build the churn analysis Jupyter notebook programmatically."""

import nbformat
from nbformat.v4 import new_notebook, new_markdown_cell, new_code_cell
import pathlib

NB_PATH = pathlib.Path("notebooks/ab_analysis.ipynb")


def md(src: str): return new_markdown_cell(src)
def code(src: str): return new_code_cell(src)


cells = [
    # ── 0 Setup ───────────────────────────────────────────────────────────────
    md("""# Telco Customer Churn Prediction
End-to-end analysis: feature engineering → model comparison → XGBoost → SHAP explainability.

**Dataset**: IBM Telco Customer Churn (7 043 rows, ~26.5% churn rate)"""),

    code("""\
import sys, pathlib, os
# Works whether nbconvert runs from repo root or notebooks/
_root = None
for _p in [pathlib.Path("."), pathlib.Path("..")]:
    if (_p / "src").exists():
        _root = _p.resolve()
        sys.path.insert(0, str(_root))
        os.chdir(_root)   # ensure relative file paths work
        break

import warnings; warnings.filterwarnings("ignore")
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from src.features import build_features, get_X_y, get_feature_names
from src.train import compare_models, train_best_model, save_model, load_model
from src.explain import global_importance, local_explanation

pd.set_option("display.max_columns", 40)
pd.set_option("display.float_format", "{:.4f}".format)
print("Setup OK")"""),

    # ── 1 Load data ───────────────────────────────────────────────────────────
    md("## 1. Load & Inspect Raw Data"),

    code("""\
import generate_data

DATA_CSV = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
if not DATA_CSV.exists():
    generate_data.main()

df_raw = pd.read_csv(DATA_CSV)
print(f"Shape: {df_raw.shape}")
print(f"Churn rate: {(df_raw['Churn'] == 'Yes').mean():.1%}")
df_raw.head(3)"""),

    code("""\
# Missing values check
missing = df_raw.isnull().sum()
print("Missing values:", missing[missing > 0].to_dict() or "none")
print("\\nDtype summary:")
print(df_raw.dtypes.value_counts())"""),

    # ── 2 Feature engineering ─────────────────────────────────────────────────
    md("""## 2. Feature Engineering

Key transformations applied by `build_features()`:
- Binary Yes/No columns → 0/1 (including "No phone/internet service" → 0)
- Categorical columns → one-hot (InternetService, Contract, PaymentMethod)
- Engineered: `num_services`, `avg_monthly_charges`, `charges_increase`, `is_new_customer`, `is_long_term`
"""),

    code("""\
df = build_features(df_raw, fit=True)
X, y = get_X_y(df)
feature_names = get_feature_names(df)

print(f"Features: {X.shape[1]}")
print(f"Churn rate after encoding: {y.mean():.1%}")
print("\\nSample features:", feature_names[:8])
X.head(3)"""),

    code("""\
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"Train: {len(X_train):,}  |  Test: {len(X_test):,}")
print(f"Train churn: {y_train.mean():.1%}  |  Test churn: {y_test.mean():.1%}")"""),

    # ── 3 Model comparison ────────────────────────────────────────────────────
    md("""## 3. Model Comparison (5-Fold Cross-Validation)

Four classifiers benchmarked on ROC-AUC and F1.
Class imbalance handled via `class_weight="balanced"` (sklearn) and `scale_pos_weight=3` (XGBoost).
"""),

    code("""\
comparison = compare_models(X_train, y_train, cv_folds=5)
print(comparison.to_string(index=False))"""),

    # ── 4 Train best model ────────────────────────────────────────────────────
    md("## 4. Train Best Model on Full Training Set"),

    code("""\
result = train_best_model(X_train, y_train, X_test, y_test, model_name="XGBoost")
print(f"AUC       : {result['auc']}")
print(f"F1        : {result['f1']}")
print(f"Precision : {result['precision']}")
print(f"Recall    : {result['recall']}")"""),

    # ── 5 Evaluation plots ────────────────────────────────────────────────────
    md("## 5. Evaluation Plots"),

    code("""\
from pathlib import Path
Path("figures").mkdir(exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# ROC
fpr, tpr = result["roc_curve"]
axes[0].plot(fpr, tpr, lw=2, label=f"AUC = {result['auc']:.3f}")
axes[0].plot([0,1],[0,1],"k--",lw=1)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("ROC Curve"); axes[0].legend()

# Precision-Recall
prec, rec = result["pr_curve"]
axes[1].plot(rec, prec, lw=2, color="darkorange")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve")

# Confusion Matrix
cm = result["confusion_matrix"]
im = axes[2].imshow(cm, cmap="Blues")
axes[2].set_xticks([0,1]); axes[2].set_xticklabels(["No Churn","Churn"])
axes[2].set_yticks([0,1]); axes[2].set_yticklabels(["No Churn","Churn"])
axes[2].set_xlabel("Predicted"); axes[2].set_ylabel("Actual")
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i,j]), ha="center", va="center",
                     color="white" if cm[i,j] > cm.max()/2 else "black")

fig.colorbar(im, ax=axes[2])
plt.tight_layout()
plt.savefig("figures/evaluation_plots.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved figures/evaluation_plots.png")"""),

    # ── 6 SHAP ────────────────────────────────────────────────────────────────
    md("""## 6. SHAP Feature Importance

SHAP (SHapley Additive exPlanations) assigns each feature a contribution to the
prediction for every customer. Mean |SHAP| gives a model-level importance ranking
that is consistent with individual explanations.
"""),

    code("""\
imp = global_importance(result["model"], X_test, feature_names)
top15 = imp.head(15)

fig, ax = plt.subplots(figsize=(9, 6))
ax.barh(top15["feature"][::-1], top15["mean_abs_shap"][::-1], color="steelblue")
ax.set_xlabel("Mean |SHAP value|")
ax.set_title("Top 15 Features by SHAP Importance")
plt.tight_layout()
plt.savefig("figures/shap_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved figures/shap_importance.png")
print(top15[["feature","mean_abs_shap"]].to_string(index=False))"""),

    # ── 7 Local explanation ───────────────────────────────────────────────────
    md("## 7. Local Explanation — Single Customer"),

    code("""\
# Pick a customer with high predicted churn probability
probs = result["model"].predict_proba(X_test)[:, 1]
high_risk_idx = probs.argmax()
X_example = X_test.iloc[[high_risk_idx]]
print(f"Customer #{high_risk_idx}  |  P(churn) = {probs[high_risk_idx]:.3f}")

local_exp = local_explanation(result["model"], X_example, feature_names)
print("\\nTop drivers:")
print(local_exp.head(8).to_string(index=False))"""),

    # ── 8 Save model ──────────────────────────────────────────────────────────
    md("## 8. Save Model"),

    code("""\
path = save_model(result["model"], feature_names)
print(f"Saved → {path}")

# Verify round-trip
model_loaded, names_loaded = load_model()
probs_reloaded = model_loaded.predict_proba(X_test)[:, 1]
assert len(probs_reloaded) == len(X_test)
print("Load verification OK")"""),

    # ── 9 Key findings ────────────────────────────────────────────────────────
    md("""## 9. Key Findings

| Metric | Value |
|--------|-------|
| Test AUC | see cell 4 output |
| Top churn driver | Contract type (Month-to-month) |
| Second driver | Tenure (short tenure → higher risk) |
| Third driver | InternetService_Fiber optic |

**Actionable insights:**
- Target Month-to-month customers with tenure < 12 months
- Prioritise customers without OnlineSecurity + TechSupport
- Electronic check users churn more — nudge toward auto-pay
"""),
]

NB_PATH.parent.mkdir(exist_ok=True)
nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
nbformat.write(nb, str(NB_PATH))
print(f"Wrote {len(cells)} cells → {NB_PATH}")
