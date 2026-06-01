#!/usr/bin/env python3
"""
Streamlit dashboard — Telco Customer Churn Risk Scorer.

Run:
    streamlit run app.py
"""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).parent))

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = pathlib.Path("models/churn_model.joblib")

st.set_page_config(
    page_title="Churn Risk Scorer",
    page_icon="📡",
    layout="wide",
)


@st.cache_resource
def load_model():
    bundle = joblib.load(MODEL_PATH)
    try:
        import shap
        explainer = shap.TreeExplainer(bundle["model"])
    except Exception:
        explainer = None
    return bundle, explainer


def _tier_color(tier):
    return {"LOW": "#2ecc71", "MODERATE": "#f39c12", "HIGH": "#e74c3c"}.get(tier, "#888")


def _predict(bundle, explainer, inputs: dict):
    feat = bundle["feature_names"]
    row = {f: 0.0 for f in feat}

    row["gender"]         = 1 if inputs["gender"] == "Male" else 0
    row["SeniorCitizen"]  = inputs["senior"]
    row["Partner"]        = 1 if inputs["partner"] == "Yes" else 0
    row["Dependents"]     = 1 if inputs["dependents"] == "Yes" else 0
    row["tenure"]         = inputs["tenure"]
    row["PhoneService"]   = 1 if inputs["phone"] == "Yes" else 0
    row["MultipleLines"]  = 1 if inputs["multi_lines"] == "Yes" else 0
    row["PaperlessBilling"] = 1 if inputs["paperless"] == "Yes" else 0
    row["MonthlyCharges"] = inputs["monthly_charges"]

    # Binary service columns
    for svc in ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
                "TechSupport", "StreamingTV", "StreamingMovies"]:
        row[svc] = 1 if inputs[svc] == "Yes" else 0

    # InternetService one-hot
    internet = inputs["internet"]
    row["InternetService_DSL"]          = 1 if internet == "DSL" else 0
    row["InternetService_Fiber optic"]  = 1 if internet == "Fiber optic" else 0
    row["InternetService_No"]           = 1 if internet == "No" else 0

    # Contract one-hot
    contract = inputs["contract"]
    row["Contract_Month-to-month"] = 1 if contract == "Month-to-month" else 0
    row["Contract_One year"]       = 1 if contract == "One year" else 0
    row["Contract_Two year"]       = 1 if contract == "Two year" else 0

    # PaymentMethod one-hot
    pay = inputs["payment"]
    row["PaymentMethod_Bank transfer (automatic)"]  = 1 if pay == "Bank transfer" else 0
    row["PaymentMethod_Credit card (automatic)"]    = 1 if pay == "Credit card" else 0
    row["PaymentMethod_Electronic check"]           = 1 if pay == "Electronic check" else 0
    row["PaymentMethod_Mailed check"]               = 1 if pay == "Mailed check" else 0

    # Engineered
    total_charges = inputs["tenure"] * inputs["monthly_charges"]
    row["TotalCharges"] = total_charges
    avg_monthly = (total_charges / inputs["tenure"]) if inputs["tenure"] > 0 else inputs["monthly_charges"]
    row["avg_monthly_charges"] = avg_monthly
    row["charges_increase"] = inputs["monthly_charges"] - avg_monthly
    row["is_new_customer"] = 1 if inputs["tenure"] < 6 else 0
    row["is_long_term"]    = 1 if inputs["tenure"] >= 24 else 0

    svc_cols = ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
    row["num_services"] = sum(row[c] for c in svc_cols)

    X = pd.DataFrame([row])[feat].astype(np.float32)

    prob = float(bundle["model"].predict_proba(X)[0, 1])
    tier = "HIGH" if prob >= 0.50 else ("MODERATE" if prob >= 0.25 else "LOW")

    shap_vals = None
    if explainer is not None:
        try:
            sv = explainer.shap_values(X)
            arr = np.array(sv[1] if isinstance(sv, list) else sv)[0]
            shap_vals = sorted(
                zip(feat, arr), key=lambda x: abs(x[1]), reverse=True
            )[:10]
        except Exception:
            pass

    return prob, tier, shap_vals, X


# ── Sidebar ──────────────────────────────────────────────────────────────────

st.sidebar.header("Customer Profile")

gender     = st.sidebar.selectbox("Gender", ["Female", "Male"])
senior     = st.sidebar.selectbox("Senior citizen?", [0, 1], format_func=lambda x: "Yes" if x else "No")
partner    = st.sidebar.selectbox("Has partner?", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has dependents?", ["No", "Yes"])

st.sidebar.subheader("Account")
tenure         = st.sidebar.slider("Tenure (months)", 0, 72, 12)
contract       = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
monthly_charges = st.sidebar.slider("Monthly charges ($)", 10.0, 120.0, 65.0, step=0.5)
paperless      = st.sidebar.selectbox("Paperless billing?", ["Yes", "No"])
payment        = st.sidebar.selectbox("Payment method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"])

st.sidebar.subheader("Services")
internet   = st.sidebar.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
phone      = st.sidebar.selectbox("Phone service?", ["Yes", "No"])
multi_lines = st.sidebar.selectbox("Multiple lines?", ["No", "Yes"])

st.sidebar.markdown("**Add-on services** (Yes / No)")
OnlineSecurity  = st.sidebar.selectbox("Online security",  ["No", "Yes"])
OnlineBackup    = st.sidebar.selectbox("Online backup",    ["No", "Yes"])
DeviceProtection= st.sidebar.selectbox("Device protection",["No", "Yes"])
TechSupport     = st.sidebar.selectbox("Tech support",     ["No", "Yes"])
StreamingTV     = st.sidebar.selectbox("Streaming TV",     ["No", "Yes"])
StreamingMovies = st.sidebar.selectbox("Streaming movies", ["No", "Yes"])

inputs = {
    "gender": gender, "senior": senior, "partner": partner, "dependents": dependents,
    "tenure": tenure, "contract": contract, "monthly_charges": monthly_charges,
    "paperless": paperless, "payment": payment,
    "internet": internet, "phone": phone, "multi_lines": multi_lines,
    "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
    "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies,
}

# ── Main panel ───────────────────────────────────────────────────────────────

st.title("📡 Telco Customer Churn Risk Scorer")
st.caption("Model: **XGBoost** · Dataset: 7,043 IBM Telco customers · AUC 0.78 · F1 0.53 (26.5% churn rate)")

bundle, explainer = load_model()
prob, tier, shap_vals, X = _predict(bundle, explainer, inputs)

col1, col2, col3 = st.columns([1, 1, 2])

with col1:
    st.metric("Churn probability", f"{prob*100:.1f}%")
    monthly_at_risk = monthly_charges * prob
    st.metric("Monthly revenue at risk", f"${monthly_at_risk:.2f}")

with col2:
    color = _tier_color(tier)
    st.markdown(
        f"<div style='background:{color};padding:12px 18px;border-radius:8px;"
        f"color:white;font-size:1.4rem;font-weight:700;text-align:center'>"
        f"{tier} CHURN RISK</div>",
        unsafe_allow_html=True,
    )

with col3:
    retention_actions = {
        "LOW":      "Routine engagement — no immediate action needed.",
        "MODERATE": "Consider a proactive outreach call or loyalty discount within 30 days.",
        "HIGH":     "Flag for retention specialist. Immediate outreach recommended — offer contract upgrade or discount.",
    }
    st.info(f"**Retention action:** {retention_actions[tier]}")
    st.markdown(
        f"""
        | Risk tier | Threshold | Action |
        |-----------|-----------|--------|
        | 🟢 LOW | < 25% | Monitor |
        | 🟡 MODERATE | 25–50% | Proactive outreach |
        | 🔴 HIGH | ≥ 50% | Immediate retention call |
        """
    )

st.divider()

if shap_vals:
    import matplotlib.pyplot as plt

    st.subheader("Top drivers for this customer")
    features = [f for f, _ in shap_vals]
    values   = [v for _, v in shap_vals]
    colors   = ["#e74c3c" if v > 0 else "#2ecc71" for v in values]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.barh(features[::-1], values[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("SHAP value (impact on churn probability)")
    ax.set_title("Feature contributions — red = increases churn risk, green = decreases risk")
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()
else:
    st.info("SHAP explainability not available for this model artifact.")

st.divider()

with st.expander("Population context (dataset baseline)"):
    st.markdown("""
    | Metric | Value |
    |--------|-------|
    | Overall churn rate | **26.5%** |
    | Median tenure | 29 months |
    | Median monthly charges | $64.76 |
    | Highest-churn contract | Month-to-month (42%) |
    | Highest-churn internet | Fiber optic (42%) |
    | Highest-churn payment | Electronic check (45%) |
    """)

with st.expander("Model info"):
    st.markdown(f"""
    - **Model file:** `{MODEL_PATH.name}`
    - **Algorithm:** XGBoost (sklearn API)
    - **Features:** {len(bundle['feature_names'])}
    - **CV ROC AUC:** 0.7802  |  **CV F1:** 0.5283
    - **Precision:** 0.44  |  **Recall:** 0.66
    - **Explainability:** TreeSHAP (Lundberg & Lee, 2017)
    - **Business metric:** Monthly revenue at risk = P(churn) × MonthlyCharges
    """)
