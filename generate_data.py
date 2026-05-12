#!/usr/bin/env python3
"""
Generate a synthetic Telco Customer Churn dataset matching the statistical
properties of the IBM Telco dataset (Kaggle / IBM Developer).

Real dataset properties reproduced:
  - 7,043 rows, 21 columns
  - Churn rate: ~26.5%
  - Churn is correlated with: Month-to-month contract, short tenure,
    high monthly charges, no tech support, electronic check payment
"""

import pathlib
import numpy as np
import pandas as pd

SEED = 42
N    = 7_043
OUT  = pathlib.Path("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

rng = np.random.default_rng(SEED)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    # ── Demographics ──────────────────────────────────────────────────────────
    gender         = rng.choice(["Male", "Female"], N, p=[0.505, 0.495])
    senior_citizen = rng.binomial(1, 0.162, N)
    partner        = rng.choice(["Yes", "No"], N, p=[0.483, 0.517])
    dependents     = rng.choice(["Yes", "No"], N, p=[0.299, 0.701])
    tenure         = rng.integers(1, 73, N)

    # ── Services ─────────────────────────────────────────────────────────────
    phone_service = rng.choice(["Yes", "No"], N, p=[0.903, 0.097])

    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], N, p=[0.535, 0.465])
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"], N, p=[0.344, 0.439, 0.217]
    )

    def internet_addon(has_internet, p_yes=0.50):
        return np.where(
            has_internet == "No", "No internet service",
            rng.choice(["Yes", "No"], N, p=[p_yes, 1 - p_yes])
        )

    online_security   = internet_addon(internet_service, 0.287)
    online_backup     = internet_addon(internet_service, 0.344)
    device_protection = internet_addon(internet_service, 0.343)
    tech_support      = internet_addon(internet_service, 0.289)
    streaming_tv      = internet_addon(internet_service, 0.384)
    streaming_movies  = internet_addon(internet_service, 0.389)

    # ── Billing ───────────────────────────────────────────────────────────────
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"], N, p=[0.550, 0.210, 0.240]
    )

    paperless_billing = rng.choice(["Yes", "No"], N, p=[0.592, 0.408])

    payment_method = rng.choice(
        ["Electronic check", "Mailed check",
         "Bank transfer (automatic)", "Credit card (automatic)"],
        N, p=[0.335, 0.228, 0.219, 0.218]
    )

    monthly_charges = np.round(
        np.where(
            internet_service == "No",
            rng.uniform(18, 30, N),
            np.where(
                internet_service == "DSL",
                rng.uniform(25, 75, N),
                rng.uniform(50, 120, N),
            )
        ), 2
    )

    total_charges = np.round(monthly_charges * tenure * rng.uniform(0.95, 1.05, N), 2)

    # ── Churn (correlated logistic model) ────────────────────────────────────
    log_odds = (
        -1.8
        + 1.4  * (contract == "Month-to-month").astype(float)
        + 0.5  * (contract == "One year").astype(float)
        - 0.04 * tenure
        + 0.02 * (monthly_charges - 65)
        + 0.5  * (internet_service == "Fiber optic").astype(float)
        + 0.4  * (online_security == "No").astype(float)
        + 0.3  * (tech_support == "No").astype(float)
        + 0.35 * (payment_method == "Electronic check").astype(float)
        + 0.15 * (paperless_billing == "Yes").astype(float)
        + rng.normal(0, 0.5, N)
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn      = np.where(rng.uniform(0, 1, N) < churn_prob, "Yes", "No")

    df = pd.DataFrame({
        "customerID":        [f"TID_{i:07d}" for i in rng.integers(1_000_000, 9_999_999, N)],
        "gender":            gender,
        "SeniorCitizen":     senior_citizen,
        "Partner":           partner,
        "Dependents":        dependents,
        "tenure":            tenure,
        "PhoneService":      phone_service,
        "MultipleLines":     multiple_lines,
        "InternetService":   internet_service,
        "OnlineSecurity":    online_security,
        "OnlineBackup":      online_backup,
        "DeviceProtection":  device_protection,
        "TechSupport":       tech_support,
        "StreamingTV":       streaming_tv,
        "StreamingMovies":   streaming_movies,
        "Contract":          contract,
        "PaperlessBilling":  paperless_billing,
        "PaymentMethod":     payment_method,
        "MonthlyCharges":    monthly_charges,
        "TotalCharges":      total_charges,
        "Churn":             churn,
    })

    df.to_csv(OUT, index=False)
    churn_rate = (df["Churn"] == "Yes").mean()
    print(f"Wrote {len(df):,} rows to {OUT}")
    print(f"  Churn rate : {churn_rate:.1%}")
    print(f"  Tenure     : min={df['tenure'].min()}  max={df['tenure'].max()}  median={df['tenure'].median():.0f}")
    print(f"  Monthly $  : min={df['MonthlyCharges'].min():.2f}  max={df['MonthlyCharges'].max():.2f}")


if __name__ == "__main__":
    main()
