"""
Feature engineering for the Telco churn dataset.

All transformations are deterministic functions of the raw DataFrame so the
same logic runs identically in training, the notebook, and the API.
"""

import numpy as np
import pandas as pd

# Columns we one-hot encode (keep all dummies — easier for SHAP to interpret)
_ONEHOT_COLS = ["InternetService", "Contract", "PaymentMethod"]

# Columns treated as binary Yes/No (also handles "No phone/internet service")
_BINARY_COLS = [
    "Partner", "Dependents", "PhoneService", "PaperlessBilling",
    "MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

# Columns we produce via get_dummies — fixed reference list built from training data
# (populated by build_training_features; reused for inference alignment)
ONEHOT_COLUMNS: list[str] = []


def _encode_binary(series: pd.Series) -> pd.Series:
    return series.map(
        {"Yes": 1, "No": 0,
         "No phone service": 0, "No internet service": 0}
    ).fillna(0).astype(int)


def build_features(df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Transform raw Telco CSV rows into model-ready features.

    Parameters
    ----------
    df  : raw DataFrame (may include 'Churn' column or not)
    fit : if True, records the one-hot column names for later alignment
          (call with fit=True on training data only)

    Returns a new DataFrame; never mutates the input.
    """
    global ONEHOT_COLUMNS

    df = df.copy()
    df = df.drop(columns=["customerID"], errors="ignore")

    # Fix TotalCharges — new customers have " " instead of a number
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Encode target
    if "Churn" in df.columns:
        df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Binary Yes/No columns
    for col in _BINARY_COLS:
        if col in df.columns:
            df[col] = _encode_binary(df[col])

    # gender → 1 = Male
    if "gender" in df.columns:
        df["gender"] = (df["gender"] == "Male").astype(int)

    # SeniorCitizen is already 0/1

    # One-hot encode multi-class categoricals
    dummies_list = []
    for col in _ONEHOT_COLS:
        if col in df.columns:
            dummies = pd.get_dummies(df[col], prefix=col)
            dummies_list.append(dummies)
            df = df.drop(columns=[col])
    if dummies_list:
        df = pd.concat([df] + dummies_list, axis=1)

    if fit:
        ONEHOT_COLUMNS = [
            c for c in df.columns
            if any(c.startswith(p + "_") for p in _ONEHOT_COLS)
        ]
    else:
        # If ONEHOT_COLUMNS wasn't populated (e.g. new process), try to
        # load persisted metadata from the saved model artifact so that
        # inference aligns deterministically with training.
        if not ONEHOT_COLUMNS:
            try:
                import joblib, pathlib
                model_path = pathlib.Path("models") / "churn_model.joblib"
                if model_path.exists():
                    obj = joblib.load(model_path)
                    ONEHOT_COLUMNS = obj.get("onehot_columns", ONEHOT_COLUMNS)
            except Exception:
                # Best-effort only — leave ONEHOT_COLUMNS empty if loading fails
                pass

        if ONEHOT_COLUMNS:
            # Add missing dummy columns as 0
            for col in ONEHOT_COLUMNS:
                if col not in df.columns:
                    df[col] = 0
            # Drop any extra dummy columns that were not present during training
            extra_dummy_cols = [
                c for c in df.columns
                if any(c.startswith(p + "_") for p in _ONEHOT_COLS) and c not in ONEHOT_COLUMNS
            ]
            if extra_dummy_cols:
                df = df.drop(columns=extra_dummy_cols)

    # ── Engineered features ───────────────────────────────────────────────────
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    df["num_services"] = df[[c for c in service_cols if c in df.columns]].sum(axis=1)

    df["avg_monthly_charges"] = np.where(
        df["tenure"] > 0,
        df["TotalCharges"] / df["tenure"],
        df["MonthlyCharges"],
    )

    df["charges_increase"] = df["MonthlyCharges"] - df["avg_monthly_charges"]
    df["is_new_customer"]  = (df["tenure"] < 6).astype(int)
    df["is_long_term"]     = (df["tenure"] >= 24).astype(int)

    return df


def get_X_y(df_transformed: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Split a transformed DataFrame into features X and target y."""
    y = df_transformed["Churn"]
    X = df_transformed.drop(columns=["Churn"])
    return X, y


def get_feature_names(df_transformed: pd.DataFrame) -> list[str]:
    return [c for c in df_transformed.columns if c != "Churn"]
