# Telco Customer Churn Prediction

An end-to-end machine learning project for identifying telecom customers at risk of churn, explaining the drivers behind each prediction, and translating model scores into retention actions.

This repository goes beyond a notebook model fit. It includes a reusable feature pipeline, cross-validated model comparison, XGBoost training, SHAP explainability, business value simulation, cohort error analysis, drift monitoring, a FastAPI prediction service, MLflow experiment tracking, and pytest coverage.

## Executive Summary

Customer churn is a high-leverage problem for subscription businesses: each saved customer preserves future recurring revenue, but retention outreach is costly if sent indiscriminately. This project builds a churn scoring system that helps a retention team answer four operational questions:

| Business question | Project answer |
| --- | --- |
| Who is likely to churn? | A supervised classifier scores each customer with `P(churn)` |
| Why is this customer high risk? | SHAP explanations identify the strongest local prediction drivers |
| Who should retention contact first? | Customers are ranked by churn risk and evaluated with cumulative gains |
| Is the model still reliable on new batches? | PSI-based drift monitoring flags shifted feature distributions |

The deployed model artifact is an XGBoost classifier trained on engineered Telco customer features and served through FastAPI.

## Dataset

The project uses the IBM Telco Customer Churn schema:

```text
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv
```

Current local dataset profile:

| Item | Value |
| --- | ---: |
| Rows | 7,043 |
| Raw columns | 21 |
| Churn rate | 26.5% |
| Train/test split | 80% / 20%, stratified |
| Engineered model features | 31 |

If the raw CSV is missing, `download_data.py` attempts public mirrors and falls back to `generate_data.py`, which creates a statistically similar synthetic Telco dataset with the same schema.

## Machine Learning Approach

### Feature Engineering

The shared feature pipeline lives in `src/features.py` and is used by training, notebooks, monitoring, and the API.

Key transformations:

- Converts binary service fields into 0/1 indicators.
- Treats `No phone service` and `No internet service` as inactive service values.
- One-hot encodes `InternetService`, `Contract`, and `PaymentMethod`.
- Converts `TotalCharges` to numeric and handles blank values.
- Adds engineered features: `num_services`, `avg_monthly_charges`, `charges_increase`, `is_new_customer`, and `is_long_term`.

### Model Comparison

`src/train.py` benchmarks four classifiers with 5-fold stratified cross-validation:

| Model | CV AUC mean | CV AUC std | CV F1 mean |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.7968 | 0.0200 | 0.5448 |
| Random Forest | 0.7900 | 0.0195 | 0.5360 |
| Gradient Boosting | 0.7866 | 0.0167 | 0.4268 |
| XGBoost | 0.7823 | 0.0155 | 0.5327 |

The notebook comparison shows Logistic Regression as the strongest cross-validated baseline by AUC. XGBoost is used for the saved deployment artifact because it supports nonlinear feature interactions, imbalance handling via `scale_pos_weight`, and efficient tree-based SHAP explanations.

### Final Model

The production artifact in `models/churn_model.joblib` is an `XGBClassifier` trained with:

- `n_estimators=300`
- `max_depth=4`
- `learning_rate=0.05`
- `subsample=0.8`
- `colsample_bytree=0.8`
- `scale_pos_weight=3`

The API decision threshold is `0.40`, intentionally lower than 0.50 to prioritize recall for retention outreach.

## Key Results

The executed churn analysis notebook reports these held-out XGBoost metrics:

| Metric | Value |
| --- | ---: |
| ROC AUC | 0.7802 |
| F1 | 0.5283 |
| Precision | 0.4403 |
| Recall | 0.6604 |

For churn prevention, recall and ranking quality matter because the business goal is to capture as many true churners as possible within an outreach budget. The project therefore includes threshold and top-N business simulations rather than stopping at AUC.

## Explainability

Model explanations are implemented in `src/explain.py` with SHAP:

- `global_importance()` ranks features by mean absolute SHAP value.
- `local_explanation()` explains a single customer prediction and powers the API response.

Top SHAP drivers from the notebook:

| Rank | Feature | Mean absolute SHAP |
| ---: | --- | ---: |
| 1 | `tenure` | 0.6120 |
| 2 | `Contract_Month-to-month` | 0.4143 |
| 3 | `InternetService_Fiber optic` | 0.3569 |
| 4 | `MonthlyCharges` | 0.3037 |
| 5 | `avg_monthly_charges` | 0.2760 |

Operational interpretation:

- Short-tenure customers are materially higher risk.
- Month-to-month contracts carry higher churn risk than annual contracts.
- Fiber customers with high monthly charges are a key retention segment.
- Missing sticky services such as online security and tech support can indicate weaker customer attachment.

## Business Impact Layer

`src/business.py` converts model scores into retention economics:

- Expected value at a probability threshold
- Threshold sweeps to identify the best decision boundary
- Top-N targeting simulation for fixed outreach capacity
- Cumulative gains analysis

Default assumptions used by the training pipeline:

| Assumption | Value |
| --- | ---: |
| Customer lifetime value saved | $600 |
| Save rate after outreach | 30% |
| Contact cost | $15 |

Generated figures include:

```text
figures/expected_value_curve.png
figures/cumulative_gains.png
```

## Error Analysis and Monitoring

`src/evaluate.py` adds calibration and cohort diagnostics:

- Probability calibration and Brier score
- Reliability diagram
- Cohort metrics by contract type
- Cohort metrics by tenure bucket

`monitor.py` computes Population Stability Index for each feature in a new customer batch:

| PSI | Interpretation |
| ---: | --- |
| `< 0.10` | Stable |
| `0.10-0.25` | Monitor |
| `> 0.25` | Retrain recommended |

## FastAPI Scoring Service

The API in `api/main.py` exposes:

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | Confirms model load status and feature count |
| `POST /predict` | Scores one customer and returns top SHAP drivers |

Example:

```bash
uvicorn api.main:app --reload
```

Open:

```text
http://localhost:8000/docs
```

## Repository Structure

```text
churn-prediction/
├── api/
│   ├── main.py
│   └── schema.py
├── data/raw/
├── figures/
├── models/
│   └── churn_model.joblib
├── notebooks/
│   ├── churn_analysis.ipynb
│   └── ab_analysis.ipynb
├── src/
│   ├── business.py
│   ├── evaluate.py
│   ├── explain.py
│   ├── features.py
│   └── train.py
├── tests/
├── download_data.py
├── generate_data.py
├── monitor.py
├── train_pipeline.py
└── requirements.txt
```

## How to Run

Install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Get data:

```bash
python download_data.py
```

Train the model:

```bash
python train_pipeline.py --figs --compare
```

Run tests:

```bash
pytest -q
```

Run drift monitoring:

```bash
python monitor.py --new-data data/raw/new_customers.csv --output reports/drift_report.csv
```

## Experiment Tracking

Training uses MLflow locally:

```bash
mlflow ui
```

Open:

```text
http://localhost:5000
```

Logged items include parameters, AUC, F1, precision, recall, Brier score, calibration slope, business metrics, figures, and model artifacts.

## Skills Demonstrated

| Area | Evidence |
| --- | --- |
| Supervised learning | Stratified train/test split, four-model benchmark |
| Feature engineering | Shared deterministic transform layer |
| Class imbalance | Balanced baselines and XGBoost `scale_pos_weight` |
| Model evaluation | ROC, PR, confusion matrix, calibration, cohort analysis |
| Explainable AI | SHAP global importance and local API explanations |
| Business analytics | Expected value curves, top-N retention targeting |
| MLOps readiness | Saved model metadata, MLflow logging, drift monitoring |
| Production API | FastAPI service with Pydantic validation |
| Testing | Unit and API tests across features, model, business logic, monitoring |

## Notes for Reviewers

This project is intentionally designed as a realistic data science deliverable rather than a single accuracy-maximizing notebook. The model is framed around decisions: who to contact, why they are at risk, what the expected value is, and when the model should be questioned due to drift.

