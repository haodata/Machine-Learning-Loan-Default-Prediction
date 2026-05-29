# Loan Default Prediction

A machine learning project that predicts whether a loan application will be **approved or rejected**, using a Random Forest classifier trained on CIBIL credit data.

---

## Results

| Metric | Score |
|--------|-------|
| Accuracy | 99.53% |
| Precision | 99.25% |
| Recall | 100.00% |
| F1 Score | 99.62% |
| ROC-AUC | 99.96% |

> **+0.0347 improvement in F1** over the CIBIL rule baseline (0.9615 → 0.9970)

---

## Project Structure

```
loan-default-prediction/
│
├── data/
│   └── loan_approval_dataset.csv   # Raw dataset (4,269 records)
│
├── notebooks/
│   └── loan_default_prediction.ipynb   # Full pipeline notebook
│
├── README.md
└── requirements.txt
```

---

## Dataset

- **Source:** CIBIL-based consumer credit dataset
- **Records:** 4,269 loan applications
- **Target:** `loan_status` — Approved (62%) / Rejected (38%)

### Original Features (13 columns)

| Column | Type | Treatment |
|--------|------|-----------|
| `loan_id` | ID | Dropped |
| `no_of_dependents` | Numeric | Kept |
| `education` | Categorical | Encoded → `education_Graduate` |
| `self_employed` | Categorical | Encoded → `self_employed_Yes` |
| `income_annum` | Numeric | Kept |
| `loan_amount` | Numeric | Kept |
| `loan_term` | Numeric | Kept |
| `cibil_score` | Numeric | Kept (raw value preserved) |
| `residential_assets_value` | Numeric | Merged → `total_assets` |
| `commercial_assets_value` | Numeric | Merged → `total_assets` |
| `luxury_assets_value` | Numeric | Merged → `total_assets` |
| `bank_asset_value` | Numeric | Merged → `total_assets` |
| `loan_status` | Target | Label encoded |

### Engineered Features (4 new)

| Feature | Formula | Rationale |
|---------|---------|-----------|
| `total_assets` | `residential + commercial + luxury + bank_asset_value` | Consolidates 4 asset columns |
| `loan_to_income_ratio` | `loan_amount / income_annum` | Loan size relative to earnings |
| `payment_to_income` | `(loan_amount / loan_term) / income_annum` | Monthly repayment burden (#2 most important feature) |
| `income_per_dependent` | `income_annum / (no_of_dependents + 1)` | Effective disposable income per household member |

**Final model input: 11 features**

---

## Pipeline

```
Raw Data (13 cols)
    │
    ├── Drop: loan_id
    ├── Encode: education, self_employed
    ├── Merge: 4 asset columns → total_assets
    ├── Engineer: loan_to_income_ratio, payment_to_income, income_per_dependent
    │
    ▼
Feature Matrix (11 cols)
    │
    ├── Train/Test Split (80/20, stratified)
    ├── Baseline 1: DummyClassifier         → F1 = 0.7671
    ├── Baseline 2: CIBIL Rule (score ≥ 550) → F1 = 0.9615
    │
    ├── Models evaluated (tuned with GridSearchCV):
    │     ├── Logistic Regression  → F1 = 0.9483  (below baseline)
    │     ├── KNN                  → F1 = 0.9571  (below baseline)
    │     ├── Gradient Boosting    → F1 = 0.9970  ✅ beats baseline
    │     └── Random Forest        → F1 = 0.9970  ✅ beats baseline
    │
    └── Final Model: Random Forest (optimal threshold = 0.60)
```

---

## Why F1, Not Accuracy?

With a 62/38 class split, a model that always predicts "Approved" would score **62% accuracy** while catching zero defaulters. This project uses **F1 Score** and **ROC-AUC** as primary metrics because they penalise this behaviour and better reflect real-world performance on imbalanced data.

---

## Why Random Forest Over Gradient Boosting?

Both achieved F1 = 0.9970. Random Forest was selected because:
- Faster to train (trees built in parallel)
- More robust to overfitting
- Easier to interpret and deploy

---

## Feature Importance

`cibil_score` dominates with **83% importance**, confirming why the CIBIL rule baseline already performs well. The engineered `payment_to_income` ratio is the second most important feature at ~6%.

---

## Confusion Matrix (Test Set)

|  | Predicted: Rejected | Predicted: Approved |
|--|--|--|
| **Actual: Rejected** | 319 ✅ | 4 ❌ (FP) |
| **Actual: Approved** | 0 ✅ | 531 ✅ |

- **FP = 4** — Bank approves a loan that will default → direct financial loss
- **FN = 0** — Zero creditworthy applicants wrongly rejected

FP is the more severe error in this domain. The model eliminates FN entirely.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

```
pandas
numpy
scikit-learn
matplotlib
seaborn
jupyter
```

### Run

```bash
jupyter notebook notebooks/loan_default_prediction.ipynb
```

---

## Tech Stack

- **Language:** Python 3
- **Libraries:** scikit-learn, pandas, numpy, matplotlib, seaborn
- **Model:** Random Forest Classifier (GridSearchCV tuned)
- **Evaluation:** F1 Score, ROC-AUC, Confusion Matrix

---

## Next Steps

1. **Deploy** — wrap the model in a REST API (e.g. FastAPI) for real-time scoring
2. **Monitor** — track data distribution drift to catch when retraining is needed
3. **Retrain** — schedule periodic retraining as new loan data accumulates
