# Telecom Churn & CLV - Marketing Analytics Case 

**Goal:** Predict customer churn and prioritize high-value, high-risk customers for data-driven retention.  
**What this shows:** end to end analytics, cleaning, feature engineering, CLV, modeling, evaluation, and business impact.

---

## Business Context

Telecoms face recurring churn (contract cancellations). The impact is not equal across customers: losing a Top-CLV customer hurts more.  
This project combines churn probability with margin adjusted CLV to focus retention spend where it drives the highest value.

---

## Data

- Public Telco churn dataset (~7k customers, 20+ features) with demographics, products, contract type, payment method, charges, and churn flag.  
- File used: `telco_customer_churn.csv`  
- Typical columns: `tenure`, `MonthlyCharges`, `InternetService`, `Contract`, `PaymentMethod`, `Churn`, etc.

> Source: “Telco Customer Churn” (Kaggle). Add instructions for downloading if you won’t commit the CSV.

---

## Methods (What I Did)

1. **Cleaning & standardization**
   - Cast `TotalCharges` → numeric; drop invalid rows; trim strings
   - Target `ChurnFlag` = 1 if churned, else 0

2. **Feature engineering (business-aligned)**
   - `ServicesCount` = count of value-added services (security/backup/tech support/streaming/etc.)
   - `HasFiber`, `IsMonthToMonth`, `IsElectronicCheck` = 0/1 flags for key behaviors
   - Rename: `tenure` → `TenureMonths`; `MonthlyCharges` → `MonthlyRevenue`

3. **CLV (profit-oriented)**
   - **Realized CLV** = `MonthlyRevenue × TenureMonths`
   - **Margin-adjusted CLV** = `CLV_realized × 0.60` (assumed gross margin)
   - **CLV segments** = quartiles: `Low`, `Mid`, `High`, `Top`

4. **EDA**
   - Churn by contract type, payment method, tenure, services
   - CLV distribution and CLV vs. churn
   - Correlations for key numeric features

5. **Modeling**
   - **Baseline:** Logistic Regression (interpretable)
   - One-hot encoding for categoricals; train/test split (75/25)
   - Metrics: ROC AUC, classification report, confusion matrix
   - Coefficient analysis → top churn drivers

6. **Value at Risk**
   - Join churn_prob with CLV to identify High/Top CLV customers with high churn risk
   - Simple impact calc: saved margin if campaign retains X% of that group

---

## Results

- **Churn rate:** ~26.6%
- **Model (Logistic Regression)**
  - ROC AUC: ~0.83 (typical)
  - Key drivers (direction):
    - `Contract: Month-to-month` (**↑ churn**)
    - `PaymentMethod: Electronic check` (**↑ churn**)
    - `TenureMonths` (**↓ churn**)
    - `ServicesCount` (**↓ churn**)
- **High-value & high-risk segment**
  - Threshold: `churn_prob ≥ 0.6` and `CLV_segment ∈ {High, Top}`
  - Customers in segment: ~250 (example)
  - **Estimated saved margin** (retain 20% of this segment): **€XX,XXX**

---

## Why the Features

- **Contract & Payment:** strongest actionable levers (move M2M → 12/24-month; nudge payment method away from “electronic check”).
- **Tenure:** early-life churn is highest → onboarding, first-90-day experience.
- **ServicesCount:** bundling increases stickiness → cross-sell value-add services.
- **CLV:** focus spend on customers where the value at risk is largest.

---

## Recommendations

1. **Contract migration:** target month-to-month customers for longer terms.  
2. **Payment behavior:** incentivize shift from electronic checks to digital payments.  
3. **Bundle strategy:** promote add on services to increase engagement.  
4. **Onboarding care:** improve early tenure experience to reduce early churn.  
5. **Value-focused retention:** prioritize Top/High CLV customers above risk threshold.

---

## Repository Structure

```
.
├── analysis.py
├── results/
│   ├── telco_clean_with_clv.csv
│   ├── test_predictions.csv
│   ├── priority_logreg.csv
│   ├── logreg_feature_importance.csv
│   └── charts/
├── data/
│   └── telco_customer_churn.csv
├── README.md
└── requirements.txt
```

---

## How to Run

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
pip install -r requirements.txt

# Run the analysis
python analysis.py
```

> Adjust `file_path` in the script to point to your dataset (recommended: `data/telco_customer_churn.csv`).

---

## Next Steps

- Compare with Random Forest or XGBoost for performance.  
- Build an interactive Streamlit dashboard.  
- Compute expected CLV for next 6 months (discounted).  
- Simulate retention scenarios for ROI estimation.

---
