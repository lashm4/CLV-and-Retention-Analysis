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

---

## Methods (What I Did)

1. **Cleaning & standardization**
2. **Feature engineering (business-aligned)**
3. **CLV computation (profit-oriented)**
4. **Exploratory Data Analysis**
5. **Modeling: Logistic Regression + Random Forest**
6. **Value at Risk (CLV × churn probability)**

---

## Results

### Logistic Regression Performance

- **Accuracy:** 0.81  
- **ROC AUC:** 0.838  
- **Precision (churners):** 0.66  
- **Recall (churners):** 0.58  
- **F1-score:** 0.61  

---

## Model Comparison: Logistic Regression vs Random Forest

To complement the interpretable Logistic Regression baseline, a second model — **Random Forest** — was trained.

### Random Forest Performance

- **Accuracy:** 0.77  
- **ROC AUC:** 0.831  
- **Precision (churners):** 0.54  
- **Recall (churners):** 0.72  
- **F1-score:** 0.62  

---

### Why Logistic Regression Performs Better

- Dataset is small (~7k rows)  
- Features are mostly linear and categorical  
- Engineered features already capture relationships cleanly  
- Random Forest overfits more easily  

➡️ Logistic Regression generalizes better  
➡️ Random Forest captures more churners (higher recall) but with many false positives  

### Feature Differences

**Logistic Regression reveals root causes.**  
**Random Forest reveals behavioral patterns and non-linear interactions.**

### Business Implications

- **Use Logistic Regression for targeted, high-ROI retention**  
- **Use Random Forest for early churn-risk detection**  
- **Best solution: combine both**  

---

## High-Value At Risk Customers

Criteria:
- **Churn probability ≥ 0.6**
- **CLV ∈ {High, Top}**

**61 High Value Customers** identified as high churn risk.

---

## Financial Impact (Retention ROI Estimation)

If a campaign retains **20%** of these high-value customers:

### **Estimated Saved Margin: €15,971**

Shows the benefits of **precision marketing**.

---

## Marketing Recommendations

1. **Contract Migration Program** (move month-to-month → 12/24 months)  
2. **Fiber Customer Experience Initiative**  
3. **Value-added service bundling**  
4. **Onboarding support for early-life customers**  
5. **Digital engagement improvement**  
6. **Targeted retention for high CLV customers**  

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
source venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

---

