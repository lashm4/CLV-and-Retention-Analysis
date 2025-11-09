# Telecom Churn & CLV – Marketing Analytics Case Study

## 🎯 Project Goal
Predict telecom customer churn **and** quantify business impact by combining churn probability with **Customer Lifetime Value (CLV)**.  
The goal is not only to predict *who* will churn, but *which churners matter most* for profit.

---

# 1. Business Context

Telecom churn is typically 20–30% per year.  
But **not all churn is equally damaging**:

- Losing a *Low CLV* user has small financial impact.  
- Losing a *Top CLV* user significantly affects profit.  

This project solves that by combining:

✔️ **Churn Prediction**  
✔️ **CLV Segmentation**  
✔️ **High‑value retention targeting**  

---

# 2. Data Description

Dataset: Public “Telco Customer Churn” dataset (~7,000 customers).  
Includes demographics, service usage, contract type, billing method, charges, and churn flag.

### Key fields used:
`gender`, `SeniorCitizen`, `Partner`, `Dependents`,  
`InternetService`, `Contract`, `PaymentMethod`,  
`MonthlyCharges`, `tenure`, `Churn`,  
and many value‑added service indicators.

---

# 3. Methods & Pipeline

## 3.1 Data Cleaning
- Convert `TotalCharges` to numeric  
- Remove invalid rows  
- Standardize categorical labels  
- Create binary churn target: **ChurnFlag**

## 3.2 Feature Engineering (Business-Aligned)
| Feature | Purpose |
|--------|---------|
| **ServicesCount** | Measures stickiness: more services = lower churn |
| **HasFiber** | Fiber users often have higher expectations; track separately |
| **IsMonthToMonth** | Main churn driver; structural instability |
| **IsElectronicCheck** | Known risky billing method |
| **CLV_realized** | Revenue × tenure |
| **CLV_realized_margin** | Adds profitability meaning |
| **CLV_segment** | Groups customers into Low–Top |

## 3.3 Modeling Approach
Two models optimized and compared:

### 1️⃣ Logistic Regression (baseline, interpretable)  
### 2️⃣ Random Forest (nonlinear, higher variance)

Both models used:

- Train/test split  
- One-hot encoding  
- ROC AUC evaluation  
- Feature importance analysis  

---

# 4. Model Results & Comparison

## 4.1 Logistic Regression – Results

**Accuracy:** 0.81  
**ROC AUC:** 0.838  
**Recall (churners):** 0.58  
**Precision (churners):** 0.66  

### Why Logistic Regression Performs Well
- Relationships between churn & features are mostly *linear and interpretable*.  
- Contract type, payment method, tenure explain churn clearly with monotonic patterns.  
- The engineered features (IsMonthToMonth, HasFiber, ServicesCount) align very well with linear coefficients.

### What It Means
This model is reliable for:
- Understanding *why* customers churn  
- Designing retention strategies  
- Explaining results to marketing leaders  

---

## 4.2 Random Forest – Results

**Accuracy:** 0.77  
**ROC AUC:** 0.831  
**Recall (churners):** 0.72  
**Precision (churners):** 0.54  

### Why Random Forest Underperformed Slightly
- Dataset is relatively small (~7k rows), forests benefit more from large datasets.  
- Signal-to-noise ratio is high → linear model is sufficient.  
- Many features are correlated (e.g., HasFiber & InternetService_Fiber) → forests struggle.  
- Churn behavior is *structural*, not random → LR captures structure better.

### What It Means
RF is great at recall (captures more churners), but:
- Lower precision = more false positives  
- Less interpretable  

---

## 4.3 Model Comparison Summary

| Model | Accuracy | AUC | Strength |
|-------|----------|------|----------|
| **Logistic Regression** | 0.81 | **0.838** | Interpretable, stable |
| **Random Forest** | 0.77 | 0.831 | Higher recall but more noise |

➡️ **Winner: Logistic Regression** (better AUC, more actionable)

---

# 5. Churn Drivers – What Causes Churn?

## 5.1 Top Drivers from Logistic Regression

### 🚩 Increases Churn Risk
- **Month-to-month contract**  
- **Fiber users** (sensitive to quality & pricing)  
- **Electronic check payment**  
- **Streaming services active**  
- **Multiple lines** (price-sensitive families)

### 🟩 Reduces Churn
- **Two-year contract**  
- **OnlineSecurity**  
- **TechSupport**  

This creates highly actionable levers for marketing, CX, and product teams.

---

# 6. High-Value At-Risk Customers

### Criteria:
- Churn probability ≥ **0.6**  
- CLV segment ∈ **High / Top**

### Result:
**61 customers** fall into this high-risk/high-value category.

These customers represent the **most important group** for retention.

---

# 7. Financial Impact Analysis

Assuming:
- Retention campaign effectiveness = **20%**
- Using **margin-adjusted CLV**

### 💰 Estimated Profit Saved:
# **€15,971**

This shows how **data-driven targeting** generates measurable financial value.

---

# 8. Marketing Recommendations (Expanded & Strategic)

## 8.1 📌 Contract Migration Program  
**Goal:** Reduce churn among month-to-month users (strongest churn driver).  
**Actions:**  
- Offer **12/24‑month discounts**, loyalty price locks  
- Bundle services with long-term contract (e.g., tech support included)  
- Provide “stay rewards” for commitment  

**Expected impact:**  
30–40% churn reduction in this segment.

---

## 8.2 📌 Fiber Customer Experience Improvement  
Fiber customers show elevated churn due to:

- Installation issues  
- Competitive offers in fiber markets  
- High performance expectations  

**Actions:**  
- Proactive quality checks  
- 24h installation follow-up  
- Dedicated fiber-quality support line  
- VIP fiber loyalty benefits  

---

## 8.3 📌 Value-Added Service Bundling
Because VAS (tech support, security) *reduce* churn:

**Actions:**
- Bundle VAS into new contract offers  
- Create “home protection” or “family bundle” packages  
- Offer 3-month free trials for VAS services  
- Add streaming bundles to reduce OTT substitution risk  

---

## 8.4 📌 Early-Life Churn Reduction (first 60–90 days)
Customers with low tenure churn the most.

**Actions:**  
- Welcome onboarding emails  
- “First 30 days check-in”  
- Tutorials for modem setup & troubleshooting  
- Free temporary upgrades in first month  

---

## 8.5 📌 Improve Digital Engagement
Paperless billing users churn more.

Reasons:
- Low engagement  
- Fewer touchpoints  
- More anonymous relationship  

**Actions:**
- Personalized app notifications  
- Reward points for digital interactions  
- New “digital-only loyalty program”  
- Goal: increase emotional engagement  

---

## 8.6 📌 Precision Targeting for High CLV Customers
Focus retention budget where value at risk is highest:

**Target signals:**
- High churn probability  
- High CLV  
- Low services count  
- Month-to-month contract  

**Actions:**
- Personalized outbound campaigns  
- Dedicated retention agent  
- High-value loyalty perks  

---

# 9. How to Improve the Models Further

### ✔️ Try XGBoost or LightGBM  
Often delivers AUC > 0.88 on this dataset.

### ✔️ Hyperparameter tuning  
Grid search / randomized search.

### ✔️ Create deeper behavioral features  
- Contract age  
- Price increases  
- Support call records (if available)  
- Competitor availability in region  

### ✔️ SMOTE  
To handle churn class imbalance.

### ✔️ Calibrate probabilities  
For more reliable retention budget allocation.

---

# 10. Repository Structure

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

# 11. How to Run

```bash
python -m venv venv
source venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
python analysis.py
```

---

# 12. Final Takeaway

> **This project proves that churn prediction + CLV segmentation enables precision retention with measurable ROI.**  
> Instead of mass marketing, the company can focus on **61 key customers** and save **€15.9k in profit** with highly targeted campaigns.

