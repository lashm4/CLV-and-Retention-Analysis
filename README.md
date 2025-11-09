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

### Churn Model Performance (Logistic Regression)

The logistic regression model provides strong and interpretable predictive power:

- **Accuracy:** 0.81  
- **ROC AUC:** 0.838  
- **Precision (churners):** 0.66  
- **Recall (churners):** 0.58  
- **F1-score (churners):** 0.61  

 **Interpretation:**
- The model correctly classifies 81% of customers.
- AUC of 0.838 indicates strong separation ability between churners and non-churners.
- Recall = 0.58 means we identify over half of all churners, suitable for a retention use case where we prefer recall > precision.
- Precision = 0.66 ensures the retention budget is not wasted on too many false positives.

---

## Key Drivers of Churn (Behavioral Insights)

These are the most influential features (based on model coefficients) and what they mean strategically:

| Feature | Impact | Marketing Interpretation |
|--------|--------|--------------------------|
| **HasFiber / InternetService_Fiber optic** | **↑ churn** | Fiber customers churn more — often due to competitive fiber pricing or installation/quality expectations. Opportunity for **fiber-specific loyalty and experience improvements**. |
| **IsMonthToMonth** | **↑ churn** | The strongest churn predictor. Month-to-month users are highly unstable. **Top priority for contract-migration campaigns**. |
| **Contract_Two year** | **↓ churn** | Long-term commitment drastically reduces churn. These customers are stable and should be nurtured. |
| **StreamingTV_Yes & StreamingMovies_Yes** | **↑ churn** | Users of OTT-like services may be more price-sensitive or exposed to competition. Opportunity for **bundle discounts or content personalization**. |
| **OnlineSecurity_Yes & TechSupport_Yes** | **↓ churn** | Value-added services increase stickiness. **Cross-selling VAS** is a direct retention lever. |
| **PaperlessBilling_Yes** | **↑ churn** | Digital-first customers churn faster; they interact less with traditional touchpoints. Great opportunity for **digital engagement campaigns**. |
| **MultipleLines_Yes** | **↑ churn** | Multi-line customers can be price-sensitive; they benefit from **multi-line discounts or family bundles**. |

---

## High-Value At Risk Customers

Using two criteria:
- **Churn probability ≥ 0.6**  
- **CLV segment ∈ {High, Top}**

we identified:

### **61 High Value Customers at High Risk of Churning**

These customers generate disproportionately higher revenue and margin. Losing them is significantly more damaging than losing low CLV users, something to watch out.

---

## Financial Impact (Retention ROI Estimation)

Assuming:
- A retention campaign can retain **20%** of these high value at risk customers,  
- And using margin-adjusted CLV,

### **Estimated Saved Margin: €15,971**

 **Interpretation:**  
With a very small, highly targeted campaign focused on only 61 customers, the business could realistically preserve nearly €16k in profit

This shows the value of precision marketing, spending budget only where ROI is highest.

---

## Marketing Recommendations

Based on the churn drivers and CLV insights, here are strategic actions:

### 1. **Contract Migration Program**
- Target month to month customers with:
  - Discounted 12- or 24-month plans  
  - “Loyalty extension” offers  
  - Free months or bonus data  
- Expected impact: lower churn by tackling the strongest predictor.

---

### 2. **Fiber Customer Experience Initiative**
Fiber customers show unusually high churn.
Focus areas:
- Proactive tech support  
- Network quality monitoring  
- Onboarding calls after installation  
- Fiber-specific loyalty perks (e.g., “fiber VIP service”)

---

### 3. **Value-Added Service Bundling**
Because:
- `OnlineSecurity_Yes` → lower churn  
- `TechSupport_Yes` → lower churn  

Offer bundles such as:
- “Home Protection Pack” (Internet + Security + Tech Support)  
- Bundle discounts for StreamingTV / Movies to counter OTT competition  

These increase stickiness and raise CLV.

---

### 4. **Reduce Early-Life Churn (Tenure-Based Triggers)**
Customers with low tenure churn the most.  
Implement:
- **Welcome journey emails**  
- First 30-day check-in  
- Proactive assistance for new customers  
- Targeted offers to improve first month experience  

---

### 5. **Digital Customer Engagement Push**
Since paperless billing users churn more:
- Launch personalized in-app messages  
- Push personalized recommendations  
- Offer loyalty points for engagement  
- Introduce “digital only loyalty rewards”

---

### 6. **Priority Retention List**
Focus retention resources on:
- High CLV  
- High predicted churn  
- Low services count  
- Month to month contracts  

This gives maximum ROI per euro spent

---

## Key Visuals (Saved in `/results/charts/`)
- ROC Curve (model quality)  
- Confusion Matrix  
- Churn by Contract Type  
- Churn by Payment Method  
- Tenure Distribution by Churn  
- CLV vs Churn (boxplot)  
- Churn Probability vs CLV (scatter plot)

---

## Final Takeaway

> By combining churn probability with CLV segmentation, we identify where churn actually hurts the business.  
> Instead of mass retention campaigns, this model enables precision targeting, saving €15,971 in margin by focusing on only 61 strategic customers.

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
- Simulate retention scenarios for ROI estimation.

---
