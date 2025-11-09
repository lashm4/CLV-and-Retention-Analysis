import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, RocCurveDisplay, ConfusionMatrixDisplay
)

# -----------------------------
# 0) PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Telecom Churn & CLV – Retention Dashboard", layout="wide")

st.title("📉 Telecom Churn & 💰 CLV - Retention Targeting Dashboard")
st.caption("Logistic Regression & Random Forest • CLV segmentation • Priority list • ROI simulator")

# -----------------------------
# 1) DATA LOAD & PREP
# -----------------------------
DATA_PATH_DEFAULT = "/Users/lashmy/Desktop/telco_customer_churn.csv"

@st.cache_data(show_spinner=False)
def load_raw_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean_and_engineer(df: pd.DataFrame, margin: float = 0.60) -> pd.DataFrame:
    # fix types
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df = df.dropna(subset=['TotalCharges']).copy()

    # strings clean
    for c in df.select_dtypes(include='object').columns:
        df[c] = df[c].astype(str).str.strip()

    # target
    df['ChurnFlag'] = (df['Churn'].str.lower() == 'yes').astype(int)

    # engineered features
    service_cols = [
        'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
    ]
    def yes_no_to_int(x):  # robust mapping
        return 1 if x == 'Yes' else 0

    df['ServicesCount'] = df[service_cols].applymap(lambda x: yes_no_to_int('Yes' if x=='Yes' else 'No')).sum(axis=1)
    df['HasFiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
    df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

    # CLV (realized + margin-adjusted) & segments
    df = df.rename(columns={'tenure':'TenureMonths', 'MonthlyCharges':'MonthlyRevenue'})
    df['CLV_realized'] = df['MonthlyRevenue'] * df['TenureMonths']
    df['CLV_realized_margin'] = df['CLV_realized'] * margin
    df['CLV_segment'] = pd.qcut(
        df['CLV_realized_margin'].rank(method='first'),
        q=4, labels=['Low','Mid','High','Top']
    )
    return df

def get_feature_lists(df: pd.DataFrame):
    keep_for_model = [
        'gender','SeniorCitizen','Partner','Dependents',
        'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
        'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
        'Contract','PaperlessBilling','PaymentMethod',
        'MonthlyRevenue','TenureMonths',
        'ServicesCount','HasFiber','IsMonthToMonth','IsElectronicCheck'
    ]
    X = df[keep_for_model]
    y = df['ChurnFlag']
    categorical = X.select_dtypes(include='object').columns.tolist()
    numeric = X.select_dtypes(exclude='object').columns.tolist()
    return X, y, categorical, numeric, keep_for_model

def make_preprocessor(categorical, numeric):
    pre = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical),
            ('num', 'passthrough', numeric)
        ]
    )
    return pre

@st.cache_resource(show_spinner=False)
def train_models(df: pd.DataFrame):
    X, y, categorical, numeric, keep = get_feature_lists(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    pre = make_preprocessor(categorical, numeric)

    # Logistic Regression
    logreg = Pipeline(steps=[
        ('preprocessor', pre),
        ('classifier', LogisticRegression(max_iter=300, solver='liblinear'))
    ])
    logreg.fit(X_train, y_train)
    lr_proba = logreg.predict_proba(X_test)[:, 1]
    lr_auc = roc_auc_score(y_test, lr_proba)

    # Random Forest
    rf = Pipeline(steps=[
        ('preprocessor', pre),
        ('classifier', RandomForestClassifier(
            n_estimators=300, max_depth=None,
            min_samples_split=5, min_samples_leaf=3,
            random_state=42, class_weight='balanced'
        ))
    ])
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_auc = roc_auc_score(y_test, rf_proba)

    meta = dict(
        X_train_idx=X_train.index, X_test_idx=X_test.index,
        y_train=y_train, y_test=y_test,
        categorical=categorical, numeric=numeric, keep=keep
    )
    metrics = {
        'logreg': {'auc': lr_auc},
        'rf': {'auc': rf_auc}
    }
    return logreg, rf, metrics, meta

# -----------------------------
# 2) SIDEBAR CONTROLS
# -----------------------------
st.sidebar.header("⚙️ Controls")

data_path = st.sidebar.text_input("CSV path", DATA_PATH_DEFAULT)
margin = st.sidebar.slider("Gross margin (for CLV)", 0.3, 0.9, 0.60, 0.01)

# model selection
model_choice = st.sidebar.radio("Model", ["Logistic Regression", "Random Forest"], index=0)

# filters
threshold = st.sidebar.slider("Churn probability threshold", 0.3, 0.9, 0.6, 0.01)
segments = st.sidebar.multiselect("CLV segments to include", ["Low","Mid","High","Top"], default=["High","Top"])
min_clv = st.sidebar.number_input("Min CLV (margin-adjusted €)", value=0.0, step=100.0)

# ROI simulator
st.sidebar.header("💵 ROI Simulator")
retention_rate = st.sidebar.slider("Expected retention rate (targeted group)", 0.05, 0.6, 0.20, 0.01)
cost_per_customer = st.sidebar.number_input("Campaign cost per targeted customer (€)", value=5.0, step=1.0)

# -----------------------------
# 3) LOAD, PREP, TRAIN
# -----------------------------
if not os.path.exists(data_path):
    st.error(f"CSV not found at `{data_path}`. Put the dataset in `data/` or update the path.")
    st.stop()

raw = load_raw_data(data_path)
df = clean_and_engineer(raw, margin=margin)

logreg, rf, metrics, meta = train_models(df)

st.success("Data loaded, features engineered, and models trained.")

# -----------------------------
# 4) METRICS & ROC VIS
# -----------------------------
left, right = st.columns([1, 1])
with left:
    st.subheader("📊 Model Performance (AUC)")
    st.metric("Logistic Regression AUC", f"{metrics['logreg']['auc']:.3f}")
    st.metric("Random Forest AUC", f"{metrics['rf']['auc']:.3f}")
    winner = "Logistic Regression" if metrics['logreg']['auc'] >= metrics['rf']['auc'] else "Random Forest"
    st.caption(f"🏁 Best AUC in this run: **{winner}**")

with right:
    st.subheader("ROC Curve")
    fig, ax = plt.subplots(figsize=(5,4))
    # re-compute ROC display for the chosen model on the test set
    X, y, categorical, numeric, keep = get_feature_lists(df)
    X_train_idx, X_test_idx = meta['X_train_idx'], meta['X_test_idx']
    y_test = meta['y_test']

    model = logreg if model_choice == "Logistic Regression" else rf
    RocCurveDisplay.from_estimator(model, X.loc[X_test_idx], y_test, ax=ax)
    ax.set_title(f"ROC — {model_choice}")
    st.pyplot(fig)

st.divider()

# -----------------------------
# 5) PRIORITY RETENTION LIST
# -----------------------------
st.subheader("🎯 Priority Retention List")

# predicted probabilities for the chosen model
model = logreg if model_choice == "Logistic Regression" else rf
proba = model.predict_proba(df[keep])[:, 1]

scored = df.copy()
scored['churn_prob'] = proba

priority = scored[
    (scored['churn_prob'] >= threshold) &
    (scored['CLV_segment'].isin(segments)) &
    (scored['CLV_realized_margin'] >= min_clv)
].sort_values(['churn_prob','CLV_realized_margin'], ascending=[False, False])

st.write(f"Customers matching filters: **{len(priority)}**")
st.dataframe(
    priority[['customerID','churn_prob','CLV_segment','CLV_realized_margin',
              'MonthlyRevenue','TenureMonths','ServicesCount','Contract','PaymentMethod']].head(50),
    use_container_width=True
)

# download button
csv_bytes = priority.to_csv(index=False).encode('utf-8')
st.download_button("⬇️ Download priority list (CSV)", data=csv_bytes, file_name="priority_retention_list.csv", mime="text/csv")

st.divider()

# -----------------------------
# 6) ROI SIMULATOR
# -----------------------------
st.subheader("💵 Retention ROI Simulator")

targeted_n = len(priority)
retained_customers = targeted_n * retention_rate
saved_margin = priority['CLV_realized_margin'].sum() * retention_rate
campaign_cost = targeted_n * cost_per_customer
net_benefit = saved_margin - campaign_cost

mcol1, mcol2, mcol3, mcol4 = st.columns(4)
mcol1.metric("Targeted customers", f"{targeted_n}")
mcol2.metric("Expected retained (est.)", f"{retained_customers:.0f}")
mcol3.metric("Saved margin (est.)", f"€{saved_margin:,.0f}")
mcol4.metric("Campaign cost", f"€{campaign_cost:,.0f}")

mcol5, _ = st.columns([1,1])
mcol5.metric("Net benefit (est.)", f"€{net_benefit:,.0f}")

st.caption("Assumes retained customers keep their **margin-adjusted CLV realized to date**. You can tune margin, threshold, and cost to stress test ROI.")

st.divider()

# -----------------------------
# 7) EXPLANATIONS
# -----------------------------
with st.expander("ℹ️ How to read these results"):
    st.markdown("""
- **Model**: choose between interpretable Logistic Regression or higher-recall Random Forest.
- **Threshold**: higher ⇒ smaller, riskier segment; lower ⇒ larger, noisier segment.
- **CLV filters**: focus on customers where losing them hurts most.
- **ROI**: experiment with retention rate and campaign cost to see business outcomes.
    """)

