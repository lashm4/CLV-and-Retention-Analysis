import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, RocCurveDisplay
from sklearn.ensemble import RandomForestClassifier


###Load data###
file_path = "data/telco_customer_churn.csv"
df = pd.read_csv(file_path)

print(os.getcwd())   
print(os.listdir())

#Quick check data
print(df.head())
print(df.info())

###Cleaning and standardization###
#convert TotalCharges to numeric (it comes as object)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
#drop rows where TotalCharges missing
df = df.dropna(subset=['TotalCharges']).copy()

#target
df['ChurnFlag'] = (df['Churn'].str.strip().str.lower() == 'yes').astype(int)

#consistent strings
for c in df.select_dtypes(include='object').columns:
    df[c] = df[c].astype(str).str.strip()

###engineered features aligned to business###
#describe the products a customer has:
service_cols = [
    'PhoneService','MultipleLines','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies'
]

def yes_no_to_int(x):
    return 1 if x == 'Yes' else 0

df['ServicesCount'] = df[service_cols].applymap(lambda x: yes_no_to_int('Yes' if x=='Yes' else 'No')).sum(axis=1)
df['HasFiber'] = (df['InternetService'] == 'Fiber optic').astype(int)
df['IsMonthToMonth'] = (df['Contract'] == 'Month-to-month').astype(int)
df['IsElectronicCheck'] = (df['PaymentMethod'] == 'Electronic check').astype(int)

#CLV
df.rename(columns={'tenure':'TenureMonths', 'MonthlyCharges':'MonthlyRevenue'}, inplace=True) #tenure:how many months the customer has been active;MonthlyCharges:what they pay each month
df['CLV_realized'] = df['MonthlyRevenue'] * df['TenureMonths'] #How much money has this customer paid us so far?

#margin-adjusted
MARGIN = 0.6
df['CLV_realized_margin'] = df['CLV_realized'] * MARGIN

#CLV segments (quartiles)
df['CLV_segment'] = pd.qcut(df['CLV_realized_margin'].rank(method='first'), q=4, labels=['Low','Mid','High','Top'])

###quick sanity/checks views###
keep_for_model = [
    'gender','SeniorCitizen','Partner','Dependents',
    'PhoneService','MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
    'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
    'Contract','PaperlessBilling','PaymentMethod',
    'MonthlyRevenue','TenureMonths',
    #engineered
    'ServicesCount','HasFiber','IsMonthToMonth','IsElectronicCheck'
]
print("Model feature count:", len(keep_for_model))

#leakage-aware drop for model
drop_in_model = ['customerID','TotalCharges','Churn','CLV_realized','CLV_realized_margin','CLV_segment']

print(df[keep_for_model + ['ChurnFlag']].head(3))

###EDA###
print("Dataset shape:", df.shape)
print("\nChurn rate:", round(df['ChurnFlag'].mean(), 3))

#Describe numeric features
print("\nNumeric feature summary:")
display(df[['MonthlyRevenue','TenureMonths','TotalCharges','CLV_realized_margin','ServicesCount']].describe())

#CLV segment counts
print("\nCLV segment distribution:")
print(df['CLV_segment'].value_counts().sort_index())

#Churn rate by contract type
plt.figure(figsize=(6,4))
sns.barplot(x='Contract', y='ChurnFlag', data=df, estimator=np.mean)
plt.title('Churn Rate by Contract Type')
plt.ylabel('Churn Rate')
plt.show()

#Churn rate by payment method
plt.figure(figsize=(7,4))
sns.barplot(x='PaymentMethod', y='ChurnFlag', data=df, estimator=np.mean)
plt.title('Churn Rate by Payment Method')
plt.xticks(rotation=30, ha='right')
plt.show()

#Churn vs. tenure (loyalty)
plt.figure(figsize=(6,4))
sns.histplot(data=df, x='TenureMonths', hue='Churn', multiple='stack', bins=30)
plt.title('Tenure Distribution by Churn')
plt.show()

#Churn vs. ServicesCount
plt.figure(figsize=(6,4))
sns.barplot(x='ServicesCount', y='ChurnFlag', data=df, estimator=np.mean)
plt.title('Churn Rate by Number of Services')
plt.ylabel('Churn Rate')
plt.show()

#CLV vs. churn
plt.figure(figsize=(6,4))
sns.boxplot(x='Churn', y='CLV_realized_margin', data=df)
plt.title('CLV (margin-adjusted) by Churn')
plt.ylabel('Profit-based CLV (€)')
plt.show()

###Quick correlation check for numeric drivers###
plt.figure(figsize=(8,6))
sns.heatmap(
    df[['MonthlyRevenue','TenureMonths','TotalCharges','ServicesCount','ChurnFlag']].corr(),
    annot=True, cmap='coolwarm', center=0
)
plt.title('Correlation Matrix (Numeric Features)')
plt.show()

###Logistic Regression###
#Split data
y = df['ChurnFlag']
X = df[keep_for_model]

#identify which columns are categorical vs numeric
categorical = X.select_dtypes(include='object').columns.tolist()
numeric = X.select_dtypes(exclude='object').columns.tolist()

print("Categorical:", categorical)
print("Numeric:", numeric)

#train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

#preprocessing pipeline
preprocess = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical),
        ('num', 'passthrough', numeric)
    ]
)

#logistic regression pipeline
logreg_model = Pipeline(steps=[
    ('preprocessor', preprocess),
    ('classifier', LogisticRegression(max_iter=300, solver='liblinear'))
])

#fit the model
logreg_model.fit(X_train, y_train)

#predict probabilities and classes
y_pred_proba = logreg_model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

#metrics
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC AUC:", round(roc_auc_score(y_test, y_pred_proba), 3))

#plot ROC curve
RocCurveDisplay.from_estimator(logreg_model, X_test, y_test)
plt.title("ROC Curve - Logistic Regression (Churn Model)")
plt.show()

###Feature importance (what drives churn)###
#get feature names after one-hot encoding
ohe = logreg_model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical)
all_feature_names = np.concatenate([cat_feature_names, numeric])

#get coefficients from logistic regression
coefficients = logreg_model.named_steps['classifier'].coef_[0]

#build dataframe
feature_importance = pd.DataFrame({
    'feature': all_feature_names,
    'coefficient': coefficients
})

#absolute importance
feature_importance['abs_coeff'] = feature_importance['coefficient'].abs()
feature_importance = feature_importance.sort_values('abs_coeff', ascending=False)

print(feature_importance.head(10))

#visualize top 10
plt.figure(figsize=(8,6))
sns.barplot(y='feature', x='coefficient', data=feature_importance.head(10))
plt.title('Top 10 Feature Coefficients (Logistic Regression)')
plt.xlabel('Coefficient Value')
plt.ylabel('Feature')
plt.show()

#attach churn probabilities to test set
X_test_results = X_test.copy()
X_test_results['churn_prob'] = y_pred_proba
X_test_results['actual_churn'] = y_test.values
X_test_results['CLV_segment'] = df.loc[X_test.index, 'CLV_segment']
X_test_results['CLV_realized_margin'] = df.loc[X_test.index, 'CLV_realized_margin']

X_test_results.head()

#threshold for "high risk"
risk_threshold = 0.6

priority_customers = X_test_results[
    (X_test_results['churn_prob'] >= risk_threshold) &
    (X_test_results['CLV_segment'].isin(['High', 'Top']))
].copy()

print(f"High-value & high-risk customers: {priority_customers.shape[0]}")
priority_customers[['MonthlyRevenue','TenureMonths','ServicesCount','churn_prob','CLV_segment']].head()

###Visualizations - predictions###
#Churn probability vs CLV
plt.figure(figsize=(7,5))
sns.scatterplot(
    data=X_test_results, x='CLV_realized_margin', y='churn_prob',
    hue='CLV_segment', alpha=0.6
)
plt.title('Churn Probability vs CLV')
plt.xlabel('Profit-based CLV (€)')
plt.ylabel('Predicted Churn Probability')
plt.show()

#Average churn probability by CLV segment
plt.figure(figsize=(6,4))
sns.barplot(x='CLV_segment', y='churn_prob', data=X_test_results, estimator=np.mean)
plt.title('Average Predicted Churn by CLV Segment')
plt.ylabel('Avg. Churn Probability')
plt.show()

#Confusion matrix
ConfusionMatrixDisplay.from_estimator(logreg_model, X_test, y_test, cmap='Blues')
plt.title("Confusion Matrix - Logistic Regression")
plt.show()

#Suppose retention campaign keeps 20% of high-risk customers
retention_rate = 0.20

if priority_customers.empty:
    print("No customers matched the 'high-value & high-risk' criteria at the current threshold.")
else:
    saved_value = (priority_customers['CLV_realized_margin'] * retention_rate).sum()
    print(f"Estimated saved margin if retaining 20% of high-value at-risk customers: €{saved_value:,.0f}")


####################################################
###RANDOM FOREST MODEL (MODEL 2)###
####################################################

print("\n============= RANDOM FOREST MODEL =============\n")

#Random Forest pipeline (same preprocessing)
rf_model = Pipeline(steps=[
    ('preprocessor', preprocess),
    ('classifier', RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=5,
        min_samples_leaf=3,
        random_state=42,
        class_weight='balanced'
    ))
])

#Train
rf_model.fit(X_train, y_train)

#Predict
rf_pred_proba = rf_model.predict_proba(X_test)[:, 1]
rf_pred = (rf_pred_proba >= 0.5).astype(int)

#Metrics
print("\nClassification Report (Random Forest):\n")
print(classification_report(y_test, rf_pred))

print("ROC AUC (Random Forest):", round(roc_auc_score(y_test, rf_pred_proba), 3))

#ROC Curve
RocCurveDisplay.from_estimator(rf_model, X_test, y_test)
plt.title("ROC Curve – Random Forest")
plt.show()

#Confusion Matrix
ConfusionMatrixDisplay.from_estimator(rf_model, X_test, y_test, cmap='Purples')
plt.title("Confusion Matrix – Random Forest")
plt.show()

###############################
###Random Forest Importance ###
###############################

#get one-hot encoded feature names
ohe = rf_model.named_steps['preprocessor'].named_transformers_['cat']
cat_feature_names = ohe.get_feature_names_out(categorical)
all_rf_features = np.concatenate([cat_feature_names, numeric])

#Extract importances
rf_importances = rf_model.named_steps['classifier'].feature_importances_

rf_feature_imp = pd.DataFrame({
    'feature': all_rf_features,
    'importance': rf_importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Random Forest Features:\n")
print(rf_feature_imp.head(10))

plt.figure(figsize=(8,6))
sns.barplot(x='importance', y='feature', data=rf_feature_imp.head(10), palette='viridis')
plt.title('Top 10 Feature Importances – Random Forest')
plt.xlabel('Importance Score')
plt.ylabel('Feature')
plt.show()

#################################################
###Compare Logistic Regression vs Random Forest###
#################################################

logistic_auc = roc_auc_score(y_test, y_pred_proba)
rf_auc = roc_auc_score(y_test, rf_pred_proba)

print("\nMODEL PERFORMANCE COMPARISON")
print("----------------------------------")
print(f"Logistic Regression AUC: {logistic_auc:.3f}")
print(f"Random Forest AUC:      {rf_auc:.3f}")

if rf_auc > logistic_auc:
    print("\n Random Forest performs BETTER than Logistic Regression.")
else:
    print("\nLogistic Regression performs BETTER (rare for churn models).")

