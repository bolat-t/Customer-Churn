# compute_results.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, classification_report, confusion_matrix
import joblib

# =============================
# 1. LOAD AND CLEAN DATA
# =============================
df = pd.read_csv("Telco-Customer-Churn.csv")

# Numeric columns
numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Convert yes/no columns to 1/0
yes_no_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn',
               'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
               'StreamingTV','StreamingMovies']
for col in yes_no_cols:
    if col in df.columns:
        df[col] = df[col].map({'Yes':1,'No':0})

# Features & target
features = ['tenure','MonthlyCharges','TotalCharges','Contract','InternetService','OnlineSecurity','TechSupport']
X = df[features].copy()
y = df['Churn']

# Encode categorical features
X['Contract'] = pd.Categorical(X['Contract'], categories=df['Contract'].unique()).codes
X['InternetService'] = pd.Categorical(X['InternetService'], categories=df['InternetService'].unique()).codes

# Fill any remaining NaNs
if X.isnull().sum().sum() > 0:
    X = X.fillna(0)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# =============================
# 2. SCALE FEATURES
# =============================
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# =============================
# 3. TRAIN MODELS
# =============================
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:,1]
    
    results[name] = {
        'model': model,
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_proba),
        'f1_score': f1_score(y_test, y_pred),
        'y_pred': y_pred
    }

# =============================
# 4. MODEL COMPARISON
# =============================
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results]
})

best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
best_model = results[best_model_name]['model']

# =============================
# 5. FEATURE IMPORTANCE
# =============================
if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
else:
    # Logistic Regression: use absolute coefficients
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': np.abs(best_model.coef_[0])
    }).sort_values('Importance', ascending=False)

# =============================
# 6. SAVE RESULTS FOR STREAMLIT
# =============================
joblib.dump(results, "results.pkl")
joblib.dump(comparison_df, "comparison_df.pkl")
joblib.dump(feature_importance, "feature_importance.pkl")
joblib.dump(best_model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")

print("✅ All models trained and results saved.")
print(f"🏆 Best model: {best_model_name} (ROC-AUC: {results[best_model_name]['roc_auc']:.3f})")
