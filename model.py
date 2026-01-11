# save_model.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

df = pd.read_csv("Telco-Customer-Churn.csv")
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
yes_no_cols = ['Partner','Dependents','PhoneService','PaperlessBilling','Churn',
               'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
               'StreamingTV','StreamingMovies']
for col in yes_no_cols:
    df[col] = df[col].map({'Yes':1,'No':0})
df['tenure_category'] = pd.cut(df['tenure'], bins=[0,12,24,48,72], labels=['0-12','12-24','24-48','48-72'])
df['has_internet'] = df['InternetService'].apply(lambda x: 0 if x=='No' else 1)

features = ['tenure','MonthlyCharges','TotalCharges','Contract','InternetService','OnlineSecurity','TechSupport']
X = df[features]
y = df['Churn']

# Encode categorical features
X['Contract'] = pd.Categorical(X['Contract'], categories=df['Contract'].unique()).codes
X['InternetService'] = pd.Categorical(X['InternetService'], categories=df['InternetService'].unique()).codes

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_scaled, y)

joblib.dump(model, "best_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved.")
