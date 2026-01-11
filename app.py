# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as snsg
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import joblib

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# =========================
# 1. Load Data
# =========================

@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")  # CSV in same folder
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode Yes/No columns
    yes_no_cols = [
        'Partner','Dependents','PhoneService','PaperlessBilling','Churn',
        'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies'
    ]
    for col in yes_no_cols:
        df[col] = df[col].map({'Yes':1,'No':0})
    
    return df

df = load_data()


st.title("📊 Telco Customer Churn Prediction")
st.write("Predict customer churn and explore insights.")

# =========================
# 2. EDA Section
# =========================
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots()
    sns.countplot(data=df, x='Churn', ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots()
    sns.barplot(x='Contract', y='Churn', data=df, estimator=lambda x: x.mean(), ax=ax)
    st.pyplot(fig)

# =========================
# 3. Prediction Section
# =========================
st.header("Predict Churn for a Customer")

# Input form
with st.form("predict_form"):
    st.subheader("Enter Customer Details")
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=850.0)
    contract_type = st.selectbox("Contract Type", df['Contract'].unique())
    internet_service = st.selectbox("Internet Service", df['InternetService'].unique())
    online_security = st.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No"])
    submit = st.form_submit_button("Predict Churn")

    if submit:
        # Encode input
        input_df = pd.DataFrame({
            'tenure':[tenure],
            'MonthlyCharges':[monthly_charges],
            'TotalCharges':[total_charges],
            'Contract':[contract_type],
            'InternetService':[internet_service],
            'OnlineSecurity':[1 if online_security=="Yes" else 0],
            'TechSupport':[1 if tech_support=="Yes" else 0]
        })
        
        # Encode categorical features
        input_df['Contract'] = pd.Categorical(input_df['Contract'], categories=df['Contract'].unique()).codes
        input_df['InternetService'] = pd.Categorical(input_df['InternetService'], categories=df['InternetService'].unique()).codes
        
        # Features used in model
        features = ['tenure','MonthlyCharges','TotalCharges','Contract','InternetService','OnlineSecurity','TechSupport']
        
        # Load pre-trained model
        model = joblib.load("best_model.pkl")
        scaler = joblib.load("scaler.pkl")
        pred_proba = model.predict_proba(scaler.transform(input_df[features]))[0,1]
        st.success(f"Predicted Churn Probability: {pred_proba*100:.2f}%")
