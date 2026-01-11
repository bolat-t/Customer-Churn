# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import shap

st.set_page_config(page_title="Telco Customer Churn", layout="wide")

# -----------------------------
# 1. Load & Preprocess Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
    # Encode Yes/No columns
    yes_no_cols = [
        'Partner','Dependents','PhoneService','PaperlessBilling','Churn',
        'OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
        'StreamingTV','StreamingMovies'
    ]
    for col in yes_no_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes':1,'No':0})
    
    return df

df = load_data()

# -----------------------------
# 2. Sidebar: Prediction Form
# -----------------------------
st.sidebar.header("🔮 Predict Customer Churn")

model_exists = os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl")

if not model_exists:
    st.sidebar.warning("⚠️ Model files not found. Train the model first.")
else:
    tenure = st.sidebar.number_input("Tenure (months)", 0, 72, 12)
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 850.0)
    contract_type = st.sidebar.selectbox("Contract Type", df['Contract'].unique())
    internet_service = st.sidebar.selectbox("Internet Service", df['InternetService'].unique())
    online_security = st.sidebar.selectbox("Online Security", ["Yes", "No"])
    tech_support = st.sidebar.selectbox("Tech Support", ["Yes", "No"])
    
    if st.sidebar.button("🎯 Predict Churn"):
        try:
            input_df = pd.DataFrame({
                'tenure':[tenure],
                'MonthlyCharges':[monthly_charges],
                'TotalCharges':[total_charges],
                'Contract':[contract_type],
                'InternetService':[internet_service],
                'OnlineSecurity':[1 if online_security=="Yes" else 0],
                'TechSupport':[1 if tech_support=="Yes" else 0]
            })
            
            # Encode categorical
            input_df['Contract'] = pd.Categorical(input_df['Contract'], categories=df['Contract'].unique()).codes
            input_df['InternetService'] = pd.Categorical(input_df['InternetService'], categories=df['InternetService'].unique()).codes
            
            features = ['tenure','MonthlyCharges','TotalCharges','Contract','InternetService','OnlineSecurity','TechSupport']
            
            model = joblib.load("best_model.pkl")
            scaler = joblib.load("scaler.pkl")
            
            pred_proba = model.predict_proba(scaler.transform(input_df[features]))[0,1]
            
            # Color-coded result card
            if pred_proba >= 0.7:
                st.markdown(f"<h3 style='color:#e74c3c'>🚨 High Risk: {pred_proba*100:.1f}% chance of churn</h3>", unsafe_allow_html=True)
            elif pred_proba >= 0.4:
                st.markdown(f"<h3 style='color:#f1c40f'>⚠️ Medium Risk: {pred_proba*100:.1f}% chance of churn</h3>", unsafe_allow_html=True)
            else:
                st.markdown(f"<h3 style='color:#2ecc71'>✅ Low Risk: {pred_proba*100:.1f}% chance of churn</h3>", unsafe_allow_html=True)
            
            # Progress bar
            st.progress(pred_proba)
            
            # SHAP explainability
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(scaler.transform(input_df[features]))
            st.subheader("Feature Impact")
            shap.initjs()
            st_shap = st.pyplot(shap.force_plot(explainer.expected_value[1], shap_values[1], input_df[features], matplotlib=True))
            
        except Exception as e:
            st.error(f"Prediction error: {str(e)}")

# -----------------------------
# 3. Main Tabs: Overview & About
# -----------------------------
tab1, tab2 = st.tabs(["📊 Overview", "📚 About"])

with tab1:
    st.title("Telco Customer Churn Dashboard")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    churn_rate = df['Churn'].mean()*100
    col1.metric("Churn Rate", f"{churn_rate:.1f}%", delta=f"{(churn_rate-20):.1f}%")
    col2.metric("Avg Tenure", f"{df['tenure'].mean():.1f} months")
    col3.metric("Avg Monthly Charges", f"${df['MonthlyCharges'].mean():.2f}")
    col4.metric("Total Customers", f"{len(df):,}")
    
    # Visualizations
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(6,3))
    sns.countplot(x='Churn', data=df, palette=['#2ecc71','#e74c3c'], ax=ax)
    ax.set_xticklabels(['Retained','Churned'])
    st.pyplot(fig)
    plt.close()
    
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots(figsize=(6,3))
    churn_by_contract = df.groupby('Contract')['Churn'].mean()
    ax.bar(churn_by_contract.index, churn_by_contract.values, color=['#3498db','#9b59b6','#e67e22'])
    ax.set_ylim(0,1)
    ax.set_ylabel("Churn Rate")
    st.pyplot(fig)
    plt.close()
    
with tab2:
    st.title("About This App")
    st.write("""
This app predicts **customer churn** using machine learning.  
It helps identify at-risk customers and provides actionable insights.

**Features:**
- Interactive churn prediction
- Dashboard with key metrics
- Feature-level prediction explanation (SHAP)
- Minimalistic, modern design

**Tech Stack:** Python, Streamlit, scikit-learn, SHAP
    """)
