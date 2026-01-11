# app.py
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import joblib
import os

st.set_page_config(page_title="Telco Customer Churn Prediction", layout="wide")

# =========================
# 1. Load Data
# =========================

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

st.title("📊 Telco Customer Churn Prediction")
st.write("Predict customer churn and explore insights.")

# =========================
# 2. EDA Section
# =========================
st.header("Exploratory Data Analysis")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Churn Distribution")
    fig, ax = plt.subplots(figsize=(6, 4))
    churn_counts = df['Churn'].value_counts()
    ax.bar(['Retained', 'Churned'], [churn_counts.get(0, 0), churn_counts.get(1, 0)], color=['#2ecc71', '#e74c3c'])
    ax.set_ylabel('Count')
    ax.set_title('Customer Churn Distribution')
    st.pyplot(fig)
    plt.close()

with col2:
    st.subheader("Churn by Contract Type")
    fig, ax = plt.subplots(figsize=(6, 4))
    # Calculate churn rate by contract type
    churn_by_contract = df.groupby('Contract')['Churn'].mean()
    ax.bar(churn_by_contract.index, churn_by_contract.values, color=['#3498db', '#9b59b6', '#e67e22'])
    ax.set_ylabel('Churn Rate')
    ax.set_title('Churn Rate by Contract Type')
    ax.set_ylim(0, 1)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close()

# Additional insights
st.subheader("📈 Key Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    churn_rate = df['Churn'].mean() * 100
    st.metric("Churn Rate", f"{churn_rate:.1f}%")

with col2:
    avg_tenure = df['tenure'].mean()
    st.metric("Avg Tenure", f"{avg_tenure:.1f} months")

with col3:
    avg_monthly = df['MonthlyCharges'].mean()
    st.metric("Avg Monthly Charges", f"${avg_monthly:.2f}")

with col4:
    total_customers = len(df)
    st.metric("Total Customers", f"{total_customers:,}")

# =========================
# 3. Prediction Section
# =========================
st.header("🔮 Predict Churn for a Customer")

# Check if model files exist
model_exists = os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl")

if not model_exists:
    st.warning("⚠️ Model files not found. Please run `python model.py` first to train and save the model.")
    st.info("The model files (best_model.pkl and scaler.pkl) need to be generated before making predictions.")
else:
    # Input form
    with st.form("predict_form"):
        st.subheader("Enter Customer Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=850.0)
            contract_type = st.selectbox("Contract Type", df['Contract'].unique())
        
        with col2:
            internet_service = st.selectbox("Internet Service", df['InternetService'].unique())
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        
        submit = st.form_submit_button("🎯 Predict Churn", use_container_width=True)

        if submit:
            try:
                # Encode input
                input_df = pd.DataFrame({
                    'tenure': [tenure],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges],
                    'Contract': [contract_type],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [1 if online_security == "Yes" else 0],
                    'TechSupport': [1 if tech_support == "Yes" else 0]
                })
                
                # Encode categorical features
                input_df['Contract'] = pd.Categorical(input_df['Contract'], categories=df['Contract'].unique()).codes
                input_df['InternetService'] = pd.Categorical(input_df['InternetService'], categories=df['InternetService'].unique()).codes
                
                # Features used in model
                features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']
                
                # Load pre-trained model
                model = joblib.load("best_model.pkl")
                scaler = joblib.load("scaler.pkl")
                
                # Make prediction
                pred_proba = model.predict_proba(scaler.transform(input_df[features]))[0, 1]
                
                # Display result with styling
                st.markdown("---")
                st.subheader("Prediction Result")
                
                if pred_proba >= 0.7:
                    st.error(f"🚨 **High Risk**: {pred_proba*100:.1f}% probability of churn")
                    st.write("**Recommendation**: Immediate intervention required. Consider offering retention incentives.")
                elif pred_proba >= 0.4:
                    st.warning(f"⚠️ **Medium Risk**: {pred_proba*100:.1f}% probability of churn")
                    st.write("**Recommendation**: Monitor closely and consider proactive engagement.")
                else:
                    st.success(f"✅ **Low Risk**: {pred_proba*100:.1f}% probability of churn")
                    st.write("**Recommendation**: Customer likely to stay. Focus on maintaining satisfaction.")
                
                # Visualise probability
                fig, ax = plt.subplots(figsize=(8, 2))
                ax.barh(['Churn Probability'], [pred_proba], color='#e74c3c' if pred_proba >= 0.5 else '#2ecc71')
                ax.set_xlim(0, 1)
                ax.set_xlabel('Probability')
                ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Make sure the model files are compatible with the input features.")

# =========================
# 4. Footer
# =========================
st.markdown("---")
st.markdown("### 📚 About This App")
st.write("""
This application predicts customer churn using machine learning. It analyses customer 
characteristics and behaviour patterns to identify customers at risk of cancelling their service.

**Key Features:**
- Real-time churn probability prediction
- Interactive data exploration
- Actionable business insights
""")