# app.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Page config with custom theme
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Remove main background */
    .stApp {
        background: none;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Card styling */
    .css-1r6slb0 {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 700;
        color: #1e293b;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 0.875rem;
        color: #64748b;
        font-weight: 500;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        box-shadow: 0 4px 16px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 24px rgba(102, 126, 234, 0.5);
    }
    
    /* Input fields */
    .stNumberInput>div>div>input,
    .stSelectbox>div>div>div {
        border-radius: 8px;
        border: 2px solid #e2e8f0;
        padding: 0.5rem;
    }
    
    /* Headers */
    h1 {
        color: #1e293b !important;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    h2 {
        color: #1e293b;
        font-weight: 600;
        margin-top: 2rem;
    }
    
    h3 {
        color: #334155;
        font-weight: 600;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: #1e293b;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background: #667eea !important;
        color: white !important;
    }
    
    /* Info/Warning/Error boxes */
    .stAlert {
        border-radius: 12px;
        border-left: 4px solid;
    }
</style>
""", unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv("Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    
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

# Header
st.markdown("<h1 style='text-align: center; font-size: 3rem; margin-bottom: 0;'>📊 Telco Churn Prediction</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: rgba(255,255,255,0.9); font-size: 1.1rem; margin-top: 0;'>AI-powered customer retention insights</p>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Tabs
tab1, tab2 = st.tabs(["📈 Dashboard", "🔮 Predict Churn"])

with tab1:
    st.markdown("### Key Metrics")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        churn_rate = df['Churn'].mean() * 100
        st.metric("Churn Rate", f"{churn_rate:.1f}%", delta="-2.3%", delta_color="inverse")
    
    with col2:
        avg_tenure = df['tenure'].mean()
        st.metric("Avg Tenure", f"{avg_tenure:.1f} mo", delta="+1.2 mo")
    
    with col3:
        avg_monthly = df['MonthlyCharges'].mean()
        st.metric("Avg Monthly", f"${avg_monthly:.2f}", delta="+$3.21")
    
    with col4:
        total_customers = len(df)
        st.metric("Total Customers", f"{total_customers:,}")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Churn Distribution")
        churn_counts = df['Churn'].value_counts()
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Retained', 'Churned'],
                y=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                marker_color=['#10b981', '#ef4444'],
                text=[churn_counts.get(0, 0), churn_counts.get(1, 0)],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Arial Black'),
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b', size=12),
            height=350,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### Churn Rate by Contract")
        churn_by_contract = df.groupby('Contract')['Churn'].mean()
        
        fig = go.Figure(data=[
            go.Bar(
                x=churn_by_contract.index,
                y=churn_by_contract.values * 100,
                marker_color=['#3b82f6', '#8b5cf6', '#f59e0b'],
                text=[f"{val*100:.1f}%" for val in churn_by_contract.values],
                textposition='auto',
                textfont=dict(size=14, color='white', family='Arial Black'),
            )
        ])
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#1e293b', size=12),
            height=350,
            margin=dict(t=20, b=40, l=40, r=20),
            xaxis=dict(showgrid=False),
            yaxis=dict(showgrid=True, gridcolor='rgba(0,0,0,0.05)', title='Churn Rate (%)'),
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Insights box
    st.markdown("### 💡 Key Insights")
    st.info("""
    - **Month-to-month contracts** show significantly higher churn rates (42.7%) compared to long-term contracts (2.8% for two-year)
    - Customers **without tech support** are more likely to churn
    - **Tenure** is a strong predictor - customers who stay longer are less likely to leave
    - **Online security** and other add-on services correlate with lower churn
    """)

with tab2:
    st.markdown("### Customer Details")
    
    model_exists = os.path.exists("best_model.pkl") and os.path.exists("scaler.pkl")
    
    if not model_exists:
        st.warning("⚠️ Model files not found. Please run `python model.py` first to train and save the model.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, help="How long the customer has been with the company")
            monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0, step=0.01)
            total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=850.0, step=0.01)
            contract_type = st.selectbox("Contract Type", df['Contract'].unique())
        
        with col2:
            internet_service = st.selectbox("Internet Service", df['InternetService'].unique())
            online_security = st.selectbox("Online Security", ["Yes", "No"])
            tech_support = st.selectbox("Tech Support", ["Yes", "No"])
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("🎯 Predict Churn Risk", use_container_width=True):
            try:
                input_df = pd.DataFrame({
                    'tenure': [tenure],
                    'MonthlyCharges': [monthly_charges],
                    'TotalCharges': [total_charges],
                    'Contract': [contract_type],
                    'InternetService': [internet_service],
                    'OnlineSecurity': [1 if online_security == "Yes" else 0],
                    'TechSupport': [1 if tech_support == "Yes" else 0]
                })
                
                input_df['Contract'] = pd.Categorical(input_df['Contract'], categories=df['Contract'].unique()).codes
                input_df['InternetService'] = pd.Categorical(input_df['InternetService'], categories=df['InternetService'].unique()).codes
                
                features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'InternetService', 'OnlineSecurity', 'TechSupport']
                
                model = joblib.load("best_model.pkl")
                scaler = joblib.load("scaler.pkl")
                
                pred_proba = model.predict_proba(scaler.transform(input_df[features]))[0, 1]
                
                st.markdown("---")
                st.markdown("### Prediction Result")
                
                # Risk level display
                if pred_proba >= 0.7:
                    st.error(f"🚨 **High Risk**: {pred_proba*100:.1f}% probability of churn")
                    risk_color = "#ef4444"
                    recommendation = "**Recommendation**: Immediate intervention required. Consider offering retention incentives, discounts, or upgraded services."
                elif pred_proba >= 0.4:
                    st.warning(f"⚠️ **Medium Risk**: {pred_proba*100:.1f}% probability of churn")
                    risk_color = "#f59e0b"
                    recommendation = "**Recommendation**: Monitor closely and consider proactive engagement. Reach out to understand satisfaction levels."
                else:
                    st.success(f"✅ **Low Risk**: {pred_proba*100:.1f}% probability of churn")
                    risk_color = "#10b981"
                    recommendation = "**Recommendation**: Customer likely to stay. Focus on maintaining satisfaction and consider upsell opportunities."
                
                st.markdown(recommendation)
                
                # Probability gauge
                fig = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = pred_proba * 100,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Churn Probability", 'font': {'size': 24, 'color': '#1e293b'}},
                    number = {'suffix': "%", 'font': {'size': 48, 'color': risk_color}},
                    gauge = {
                        'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                        'bar': {'color': risk_color},
                        'bgcolor': "white",
                        'borderwidth': 2,
                        'bordercolor': "gray",
                        'steps': [
                            {'range': [0, 40], 'color': 'rgba(16, 185, 129, 0.2)'},
                            {'range': [40, 70], 'color': 'rgba(245, 158, 11, 0.2)'},
                            {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 70
                        }
                    }
                ))
                
                fig.update_layout(
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    height=300,
                    margin=dict(t=60, b=20, l=40, r=40)
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Risk factors
                st.markdown("#### 🎯 Key Risk Factors")
                risk_factors = []
                
                if contract_type == 'Month-to-month':
                    risk_factors.append("🔴 Month-to-month contract (highest churn rate)")
                if tenure < 12:
                    risk_factors.append(f"🟠 Low tenure ({tenure} months)")
                if online_security == 'No':
                    risk_factors.append("🟡 No online security service")
                if tech_support == 'No':
                    risk_factors.append("🟡 No tech support service")
                if monthly_charges > 70:
                    risk_factors.append("🟡 Higher than average monthly charges")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.markdown(f"- {factor}")
                else:
                    st.markdown("✅ No major risk factors identified")
                
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Make sure the model files (best_model.pkl and scaler.pkl) are in the same directory as this script.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: rgba(255,255,255,0.7); padding: 2rem 0;'>
    <p style='margin: 0; font-size: 0.9rem;'>Built with Streamlit • Powered by Machine Learning</p>
    <p style='margin: 0.5rem 0 0 0; font-size: 0.8rem;'>Customer Churn Prediction Dashboard © 2026</p>
</div>
""", unsafe_allow_html=True)