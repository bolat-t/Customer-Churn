# Customer Churn Prediction - Complete ML Pipeline
# A production-ready machine learning solution for customer retention

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, 
                             roc_curve, precision_recall_curve, f1_score)
import warnings
warnings.filterwarnings('ignore')

# Set style for visualisations
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("=" * 80)
print("CUSTOMER CHURN PREDICTION SYSTEM")
print("End-to-End Machine Learning Pipeline for Proactive Retention")
print("=" * 80)

# ============================================================================
# 1. DATA GENERATION & EXPLORATION
# ============================================================================

def generate_telco_churn_data(n_samples=5000, random_state=42):
    """
    Generate realistic telecommunications customer churn data
    Simulates customer behaviour patterns and churn indicators
    """
    np.random.seed(random_state)
    
    data = {
        # Demographics
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'age': np.random.normal(45, 15, n_samples).clip(18, 80).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        
        # Account information
        'tenure_months': np.random.exponential(24, n_samples).clip(1, 72).astype(int),
        'contract_type': np.random.choice(['Month-to-Month', 'One Year', 'Two Year'], 
                                         n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic', 'Mailed Check', 'Bank Transfer', 'Credit Card'],
                                          n_samples, p=[0.35, 0.20, 0.25, 0.20]),
        
        # Services
        'internet_service': np.random.choice(['DSL', 'Fibre', 'No'], n_samples, p=[0.35, 0.40, 0.25]),
        'online_security': np.random.choice(['Yes', 'No', 'No Service'], n_samples, p=[0.3, 0.45, 0.25]),
        'tech_support': np.random.choice(['Yes', 'No', 'No Service'], n_samples, p=[0.3, 0.45, 0.25]),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No Service'], n_samples, p=[0.35, 0.40, 0.25]),
        
        # Usage & billing
        'monthly_charges': np.random.gamma(2, 30, n_samples).clip(20, 150),
        'total_charges': None,  # Will calculate based on tenure
        
        # Support interactions
        'support_calls': np.random.poisson(2.5, n_samples).clip(0, 10),
        'late_payments': np.random.poisson(1.2, n_samples).clip(0, 6),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate total charges
    df['total_charges'] = df['monthly_charges'] * df['tenure_months'] + np.random.normal(0, 50, n_samples)
    df['total_charges'] = df['total_charges'].clip(50, None)
    
    # Generate churn with realistic patterns
    churn_probability = (
        0.02 +  # Base rate
        (df['contract_type'] == 'Month-to-Month') * 0.35 +
        (df['tenure_months'] < 6) * 0.25 +
        (df['support_calls'] > 3) * 0.15 +
        (df['late_payments'] > 2) * 0.20 +
        (df['online_security'] == 'No') * 0.08 +
        (df['tech_support'] == 'No') * 0.08 +
        (df['monthly_charges'] > 80) * 0.10 -
        (df['tenure_months'] > 36) * 0.25
    ).clip(0, 0.95)
    
    df['churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    
    return df

# Generate dataset
print("\n[1/7] Generating Customer Dataset...")
df = generate_telco_churn_data(n_samples=5000)
print(f"✓ Generated {len(df):,} customer records")
print(f"✓ Churn Rate: {df['churn'].mean()*100:.1f}%")

# Display sample
print("\nSample Records:")
print(df.head(3).to_string())

# ============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("[2/7] Exploratory Data Analysis")
print("=" * 80)

# Basic statistics
print("\nDataset Overview:")
print(f"  Rows: {df.shape[0]:,}")
print(f"  Columns: {df.shape[1]}")
print(f"  Churned Customers: {df['churn'].sum():,} ({df['churn'].mean()*100:.1f}%)")
print(f"  Retained Customers: {(~df['churn'].astype(bool)).sum():,} ({(1-df['churn'].mean())*100:.1f}%)")

# Key insights
print("\n📊 Key Business Insights:")
print(f"  • Average Customer Tenure: {df['tenure_months'].mean():.1f} months")
print(f"  • Average Monthly Charges: ${df['monthly_charges'].mean():.2f}")
print(f"  • Customers on Month-to-Month: {(df['contract_type'] == 'Month-to-Month').mean()*100:.1f}%")

# Churn analysis by key features
print("\n🎯 Churn Rates by Segment:")
for col in ['contract_type', 'internet_service', 'online_security']:
    print(f"\n  {col.replace('_', ' ').title()}:")
    churn_by_segment = df.groupby(col)['churn'].agg(['mean', 'count'])
    for idx, row in churn_by_segment.iterrows():
        print(f"    {idx:20s}: {row['mean']*100:5.1f}% (n={row['count']:,})")

# ============================================================================
# 3. FEATURE ENGINEERING
# ============================================================================

print("\n" + "=" * 80)
print("[3/7] Feature Engineering")
print("=" * 80)

# Create working copy
df_model = df.copy()

# Drop customer ID
df_model = df_model.drop('customer_id', axis=1)

# Create new features
print("\n🔧 Creating Advanced Features...")

# Tenure categories
df_model['tenure_category'] = pd.cut(df_model['tenure_months'], 
                                     bins=[0, 6, 12, 24, 48, 100],
                                     labels=['0-6mo', '6-12mo', '1-2yr', '2-4yr', '4yr+'])

# Value segment
df_model['customer_value'] = pd.qcut(df_model['total_charges'], 
                                     q=4, labels=['Low', 'Medium', 'High', 'Premium'])

# Risk indicators
df_model['high_support_calls'] = (df_model['support_calls'] >= 4).astype(int)
df_model['has_late_payments'] = (df_model['late_payments'] > 0).astype(int)
df_model['high_monthly_charges'] = (df_model['monthly_charges'] > df_model['monthly_charges'].median()).astype(int)

# Service engagement score
df_model['services_count'] = (
    (df_model['online_security'] == 'Yes').astype(int) +
    (df_model['tech_support'] == 'Yes').astype(int) +
    (df_model['streaming_tv'] == 'Yes').astype(int)
)

print(f"✓ Created {5} new engineered features")

# Encode categorical variables
print("\n🏷️  Encoding Categorical Variables...")
categorical_cols = df_model.select_dtypes(include=['object', 'category']).columns
categorical_cols = categorical_cols.drop('churn', errors='ignore')

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

print(f"✓ Encoded {len(categorical_cols)} categorical variables")

# ============================================================================
# 4. MODEL TRAINING & EVALUATION
# ============================================================================

print("\n" + "=" * 80)
print("[4/7] Model Training & Evaluation")
print("=" * 80)

# Prepare data
X = df_model.drop('churn', axis=1)
y = df_model['churn']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n📦 Data Split:")
print(f"  Training Set: {len(X_train):,} samples")
print(f"  Test Set: {len(X_test):,} samples")
print(f"  Churn Rate (Train): {y_train.mean()*100:.1f}%")
print(f"  Churn Rate (Test): {y_test.mean()*100:.1f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define models
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

# Train and evaluate models
print("\n🤖 Training Models...")
results = {}

for name, model in models.items():
    print(f"\n  Training {name}...")
    
    # Use scaled data for Logistic Regression, original for tree-based
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    accuracy = model.score(X_test_scaled if name == 'Logistic Regression' else X_test, y_test)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'model': model,
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'f1_score': f1,
        'y_pred': y_pred,
        'y_proba': y_proba
    }
    
    print(f"    ✓ Accuracy: {accuracy:.3f}")
    print(f"    ✓ ROC-AUC: {roc_auc:.3f}")
    print(f"    ✓ F1-Score: {f1:.3f}")

# ============================================================================
# 5. MODEL COMPARISON & SELECTION
# ============================================================================

print("\n" + "=" * 80)
print("[5/7] Model Comparison & Selection")
print("=" * 80)

# Compare models
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results],
    'ROC-AUC': [results[m]['roc_auc'] for m in results],
    'F1-Score': [results[m]['f1_score'] for m in results]
})

print("\n📊 Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# Select best model based on ROC-AUC
best_model_name = comparison_df.loc[comparison_df['ROC-AUC'].idxmax(), 'Model']
best_model = results[best_model_name]['model']

print(f"\n🏆 Best Model: {best_model_name}")
print(f"   ROC-AUC Score: {results[best_model_name]['roc_auc']:.3f}")

# Detailed classification report for best model
print(f"\n📋 Detailed Classification Report ({best_model_name}):")
print(classification_report(y_test, results[best_model_name]['y_pred'], 
                          target_names=['Retained', 'Churned']))

# Confusion matrix
cm = confusion_matrix(y_test, results[best_model_name]['y_pred'])
print("\n🎯 Confusion Matrix:")
print(f"                  Predicted")
print(f"                Retained  Churned")
print(f"Actual Retained  {cm[0,0]:6d}   {cm[0,1]:6d}")
print(f"       Churned   {cm[1,0]:6d}   {cm[1,1]:6d}")

# ============================================================================
# 6. FEATURE IMPORTANCE ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("[6/7] Feature Importance Analysis")
print("=" * 80)

if best_model_name in ['Random Forest', 'Gradient Boosting']:
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("\n📈 Top 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:30s}: {row['Importance']:.4f}")
    
    # Save for visualisation
    top_features = feature_importance.head(15)
else:
    # For logistic regression, use coefficients
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': np.abs(best_model.coef_[0])
    }).sort_values('Coefficient', ascending=False)
    
    print("\n📈 Top 10 Most Influential Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['Feature']:30s}: {row['Coefficient']:.4f}")
    
    top_features = feature_importance.head(15)

# ============================================================================
# 7. BUSINESS INSIGHTS & RECOMMENDATIONS
# ============================================================================

print("\n" + "=" * 80)
print("[7/7] Business Insights & Recommendations")
print("=" * 80)

# Calculate potential savings
y_proba_best = results[best_model_name]['y_proba']
high_risk_threshold = 0.7
high_risk_customers = (y_proba_best >= high_risk_threshold).sum()
avg_customer_value = df['total_charges'].mean()
retention_rate = 0.25  # Assume 25% of contacted high-risk customers can be retained
retention_cost_per_customer = 50

potential_saves = high_risk_customers * retention_rate * avg_customer_value
intervention_cost = high_risk_customers * retention_cost_per_customer
net_benefit = potential_saves - intervention_cost

print("\n💰 Financial Impact Analysis:")
print(f"  • High-Risk Customers Identified: {high_risk_customers:,}")
print(f"  • Average Customer Lifetime Value: ${avg_customer_value:,.2f}")
print(f"  • Potential Revenue at Risk: ${high_risk_customers * avg_customer_value:,.2f}")
print(f"  • Expected Saves (25% retention): ${potential_saves:,.2f}")
print(f"  • Intervention Cost: ${intervention_cost:,.2f}")
print(f"  • Net Benefit: ${net_benefit:,.2f}")

print("\n🎯 Key Strategic Recommendations:")
print("\n  1. PROACTIVE RETENTION PROGRAMME")
print("     • Target customers with churn probability > 70%")
print("     • Implement personalised retention offers")
print("     • Expected ROI: {:.1f}x".format(net_benefit / intervention_cost if intervention_cost > 0 else 0))

print("\n  2. CONTRACT OPTIMISATION")
print("     • Focus on converting Month-to-Month to annual contracts")
print("     • Offer incentives for long-term commitments")
print("     • Target early-tenure customers (0-6 months)")

print("\n  3. SERVICE ENHANCEMENT")
print("     • Promote online security and tech support services")
print("     • Reduce support call volume through better UX")
print("     • Implement proactive service monitoring")

print("\n  4. PAYMENT EXPERIENCE")
print("     • Streamline payment processes to reduce late payments")
print("     • Offer auto-pay discounts")
print("     • Send proactive payment reminders")

print("\n" + "=" * 80)
print("✅ ANALYSIS COMPLETE")
print("=" * 80)
print("\nModel successfully trained and evaluated!")
print(f"Best performing model: {best_model_name}")
print(f"Ready for production deployment and dashboard integration")
print("\nNext Steps:")
print("  1. Deploy model to production environment")
print("  2. Build real-time churn prediction dashboard")
print("  3. Integrate with CRM for automated interventions")
print("  4. Set up model monitoring and retraining pipeline")
print("=" * 80)