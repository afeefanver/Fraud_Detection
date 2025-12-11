import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
import plotly.graph_objects as go

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# LOAD MODEL AND ARTIFACTS
# ============================================================================
@st.cache_resource
def load_model_artifacts():
    """Load the trained model and related artifacts"""
    try:
        model = load('fraud_model.joblib')
        feature_columns = load('feature_columns.joblib')
        model_metadata = load('model_metadata.joblib')
        return model, feature_columns, model_metadata
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model, feature_columns, metadata = load_model_artifacts()

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .success-box {
        background-color: #d4edda;
        border-left: 5px solid #28a745;
        padding: 1rem;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 1rem;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# HEADER
# ============================================================================
st.markdown('<h1 class="main-header">🔒 Fraud Detection System</h1>', unsafe_allow_html=True)

model_type = metadata['model_type']
f1_val = metadata['f1_score']
precision_val = metadata['precision']
recall_val = metadata['recall']

st.markdown(f"""
<div style='text-align: center; margin-bottom: 2rem;'>
    This system uses <b>{model_type}</b> to predict fraudulent transactions in real-time.<br>
    <b>Model Performance:</b> F1-Score: {f1_val:.4f} | Precision: {precision_val:.4f} | Recall: {recall_val:.4f}
</div>
""", unsafe_allow_html=True)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.header("📊 Model Information")
st.sidebar.metric("Model Type", metadata['model_type'])
st.sidebar.metric("F1-Score", f"{metadata['f1_score']:.4f}")
st.sidebar.metric("Precision", f"{metadata['precision']:.4f}")
st.sidebar.metric("Recall", f"{metadata['recall']:.4f}")
st.sidebar.metric("Accuracy", f"{metadata['accuracy']:.4f}")
st.sidebar.metric("Features Used", metadata['n_features'])

st.sidebar.markdown("---")
st.sidebar.info("""
**How to use:**
1. Enter transaction details
2. Click 'Detect Fraud'
3. Review the prediction result
4. Check fraud probability score
""")

# ============================================================================
# MAIN CONTENT - TABS
# ============================================================================
tab1, tab2 = st.tabs(["🔍 Single Transaction", "📁 Batch Processing"])

# ============================================================================
# TAB 1: SINGLE TRANSACTION PREDICTION
# ============================================================================
with tab1:
    st.header("🔍 Single Transaction Fraud Detection")
    
    # Create two columns for input
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Transaction Details")
        step = st.number_input("⏰ Time Step (Hours)", min_value=1, max_value=744, value=1, 
                               help="Time step in hours (1 step = 1 hour)")
        
        transaction_type = st.selectbox("💳 Transaction Type", 
                                        ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
                                        help="Type of transaction")
        
        amount = st.number_input("💰 Transaction Amount ($)", min_value=0.0, value=1000.0, 
                                step=100.0, help="Amount of money in transaction")
        
        oldbalance_org = st.number_input("🏦 Origin Old Balance ($)", min_value=0.0, 
                                         value=5000.0, step=100.0,
                                         help="Origin account balance before transaction")
        
        newbalance_org = st.number_input("🏦 Origin New Balance ($)", min_value=0.0, 
                                         value=4000.0, step=100.0,
                                         help="Origin account balance after transaction")

    with col2:
        st.subheader("Destination Details")
        oldbalance_dest = st.number_input("🎯 Destination Old Balance ($)", min_value=0.0, 
                                          value=3000.0, step=100.0,
                                          help="Destination account balance before transaction")
        
        newbalance_dest = st.number_input("🎯 Destination New Balance ($)", min_value=0.0, 
                                          value=4000.0, step=100.0,
                                          help="Destination account balance after transaction")
        
        is_merchant = st.checkbox("🏪 Is Destination a Merchant?",
                                 help="Check if destination account is a merchant")
        
        st.markdown("<br>", unsafe_allow_html=True)

    # Add some spacing
    st.markdown("<br>", unsafe_allow_html=True)

    # ============================================================================
    # PREDICTION BUTTON
    # ============================================================================
    if st.button("🔍 DETECT FRAUD", type="primary", use_container_width=True):
        # Feature engineering
        orig_balance_change = oldbalance_org - newbalance_org
        dest_balance_change = newbalance_dest - oldbalance_dest
        error_balance_orig = newbalance_org + amount - oldbalance_org
        error_balance_dest = oldbalance_dest + amount - newbalance_dest
        amount_to_oldbalance_orig = amount / (oldbalance_org + 1)
        amount_to_oldbalance_dest = amount / (oldbalance_dest + 1)
        is_orig_balance_zero = 1 if oldbalance_org == 0 else 0
        is_dest_balance_zero = 1 if oldbalance_dest == 0 else 0
        is_merchant_flag = 1 if is_merchant else 0
        hour_sin = np.sin(2 * np.pi * step / 24)
        hour_cos = np.cos(2 * np.pi * step / 24)
        
        # Create feature dictionary
        features = {
            'step': step,
            'amount': amount,
            'oldbalanceOrg': oldbalance_org,
            'newbalanceOrig': newbalance_org,
            'oldbalanceDest': oldbalance_dest,
            'newbalanceDest': newbalance_dest,
            'orig_balance_change': orig_balance_change,
            'dest_balance_change': dest_balance_change,
            'error_balance_orig': error_balance_orig,
            'error_balance_dest': error_balance_dest,
            'amount_to_oldbalance_orig': amount_to_oldbalance_orig,
            'amount_to_oldbalance_dest': amount_to_oldbalance_dest,
            'is_orig_balance_zero': is_orig_balance_zero,
            'is_dest_balance_zero': is_dest_balance_zero,
            'is_merchant': is_merchant_flag,
            'hour_sin': hour_sin,
            'hour_cos': hour_cos,
            'type_CASH_IN': 1 if transaction_type == 'CASH_IN' else 0,
            'type_CASH_OUT': 1 if transaction_type == 'CASH_OUT' else 0,
            'type_DEBIT': 1 if transaction_type == 'DEBIT' else 0,
            'type_PAYMENT': 1 if transaction_type == 'PAYMENT' else 0,
            'type_TRANSFER': 1 if transaction_type == 'TRANSFER' else 0
        }
        
        # Create DataFrame
        input_data = pd.DataFrame([features])
        input_features = input_data[feature_columns]
        
        # Make prediction
        with st.spinner('Analyzing transaction...'):
            prediction = model.predict(input_features)[0]
            probability = model.predict_proba(input_features)[0][1]
        
        # Display results
        st.markdown("---")
        st.header("📊 Prediction Results")
        
        # Create three columns for metrics
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        
        with metric_col1:
            if prediction == 1:
                st.markdown("""
                    <div class='danger-box'>
                        <h2 style='color: #dc3545; text-align: center;'>🚨 FRAUD DETECTED</h2>
                        <p style='text-align: center;'>This transaction is likely fraudulent!</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                    <div class='success-box'>
                        <h2 style='color: #28a745; text-align: center;'>✅ LEGITIMATE</h2>
                        <p style='text-align: center;'>This transaction appears legitimate.</p>
                    </div>
                """, unsafe_allow_html=True)
        
        prob_pct = probability * 100
        delta_val = (probability - 0.5) * 100
        
        with metric_col2:
            st.metric("Fraud Probability", f"{prob_pct:.2f}%", 
                     delta=f"{delta_val:.1f}% from threshold")
        
        with metric_col3:
            if probability > 0.7:
                risk_level = "🔴 HIGH RISK"
            elif probability > 0.3:
                risk_level = "🟡 MEDIUM RISK"
            else:
                risk_level = "🟢 LOW RISK"
            st.metric("Risk Level", risk_level)
        
        # Fraud probability gauge
        st.subheader("Fraud Probability Meter")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=probability * 100,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': "Fraud Probability (%)", 'font': {'size': 24}},
            gauge={
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "darkred" if probability > 0.5 else "orange"},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "gray",
                'steps': [
                    {'range': [0, 30], 'color': '#d4edda'},
                    {'range': [30, 70], 'color': '#fff3cd'},
                    {'range': [70, 100], 'color': '#f8d7da'}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 50
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="white",
            font={'color': "darkblue", 'family': "Arial"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Transaction details summary
        st.subheader("📋 Transaction Summary")
        summary_col1, summary_col2 = st.columns(2)
        
        amount_fmt = f"{amount:,.2f}"
        orig_change_fmt = f"{orig_balance_change:,.2f}"
        
        with summary_col1:
            st.info(f"""
            **Transaction Information:**
            - Type: {transaction_type}
            - Amount: ${amount_fmt}
            - Time Step: {step} hours
            - Origin Balance Change: ${orig_change_fmt}
            """)
        
        dest_change_fmt = f"{dest_balance_change:,.2f}"
        merchant_text = 'Yes' if is_merchant else 'No'
        abs_error_orig = abs(error_balance_orig)
        abs_error_dest = abs(error_balance_dest)
        error_orig_fmt = f"{abs_error_orig:,.2f}"
        error_dest_fmt = f"{abs_error_dest:,.2f}"
        
        with summary_col2:
            st.info(f"""
            **Account Details:**
            - Destination Balance Change: ${dest_change_fmt}
            - Is Merchant: {merchant_text}
            - Balance Error (Origin): ${error_orig_fmt}
            - Balance Error (Dest): ${error_dest_fmt}
            """)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        if prediction == 1:
            st.warning("""
            **⚠️ Recommended Actions:**
            - Flag this transaction for manual review
            - Contact the account holder to verify
            - Temporarily freeze the account if necessary
            - Investigate recent account activity
            - Check for other suspicious patterns
            """)
        else:
            st.success("""
            **✅ Recommended Actions:**
            - Process the transaction normally
            - No immediate action required
            - Continue monitoring account activity
            """)

# ============================================================================
# TAB 2: BATCH PROCESSING
# ============================================================================
with tab2:
    st.header("📁 Batch Transaction Processing")
    
    st.markdown("""
    Upload a CSV file containing multiple transactions to analyze them all at once.
    
    **Required columns in your CSV:**
    - `step`, `type`, `amount`, `nameOrig`, `oldbalanceOrg`, `newbalanceOrig`, 
    - `nameDest`, `oldbalanceDest`, `newbalanceDest`
    """)
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'], 
                                     help="Upload a CSV file with transaction data")
    
    if uploaded_file is not None:
        try:
            # Read the CSV file
            df_batch = pd.read_csv(uploaded_file)
            
            batch_len = len(df_batch)
            st.success(f"✅ File uploaded successfully! Found {batch_len:,} transactions.")
            
            # Show preview
            with st.expander("📋 Preview uploaded data (first 10 rows)"):
                st.dataframe(df_batch.head(10))
            
            # Process button
            if st.button("🚀 ANALYZE ALL TRANSACTIONS", type="primary", use_container_width=True):
                with st.spinner('Analyzing all transactions... This may take a moment.'):
                    # Feature engineering for batch
                    df_batch['orig_balance_change'] = df_batch['oldbalanceOrg'] - df_batch['newbalanceOrig']
                    df_batch['dest_balance_change'] = df_batch['newbalanceDest'] - df_batch['oldbalanceDest']
                    df_batch['error_balance_orig'] = df_batch['newbalanceOrig'] + df_batch['amount'] - df_batch['oldbalanceOrg']
                    df_batch['error_balance_dest'] = df_batch['oldbalanceDest'] + df_batch['amount'] - df_batch['newbalanceDest']
                    df_batch['amount_to_oldbalance_orig'] = df_batch['amount'] / (df_batch['oldbalanceOrg'] + 1)
                    df_batch['amount_to_oldbalance_dest'] = df_batch['amount'] / (df_batch['oldbalanceDest'] + 1)
                    df_batch['is_orig_balance_zero'] = (df_batch['oldbalanceOrg'] == 0).astype(int)
                    df_batch['is_dest_balance_zero'] = (df_batch['oldbalanceDest'] == 0).astype(int)
                    
                    # Handle merchant flag
                    if 'nameDest' in df_batch.columns:
                        df_batch['is_merchant'] = df_batch['nameDest'].str.startswith('M').astype(int)
                    else:
                        df_batch['is_merchant'] = 0
                    
                    # Time features
                    df_batch['hour_sin'] = np.sin(2 * np.pi * df_batch['step'] / 24)
                    df_batch['hour_cos'] = np.cos(2 * np.pi * df_batch['step'] / 24)
                    
                    # One-hot encoding for transaction types
                    type_dummies = pd.get_dummies(df_batch['type'], prefix='type')
                    df_batch = pd.concat([df_batch, type_dummies], axis=1)
                    
                    # Ensure all required type columns exist
                    for t_type in ['type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']:
                        if t_type not in df_batch.columns:
                            df_batch[t_type] = 0
                    
                    # Select features for prediction
                    batch_features = df_batch[feature_columns]
                    
                    # Make predictions
                    predictions = model.predict(batch_features)
                    probabilities = model.predict_proba(batch_features)[:, 1]
                    
                    # Add predictions to dataframe
                    df_batch['Fraud_Prediction'] = predictions
                    df_batch['Fraud_Probability'] = probabilities
                    df_batch['Result'] = df_batch['Fraud_Prediction'].map({0: 'Legitimate', 1: 'Fraud'})
                    df_batch['Risk_Level'] = pd.cut(probabilities, 
                                                     bins=[0, 0.3, 0.7, 1.0],
                                                     labels=['Low', 'Medium', 'High'])
                
                # Display summary statistics
                st.markdown("---")
                st.header("📊 Analysis Summary")
                
                # Key metrics in columns
                col1, col2, col3, col4 = st.columns(4)
                
                total_transactions = len(df_batch)
                fraud_count = int((predictions == 1).sum())
                legitimate_count = int((predictions == 0).sum())
                fraud_percentage = (fraud_count / total_transactions) * 100
                
                with col1:
                    st.metric("📝 Total Transactions", f"{total_transactions:,}")
                
                fraud_pct_fmt = f"{fraud_percentage:.1f}%"
                with col2:
                    st.metric("🚨 Fraudulent", f"{fraud_count:,}", 
                             delta=fraud_pct_fmt, delta_color="inverse")
                
                legit_pct = (legitimate_count/total_transactions)*100
                legit_pct_fmt = f"{legit_pct:.1f}%"
                with col3:
                    st.metric("✅ Legitimate", f"{legitimate_count:,}",
                             delta=legit_pct_fmt)
                
                avg_fraud_prob = probabilities.mean() * 100
                avg_fraud_fmt = f"{avg_fraud_prob:.2f}%"
                with col4:
                    st.metric("📈 Avg Fraud Probability", avg_fraud_fmt)
                
                # Visualization: Fraud Distribution
                st.subheader("📊 Fraud Distribution")
                viz_col1, viz_col2 = st.columns(2)
                
                with viz_col1:
                    # Pie chart
                    fig_pie = go.Figure(data=[go.Pie(
                        labels=['Legitimate', 'Fraud'],
                        values=[legitimate_count, fraud_count],
                        marker=dict(colors=['#2ecc71', '#e74c3c']),
                        hole=0.4
                    )])
                    fig_pie.update_layout(
                        title="Transaction Distribution",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with viz_col2:
                    # Risk level distribution
                    risk_counts = df_batch['Risk_Level'].value_counts()
                    fig_bar = go.Figure(data=[go.Bar(
                        x=risk_counts.index,
                        y=risk_counts.values,
                        marker=dict(color=['#2ecc71', '#f39c12', '#e74c3c'])
                    )])
                    fig_bar.update_layout(
                        title="Risk Level Distribution",
                        xaxis_title="Risk Level",
                        yaxis_title="Count",
                        height=300,
                        margin=dict(l=20, r=20, t=40, b=20)
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                
                # Fraud by transaction type
                st.subheader("💳 Fraud Analysis by Transaction Type")
                fraud_by_type = df_batch.groupby('type').agg({
                    'Fraud_Prediction': ['sum', 'count', 'mean']
                }).round(4)
                fraud_by_type.columns = ['Fraud_Count', 'Total_Transactions', 'Fraud_Rate']
                fraud_by_type['Fraud_Rate'] = fraud_by_type['Fraud_Rate'] * 100
                fraud_by_type = fraud_by_type.reset_index()
                
                st.dataframe(fraud_by_type, use_container_width=True)
                
                # Show fraudulent transactions
                st.subheader("🚨 Flagged Fraudulent Transactions")
                fraud_transactions = df_batch[df_batch['Fraud_Prediction'] == 1].copy()
                
                if len(fraud_transactions) > 0:
                    # Select relevant columns to display
                    display_cols = ['step', 'type', 'amount', 'nameOrig', 'nameDest', 
                                   'Fraud_Probability', 'Risk_Level', 'Result']
                    display_cols = [col for col in display_cols if col in fraud_transactions.columns]
                    
                    st.dataframe(
                        fraud_transactions[display_cols].sort_values('Fraud_Probability', ascending=False),
                        use_container_width=True
                    )
                    
                    # Statistics on flagged transactions
                    total_amount_risk = fraud_transactions['amount'].sum()
                    avg_amount = fraud_transactions['amount'].mean()
                    max_prob = fraud_transactions['Fraud_Probability'].max() * 100
                    
                    total_risk_fmt = f"{total_amount_risk:,.2f}"
                    avg_amt_fmt = f"{avg_amount:,.2f}"
                    max_prob_fmt = f"{max_prob:.2f}"
                    
                    st.info(f"""
                    **Flagged Transaction Statistics:**
                    - Total Amount at Risk: ${total_risk_fmt}
                    - Average Transaction Amount: ${avg_amt_fmt}
                    - Highest Risk Transaction: {max_prob_fmt}% probability
                    """)
                else:
                    st.success("✅ No fraudulent transactions detected!")
                
                # Download options
                st.subheader("📥 Download Results")
                
                download_col1, download_col2 = st.columns(2)
                
                with download_col1:
                    # Download all results
                    csv_all = df_batch.to_csv(index=False)
                    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                    st.download_button(
                        label="📥 Download All Results (CSV)",
                        data=csv_all,
                        file_name=f"fraud_analysis_all_{timestamp}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
                with download_col2:
                    # Download only fraudulent transactions
                    if len(fraud_transactions) > 0:
                        csv_fraud = fraud_transactions.to_csv(index=False)
                        timestamp2 = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')
                        st.download_button(
                            label="📥 Download Fraud Only (CSV)",
                            data=csv_fraud,
                            file_name=f"fraud_flagged_{timestamp2}.csv",
                            mime="text/csv",
                            use_container_width=True
                        )
                
        except Exception as e:
            st.error(f"❌ Error processing file: {str(e)}")
            st.info("Please make sure your CSV file has the correct column names and format.")
    
    else:
        # Show sample format
        st.info("📝 **Sample CSV Format:**")
        sample_data = pd.DataFrame({
            'step': [1, 2, 3],
            'type': ['TRANSFER', 'CASH_OUT', 'PAYMENT'],
            'amount': [1000.0, 2500.0, 500.0],
            'nameOrig': ['C123456', 'C234567', 'C345678'],
            'oldbalanceOrg': [5000.0, 3000.0, 1500.0],
            'newbalanceOrig': [4000.0, 500.0, 1000.0],
            'nameDest': ['C654321', 'C765432', 'M876543'],
            'oldbalanceDest': [3000.0, 1000.0, 5000.0],
            'newbalanceDest': [4000.0, 3500.0, 5500.0]
        })
        st.dataframe(sample_data)

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🔒 Fraud Detection System v1.0 | Built with Streamlit</p>
    <p>⚠️ This is a machine learning prediction and should be used as part of a comprehensive fraud detection strategy.</p>
</div>
""", unsafe_allow_html=True)