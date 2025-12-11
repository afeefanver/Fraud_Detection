# ============================================================================
# FRAUD DETECTION - TRAIN AND SAVE BEST MODEL
# ============================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from joblib import dump
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("TRAINING BEST MODEL FOR FRAUD DETECTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/6] Loading data...")
df = pd.read_csv('D:\BIA\Capstone_Project_Fraud\Fraud_Analysis_Dataset.csv')
print(f"✓ Loaded {len(df):,} transactions")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/6] Engineering features...")

# Balance-related features
df['orig_balance_change'] = df['oldbalanceOrg'] - df['newbalanceOrig']
df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']

# Error features (inconsistency detection)
df['error_balance_orig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
df['error_balance_dest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']

# Ratio features
df['amount_to_oldbalance_orig'] = df['amount'] / (df['oldbalanceOrg'] + 1)
df['amount_to_oldbalance_dest'] = df['amount'] / (df['oldbalanceDest'] + 1)

# Binary features
df['is_orig_balance_zero'] = (df['oldbalanceOrg'] == 0).astype(int)
df['is_dest_balance_zero'] = (df['oldbalanceDest'] == 0).astype(int)
df['is_merchant'] = df['nameDest'].str.startswith('M').astype(int)

# Time features (circular encoding)
df['hour_sin'] = np.sin(2 * np.pi * df['step'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['step'] / 24)

# One-hot encoding for transaction types
type_dummies = pd.get_dummies(df['type'], prefix='type')
df = pd.concat([df, type_dummies], axis=1)

print("✓ Created engineered features")

# ============================================================================
# 3. SELECT FEATURES
# ============================================================================
print("\n[3/6] Selecting features...")

# Best features identified from analysis (adjust based on your results)
selected_features = [
    'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 
    'oldbalanceDest', 'newbalanceDest',
    'orig_balance_change', 'dest_balance_change',
    'error_balance_orig', 'error_balance_dest',
    'amount_to_oldbalance_orig', 'amount_to_oldbalance_dest',
    'is_orig_balance_zero', 'is_dest_balance_zero', 'is_merchant',
    'hour_sin', 'hour_cos',
    'type_CASH_IN', 'type_CASH_OUT', 'type_DEBIT', 
    'type_PAYMENT', 'type_TRANSFER'
]

X = df[selected_features]
y = df['isFraud']

print(f"✓ Using {len(selected_features)} features")

# ============================================================================
# 4. TRAIN-TEST SPLIT
# ============================================================================
print("\n[4/6] Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Train set: {X_train.shape[0]:,} samples")
print(f"✓ Test set: {X_test.shape[0]:,} samples")

# ============================================================================
# 5. TRAIN BEST MODEL (Random Forest based on typical best performance)
# ============================================================================
print("\n[5/6] Training Random Forest model...")

# Initialize the best model
# Note: Adjust hyperparameters based on your analysis results
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42,
    class_weight='balanced',
    n_jobs=-1
)

# Train the model
model.fit(X_train, y_train)

# Evaluate on test set
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\n✓ Model trained successfully!")
print(f"\nModel Performance:")
print(f"  Accuracy:  {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall:    {recall:.4f}")
print(f"  F1-Score:  {f1:.4f}")
print(f"  ROC-AUC:   {roc_auc:.4f}")

# ============================================================================
# 6. SAVE MODEL AND ARTIFACTS
# ============================================================================
print("\n[6/6] Saving model and artifacts...")

# Save the trained model
dump(model, 'fraud_model.joblib')
print("✓ Model saved: fraud_model.joblib")

# Save feature columns list
dump(selected_features, 'feature_columns.joblib')
print("✓ Feature columns saved: feature_columns.joblib")

# Save label encoder for transaction types (if needed for deployment)
label_encoder = LabelEncoder()
label_encoder.fit(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
dump(label_encoder, 'label_encoder.joblib')
print("✓ Label encoder saved: label_encoder.joblib")

# Save model metadata
model_metadata = {
    'model_type': 'Random Forest',
    'n_features': len(selected_features),
    'accuracy': accuracy,
    'precision': precision,
    'recall': recall,
    'f1_score': f1,
    'roc_auc': roc_auc,
    'features': selected_features
}
dump(model_metadata, 'model_metadata.joblib')
print("✓ Model metadata saved: model_metadata.joblib")

print("\n" + "="*80)
print("MODEL TRAINING COMPLETE!")
print("="*80)
print("\nSaved Files:")
print("  1. fraud_model.joblib - Trained Random Forest model")
print("  2. feature_columns.joblib - List of features used")
print("  3. label_encoder.joblib - Label encoder for transaction types")
print("  4. model_metadata.joblib - Model performance metrics")
print("\nNext Step: Run 'streamlit run app.py' to deploy the model")
print("="*80)