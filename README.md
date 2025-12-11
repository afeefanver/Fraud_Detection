# 🔒 Fraud Detection System — Machine Learning + Streamlit

A complete end-to-end Fraud Detection System built using Machine Learning (Random Forest) and a fully interactive Streamlit Dashboard for real-time and batch transaction fraud analysis.

This model predicts whether a financial transaction is fraudulent or legitimate using advanced feature engineering and supervised learning.

## ⭐ Key Features

- End-to-End ML Pipeline
- Extensive Feature Engineering

- Random Forest Classifier with Balanced Class Weights

- Real-time Fraud Prediction (Single Transaction)

- Batch Prediction for Multiple Transactions

- Fraud Probability Gauge & Risk Levels

- Downloadable CSV Results

- Clean & Responsive UI built with Streamlit

- Model Performance Summary included

## 📂 Project Structure

Fraud_Detection/
│── fraud_main.py                # Training script
│── fraud_app.py                 # Streamlit application
│── Fraud.ipynb                  # Full EDA + Model Building Notebook
│── fraud_model.joblib           # Saved ML model
│── feature_columns.joblib       # Feature list used by the model
│── model_metadata.joblib        # Stores performance metrics
│── label_encoder.joblib         # Encoder for transaction types
│── README.md                    # Project documentation


## 📘 1. Project Overview

Financial fraud causes massive losses globally. This project uses machine learning to:

- Detect fraudulent transactions

- Provide probability-based risk scores

- Analyze thousands of transactions at once

- Help organizations prevent financial loss

## 📊 2. Dataset Description

The dataset includes the following columns:

- step – Time step (hours)

- type – Transaction type

- amount – Amount transferred

- oldbalanceOrg, newbalanceOrig

- oldbalanceDest, newbalanceDest

- nameOrig, nameDest

- isFraud – Target variable (1 = fraud, 0 = legitimate)

Dataset size: large-scale financial transactions dataset.

## 🧠 3. Feature Engineering
### 🔹 Balance Behavior

- orig_balance_change

- dest_balance_change

### 🔹 Inconsistency Checks

- error_balance_orig

- error_balance_dest

### 🔹 Ratio Features

- amount_to_oldbalance_orig

- amount_to_oldbalance_dest

### 🔹 Flags

- is_orig_balance_zero

- is_dest_balance_zero

- is_merchant

### 🔹 Time Features (Circular Encoding)

- hour_sin

- hour_cos

### 🔹 One-Hot Encoded Transaction Types

- type_CASH_IN

- type_CASH_OUT

- type_DEBIT

- type_PAYMENT

- type_TRANSFER

These features significantly improved model performance.
