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
- 
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

## 🤖 4. Model Training (fraud_main.py)

The primary ML model used:

### Random Forest Classifier
- n_estimators = 100
- max_depth = 15
- class_weight = 'balanced'
- n_jobs = -1

### Metrics (Example)
Metric	Score
Accuracy	~0.99
Precision	High
Recall	Strong
F1 Score	Balanced performance
ROC-AUC	Excellent

All results are saved in model_metadata.joblib.

## 🖥️ 5. Streamlit Web App (fraud_app.py)

The Streamlit UI offers two main modes:

### 🔍 Single Transaction Prediction
- Input transaction details manually
- Real-time fraud prediction
- Fraud probability gauge
- Risk level classification
- Summary and recommended actions

### 📁 Batch Processing (CSV Upload)
- Upload thousands of transactions
- Auto feature engineering
- Fraud prediction for each entry
- Overall statistics:
  - Fraud percentage
  - Legitimate vs Fraud counts
  - Risk distribution
  - Fraud by transaction type
- Downloadable results CSV

## 🚀 6. How to Run the Project
1. Clone the Repository
git clone https://github.com/afeefanver/Fraud_Detection.git
cd Fraud_Detection

2. Install Dependencies
pip install -r requirements.txt

3. Train Model (Optional)
python fraud_main.py

4. Run Streamlit App
streamlit run fraud_app.py

## 📸 7. Screenshots (Add later)

You can add screenshots like:

![App Screenshot](images/app_home.png)
![Fraud Gauge](images/fraud_gauge.png)
![Batch Processing](images/batch_results.png)

## 📈 8. Results & Insights

- Fraud transactions often involve:
 - Zero origin balance
 - Significant balance errors
 - Merchant accounts
- Random Forest performed best vs Logistic Regression, SVM, etc.
- Custom-engineered features improved fraud detection accuracy significantly.

The system is suitable for production-level deployment with minimal changes.

## 🔮 9. Future Improvements

- Add SHAP explainability
- Deploy on AWS / GCP / Render
- Add REST API endpoints
- Add alerting system for high-risk transactions
- Add user authentication & admin dashboard

## 🏆 10. Author

Afeef Anver
AI/ML Developer | Python | Data Science
