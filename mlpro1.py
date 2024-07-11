import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load('loan_approval_model.pkl')
# Title
st.title("Bank Loan Approval Prediction")

# Input fields
age = st.number_input("Age", min_value=18, max_value=100, value=30)
income = st.number_input("Annual Income", min_value=0, max_value=1000000, value=50000)
loan_amount = st.number_input("Loan Amount", min_value=0, max_value=1000000, value=10000)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)

# Predict button
if st.button("Predict"):
    features = np.array([[age, income, loan_amount, credit_score]])
    prediction = model.predict(features)
    approval_status = "Approved" if prediction[0] == 1 else "Rejected"
    
    st.write(f"Loan Approval Status: {approval_status}")
