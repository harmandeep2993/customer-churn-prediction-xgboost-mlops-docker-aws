import streamlit as st
import pandas as pd
from src.predict import load_model, predict_churn

st.title("Customer Churn Prediction")

# Load model once
model = load_model()

# Input fields
st.header("Customer Information")
gender = st.selectbox("Gender", ["Male", "Female"])
senior = st.selectbox("Senior Citizen", ["Yes", "No"])
partner = st.selectbox("Partner", ["Yes", "No"])
dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (Months)", 0, 72, 12)
monthly_charge = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)
total_charge = st.number_input("Total Charges", 0.0, 10000.0, 2500.0)
contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
TotalCharges = st.text_input("Total Charges")
submitted = st.form_submit_button("Predict")
# Add more inputs as needed...

# Predict
if st.button("Predict Churn"):
    input_dict = {
        "gender": gender,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "MonthlyCharges": monthly_charge,
        "TotalCharges": total_charge,
        "Contract": contract,
        "PaymentMethod": payment,
        # Add any other required features here
    }

    input_df = pd.DataFrame([input_dict])
    pred, prob = predict_churn(input_df, model)

    st.subheader("Prediction Result")
    st.write("Churn" if pred[0] == 1 else "No Churn")
    st.write(f"Probability of churn: {prob[0]:.2f}")