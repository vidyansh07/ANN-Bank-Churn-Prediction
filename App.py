import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model & scaler
model = load_model("model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))

st.set_page_config(page_title="Customer Churn Dashboard")

st.title("🏦 Customer Churn Prediction Dashboard")

st.write("Predict whether a customer will churn using ANN")

# Inputs
credit_score = st.number_input("Credit Score")
age = st.number_input("Age")
balance = st.number_input("Balance")
salary = st.number_input("Estimated Salary")

# Predict button
if st.button("Predict"):
    data = np.array([[credit_score, age, balance, salary]])
    data = scaler.transform(data)

    prediction = model.predict(data)[0][0]

    churn_prob = prediction * 100
    retention = 100 - churn_prob

    if churn_prob < 50:
        status = "Low Risk"
        st.success(f"Target Status: {status}")
    else:
        status = "High Risk"
        st.error(f"Target Status: {status}")

    st.write(f"Churn Risk: {churn_prob:.2f}%")
    st.write(f"Retention Probability: {retention:.2f}%")
