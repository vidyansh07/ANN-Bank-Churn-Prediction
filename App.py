import streamlit as st
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model

def load_components():
    """Load model and preprocessing objects."""
    try:
        model = load_model('churn_model.keras')
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        with open('label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('column_transformer.pkl', 'rb') as f:
            column_transformer = pickle.load(f)
        return model, scaler, label_encoder, column_transformer
    except FileNotFoundError:
        st.warning("Model or preprocessing files not found. Please ensure you have run Main.py to train the model and save the required files.")
        return None, None, None, None
    except Exception as e:
        st.error(f"Error loading model or preprocessors: {e}")
        return None, None, None, None

def main():
    st.set_page_config(page_title="Bank Churn Prediction", page_icon="🏦", layout="centered")
    
    st.title("🏦 Bank Churn Prediction")
    st.write("Enter the customer's details below to predict their likelihood of churning (exiting the bank).")

    model_components = load_components()
    if model_components[0] is None:
        st.stop()
        
    model, scaler, label_encoder, column_transformer = model_components

    # Customer inputs via Streamlit form for better UI
    with st.form("prediction_form"):
        st.subheader("Customer Details")
        
        col1, col2 = st.columns(2)
        with col1:
            credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
            geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
            gender = st.selectbox("Gender", ["Female", "Male"])
            age = st.number_input("Age", min_value=18, max_value=100, value=40)
            tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
            
        with col2:
            balance = st.number_input("Balance", min_value=0.0, value=60000.0, format="%.2f")
            num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
            has_cr_card = st.selectbox("Has Credit Card?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            is_active_member = st.selectbox("Is Active Member?", [0, 1], format_func=lambda x: "Yes" if x == 1 else "No")
            estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0, format="%.2f")

        submitted = st.form_submit_button("Predict Churn")

    if submitted:
        # Preprocess input sequence exactly as during training:
        # Order: CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
        input_data = [[credit_score, geography, gender, age, tenure, balance, num_of_products, has_cr_card, is_active_member, estimated_salary]]
        
        try:
            # 1. Encode Gender
            # label_encoder expects a 1D array/list
            input_data[0][2] = label_encoder.transform([input_data[0][2]])[0]
            
            # 2. One-Hot Encode Geography
            input_df = np.array(input_data, dtype=object)
            input_encoded = column_transformer.transform(input_df)
            
            # 3. Scale Data
            input_scaled = scaler.transform(input_encoded)
            
            # 4. Prediction
            prediction = model.predict(input_scaled)
            churn_probability = prediction[0][0]
            
            st.divider()
            st.subheader("Prediction Result")
            
            # Display results with dynamic styling
            if churn_probability > 0.5:
                st.error(f"⚠️ High Risk! The customer is likely to **CHURN**. (Probability: {churn_probability:.2%})")
            else:
                st.success(f"✅ Low Risk. The customer is likely to **STAY**. (Probability: {churn_probability:.2%})")
                
        except ValueError as ve:
            st.error(f"Value Error during prediction (Possible mismatch in categorical values like Gender or Geography): {ve}")
        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")

if __name__ == '__main__':
    main()
