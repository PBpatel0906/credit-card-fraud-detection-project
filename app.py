#18# streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# Load saved model
model = joblib.load("C:\Users\pbpat\Desktop\visual\fraud_model_train.py")

st.title("Credit Card Fraud Detection")

st.write("""
This app predicts whether a transaction is fraudulent based on its features.
""")

# Create input fields dynamically
st.sidebar.header("Transaction Features Input")

def user_input_features():
    # Example features from the dataset (modify according to your dataset)
    V1 = st.sidebar.number_input("V1", value=0.0)
    V2 = st.sidebar.number_input("V2", value=0.0)
    V3 = st.sidebar.number_input("V3", value=0.0)
    V4 = st.sidebar.number_input("V4", value=0.0)
    V5 = st.sidebar.number_input("V5", value=0.0)
    Amount = st.sidebar.number_input("Amount", value=0.0)
    
    data = {
        'V1': V1,
        'V2': V2,
        'V3': V3,
        'V4': V4,
        'V5': V5,
        'Amount': Amount
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()
st.subheader("User Input Features")
st.write(input_df)#19
# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.subheader("Prediction")
fraud_label = "Fraudulent Transaction" if prediction[0] == 1 else "Legitimate Transaction"
st.write(fraud_label)

st.subheader("Prediction Probability")
st.write(f"Probability of Fraud: {prediction_proba[0][1]:.2f}")