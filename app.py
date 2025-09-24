# Streamlit Credit Card Fraud Detection App

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# ------------------------------
# Load Dataset
# ------------------------------
@st.cache_data
def load_data():
    data = pd.read_csv("creditcard.csv")
  

    return data

# ------------------------------
# Train Model Function
# ------------------------------
@st.cache_resource
def train_model(data):
    X = data.drop("Class", axis=1)
    y = data["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, acc, report

# ------------------------------
# Save and Load Model
# ------------------------------
MODEL_FILE = "fraud_model_train.pkl"

def save_model(model, scaler):
    joblib.dump({"model": model, "scaler": scaler}, MODEL_FILE)

def load_model():
    return joblib.load(MODEL_FILE)

# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.title("💳 Credit Card Fraud Detection")
    st.write("Detect fraudulent credit card transactions using a machine learning model.")

    menu = ["Train Model", "Predict", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Train Model":
        st.subheader("Train a New Model")
        uploaded_file = st.file_uploader("Upload CSV file", type="csv")
        if uploaded_file:
            data = load_data(uploaded_file)
            st.write("Dataset shape:", data.shape)
            model, scaler, acc, report = train_model(data)
            save_model(model, scaler)
            st.success(f"Model trained successfully with Accuracy: {acc:.4f}")
        else:
            st.warning("Please upload a CSV file to train the model.")

    elif choice == "Predict":
        st.subheader("Make a Prediction")
        try:
            model_data = load_model()
            model = model_data["model"]
            scaler = model_data["scaler"]
        except FileNotFoundError:
            st.error("Model not found. Please train the model first.")
            return

        st.write("Enter transaction details:")

        # Example input: Time, Amount, first 5 PCA features
        time = st.number_input("Time", min_value=0.0, step=1.0)
        amount = st.number_input("Amount", min_value=0.0, step=1.0)
        v_features = [st.number_input(f"V{i}", step=0.1) for i in range(1, 6)]

        input_data = np.array([time] + v_features + [amount]).reshape(1, -1)

        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            result = "Fraud ❌" if prediction[0] == 1 else "Legit ✅"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    else:
        st.subheader("About")
        st.write(
            """
            This app uses a machine learning model to detect credit card fraud.
            Upload your dataset to train a new model, or use the saved model to make predictions.
            """
        )

if __name__ == "__main__":
    main()
