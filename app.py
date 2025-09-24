# Day 18 - Streamlit Credit Card Fraud Detection App

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

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluation
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    return model, scaler, acc, report

# ------------------------------
# Save Model
# ------------------------------
def save_model(model, scaler):
    joblib.dump({"model": model, "scaler": scaler}, "fraud_model_train.pkl")

# ------------------------------
# Load Saved Model
# ------------------------------
def load_model():
    return joblib.load("fraud_model_train.pkl")

# ------------------------------
# Streamlit UI
# ------------------------------
def main():
    st.title("💳 Credit Card Fraud Detection - Day 18 🚀")
    st.write("A simple ML pipeline with Streamlit")

    menu = ["Train Model", "Predict", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    if choice == "Train Model":
        st.subheader("Training the Model")
        data = load_data()
        st.write("Dataset shape:", data.shape)

        model, scaler, acc, report = train_model(data)
        save_model(model, scaler)

        st.success(f"✅ Model trained successfully with Accuracy: {acc:.4f}")
        st.json(report)

    elif choice == "Predict":
        st.subheader("Make a Prediction")
        model_data = load_model()
        model = model_data["model"]
        scaler = model_data["scaler"]

        st.write("Enter transaction details:")

        # Example features (only Time, Amount, and first 5 PCA components for demo)
        time = st.number_input("Time", min_value=0.0, max_value=100000.0, step=1.0)
        amount = st.number_input("Amount", min_value=0.0, max_value=10000.0, step=1.0)
        v_features = [st.number_input(f"V{i}", step=0.1) for i in range(1, 6)]

        input_data = np.array([time] + v_features + [amount]).reshape(1, -1)

        # Scale input (handling shape mismatch by padding zeros if needed)
        try:
            input_scaled = scaler.transform(input_data)
            prediction = model.predict(input_scaled)
            result = "Fraud ❌" if prediction[0] == 1 else "Legit ✅"
            st.success(f"Prediction: {result}")
        except Exception as e:
            st.error(f"Error in prediction: {e}")

    else:
        st.subheader("About this App")
        st.write("Day 18 Challenge: Building a Streamlit App with Model Training & Prediction")

if __name_ == "__main__":
    main()
