import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import os
from datetime import datetime

# ------------------------------
# Page Configuration
# ------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ------------------------------
# Custom CSS
# ------------------------------
def add_custom_css():
    st.markdown("""
    <style>
    /* Main background and font */
    .stApp {
        background-color: #ffffff;  /* White background */
        color: #1a1a1a;             /* Dark font for readability */
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(230, 230, 230, 0.9);
        backdrop-filter: blur(10px);
        border-radius: 10px;
    }

    /* Main content container */
    .block-container {
        background: #ffffff;         /* White container */
        border-radius: 15px;
        padding: 2rem;
        margin: 1rem;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
        color: #1a1a1a;             /* Dark font color */
    }

    /* Metric cards */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    /* Buttons styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(0, 0, 0, 0.3);
    }

    /* Titles */
    .main-title {
        text-align: center;
        color: #1a1a1a;
        font-size: 3rem;
        font-weight: bold;
        margin-bottom: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# ------------------------------
# Data Loading
# ------------------------------
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {}
    data['Time'] = np.random.randint(0, 172800, n_samples)
    data['Amount'] = np.random.exponential(88.35, n_samples)
    for i in range(1, 29):
        data[f'V{i}'] = np.random.normal(0, 1, n_samples)
    data['Class'] = np.random.choice([0, 1], n_samples, p=[0.998, 0.002])
    return pd.DataFrame(data)

def load_uploaded_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(('.xlsx', '.xls')):
            data = pd.read_excel(uploaded_file)
        else:
            st.error("Unsupported file format.")
            return None

        required_cols = ['Class'] + [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}")
            st.info("Using sample data instead.")
            return load_sample_data()
        return data
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# ------------------------------
# Model Training
# ------------------------------
@st.cache_resource
def train_model(data):
    """Train the fraud detection model"""
    try:
        X = data.drop("Class", axis=1)
        y = data["Class"]

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        # Safe predict
        if len(np.unique(y_train)) == 1:
            # Only one class present
            y_pred = np.zeros_like(y_test)
            y_pred_proba = np.zeros_like(y_test)
        else:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba_raw = model.predict_proba(X_test_scaled)
            if y_pred_proba_raw.shape[1] == 2:
                y_pred_proba = y_pred_proba_raw[:, 1]
            else:
                y_pred_proba = np.zeros_like(y_test)

        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        return {
            'model': model, 
            'scaler': scaler, 
            'accuracy': acc, 
            'report': report,
            'confusion_matrix': cm,
            'feature_names': X.columns.tolist(),
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba,
            'test_actual': y_test
        }

    except Exception as e:
        st.error(f"Error training model: {str(e)}")
        return None
# ------------------------------
# Model Persistence
# ------------------------------
MODEL_FILE = "fraud_model_train.pkl"

def save_model(model_data):
    try:
        joblib.dump(model_data, MODEL_FILE)
        return True
    except Exception as e:
        st.error(f"Error saving model: {str(e)}")
        return False

def load_model():
    try:
        if os.path.exists(MODEL_FILE):
            return joblib.load(MODEL_FILE)
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# ------------------------------
# Visualization
# ------------------------------
def plot_confusion_matrix(cm):
    fig = px.imshow(cm, labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Legitimate', 'Fraud'], y=['Legitimate', 'Fraud'],
                    color_continuous_scale='Blues', title="Confusion Matrix")
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(x=j, y=i, text=str(cm[i][j]), showarrow=False, font=dict(color="white", size=14))
    return fig

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feature_df = pd.DataFrame({'feature': feature_names, 'importance': importance}).sort_values('importance', ascending=True).tail(10)
    fig = px.bar(feature_df, x='importance', y='feature', orientation='h', title="Top 10 Feature Importance")
    fig.update_layout(height=400)
    return fig

def plot_prediction_distribution(probabilities, actual):
    df = pd.DataFrame({'Probability': probabilities, 'Actual': ['Fraud' if x==1 else 'Legitimate' for x in actual]})
    fig = px.histogram(df, x='Probability', color='Actual', nbins=50, title="Prediction Probability Distribution", labels={'Probability': 'Fraud Probability'})
    return fig

# ------------------------------
# Main App
# ------------------------------
def main():
    add_custom_css()
    st.markdown('<h1 class="main-title floating">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)

    st.sidebar.markdown("## 🎯 Navigation")
    menu = ["Dashboard", "Train Model", "Predict", "Analytics", "About"]
    choice = st.sidebar.selectbox("Choose an option", menu)

    # Dashboard
    if choice == "Dashboard":
        st.subheader("📈 System Overview")
        sample_data = load_sample_data()
        st.write(f"Total transactions: {len(sample_data)}")
        st.write(f"Fraud cases: {sample_data['Class'].sum()}")
        st.write(f"Legitimate cases: {len(sample_data)-sample_data['Class'].sum()}")

    # Train Model# -------------------------------
# Train Model Section
# -------------------------------
elif choice == "Train Model":
    st.subheader("🚀 Train a New Model")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        uploaded_file = st.file_uploader(
            "Upload your dataset (CSV or Excel)", 
            type=['csv', 'xlsx', 'xls'],
            help="Upload a file containing credit card transaction data"
        )
    
    with col2:
        use_sample = st.checkbox("Use Sample Data", value=True)
    
    # Load data safely
    if uploaded_file and not use_sample:
        try:
            if uploaded_file.name.endswith('.csv'):
                data = pd.read_csv(uploaded_file)
            else:
                data = pd.read_excel(uploaded_file)
            
            required_cols = ['Class'] + [f'V{i}' for i in range(1, 29)] + ['Time', 'Amount']
            missing_cols = [col for col in required_cols if col not in data.columns]
            
            if missing_cols:
                st.warning(f"Missing columns: {missing_cols}")
                st.info("Using sample data instead.")
                data = load_sample_data()
                
        except Exception as e:
            st.error(f"Error loading uploaded file: {e}")
            data = load_sample_data()
    else:
        data = load_sample_data()
        st.info("Using sample data for demonstration")
    
    # Show dataset info
    st.write(f"📊 Dataset shape: {data.shape}")
    with st.expander("👀 Data Preview"):
        st.dataframe(data.head())
        st.write(f"*Fraud cases:* {data['Class'].sum()}")
        st.write(f"*Legitimate cases:* {len(data) - data['Class'].sum()}")
    
    # Start training
    if st.button("🔥 Start Training", type="primary"):
        with st.spinner("🔄 Training model... This may take a few minutes."):
            model_data = train_model(data)
            
            if model_data:
                # Save model
                if save_model(model_data):
                    st.success(f"✅ Model trained and saved successfully!")
                    st.balloons()
                    
                    # Display metrics
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Accuracy", f"{model_data['accuracy']:.4f}")
                        st.metric("Precision", f"{model_data['report']['1']['precision']:.4f}" 
                                  if '1' in model_data['report'] else "N/A")
                    with col2:
                        st.metric("Recall", f"{model_data['report']['1']['recall']:.4f}" 
                                  if '1' in model_data['report'] else "N/A")
                        st.metric("F1-Score", f"{model_data['report']['1']['f1-score']:.4f}" 
                                  if '1' in model_data['report'] else "N/A")
                    
                    # Confusion matrix
                    fig = plot_confusion_matrix(model_data['confusion_matrix'])
                    st.plotly_chart(fig, use_container_width=True)
                    
                else:
                    st.error("❌ Failed to save model. Check permissions.")
            else:
                st.error("❌ Model training failed. Check the dataset.")
    # Predict
    elif choice == "Predict":
        st.subheader("🔍 Make a Prediction")
        model_data = load_model()
        if not model_data:
            st.error("❌ No trained model found.")
            return

        st.success("✅ Model loaded successfully!")
        # Manual Entry Prediction
        time = st.number_input("Time", min_value=0.0, step=1.0, value=0.0)
        amount = st.number_input("Amount", min_value=0.0, step=0.1, value=100.0)
        v_features = []
        v_cols = st.columns(4)
        for i in range(1, 29):
            col_idx = (i-1) % 4
            with v_cols[col_idx]:
                val = st.number_input(f"V{i}", step=0.1, value=0.0, key=f"v{i}")
                v_features.append(val)

        if st.button("🔮 Predict Transaction"):
            input_data = np.array([time] + v_features + [amount]).reshape(1, -1)
            input_scaled = model_data['scaler'].transform(input_data)
            prediction = model_data['model'].predict(input_scaled)
            probability = model_data['model'].predict_proba(input_scaled)[0,1]
            if prediction[0] == 1:
                st.error("🚨 FRAUD DETECTED ❌")
            else:
                st.success("✅ LEGITIMATE TRANSACTION")
            st.metric("Fraud Probability", f"{probability:.4f}")

    # Analytics
    elif choice == "Analytics":
        st.subheader("📊 Model Analytics")
        model_data = load_model()
        if not model_data:
            st.error("❌ No trained model found.")
            return
        fig = plot_feature_importance(model_data['model'], model_data['feature_names'])
        st.plotly_chart(fig, use_container_width=True)

    # About
    else:
        st.subheader("ℹ About This Application")
        st.write("This app detects credit card fraud using Random Forest classifier.")

# ------------------------------
# Run App
# ------------------------------
if __name__ == "__main__":
    main()
