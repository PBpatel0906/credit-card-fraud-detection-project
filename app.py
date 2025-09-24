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

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS
# -------------------------------
def add_custom_css():
    st.markdown("""
    <style>
    .stApp {
        background-color: #ffffff;
        color: #111111;
    }
    .css-1d391kg { background: rgba(255,255,255,0.9); backdrop-filter: blur(10px); border-radius:10px; }
    .block-container { background: rgba(255,255,255,0.95); border-radius:15px; padding:2rem; margin:1rem; box-shadow:0 8px 32px rgba(31,38,135,0.05); border:1px solid rgba(0,0,0,0.05); }
    .metric-card { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding:1rem; border-radius:10px; color:white; text-align:center; margin:0.5rem; box-shadow:0 4px 15px rgba(0,0,0,0.2); }
    .stButton > button { background: linear-gradient(45deg,#667eea 0%,#764ba2 100%); color:white; border:none; border-radius:20px; padding:0.5rem 2rem; font-weight:bold; }
    .main-title { text-align:center; color:#2c3e50; margin-bottom:2rem; font-size:2.5rem; font-weight:bold; }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------
# Model File
# -------------------------------
MODEL_FILE = "fraud_model_train.pkl"

# -------------------------------
# Sample Data
# -------------------------------
@st.cache_data
def load_sample_data():
    np.random.seed(42)
    n_samples = 1000
    data = {}
    data['Time'] = np.random.randint(0,172800,n_samples)
    data['Amount'] = np.random.exponential(88.35,n_samples)
    for i in range(1,29):
        data[f'V{i}'] = np.random.normal(0,1,n_samples)
    data['Class'] = np.random.choice([0,1], n_samples, p=[0.998,0.002])
    return pd.DataFrame(data)

# -------------------------------
# Load Uploaded Data
# -------------------------------
def load_uploaded_data(uploaded_file):
    try:
        if uploaded_file.name.endswith('.csv'):
            data = pd.read_csv(uploaded_file)
        else:
            data = pd.read_excel(uploaded_file)
        required_cols = ['Class'] + [f'V{i}' for i in range(1,29)] + ['Time','Amount']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            st.warning(f"Missing columns: {missing_cols}. Using sample data instead.")
            return load_sample_data()
        return data
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return load_sample_data()

# -------------------------------
# Train Model
# -------------------------------
@st.cache_resource
def train_model(data):
    try:
        X = data.drop("Class", axis=1)
        y = data["Class"]
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        model.fit(X_train_scaled, y_train)

        # Handle single-class safely
        if len(np.unique(y_train))==1:
            y_pred = np.zeros_like(y_test)
            y_pred_proba = np.zeros_like(y_test)
        else:
            y_pred = model.predict(X_test_scaled)
            y_pred_proba_raw = model.predict_proba(X_test_scaled)
            if y_pred_proba_raw.shape[1]==2:
                y_pred_proba = y_pred_proba_raw[:,1]
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
        st.error(f"Error training model: {e}")
        return None

# -------------------------------
# Save & Load Model
# -------------------------------
def save_model(model_data):
    try:
        joblib.dump(model_data, MODEL_FILE)
        return True
    except Exception as e:
        st.error(f"Error saving model: {e}")
        return False

def load_model():
    try:
        if os.path.exists(MODEL_FILE):
            return joblib.load(MODEL_FILE)
        return None
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# -------------------------------
# Visualization
# -------------------------------
def plot_confusion_matrix(cm):
    fig = px.imshow(cm, labels=dict(x="Predicted",y="Actual",color="Count"),
                    x=['Legitimate','Fraud'],y=['Legitimate','Fraud'],
                    color_continuous_scale='Blues', title="Confusion Matrix")
    for i in range(len(cm)):
        for j in range(len(cm[0])):
            fig.add_annotation(x=j,y=i,text=str(cm[i][j]),showarrow=False,font=dict(color="white"))
    return fig

def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    df = pd.DataFrame({'feature':feature_names,'importance':importance}).sort_values('importance',ascending=True).tail(10)
    fig = px.bar(df, x='importance', y='feature', orientation='h', title="Top 10 Feature Importance")
    fig.update_layout(height=400)
    return fig

def plot_prediction_distribution(probabilities, actual):
    df = pd.DataFrame({'Probability':probabilities,'Actual':['Fraud' if x==1 else 'Legitimate' for x in actual]})
    fig = px.histogram(df, x='Probability', color='Actual', nbins=50, title="Prediction Probability Distribution")
    return fig

# -------------------------------
# Main App
# -------------------------------
def main():
    add_custom_css()
    st.markdown('<h1 class="main-title">💳 Credit Card Fraud Detection</h1>', unsafe_allow_html=True)

    menu = ["🏠 Dashboard","🔧 Train Model","🔮 Predict","📊 Analytics","ℹ About"]
    choice = st.sidebar.selectbox("Choose an option", menu)

    # -------------------------------
    # Dashboard
    if choice=="🏠 Dashboard":
        st.subheader("📈 System Overview")
        sample_data = load_sample_data()
        st.write(f"📊 Sample Dataset shape: {sample_data.shape}")
        st.write(f"*Fraud cases:* {sample_data['Class'].sum()}")
        st.write(f"*Legitimate cases:* {len(sample_data)-sample_data['Class'].sum()}")

    # -------------------------------
    # Train Model
    elif choice=="🔧 Train Model":
        uploaded_file = st.file_uploader("Upload CSV/Excel dataset", type=['csv','xlsx','xls'])
        use_sample = st.checkbox("Use Sample Data", value=True)
        if uploaded_file and not use_sample:
            data = load_uploaded_data(uploaded_file)
        else:
            data = load_sample_data()
            st.info("Using sample data for demonstration")
        st.write(f"📊 Dataset shape: {data.shape}")
        with st.expander("👀 Data Preview"):
            st.dataframe(data.head())
        if st.button("🔥 Train Model"):
            model_data = train_model(data)
            if model_data and save_model(model_data):
                st.success("✅ Model trained and saved successfully!")
            else:
                st.error("❌ Model training failed")

    # -------------------------------
    # Predict
    elif choice=="🔮 Predict":
        model_data = load_model()
        if not model_data:
            st.error("❌ No trained model found. Train a model first.")
            return
        st.success("✅ Model loaded!")
        st.markdown("""<div style="background-color:#ffffff; color:#111111; padding:20px; border-radius:10px;">""", unsafe_allow_html=True)
        method = st.radio("Input method:", ["Manual Entry","Upload CSV","Random Sample"])
        
        if method=="Manual Entry":
            time = st.number_input("Time", 0.0, 172800.0, 0.0)
            amount = st.number_input("Amount", 0.0, 1000000.0, 100.0)
            v_features = [st.number_input(f"V{i}", -10.0, 10.0, 0.0) for i in range(1,29)]
            if st.button("Predict Transaction"):
                input_data = np.array([time]+v_features+[amount]).reshape(1,-1)
                input_scaled = model_data['scaler'].transform(input_data)
                prediction = model_data['model'].predict(input_scaled)[0]
                probability = model_data['model'].predict_proba(input_scaled)[0,1] if model_data['model'].n_classes_>1 else 0.0
                if prediction==1:
                    st.error("🚨 FRAUD DETECTED")
                else:
                    st.success("✅ Legitimate Transaction")
                st.metric("Fraud Probability", f"{probability:.4f}")
                st.markdown("</div>", unsafe_allow_html= True)
    # -------------------------------
    # Analytics
    elif choice=="📊 Analytics":
        model_data = load_model()
        if not model_data:
            st.error("❌ No trained model found. Train a model first.")
            return
        st.subheader("📊 Feature Importance")
        st.plotly_chart(plot_feature_importance(model_data['model'], model_data['feature_names']), use_container_width=True)
        st.subheader("📊 Prediction Distribution")
        st.plotly_chart(plot_prediction_distribution(model_data['test_probabilities'], model_data['test_actual']), use_container_width=True)

    # -------------------------------
    # About
    else:
        st.subheader("ℹ About This Application")
        st.markdown(f"""
        This application detects fraudulent credit card transactions using machine learning.
        - *Developer*: AI-Powered Fraud Detection
        - *Version*: 2.0
        - *Last Updated*: {datetime.now().strftime('%Y-%m-%d')}
        """)

if __name__ == "__main__":
    main()
