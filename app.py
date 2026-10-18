import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier         

# Custom CSS to style buttons nicely
st.markdown("""
    <style>
    /* Change primary button background color */
    div.stButton > button {
        background-color: #0066cc; /* Professional Blue */
        color: white;
        border-radius: 6px;
        border: none;
    }
    div.stButton > button:hover {
        background-color: #0052a3; /* Darker blue on hover */
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="wide",  # Wide layout for a professional dashboard feel
    initial_sidebar_state="expanded"
)

# --- 1. Project Overview / Description Section ---
st.title("📊 Telco Customer Churn Predictor")
st.markdown(
    """
    **Project Overview:** This interactive machine learning application predicts customer churn for telecommunications companies. 
    Powered by a production-grade **CatBoost Pipeline**, it analyzes customer demographics, account details, and service usage 
    to identify high-risk subscribers, enabling proactive retention strategies.
    """
)
st.markdown("---")

# --- 2. Model Performance Metrics Row ---
# Displaying Precision, Recall, and F1-Score right at the top
st.markdown("##### 📈 Model Performance Metrics")
perf_col1, perf_col2, perf_col3 = st.columns(3)
with perf_col1:
    st.metric(label="Precision", value="0.56", delta="Optimized")
with perf_col2:
    st.metric(label="Recall", value="0.69", delta="Balanced")
with perf_col3:
    st.metric(label="F1-Score", value="0.62", delta="Robust")

st.markdown("---")

# Load the model using caching
@st.cache_resource
def load_model():
    return joblib.load('churn_master_pipeline.pkl')

# Initialize the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model file: {e}")

# Sidebar for User Input & Model Info
st.sidebar.header("📝 Customer Details")

def get_user_input():
    # Categorical Inputs
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber Optic", "No"])
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    senior_citizen = st.sidebar.selectbox("Senior Citizen", ["Yes", "No"])
    partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])
    phone_service = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.sidebar.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.sidebar.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.sidebar.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.sidebar.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.sidebar.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    # Numerical Inputs
    tenure = st.sidebar.slider("Tenure (in months)", 0, 72, 12)
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.sidebar.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.sidebar.number_input("Monthly Charges ($)", 0.0, 200.0, 70.0)
    total_charges = st.sidebar.number_input("Total Charges ($)", 0.0, 10000.0, 800.0)

    # Create the DataFrame with ALL columns in the exact order required
    data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# Balanced Two-Column Layout for Main Content
col1, col2 = st.columns([1.3, 0.7], gap="large")

with col1:
    st.subheader("📋 Selected Customer Profile")
    # Split the 19 input features into two equal columns to eliminate scrolling
    features_dict = input_df.iloc[0].to_dict()
    items = list(features_dict.items())
    mid_point = len(items) // 2
    
    p_col1, p_col2 = st.columns(2)
    
    with p_col1:
        for key, val in items[:mid_point]:
            st.markdown(f"**{key}:** {val}")
            
    with p_col2:
        for key, val in items[mid_point:]:
            st.markdown(f"**{key}:** {val}")

with col2:
    st.subheader("🎯 Prediction Panel")
    st.write("Click below to run model inference.")
    
    # Styled primary button
    if st.button("🚀 Predict Churn Probability", use_container_width=True):
        try:
            # Make prediction
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]  # Probability of churn

            st.markdown("### Analysis Result")
            
            # Prominent metric display
            st.metric(label="Churn Probability", value=f"{probability:.1%}")

            # Conditional alert callouts
            if prediction[0] == 1 or probability > 0.5:
                st.error("🚨 **High Churn Risk:** This customer is likely to churn.")
            else:
                st.success("✅ **Low Churn Risk:** This customer is stable.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            # Conditional alert callouts
            if prediction[0] == 1 or probability > 0.5:
                st.error("🚨 **High Churn Risk:** This customer is likely to churn.")
            else:
                st.success("✅ **Low Churn Risk:** This customer is stable.")

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
        
