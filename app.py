import streamlit as st
import pandas as pd
import joblib
from catboost import CatBoostClassifier         

# Page Configuration
st.set_page_config(
    page_title="Telco Churn Predictor",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.title("Telco Churn Predictor")
st.write("Enter Customer Detail below to Predict the Probability of Churn")

# Load the model using caching
@st.cache_resource
def load_model():
  # Use joblib for .pkl files
  return joblib.load('churn_master_pipeline.pkl')
  

# Initialize the model
model = load_model()

# Sidebar for User Input
st.sidebar.header("Customer Details")

def get_user_input():
    # Categorical Inputs
    internet_service = st.sidebar.selectbox("Internet Service", ["DSL","Fiber Optic", "No"])
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
    monthly_charges = st.sidebar.number_input("Monthly Charges", 0.0, 200.0, 70.0)
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

# Display the selected Inputs back to the user
st.subheader("Selected Customer Profile")
st.dataframe(input_df)

# Prediction Logic
st.subheader("Prediction Result")

if st.button("Prediction Churn Probability"):
    try:
        #Load the Pipeline Model
        model = joblib.load("churn_master_pipeline.pkl")

        # Make prediction
        prediction = model.predict(input_df)
        probability = model.predict_proba(input_df)[0][1]  # Probability of churn


    
        # Display the prediction result
        if prediction[0] == 1:
            st.error(f"🚨 The Customer is likely to churn. (Risk: {probability: .1%})")
        else:
            st.success(f"✅ The Customer is unlikely to churn. (Risk: {probability: .1%})")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        