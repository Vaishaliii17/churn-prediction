import os
import pandas as pd
import joblib
import streamlit as st

st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("📊 Telecom Churn Prediction App")

# Load model
MODEL_PATH = os.path.join('models','churn_model.pkl')
model = joblib.load(MODEL_PATH)

st.sidebar.header("Enter Customer Details")

def user_input():
    data = {}
    data['gender'] = st.sidebar.selectbox("Gender", ['Female','Male'])
    data['SeniorCitizen'] = st.sidebar.selectbox("Senior Citizen", [0,1])
    data['Partner'] = st.sidebar.selectbox("Partner", ['Yes','No'])
    data['Dependents'] = st.sidebar.selectbox("Dependents", ['Yes','No'])
    data['tenure'] = st.sidebar.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
    data['PhoneService'] = st.sidebar.selectbox("Phone Service", ['Yes','No'])
    data['MultipleLines'] = st.sidebar.selectbox("Multiple Lines", ['Yes','No','No phone service'])
    data['InternetService'] = st.sidebar.selectbox("Internet Service", ['DSL','Fiber optic','No'])
    data['OnlineSecurity'] = st.sidebar.selectbox("Online Security", ['Yes','No','No internet service'])
    data['OnlineBackup'] = st.sidebar.selectbox("Online Backup", ['Yes','No','No internet service'])
    data['DeviceProtection'] = st.sidebar.selectbox("Device Protection", ['Yes','No','No internet service'])
    data['TechSupport'] = st.sidebar.selectbox("Tech Support", ['Yes','No','No internet service'])
    data['StreamingTV'] = st.sidebar.selectbox("Streaming TV", ['Yes','No','No internet service'])
    data['StreamingMovies'] = st.sidebar.selectbox("Streaming Movies", ['Yes','No','No internet service'])
    data['Contract'] = st.sidebar.selectbox("Contract", ['Month-to-month','One year','Two year'])
    data['PaperlessBilling'] = st.sidebar.selectbox("Paperless Billing", ['Yes','No'])
    data['PaymentMethod'] = st.sidebar.selectbox("Payment Method", ['Electronic check','Mailed check','Bank transfer (automatic)','Credit card (automatic)'])
    data['MonthlyCharges'] = st.sidebar.number_input("Monthly Charges", min_value=0.0, value=70.0)
    data['TotalCharges'] = st.sidebar.number_input("Total Charges", min_value=0.0, value=800.0)
    return pd.DataFrame([data])

input_df = user_input()

st.subheader("Customer Input")
st.write(input_df)

if st.button("Predict Churn"):
    pred = model.predict(input_df)
    pred_proba = model.predict_proba(input_df)[:,1]
    
    st.subheader("Prediction")
    st.write(f"Churn: {'Yes' if pred[0]==1 else 'No'}")
    st.write(f"Probability: {pred_proba[0]:.2f}")