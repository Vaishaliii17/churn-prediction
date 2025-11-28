import os
import pandas as pd
import joblib

MODEL_PATH = os.path.join('models', 'churn_model.pkl')
print('Loading model from', MODEL_PATH)
model = joblib.load(MODEL_PATH)

# Example new customer data (fill with actual values)
new_data = pd.DataFrame([{
    'gender':'Female',
    'SeniorCitizen':0,
    'Partner':'Yes',
    'Dependents':'No',
    'tenure':12,
    'PhoneService':'Yes',
    'MultipleLines':'No',
    'InternetService':'Fiber optic',
    'OnlineSecurity':'No',
    'OnlineBackup':'Yes',
    'DeviceProtection':'No',
    'TechSupport':'No',
    'StreamingTV':'Yes',
    'StreamingMovies':'No',
    'Contract':'Month-to-month',
    'PaperlessBilling':'Yes',
    'PaymentMethod':'Electronic check',
    'MonthlyCharges':70.35,
    'TotalCharges':845.5
}])

# Make prediction
pred = model.predict(new_data)
pred_proba = model.predict_proba(new_data)[:,1]

print(f'Churn Prediction: {"Yes" if pred[0]==1 else "No"}')
print(f'Churn Probability: {pred_proba[0]:.2f}')
