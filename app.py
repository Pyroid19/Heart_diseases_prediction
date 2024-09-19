import streamlit as st
import numpy as np
import joblib

# Load the model
loaded_model = joblib.load('RidgeClassifier.pkl')

def predict_heart_disease(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall):
    sample_data = np.array([[age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall]])
    prediction = loaded_model.predict(sample_data)
    return 'Heart Disease' if prediction[0] == 1 else 'No Heart Disease'

# Streamlit App Interface
st.title("Heart Disease Prediction")

# Input parameters on the main page
age = st.number_input('Age', min_value=1, max_value=120, value=50)
sex = st.selectbox('Sex (0: Female, 1: Male)', [0, 1])
cp = st.number_input('Chest Pain Type (0-3)', min_value=0, max_value=3, value=0)
trtbps = st.number_input('Resting Blood Pressure (mm Hg)', min_value=80, max_value=200, value=120)
chol = st.number_input('Cholesterol (mg/dl)', min_value=100, max_value=400, value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1: True, 0: False)', [0, 1])
restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
thalachh = st.number_input('Maximum Heart Rate Achieved', min_value=60, max_value=220, value=150)
exng = st.selectbox('Exercise Induced Angina (1: Yes, 0: No)', [0, 1])
oldpeak = st.number_input('Oldpeak (ST depression induced by exercise)', min_value=0.0, max_value=10.0, value=1.0)
slp = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
caa = st.number_input('Number of Major Vessels (0-4)', min_value=0, max_value=4, value=0)
thall = st.selectbox('Thalassemia (1: Normal, 2: Fixed Defect, 3: Reversible Defect)', [1, 2, 3])

# Button for prediction
if st.button('Predict'):
    result = predict_heart_disease(age, sex, cp, trtbps, chol, fbs, restecg, thalachh, exng, oldpeak, slp, caa, thall)
    st.success(f"The model predicts: {result}")
