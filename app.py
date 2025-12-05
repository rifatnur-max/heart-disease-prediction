import streamlit as st
import pandas as pd
import joblib

model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('heart_disease_scaler.pkl')
expected_columns = joblib.load('heart_disease_columns.pkl')

st.title("Heart Disease Prediction App")
st.markdown("Provide the following details")

age = st.slider("Age", 17, 120, 40)
gender = st.selectbox("Gender", ["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", ["0", "1", "2", "3"])
resting_bp = st.slider("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["Yes", "No"])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
exercise_induced_angina = st.selectbox("Exercise Induced Angina", ["Yes", "No"])
oldpeak = st.slider("Oldpeak (ST depression)", 0.0, 10.0, 1.0)
thalach = st.number_input("Thalach (Maximum Heart Rate)", 60, 220, 150)
slope = st.selectbox("Slope of the Peak Exercise ST Segment", ["0", "1", "2"])
ca = st.selectbox("Number of Major Vessels Colored by Fluoroscopy", ["0", "1", "2", "3"])
thal = st.selectbox("Thalassemia", ["3", "6", "7"])


if st.button("Predict"):
    # Map UI inputs to the numeric feature names expected by the model
    restecg_map = {"Normal": 0, "ST": 1, "LVH": 2}

    raw_input = {
        "age": age,
        "sex": 1 if gender == "Male" else 0,
        "cp": int(chest_pain_type),
        "trestbps": resting_bp,
        "chol": cholesterol,
        "fbs": 1 if fasting_blood_sugar == "Yes" else 0,
        "restecg": restecg_map.get(resting_ecg, 0),
        "thalach": thalach,
        "exang": 1 if exercise_induced_angina == "Yes" else 0,
        "oldpeak": oldpeak,
        "slope": int(slope),
        "ca": int(ca),
        "thal": int(thal),
    }

    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns are present (missing ones -> 0) and order matches
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[expected_columns]

    # Scale and predict
    try:
        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]

        if prediction == 1:
            st.error("The model predicts that you may have heart disease.")
        else:
            st.success("The model predicts that you are unlikely to have heart disease.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")