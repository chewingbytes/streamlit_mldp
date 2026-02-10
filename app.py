import os
import json
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Student Depression Risk", page_icon="🧠", layout="centered")

BASE_DIR = os.path.dirname(__file__)
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "depression_gbc.pkl"),
    os.path.join(BASE_DIR, "model_artifacts", "depression_gbc.pkl"),
]
FEATURE_CANDIDATES = [
    os.path.join(BASE_DIR, "feature_columns.json"),
    os.path.join(BASE_DIR, "model_artifacts", "feature_columns.json"),
]

@st.cache_resource
def load_artifacts():
    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    feature_path = next((p for p in FEATURE_CANDIDATES if os.path.exists(p)), None)

    if model_path is None:
        st.error(
            "Model file not found. Upload depression_gbc.pkl in the app folder or model_artifacts/."
        )
        st.stop()
    if feature_path is None:
        st.error(
            "Feature columns file not found. Upload feature_columns.json in the app folder or model_artifacts/."
        )
        st.stop()

    model = joblib.load(model_path)
    with open(feature_path, "r", encoding="utf-8") as f:
        feature_columns = json.load(f)
    return model, feature_columns

model, feature_columns = load_artifacts()

st.title("Student Depression Risk Screening")
st.write("Provide inputs to estimate depression risk. This is **not** a medical diagnosis.")

with st.form("input_form"):
    st.subheader("Student Profile")
    age = st.slider("Age", 18, 35, 22)
    gender = st.selectbox("Gender", ["Male", "Female"])
    city_region = st.selectbox("City Region", ["Central India", "East India", "North India", "South India", "West India"])
    degree = st.selectbox("Education Level", ["Undergraduate", "Post Graduate", "Professional"])

    st.subheader("Academic & Lifestyle")
    academic_pressure = st.slider("Academic Pressure (0 = None, 10 = Extreme)", 0.0, 10.0, 5.0)
    cgpa = st.slider("CGPA (0.0 to 10.0)", 0.0, 10.0, 7.0)
    study_satisfaction = st.slider("Study Satisfaction (0 = Low, 10 = High)", 0.0, 10.0, 5.0)
    work_study_hours = st.slider("Work/Study Hours per Day", 0.0, 24.0, 6.0)
    financial_stress = st.slider("Financial Stress (0 = None, 10 = Extreme)", 0.0, 10.0, 5.0)
    sleep_duration = st.selectbox(
        "Sleep Duration",
        ["Less than 5 hours", "5-6 hours", "7-8 hours", "More than 8 hours"],
    )
    dietary_habits = st.selectbox("Dietary Habits", ["Healthy", "Moderate", "Unhealthy"])

    st.subheader("Mental Health Indicators")
    suicidal_thoughts = st.selectbox("Have you ever had suicidal thoughts?", ["No", "Yes"])
    family_history = st.selectbox("Family history of mental illness?", ["No", "Yes"])

    submitted = st.form_submit_button("Predict Risk")

if submitted:
    try:
        raw_input = {
            "Age": age,
            "Academic Pressure": academic_pressure,
            "CGPA": cgpa,
            "Study Satisfaction": study_satisfaction,
            "Work/Study Hours": work_study_hours,
            "Financial Stress": financial_stress,
            "Gender": gender,
            "Sleep Duration": sleep_duration,
            "Dietary Habits": dietary_habits,
            "Have you ever had suicidal thoughts ?": suicidal_thoughts,
            "Family History of Mental Illness": family_history,
            "New_Degree": degree,
            "City_Region": city_region,
        }

        input_df = pd.DataFrame([raw_input])
        input_encoded = pd.get_dummies(
            input_df,
            columns=[
                "Gender",
                "Sleep Duration",
                "Dietary Habits",
                "Have you ever had suicidal thoughts ?",
                "Family History of Mental Illness",
                "New_Degree",
                "City_Region",
            ],
        )

        input_aligned = input_encoded.reindex(columns=feature_columns, fill_value=0)

        proba = model.predict_proba(input_aligned)[0][1]
        prediction = int(proba >= 0.5)

        st.subheader("Prediction")
        st.write(f"Estimated probability of depression: **{proba:.2%}**")
        if prediction == 1:
            st.warning("Higher risk detected. Consider reaching out to a counselor or support services.")
        else:
            st.success("Lower risk detected based on the provided inputs.")
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
