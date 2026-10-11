from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Heart Disease Prediction", page_icon="❤️")
st.title("❤️ Heart Disease Prediction")
st.warning("Educational demonstration only. Not a medical diagnosis or clinical decision tool.")
@st.cache_resource
def get_model():
    return joblib.load(Path(__file__).with_name("heart_disease_model.joblib"))
model_data = get_model()
with st.form("prediction"):
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=54)
    bp = st.number_input("Resting blood pressure (mm Hg)", min_value=50, max_value=300, value=130)
    cholesterol = st.number_input("Serum cholesterol (mg/dl)", min_value=50, max_value=700, value=240)
    max_hr = st.number_input("Maximum heart rate achieved (beats/min)", min_value=40, max_value=250, value=150)
    submitted = st.form_submit_button("Predict")
if submitted:
    X = pd.DataFrame([[age, bp, cholesterol, max_hr]], columns=model_data["features"])
    pred = int(model_data["pipeline"].predict(X)[0])
    if pred:
        st.warning("Model prediction: heart disease present (dataset label 1).")
    else:
        st.success("Model prediction: no heart disease detected (dataset label 0).")
    st.caption("This prediction is not medically validated and cannot rule disease in or out.")
