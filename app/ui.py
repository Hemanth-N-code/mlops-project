import streamlit as st
import requests

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")

st.title("❤️ Heart Disease Prediction")
st.write("Enter patient details and click Predict")

# --- Input fields (13 features) ---
age = st.number_input("Age", 1, 120, 50)
sex = st.selectbox("Sex (0=female, 1=male)", [0, 1])
cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 600, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 (1=true)", [0, 1])
restecg = st.selectbox("Rest ECG (0-2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1=yes)", [0, 1])
oldpeak = st.number_input("Oldpeak", 0.0, 10.0, 1.0, step=0.1)
slope = st.selectbox("Slope (0-2)", [0, 1, 2])
ca = st.selectbox("Number of Vessels (0-3)", [0, 1, 2, 3])
thal = st.selectbox("Thal (0-2)", [0, 1, 2])

features = [
    age, sex, cp, trestbps, chol, fbs, restecg,
    thalach, exang, oldpeak, slope, ca, thal
]

# --- Predict button ---
if st.button("Predict"):
    try:
        response = requests.post(
            "http://127.0.0.1:5000/predict",
            json={"features": features}
        )

        result = response.json()

        if "prediction" in result:
            st.success(f"Prediction: {result['prediction']}")
            st.info(f"Confidence: {result['confidence']:.2f}")
        else:
            st.warning(result)

    except Exception as e:
        st.error(f"Error: {e}")