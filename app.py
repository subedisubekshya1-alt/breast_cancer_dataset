import streamlit as st
import joblib
import numpy as np

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

st.markdown("""<style>
.stApp {background-color: #FFE4EC;}
.stApp, .stApp h1, .stApp p, .stApp label {color: #0B2545 !important;}
[data-testid="stNumberInputContainer"], [data-testid="stNumberInputContainer"] input, [data-testid="stNumberInputContainer"] button {background-color: #FFFFFF !important; color: #38BDF8 !important;}
div.stButton > button {background-color: #FF8A80 !important; color: #0B2545 !important; font-weight: bold;}
.box {padding: 16px; border-radius: 10px; font-size: 20px; font-weight: bold; text-align: center; border: 2px solid #0B2545;}
</style>""", unsafe_allow_html=True)

model, scaler = joblib.load("model.pkl"), joblib.load("scaler.pkl")

st.markdown("<h1 style='text-align:center;'>Breast Cancer Prediction</h1><p style='text-align:center;'>Enter your measurements in the cells below.</p>", unsafe_allow_html=True)

features = [
    'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean',
    'compactness_mean', 'concavity_mean', 'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
    'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
    'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se',
    'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst',
    'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst'
]

cols = st.columns(3)
inputs = [cols[i % 3].number_input(f, min_value=0.0, value=0.0, step=0.0001, format="%g") for i, f in enumerate(features)]

if st.button("Predict Diagnosis", use_container_width=True):
    pred = model.predict(scaler.transform(np.array([inputs])))[0]
    bg, label = ("#FFCDD2", "True positive (M)") if pred == 1 else ("#FFFFFF", "False negative (B)")
    st.markdown(f"<div class='box' style='background:{bg};'>Result: {label}</div>", unsafe_allow_html=True)

