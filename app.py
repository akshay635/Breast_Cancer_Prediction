# -*- coding: utf-8 -*-
"""
Created on Tue Jan  6 12:12:16 2026

@author: aksha
"""

import pandas as pd
import numpy as np
import joblib
import streamlit as st
import json


def load_model():
    return joblib.load("C:/Users/aksha/breast_cancer_project/breast_cancer_model_v1.joblib")

model = load_model()

st.title('Breast Cancer Prediction')

# Load feature names
with open("feature_names.json") as f:
    feature_names = json.load(f)
    
# Load medians json file
with open("medians.json") as f:
    medians = json.load(f)

with st.form("Cancer data fillup form"):
    st.header('Inside the form')
    FEATURE_RANGES = {
    "mean radius": (6.0, 28.0),
    "mean texture": (9.0, 39.0),
    "mean perimeter": (43.0, 189.0),
    "mean area": (143.0, 2501.0),
    "mean concavity": (0.0, 0.43),
    "mean concave points": (0.0, 0.20),
    "mean compactness": (0.0, 0.35),
    "mean smoothness": (0.05, 0.17)
    }

    inputs = {}

    for feature, (low, high) in FEATURE_RANGES.items():
        inputs[feature] = st.slider(
                            feature,
                            min_value=float(low),
                            max_value=float(high),
                            value=float(medians[feature])
    )
    
    submitted = st.form_submit_button('Submit')
    

if submitted:
    df = pd.DataFrame([medians | inputs])
    df = df.reindex(columns=feature_names)

    proba = model.predict_proba(df)[0]
    class_map = dict(zip(model.classes_, proba))

    pred_proba_malignant = class_map[0]
    pred_proba_benign = class_map[1]

    st.subheader('Breast Cancer Risk Assessment')

    if pred_proba_benign >= 0.75:
        st.success(
            f"Benign (non-cancerous) with probability {pred_proba_benign:.2%}"
        )
        st.markdown("**Suggested Action:** Routine monitoring only")

    elif 0.50 <= pred_proba_benign < 0.75:
        st.warning(
            f"Benign but close to malignant — probability {pred_proba_benign:.2%}"
        )
        st.markdown("**Suggested Action:** Medical consultation recommended")

    else:
        st.error(
            f"Malignant (cancerous) with probability {pred_proba_malignant:.2%}"
        )
        st.markdown(
            "**Suggested Action:** Immediate medical intervention required"
        )
        
    st.write("Changed features:", inputs)
    st.write("changed data", df)
    st.write("Full row variance:", np.var(df))

        
    
