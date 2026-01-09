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


rf = joblib.load("final_random_forest.joblib")

dt = joblib.load("final_decision_tree.joblib")

common_features = ['worst concavity', 'mean radius', 'worst compactness', 'mean area']

st.title('Breast Cancer Analysis')

# Load feature names
with open("feature_names.json") as f:
    feature_names = json.load(f)
    
# Load medians json file
with open("medians.json") as f:
    medians = json.load(f)

with st.form("Cancer data fillup form"):
    st.header('Inside the form')
    FEATURE_RANGES = {
    "worst concavity": (0.0, 1.25),
    "mean radius": (6.0, 28.0),
    "worst compactness": (0.027, 1.058),
    "mean area": (143.0, 2501.0),
    }

    inputs = {}

    for feature, (low, high) in FEATURE_RANGES.items():
        inputs[feature] = st.slider(
                            feature,
                            min_value=float(low),
                            max_value=float(high)
    )
    
    submitted = st.form_submit_button('Submit')
    

if submitted:
    df = pd.DataFrame([inputs])
    
    
    proba = rf.predict_proba(df)[0][1]
    proba_rf = rf.predict_proba(df)[0]
    class_map = dict(zip(rf.classes_, proba_rf))

    pred_proba_malignant = class_map[1]
    pred_proba_benign = class_map[0]

    st.subheader('Breast Cancer Risk Assessment')

    if proba <= 0.25:
        st.success(
            f"Benign (non-cancerous) with probability {pred_proba_benign:.2%}"
        )
        st.markdown("**Suggested Action:** Routine monitoring only")

    elif 0.25 < proba <= 0.50:
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
    
    
    st.write(f'pred_proba_malignant: {pred_proba_malignant:.2%}')

    st.write(f'pred_proba_benign: {pred_proba_benign:.2%}')
