
import streamlit as st
import numpy as np
from src.predict import predict

st.title("Predictive Maintenance Dashboard")

sequence = np.random.rand(20, 3).tolist()

if st.button("Predict Failure"):
    prob = predict(sequence)
    st.write(f"Failure Probability: {prob}")
