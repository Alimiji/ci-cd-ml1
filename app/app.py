# app/app.py
import streamlit as st
import requests

st.title("Demo MLOps – Prédiction")

f1 = st.number_input("Feature 1", value=0.0)
f2 = st.number_input("Feature 2", value=0.0)

if st.button("Prédire"):
    resp = requests.post(
        "https://ton-api.onrender.com/predict",
        json={"feature1": f1, "feature2": f2}
    )
    st.write("Résultat :", resp.json())

