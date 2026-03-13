import streamlit as st
import pickle
import os
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "combined_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, "rb"))
vectorizer = pickle.load(open(vectorizer_path, "rb"))

st.title("AI Text Detection System")

st.write("Paste text below to check if it is AI generated.")

text = st.text_area("Enter text")

if st.button("Analyze"):

    if text.strip() == "":
        st.warning("Please enter some text")

    else:
        X = vectorizer.transform([text])

        prob = model.predict_proba(X)[0][1]

        ai_percent = prob * 100
        human_percent = 100 - ai_percent

        st.subheader("Detection Result")

        st.write(f"AI Generated: {ai_percent:.2f}%")
        st.write(f"Human Written: {human_percent:.2f}%")