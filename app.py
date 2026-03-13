import streamlit as st
import pickle
import numpy as np

# load models
model = pickle.load(open("models/combined_model.pkl", "rb"))
vectorizer = pickle.load(open("models/tfidf_vectorizer.pkl", "rb"))

st.title("AI Text Detection System")

st.write("Paste text below to analyze whether it is AI generated.")

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