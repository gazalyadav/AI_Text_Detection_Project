import streamlit as st
import joblib
import os
import numpy as np
from pypdf import PdfReader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "models", "combined_model.joblib")
vectorizer_path = os.path.join(BASE_DIR, "models", "tfidf_vectorizer.joblib")

@st.cache_resource
def load_artifacts():
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

model, vectorizer = load_artifacts()

st.title("AI Text Detection System")
st.write("Upload a PDF or paste text to detect AI-generated content.")

# ---------- TEXT INPUT ----------
text_input = st.text_area("Paste Text Here")

# ---------- PDF UPLOAD ----------
uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

def extract_pdf_text(file):
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

if st.button("Analyze"):

    if uploaded_file is not None:
        text = extract_pdf_text(uploaded_file)
    else:
        text = text_input

    if text.strip() == "":
        st.warning("No text detected.")
    else:
        paragraphs = [p.strip() for p in text.split("\n") if len(p.strip()) > 20]

        scores = []
        results = []

        for p in paragraphs:
            X = vectorizer.transform([p])
            prob = model.predict_proba(X)[0][1]

            scores.append(prob)

            results.append({
                "text": p,
                "ai_prob": prob
            })

        ai_percent = np.mean(scores) * 100
        human_percent = 100 - ai_percent

        st.subheader("Overall Result")

        st.metric("AI Generated", f"{ai_percent:.2f}%")
        st.metric("Human Written", f"{human_percent:.2f}%")

        st.subheader("Paragraph Analysis")

        for r in results:
            if r["ai_prob"] > 0.6:
                st.markdown(
                    f"<div style='background:#ffcccc;padding:10px;border-radius:5px;'>"
                    f"<b>AI ({r['ai_prob']*100:.1f}%)</b><br>{r['text']}</div>",
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"<div style='background:#ccffcc;padding:10px;border-radius:5px;'>"
                    f"<b>Human ({(1-r['ai_prob'])*100:.1f}%)</b><br>{r['text']}</div>",
                    unsafe_allow_html=True
                )