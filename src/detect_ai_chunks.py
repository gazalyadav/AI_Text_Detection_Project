import joblib
import spacy
import numpy as np
from scipy.sparse import hstack
from scipy.sparse import csr_matrix

nlp = spacy.load("en_core_web_sm")

# load model and vectorizer
model = joblib.load("../models/combined_model.pkl")
vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")


def extract_features(text):

    doc = nlp(text)

    sentences = list(doc.sents)
    tokens = [t for t in doc if not t.is_space]

    sentence_lengths = [len(sent) for sent in sentences]

    avg_sentence_length = np.mean(sentence_lengths) if sentence_lengths else 0
    sentence_var = np.var(sentence_lengths) if sentence_lengths else 0

    words = [t.text.lower() for t in tokens if t.is_alpha]
    unique_words = len(set(words))
    total_words = len(words)

    type_token_ratio = unique_words / total_words if total_words else 0

    stopwords = len([t for t in tokens if t.is_stop])
    stopword_ratio = stopwords / len(tokens) if tokens else 0

    punctuation = len([t for t in tokens if t.is_punct])
    punctuation_ratio = punctuation / len(tokens) if tokens else 0

    return [
        avg_sentence_length,
        sentence_var,
        type_token_ratio,
        stopword_ratio,
        punctuation_ratio
    ]


def detect_ai(text):

    paragraphs = text.split("\n\n")

    ai_scores = []

    for i, p in enumerate(paragraphs):

        tfidf = vectorizer.transform([p])

        style = csr_matrix([extract_features(p)])

        combined = hstack([tfidf, style])

        prob = model.predict_proba(combined)[0][1]

        ai_scores.append(prob)

        label = "AI" if prob > 0.5 else "Human"

        print(f"Paragraph {i+1}: {label} (AI probability: {prob:.2f})")

    overall_ai = sum(ai_scores) / len(ai_scores)

    ai_percentage = overall_ai * 100
    human_percentage = 100 - ai_percentage

    print("\nFinal Result:")
    print(f"AI Used: {ai_percentage:.2f}%")
    print(f"Human Written: {human_percentage:.2f}%")


# Example test text
sample_text = """
Artificial intelligence is rapidly transforming industries around the world. Many companies are adopting AI technologies to improve efficiency and decision making.

However, there are concerns about job displacement and ethical issues related to automation. Governments and organizations must carefully manage these challenges.

Technology has always changed the nature of work, but humans have continuously adapted by developing new skills and industries.
"""

detect_ai(sample_text)