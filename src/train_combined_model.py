import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from scipy.sparse import hstack
from scipy.sparse import csr_matrix

# load dataset
df = pd.read_csv("../data/processed/dataset_with_stylometry.csv")

X_text = df["text"]

# stylometric features
stylometric_features = df[[
    "avg_sentence_length",
    "sentence_variance",
    "type_token_ratio",
    "stopword_ratio",
    "punctuation_ratio"
]]

y = df["label"]

# train test split
X_train_text, X_test_text, X_train_style, X_test_style, y_train, y_test = train_test_split(
    X_text,
    stylometric_features,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# TF-IDF vectorizer
vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1,2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train_text)
X_test_tfidf = vectorizer.transform(X_test_text)

# convert stylometric features to sparse matrix
X_train_style = csr_matrix(X_train_style.values)
X_test_style = csr_matrix(X_test_style.values)

# combine features
X_train_combined = hstack([X_train_tfidf, X_train_style])
X_test_combined = hstack([X_test_tfidf, X_test_style])

# train classifier
model = LogisticRegression(max_iter=2000)
model.fit(X_train_combined, y_train)

# predictions
y_pred = model.predict(X_test_combined)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))

import joblib

joblib.dump(model, "../models/combined_model.pkl")
joblib.dump(vectorizer, "../models/tfidf_vectorizer.pkl")

print("Model saved successfully.")