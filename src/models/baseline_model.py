import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import hstack, csr_matrix
from sklearn.metrics import (
    accuracy_score, f1_score,
    roc_auc_score, classification_report
)
import pickle
import os

# ── columns we engineered in the previous step ──────────────────────────────
FEATURE_COLS = [
    "num_sentences", "num_words", "avg_sent_length",
    "type_token_ratio", "avg_word_length",
    "burstiness", "punctuation_density",
    "repetition_score", "avg_paragraph_length",
]

def load_split(path):
    df = pd.read_csv(path)
    df["cleaned_text"] = df["cleaned_text"].fillna("")
    df[FEATURE_COLS]   = df[FEATURE_COLS].fillna(0)
    return df

def train():
    print("Loading data...")
    train = load_split("data/splits/train_features.csv")
    val   = load_split("data/splits/val_features.csv")
    test  = load_split("data/splits/test_features.csv")

    # ── TF-IDF on raw text ───────────────────────────────────────────────────
    print("Fitting TF-IDF...")
    tfidf = TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=50_000,
        sublinear_tf=True
    )
    X_train_tfidf = tfidf.fit_transform(train["cleaned_text"])
    X_val_tfidf   = tfidf.transform(val["cleaned_text"])
    X_test_tfidf  = tfidf.transform(test["cleaned_text"])

    # ── Handcrafted numeric features ─────────────────────────────────────────
    scaler = StandardScaler()
    X_train_num = csr_matrix(
        scaler.fit_transform(train[FEATURE_COLS].values)
    )
    X_val_num   = csr_matrix(scaler.transform(val[FEATURE_COLS].values))
    X_test_num  = csr_matrix(scaler.transform(test[FEATURE_COLS].values))

    # ── Combine both ─────────────────────────────────────────────────────────
    X_train = hstack([X_train_tfidf, X_train_num])
    X_val   = hstack([X_val_tfidf,   X_val_num])
    X_test  = hstack([X_test_tfidf,  X_test_num])

    y_train, y_val, y_test = train["label"], val["label"], test["label"]

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, C=1.0, n_jobs=-1)
    model.fit(X_train, y_train)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    for name, X, y in [("Val", X_val, y_val), ("Test", X_test, y_test)]:
        preds  = model.predict(X)
        probs  = model.predict_proba(X)[:, 1]
        acc    = accuracy_score(y, preds)
        f1     = f1_score(y, preds)
        auc    = roc_auc_score(y, probs)
        print(f"\n── {name} Results ──────────────────")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  F1 Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y, preds,
              target_names=["Human", "AI"]))

    # ── Save model artifacts ──────────────────────────────────────────────────
    os.makedirs("src/models/saved", exist_ok=True)
    with open("src/models/saved/baseline_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("src/models/saved/tfidf.pkl", "wb") as f:
        pickle.dump(tfidf, f)
    with open("src/models/saved/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    print("\nModel saved to src/models/saved/")

if __name__ == "__main__":
    train()