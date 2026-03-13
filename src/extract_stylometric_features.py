import pandas as pd
import spacy
import numpy as np

# load spacy model
nlp = spacy.load("en_core_web_sm")

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

# load dataset
df = pd.read_csv("../data/processed/balanced_dataset_binary.csv")

features = df["text"].apply(extract_features)

features_df = pd.DataFrame(
    features.tolist(),
    columns=[
        "avg_sentence_length",
        "sentence_variance",
        "type_token_ratio",
        "stopword_ratio",
        "punctuation_ratio"
    ]
)

final_df = pd.concat([df, features_df], axis=1)

final_df.to_csv("../data/processed/dataset_with_stylometry.csv", index=False)

print("Stylometric features extracted.")
print(final_df.head())
