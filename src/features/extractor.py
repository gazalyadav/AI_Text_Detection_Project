import pandas as pd
import numpy as np
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import os

nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)

class FeatureExtractor:

    def burstiness(self, text: str) -> float:
        """AI text tends to be uniform; human text is bursty."""
        sentences = sent_tokenize(text)
        if len(sentences) < 2:
            return 0.0
        lengths = [len(word_tokenize(s)) for s in sentences]
        mean = np.mean(lengths)
        std  = np.std(lengths)
        if mean == 0:
            return 0.0
        return round((std - mean) / (std + mean + 1e-9), 4)

    def punctuation_density(self, text: str) -> float:
        if not text:
            return 0.0
        puncts = sum(1 for c in text if c in '.,!?;:')
        return round(puncts / len(text), 4)

    def repetition_score(self, text: str) -> float:
        """How often words repeat — AI tends to repeat phrases."""
        words = word_tokenize(text.lower())
        words = [w for w in words if w.isalpha()]
        if not words:
            return 0.0
        unique = len(set(words))
        return round(1 - (unique / len(words)), 4)

    def avg_paragraph_length(self, text: str) -> float:
        paragraphs = [p.strip() for p in text.split('\n') if p.strip()]
        if not paragraphs:
            return 0.0
        lengths = [len(word_tokenize(p)) for p in paragraphs]
        return round(np.mean(lengths), 2)

    def extract(self, text: str) -> dict:
        if not isinstance(text, str) or not text.strip():
            return {
                "burstiness": 0.0,
                "punctuation_density": 0.0,
                "repetition_score": 0.0,
                "avg_paragraph_length": 0.0,
            }
        return {
            "burstiness"           : self.burstiness(text),
            "punctuation_density"  : self.punctuation_density(text),
            "repetition_score"     : self.repetition_score(text),
            "avg_paragraph_length" : self.avg_paragraph_length(text),
        }


def extract_features(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    extractor = FeatureExtractor()
    print("Extracting features...")

    features = df["cleaned_text"].apply(extractor.extract)
    feat_df  = pd.DataFrame(features.tolist())

    df = pd.concat([df, feat_df], axis=1)
    df.to_csv(output_path, index=False)
    print(f"Done! Saved to {output_path}")


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        extract_features(
            input_path  = f"data/splits/{split}_clean.csv",
            output_path = f"data/splits/{split}_features.csv"
        )
        