import re
import nltk
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize

# Download required nltk data
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
nltk.download("stopwords", quiet=True)

class TextPreprocessor:
    def __init__(self):
        pass

    def clean(self, text: str) -> str:
        if not isinstance(text, str):
            return ""
        text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
        text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\'\"]', '', text)  # remove special chars
        text = re.sub(r'\s+', ' ', text).strip()           # normalize whitespace
        return text

    def get_stats(self, text: str) -> dict:
        sentences = sent_tokenize(text)
        words = word_tokenize(text)
        words_clean = [w for w in words if w.isalpha()]

        avg_sent_len = (
            sum(len(word_tokenize(s)) for s in sentences) / len(sentences)
            if sentences else 0
        )
        unique_words = set(w.lower() for w in words_clean)
        ttr = len(unique_words) / len(words_clean) if words_clean else 0

        return {
            "num_sentences"  : len(sentences),
            "num_words"      : len(words_clean),
            "avg_sent_length": round(avg_sent_len, 2),
            "type_token_ratio": round(ttr, 4),   # vocabulary richness
            "avg_word_length": round(
                sum(len(w) for w in words_clean) / len(words_clean), 2
            ) if words_clean else 0,
        }

    def process(self, text: str) -> dict:
        cleaned = self.clean(text)
        stats   = self.get_stats(cleaned)
        return {"cleaned_text": cleaned, **stats}


def preprocess_csv(input_path: str, output_path: str):
    print(f"Loading {input_path}...")
    df = pd.read_csv(input_path)

    preprocessor = TextPreprocessor()
    print("Cleaning and extracting stats...")

    results = df["text"].apply(preprocessor.process)
    stats_df = pd.DataFrame(results.tolist())

    df = pd.concat([df, stats_df], axis=1)
    df = df[df["cleaned_text"].str.len() > 50].reset_index(drop=True)  # drop very short texts

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df)} samples to {output_path}")
    return df


if __name__ == "__main__":
    for split in ["train", "val", "test"]:
        preprocess_csv(
            input_path  = f"data/splits/{split}.csv",
            output_path = f"data/splits/{split}_clean.csv"
        )