import pandas as pd
import re

file_path = "../data/raw/ai_texts.txt"

with open(file_path, "r", encoding="utf-8") as f:
    content = f.read()

# split using regex to catch separators even with spaces
essays = re.split(r"\n\s*={6}\s*\n", content)

# clean essays
essays = [e.strip() for e in essays if len(e.strip()) > 100]

ai_df = pd.DataFrame({
    "text": essays,
    "label": 1
})

print("Total AI Essays:", len(ai_df))

ai_df.to_csv("../data/processed/ai_dataset.csv", index=False)

print("Saved to data/processed/ai_dataset.csv")