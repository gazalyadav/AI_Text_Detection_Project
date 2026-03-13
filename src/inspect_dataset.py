import pandas as pd

file_path = "../data/raw/training_set_rel3.tsv"

# Load dataset
df = pd.read_csv(file_path, sep="\t", encoding="latin1")

# Add word count
df["word_count"] = df["essay"].apply(lambda x: len(str(x).split()))

# Filter essays between 150 and 600 words
df = df[(df["word_count"] >= 150) & (df["word_count"] <= 600)]

# Keep only essay column
human_df = df[["essay"]].copy()
human_df.rename(columns={"essay": "text"}, inplace=True)

# Add label
human_df["label"] = 0

# Save processed dataset
human_df.to_csv("../data/processed/human_dataset.csv", index=False)

print("Final Human Dataset Shape:", human_df.shape)
print("Saved to data/processed/human_dataset.csv")