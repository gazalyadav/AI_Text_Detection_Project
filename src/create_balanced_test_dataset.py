import pandas as pd
import pandas as pd

# Load datasets
human_df = pd.read_csv("../data/processed/human_dataset.csv")
ai_df = pd.read_csv("../data/processed/ai_dataset.csv")

print("Human samples available:", len(human_df))
print("AI samples available:", len(ai_df))

# Sample equal number of human essays
human_sampled = human_df.sample(n=len(ai_df), random_state=42)

# Combine datasets
balanced_df = pd.concat([human_sampled, ai_df], ignore_index=True)

# Shuffle dataset
balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced dataset size:", len(balanced_df))
print(balanced_df["label"].value_counts())

# Save dataset
balanced_df.to_csv("../data/processed/balanced_dataset_binary.csv", index=False)

print("\nDataset saved successfully.")