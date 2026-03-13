import pandas as pd

# Load human dataset
human_df = pd.read_csv("../data/processed/human_dataset.csv")

# Load AI dataset
ai_df = pd.read_csv("../data/processed/ai_dataset.csv")

print("Human samples:", len(human_df))
print("AI samples:", len(ai_df))

# Combine
combined_df = pd.concat([human_df, ai_df], ignore_index=True)

# Shuffle dataset
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("Total combined samples:", len(combined_df))
print(combined_df["label"].value_counts())

# Save final dataset
combined_df.to_csv("../data/processed/final_dataset_binary.csv", index=False)

print("Saved to data/processed/final_dataset_binary.csv")