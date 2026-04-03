import pandas as pd

df = pd.read_csv("data/raw/dataset.csv")
print(f"Full size: {len(df)}")

# Balance: 25k human + 25k AI
human = df[df.label == 0].sample(25000, random_state=42)
ai    = df[df.label == 1].sample(25000, random_state=42)

df_small = pd.concat([human, ai]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"Trimmed size: {len(df_small)}")

df_small.to_csv("data/raw/dataset.csv", index=False)
print("Done! Saved 50k balanced samples.")