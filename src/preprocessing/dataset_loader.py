from datasets import load_dataset
import pandas as pd
import os

def download_and_save():
    print("Downloading dataset...")
    dataset = load_dataset("artem9k/ai-text-detection-pile", split="train")

    records = []
    for item in dataset:
        text = item.get("text", "").strip()
        source = item.get("source", "")
        if not text:
            continue
        label = 0 if source == "human" else 1
        records.append({"text": text, "label": label})

    df = pd.DataFrame(records)
    df = df.drop_duplicates(subset="text").reset_index(drop=True)

    print(f"Total samples: {len(df)}")
    print(f"Human: {(df.label == 0).sum()}, AI: {(df.label == 1).sum()}")

    os.makedirs("data/raw", exist_ok=True)
    df.to_csv("data/raw/dataset.csv", index=False)
    print("Saved to data/raw/dataset.csv")

if __name__ == "__main__":
    download_and_save()