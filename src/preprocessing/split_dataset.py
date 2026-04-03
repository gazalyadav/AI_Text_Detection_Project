import pandas as pd
from sklearn.model_selection import train_test_split
import os

def split_and_save():
    print("Loading dataset...")
    df = pd.read_csv("data/raw/dataset.csv")
    print(f"Total: {len(df)} | Human: {(df.label==0).sum()} | AI: {(df.label==1).sum()}")

    # 70% train, 15% val, 15% test
    train, temp = train_test_split(df, test_size=0.30, random_state=42, stratify=df.label)
    val, test   = train_test_split(temp, test_size=0.50, random_state=42, stratify=temp.label)

    os.makedirs("data/splits", exist_ok=True)
    train.to_csv("data/splits/train.csv", index=False)
    val.to_csv("data/splits/val.csv",     index=False)
    test.to_csv("data/splits/test.csv",   index=False)

    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)}")
    print("Splits saved to data/splits/")

if __name__ == "__main__":
    split_and_save()