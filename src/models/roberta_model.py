import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import os

# ── Config ────────────────────────────────────────────────────────────────────
MODEL_NAME = "roberta-base"
MAX_LEN    = 256
BATCH_SIZE = 16
EPOCHS     = 3
LR         = 2e-5
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, tokenizer):
        self.texts  = df["cleaned_text"].fillna("").tolist()
        self.labels = df["label"].tolist()
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            max_length=MAX_LEN,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        return {
            "input_ids"     : enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label"         : torch.tensor(self.labels[idx], dtype=torch.long)
        }

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            logits = outputs.logits
            probs  = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds  = logits.argmax(dim=1).cpu().numpy()

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, f1, auc

# ── Train ─────────────────────────────────────────────────────────────────────
def train():
    print("Loading data...")
    train_df = pd.read_csv("data/splits/train_features.csv")
    val_df   = pd.read_csv("data/splits/val_features.csv")
    test_df  = pd.read_csv("data/splits/test_features.csv")

    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model     = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    ).to(DEVICE)

    train_loader = DataLoader(TextDataset(train_df, tokenizer),
                              batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_df, tokenizer),
                              batch_size=BATCH_SIZE)
    test_loader  = DataLoader(TextDataset(test_df, tokenizer),
                              batch_size=BATCH_SIZE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    criterion = torch.nn.CrossEntropyLoss()

    best_val_f1 = 0
    os.makedirs("src/models/saved", exist_ok=True)

    print("\nFine-tuning RoBERTa...\n")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids      = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels         = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()

            if (i + 1) % 100 == 0:
                print(f"  Epoch {epoch} | Step {i+1}/{len(train_loader)} "
                      f"| Loss: {total_loss/(i+1):.4f}")

        val_acc, val_f1, val_auc = evaluate(model, val_loader)
        print(f"\nEpoch {epoch}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} "
              f"| Val AUC: {val_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained("src/models/saved/roberta")
            tokenizer.save_pretrained("src/models/saved/roberta")
            print(f"  ✓ Best model saved (F1={best_val_f1:.4f})\n")

    # Final test evaluation
    print("Loading best model for test evaluation...")
    model = AutoModelForSequenceClassification.from_pretrained(
        "src/models/saved/roberta"
    ).to(DEVICE)
    test_acc, test_f1, test_auc = evaluate(model, test_loader)
    print(f"\n── Test Results ──────────────────")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  F1 Score : {test_f1:.4f}")
    print(f"  ROC-AUC  : {test_auc:.4f}")

if __name__ == "__main__":
    train()