import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
import pickle
import os
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# ── Config ────────────────────────────────────────────────────────────────────
MAX_VOCAB  = 30_000
MAX_LEN    = 256
EMBED_DIM  = 128
HIDDEN_DIM = 128
BATCH_SIZE = 64
EPOCHS     = 5
LR         = 1e-3
DEVICE     = "mps" if torch.backends.mps.is_available() else "cpu"  # Apple Silicon
print(f"Using device: {DEVICE}")

# ── Tokenizer ─────────────────────────────────────────────────────────────────
class Vocabulary:
    def __init__(self, max_size=MAX_VOCAB):
        self.max_size  = max_size
        self.word2idx  = {"<PAD>": 0, "<UNK>": 1}
        self.idx2word  = {0: "<PAD>", 1: "<UNK>"}

    def build(self, texts):
        counter = Counter()
        for t in texts:
            counter.update(t.lower().split())
        for word, _ in counter.most_common(self.max_size - 2):
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx]  = word

    def encode(self, text, max_len=MAX_LEN):
        tokens = text.lower().split()[:max_len]
        ids    = [self.word2idx.get(t, 1) for t in tokens]
        # pad or truncate
        ids += [0] * (max_len - len(ids))
        return ids[:max_len]

# ── Dataset ───────────────────────────────────────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, df, vocab):
        self.encodings = [vocab.encode(t) for t in df["cleaned_text"].fillna("")]
        self.labels    = df["label"].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.encodings[idx], dtype=torch.long),
            torch.tensor(self.labels[idx],    dtype=torch.float),
        )

# ── BiLSTM Model ──────────────────────────────────────────────────────────────
class BiLSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm      = nn.LSTM(embed_dim, hidden_dim, batch_first=True,
                                 bidirectional=True, num_layers=2,
                                 dropout=0.3)
        self.dropout   = nn.Dropout(0.4)
        self.fc        = nn.Linear(hidden_dim * 2, 1)

    def forward(self, x):
        emb = self.dropout(self.embedding(x))
        out, (h, _) = self.lstm(emb)
        # concat last forward + backward hidden state
        h = torch.cat((h[-2], h[-1]), dim=1)
        return self.fc(self.dropout(h)).squeeze(1)

# ── Train loop ────────────────────────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    all_preds, all_probs, all_labels = [], [], []
    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            probs  = torch.sigmoid(logits).cpu().numpy()
            preds  = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(y.cpu().numpy())
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return acc, f1, auc

def train():
    # Load data
    train_df = pd.read_csv("data/splits/train_features.csv")
    val_df   = pd.read_csv("data/splits/val_features.csv")
    test_df  = pd.read_csv("data/splits/test_features.csv")

    # Build vocabulary
    print("Building vocabulary...")
    vocab = Vocabulary()
    vocab.build(train_df["cleaned_text"].fillna("").tolist())
    print(f"Vocab size: {len(vocab.word2idx)}")

    # Datasets & loaders
    train_ds = TextDataset(train_df, vocab)
    val_ds   = TextDataset(val_df,   vocab)
    test_ds  = TextDataset(test_df,  vocab)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE)

    # Model
    model     = BiLSTMClassifier(len(vocab.word2idx), EMBED_DIM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    best_val_f1 = 0
    print("\nTraining BiLSTM...\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        val_acc, val_f1, val_auc = evaluate(model, val_loader)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} "
              f"| Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            os.makedirs("src/models/saved", exist_ok=True)
            torch.save(model.state_dict(), "src/models/saved/bilstm_model.pt")
            print(f"  ✓ Best model saved (F1={best_val_f1:.4f})")

    # Test evaluation
    print("\nLoading best model for test evaluation...")
    model.load_state_dict(torch.load("src/models/saved/bilstm_model.pt",
                                      map_location=DEVICE))
    test_acc, test_f1, test_auc = evaluate(model, test_loader)
    print(f"\n── Test Results ──────────────────")
    print(f"  Accuracy : {test_acc:.4f}")
    print(f"  F1 Score : {test_f1:.4f}")
    print(f"  ROC-AUC  : {test_auc:.4f}")

    # Save vocabulary
    with open("src/models/saved/vocab.pkl", "wb") as f:
        pickle.dump(vocab, f)
    print("\nVocabulary saved to src/models/saved/vocab.pkl")

if __name__ == "__main__":
    train()