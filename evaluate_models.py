"""
evaluate_models.py  --  Item #1 (+ trivial baselines) of the Eval/Ablation plan.

Loads all 7 trained BiLSTM checkpoints, scores them on the HELD-OUT TEST split
(not validation), and reports the metrics the paper's headline table needs:
accuracy, precision, recall, macro-F1, and ROC-AUC, plus a confusion matrix per
model. Also runs two control baselines (majority class, length-only logistic
regression) so SHAPE's strength can be checked against a trivial length signal.

Runs on CPU in a few seconds -- no GPU needed.

Usage (from the project root, with the training venv active):
    pip install scikit-learn matplotlib        # if not already present
    python evaluate_models.py

Outputs (written to ./eval_out/):
    test_results.csv          one row per model + baseline, all metrics
    test_results.md           same table, markdown (paste into the report)
    confusion_<name>.png      confusion matrix per model
    roc_curves.png            ROC curves, all models overlaid
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support, f1_score,
    roc_auc_score, confusion_matrix, roc_curve,
)
from sklearn.linear_model import LogisticRegression

MODEL_DIR = Path("models")
DATA_PATH = "Dataset + Feature/sequences.parquet"
OUT_DIR = Path("eval_out")
OUT_DIR.mkdir(exist_ok=True)
COL = {"pos": "pos_ids", "dep": "dep_ids", "shape": "shape_ids"}
PAD_IDX = 0

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)


# --------------------------------------------------------------------------- #
# Model (identical to training / load_models.ipynb)
# --------------------------------------------------------------------------- #
class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_sizes, emb_dim, hidden_dim, dropout):
        super().__init__()
        self.channels = list(vocab_sizes.keys())
        self.embeddings = nn.ModuleDict({
            ch: nn.Embedding(size, emb_dim, padding_idx=PAD_IDX)
            for ch, size in vocab_sizes.items()
        })
        self.lstm = nn.LSTM(emb_dim * len(self.channels), hidden_dim,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, 2)

    def forward(self, batch, lengths):
        embs = [self.embeddings[ch](batch[ch]) for ch in self.channels]
        x = torch.cat(embs, dim=-1) if len(embs) > 1 else embs[0]
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
        mask = (torch.arange(out.size(1), device=out.device)[None, :]
                < lengths.to(out.device)[:, None]).unsqueeze(-1)
        pooled = (out * mask).sum(1) / lengths.to(out.device).clamp(min=1)[:, None]
        return self.fc(self.dropout(pooled))


def load_one(name):
    ckpt = torch.load(MODEL_DIR / f"bilstm_{name}.pt", map_location=device)
    h = ckpt["hparams"]
    model = BiLSTMTagger(ckpt["vocab_sizes"], h["emb_dim"], h["hidden_dim"], h["dropout"])
    model.load_state_dict(ckpt["state_dict"])
    model.max_len = h["max_len"]
    return model.to(device).eval()


models = {
    p.stem.replace("bilstm_", ""): load_one(p.stem.replace("bilstm_", ""))
    for p in sorted(MODEL_DIR.glob("bilstm_*.pt"))
}
print("loaded models:", ", ".join(models))


# --------------------------------------------------------------------------- #
# Test split (the same indices saved at training time)
# --------------------------------------------------------------------------- #
split = json.loads((MODEL_DIR / "split.json").read_text())
df = pd.read_parquet(DATA_PATH)
train_df = df.iloc[split["train"]].reset_index(drop=True)
test_df = df.iloc[split["test"]].reset_index(drop=True)
print(f"test set: {len(test_df)} articles  "
      f"label balance: {test_df['label'].value_counts().to_dict()}")

MAX_LEN = next(iter(models.values())).max_len


class TestSet(Dataset):
    def __init__(self, frame, channels):
        self.channels = channels
        self.data = {ch: [np.array(s[:MAX_LEN], dtype=np.int64) for s in frame[COL[ch]]]
                     for ch in channels}
        self.labels = frame["label"].to_numpy(dtype=np.int64)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        seqs = {ch: torch.from_numpy(self.data[ch][i]) for ch in self.channels}
        return seqs, len(seqs[self.channels[0]]), self.labels[i]


def _collate(channels):
    def collate(batch):
        seqs, lengths, labels = zip(*batch)
        padded = {ch: pad_sequence([s[ch] for s in seqs], batch_first=True,
                                   padding_value=PAD_IDX) for ch in channels}
        return padded, torch.tensor(lengths), torch.tensor(labels)
    return collate


def predict(model, batch_size=64):
    """Return (y_true, y_pred, y_prob) on the full test set. y_prob = P(label==1)."""
    ds = TestSet(test_df, model.channels)
    loader = DataLoader(ds, batch_size=batch_size, collate_fn=_collate(model.channels))
    y_true, y_pred, y_prob = [], [], []
    with torch.no_grad():
        for batch, lengths, labels in loader:
            batch = {ch: t.to(device) for ch, t in batch.items()}
            logits = model(batch, lengths.to(device))
            prob1 = torch.softmax(logits, dim=1)[:, 1]
            y_pred.extend(logits.argmax(1).cpu().tolist())
            y_prob.extend(prob1.cpu().tolist())
            y_true.extend(labels.tolist())
    return np.array(y_true), np.array(y_pred), np.array(y_prob)


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #
def metric_row(name, y_true, y_pred, y_prob=None):
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0)
    row = {
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": p,
        "recall_macro": r,
        "f1_macro": f1,
        "f1_pos": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_true, y_prob) if y_prob is not None else np.nan,
    }
    return row


rows, roc_data = [], {}
preds_cache = {}  # keep for McNemar later (item #5)

for name, model in models.items():
    y_true, y_pred, y_prob = predict(model)
    rows.append(metric_row(name, y_true, y_pred, y_prob))
    roc_data[name] = (y_true, y_prob)
    preds_cache[name] = y_pred

    cm = confusion_matrix(y_true, y_pred)
    np.save(OUT_DIR / f"cm_{name}.npy", cm)

# ---- baselines ------------------------------------------------------------ #
y_true_global = roc_data[next(iter(roc_data))][0]

# (a) majority class
majority = np.bincount(train_df["label"]).argmax()
rows.append(metric_row("baseline_majority",
                       y_true_global,
                       np.full_like(y_true_global, majority),
                       None))

# (b) length-only logistic regression (n_tokens) -- control for the worry that
#     SHAPE merely proxies article length.
lr = LogisticRegression(max_iter=1000)
lr.fit(train_df[["n_tokens"]].to_numpy(), train_df["label"].to_numpy())
len_pred = lr.predict(test_df[["n_tokens"]].to_numpy())
len_prob = lr.predict_proba(test_df[["n_tokens"]].to_numpy())[:, 1]
rows.append(metric_row("baseline_length_only", y_true_global, len_pred, len_prob))
preds_cache["baseline_length_only"] = len_pred

# --------------------------------------------------------------------------- #
# Save table
# --------------------------------------------------------------------------- #
res = pd.DataFrame(rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
res_round = res.copy()
for c in res.columns:
    if c != "model":
        res_round[c] = res_round[c].astype(float).round(4)

res_round.to_csv(OUT_DIR / "test_results.csv", index=False)
try:
    (OUT_DIR / "test_results.md").write_text(res_round.to_markdown(index=False))
except ImportError:
    print("(install 'tabulate' to also get the markdown table; CSV written.)")
np.save(OUT_DIR / "test_preds.npy", preds_cache, allow_pickle=True)

print("\n==== TEST-SET RESULTS (sorted by macro-F1) ====")
print(res_round.to_string(index=False))

# --------------------------------------------------------------------------- #
# Figures (optional -- skipped gracefully if matplotlib missing)
# --------------------------------------------------------------------------- #
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ROC curves
    plt.figure(figsize=(6, 6))
    for name, (yt, yp) in roc_data.items():
        fpr, tpr, _ = roc_curve(yt, yp)
        auc = roc_auc_score(yt, yp)
        plt.plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=0.8)
    plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
    plt.title("ROC -- structural channel ablations (test set)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(OUT_DIR / "roc_curves.png", dpi=150)
    plt.close()

    # confusion matrices
    for name in models:
        cm = np.load(OUT_DIR / f"cm_{name}.npy")
        plt.figure(figsize=(3.2, 3))
        plt.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            plt.text(j, i, str(v), ha="center", va="center",
                     color="white" if v > cm.max() / 2 else "black")
        plt.xticks([0, 1]); plt.yticks([0, 1])
        plt.xlabel("predicted"); plt.ylabel("true"); plt.title(name)
        plt.tight_layout()
        plt.savefig(OUT_DIR / f"confusion_{name}.png", dpi=150)
        plt.close()
    print(f"\nFigures + tables written to {OUT_DIR.resolve()}")
except ImportError:
    print("\n(matplotlib not installed -- CSV/MD tables written, figures skipped.)")
