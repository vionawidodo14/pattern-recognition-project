"""
train_multiseed.py  --  Items #4 & #5 of the Eval/Ablation plan.

Retrains all 7 channel combos over several seeds so single-run differences
(e.g. dep_shape 0.9315 vs pos_shape 0.9325) can be reported as mean +/- std
instead of noise, and reports TEST metrics (not validation).

Design choice that matters: the train/val/test SPLIT is held FIXED across every
seed and every model (reused from models/split.json). Only the training seed
(weight init + batch shuffling) varies. This keeps the test set identical for
all runs, which (a) makes mean +/- std reflect pure training variance and
(b) leaves a single fixed test set so paired tests like McNemar (item #5) stay
valid. If split.json is missing, a split is created once and saved.

Built for Colab GPU (set the runtime to GPU). On GPU the full sweep
(7 combos x N seeds x 6 epochs) is a few minutes. It also runs on CPU, slower.

Usage:
    python train_multiseed.py --seeds 0 1 2 3 4
Outputs (./eval_out/):
    multiseed_runs.csv        one row per (model, seed): test acc/precision/recall/f1/auc
    multiseed_summary.csv     mean +/- std per model, sorted by mean test macro-F1
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_sequence
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                             precision_recall_fscore_support)

# ------- paths / hyperparameters (match train_ablation.ipynb) -------------- #
DATA_PATH = "Dataset + Feature/sequences.parquet"
MODEL_DIR = Path("models")
OUT_DIR = Path("eval_out"); OUT_DIR.mkdir(exist_ok=True)
SPLIT_SEED = 404           # only used if split.json must be created

MAX_LEN = 160
EMB_DIM = 48
HIDDEN_DIM = 64
DROPOUT = 0.3
BATCH_SIZE = 128
EPOCHS = 6
LR = 1e-3
PATIENCE = 2

CHANNELS = ["pos", "dep", "shape"]
COL = {"pos": "pos_ids", "dep": "dep_ids", "shape": "shape_ids"}
PAD_IDX = 0
COMBOS = {
    "pos": ["pos"], "dep": ["dep"], "shape": ["shape"],
    "pos_dep": ["pos", "dep"], "pos_shape": ["pos", "shape"],
    "dep_shape": ["dep", "shape"], "pos_dep_shape": ["pos", "dep", "shape"],
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------- data -------------------------------------- #
df = pd.read_parquet(DATA_PATH)
vocab_sizes = {ch: int(max(seq.max() for seq in df[COL[ch]])) + 1 for ch in CHANNELS}

split_path = MODEL_DIR / "split.json"
if split_path.exists():
    split = json.loads(split_path.read_text())
    train_idx = np.array(split["train"]); val_idx = np.array(split["val"])
    test_idx = np.array(split["test"])
    print("reusing fixed split from split.json")
else:
    idx = np.arange(len(df))
    train_idx, temp = train_test_split(idx, test_size=0.2, random_state=SPLIT_SEED,
                                       stratify=df["label"])
    val_idx, test_idx = train_test_split(temp, test_size=0.5, random_state=SPLIT_SEED,
                                         stratify=df["label"].iloc[temp])
    MODEL_DIR.mkdir(exist_ok=True)
    split_path.write_text(json.dumps({"seed": SPLIT_SEED, "train": train_idx.tolist(),
                                      "val": val_idx.tolist(), "test": test_idx.tolist()}))
    print("created and saved new split.json")

train_df = df.iloc[train_idx].reset_index(drop=True)
val_df = df.iloc[val_idx].reset_index(drop=True)
test_df = df.iloc[test_idx].reset_index(drop=True)


class TagDataset(Dataset):
    def __init__(self, frame, channels):
        self.channels = channels
        self.data = {ch: [np.array(s[:MAX_LEN], dtype=np.int64) for s in frame[COL[ch]]]
                     for ch in channels}
        self.labels = frame["label"].to_numpy(dtype=np.int64)

    def __len__(self): return len(self.labels)

    def __getitem__(self, i):
        seqs = {ch: torch.from_numpy(self.data[ch][i]) for ch in self.channels}
        return seqs, len(seqs[self.channels[0]]), self.labels[i]


def make_collate(channels):
    def collate(batch):
        seqs, lengths, labels = zip(*batch)
        padded = {ch: pad_sequence([s[ch] for s in seqs], batch_first=True,
                                   padding_value=PAD_IDX) for ch in channels}
        return padded, torch.tensor(lengths), torch.tensor(labels)
    return collate


class BiLSTMTagger(nn.Module):
    def __init__(self, vocab_sizes_active):
        super().__init__()
        self.channels = list(vocab_sizes_active.keys())
        self.embeddings = nn.ModuleDict({
            ch: nn.Embedding(size, EMB_DIM, padding_idx=PAD_IDX)
            for ch, size in vocab_sizes_active.items()})
        self.lstm = nn.LSTM(EMB_DIM * len(self.channels), HIDDEN_DIM,
                            batch_first=True, bidirectional=True)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(HIDDEN_DIM * 2, 2)

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


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    preds, golds = [], []
    with torch.set_grad_enabled(train):
        for batch, lengths, labels in loader:
            batch = {ch: t.to(device) for ch, t in batch.items()}
            lengths, labels = lengths.to(device), labels.to(device)
            logits = model(batch, lengths)
            loss = criterion(logits, labels)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            preds.extend(logits.argmax(1).cpu().tolist())
            golds.extend(labels.cpu().tolist())
    return f1_score(golds, preds)


@torch.no_grad()
def eval_test(model):
    ds = TagDataset(test_df, model.channels)
    loader = DataLoader(ds, batch_size=64, collate_fn=make_collate(model.channels))
    yt, yp, ypr = [], [], []
    for batch, lengths, labels in loader:
        batch = {ch: t.to(device) for ch, t in batch.items()}
        logits = model(batch, lengths.to(device))
        ypr.extend(torch.softmax(logits, 1)[:, 1].cpu().tolist())
        yp.extend(logits.argmax(1).cpu().tolist())
        yt.extend(labels.tolist())
    yt, yp, ypr = np.array(yt), np.array(yp), np.array(ypr)
    p, r, f1, _ = precision_recall_fscore_support(yt, yp, average="macro", zero_division=0)
    return {"test_acc": accuracy_score(yt, yp), "test_precision": p,
            "test_recall": r, "test_f1": f1, "test_auc": roc_auc_score(yt, ypr)}


def train_one(channels, seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    vsz = {ch: vocab_sizes[ch] for ch in channels}
    collate = make_collate(channels)
    tr = DataLoader(TagDataset(train_df, channels), batch_size=BATCH_SIZE,
                    shuffle=True, collate_fn=collate)
    vl = DataLoader(TagDataset(val_df, channels), batch_size=BATCH_SIZE,
                    collate_fn=collate)
    model = BiLSTMTagger(vsz).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    best_f1, best_state, stale = -1, None, 0
    for _ in range(EPOCHS):
        run_epoch(model, tr, criterion, opt)
        vf1 = run_epoch(model, vl, criterion)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    model.load_state_dict(best_state)
    return eval_test(model.eval())


def main(seeds):
    print(f"device: {device}   seeds: {seeds}")
    runs = []
    for name, chans in COMBOS.items():
        for seed in seeds:
            m = train_one(chans, seed)
            m.update({"model": name, "seed": seed})
            runs.append(m)
            print(f"  {name:<15} seed {seed}  test f1 {m['test_f1']:.4f}  "
                  f"acc {m['test_acc']:.4f}  auc {m['test_auc']:.4f}")

    runs_df = pd.DataFrame(runs)
    runs_df.to_csv(OUT_DIR / "multiseed_runs.csv", index=False)

    metrics = ["test_acc", "test_precision", "test_recall", "test_f1", "test_auc"]
    summary = (runs_df.groupby("model")[metrics]
               .agg(["mean", "std"]).round(4)
               .sort_values(("test_f1", "mean"), ascending=False))
    summary.to_csv(OUT_DIR / "multiseed_summary.csv")
    print("\n==== mean +/- std over seeds (test set) ====")
    print(summary.to_string())
    print(f"\nWritten to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    main(ap.parse_args().seeds)
