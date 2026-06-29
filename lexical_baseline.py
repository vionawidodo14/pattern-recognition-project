"""
lexical_baseline.py  --  Item #2 of the Eval/Ablation plan.

The central claim is that structure-only models "competitively" separate fake
from real. That claim is empty without a LEXICAL upper-reference: a model that
DOES see the words. This trains a standard TF-IDF + Logistic Regression on the
raw article text (title + body), on the SAME train/test split as the structural
models, and reports the same metrics so the numbers are directly comparable.

CPU-only, no torch. Runs in a few seconds.

Usage (project root, training venv active):
    python lexical_baseline.py

Outputs (./eval_out/):
    lexical_results.csv     metrics for the lexical model (+ appended to the
                            structural table if test_results.csv is present)
    lexical_preds.npy       test-set predictions (for McNemar in significance_tests.py)
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             f1_score, roc_auc_score)

CSV_PATH = "Dataset + Feature/welfake_clean.csv"
SPLIT_PATH = Path("models") / "split.json"
OUT_DIR = Path("eval_out"); OUT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------------------------------- #
# Same split as the structural models (positional indices into the dataset).
# welfake_clean.csv is row-aligned with sequences.parquet, so the indices match.
# --------------------------------------------------------------------------- #
split = json.loads(SPLIT_PATH.read_text())
df = pd.read_csv(CSV_PATH)
text = (df["title"].fillna("") + " " + df["text"].fillna(""))  # same field as features
y = df["label"].to_numpy()

train_idx, test_idx = np.array(split["train"]), np.array(split["test"])
X_train_txt, X_test_txt = text.iloc[train_idx], text.iloc[test_idx]
y_train, y_test = y[train_idx], y[test_idx]
print(f"train {len(train_idx)}  test {len(test_idx)}  "
      f"test balance {pd.Series(y_test).value_counts().to_dict()}")

# --------------------------------------------------------------------------- #
# TF-IDF (word 1-2 grams) + Logistic Regression -- a strong, standard lexical
# baseline. Vectorizer is fit on TRAIN ONLY (no test leakage).
# --------------------------------------------------------------------------- #
vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                      sublinear_tf=True, strip_accents="unicode")
Xtr = vec.fit_transform(X_train_txt)
Xte = vec.transform(X_test_txt)
print(f"tf-idf vocabulary: {len(vec.vocabulary_):,} features")

clf = LogisticRegression(max_iter=2000, C=4.0)
clf.fit(Xtr, y_train)

y_pred = clf.predict(Xte)
y_prob = clf.predict_proba(Xte)[:, 1]

p, r, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="macro",
                                              zero_division=0)
row = {
    "model": "lexical_tfidf_lr",
    "accuracy": accuracy_score(y_test, y_pred),
    "precision_macro": p,
    "recall_macro": r,
    "f1_macro": f1,
    "f1_pos": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob),
}
res = pd.DataFrame([row]).round(4)
res.to_csv(OUT_DIR / "lexical_results.csv", index=False)
np.save(OUT_DIR / "lexical_preds.npy", {"lexical_tfidf_lr": y_pred}, allow_pickle=True)

print("\n==== LEXICAL BASELINE (test set) ====")
print(res.to_string(index=False))

# --------------------------------------------------------------------------- #
# Side-by-side with the structural results, if they exist.
# --------------------------------------------------------------------------- #
struct_path = OUT_DIR / "test_results.csv"
if struct_path.exists():
    struct = pd.read_csv(struct_path)
    combined = (pd.concat([struct, res], ignore_index=True)
                .sort_values("f1_macro", ascending=False).reset_index(drop=True))
    combined.to_csv(OUT_DIR / "all_results.csv", index=False)
    print("\n==== STRUCTURAL vs LEXICAL (sorted by macro-F1) ====")
    print(combined.to_string(index=False))

    best_struct = struct.loc[struct["f1_macro"].idxmax()]
    gap = row["f1_macro"] - best_struct["f1_macro"]
    print(f"\nBest structural: {best_struct['model']} "
          f"(macro-F1 {best_struct['f1_macro']:.4f})")
    print(f"Lexical:         {row['f1_macro']:.4f}  "
          f"=> lexical leads by {gap:+.4f} macro-F1")
    print("Interpretation: a small positive gap supports the 'structure is "
          "competitive' claim; a large gap argues structure leaves signal on "
          "the table.")
else:
    print("\n(run evaluate_models.py first to get the structural side-by-side.)")
