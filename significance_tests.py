"""
significance_tests.py  --  Item #5 of the Eval/Ablation plan.

The top structural models sit within ~0.03 macro-F1 of each other on the test
set (e.g. pos_dep_shape 0.945 vs pos_shape 0.941). A raw F1 gap that small is
meaningless without a significance test. McNemar's test is the right tool: it
compares two classifiers on the SAME test items by looking only at the cases
where they disagree, so it directly answers "is model A reliably better than
model B, or is this noise?"

This loads the cached test-set predictions written by evaluate_models.py
(and, if present, lexical_baseline.py), reconstructs the ground-truth labels
from the fixed split, and runs McNemar on every pair.

CPU-only, no torch. Needs scipy (installed automatically with scikit-learn).

Usage (project root, venv active):
    python evaluate_models.py        # must have been run -> test_preds.npy
    python lexical_baseline.py       # optional -> lexical_preds.npy
    python significance_tests.py

Outputs (./eval_out/):
    mcnemar_pvalues.csv     full pairwise p-value matrix
    mcnemar_summary.txt     readable verdict for the key comparisons
"""

import json
from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
import scipy.stats as st

OUT_DIR = Path("eval_out")
DATA_PATH = "Dataset + Feature/sequences.parquet"
SPLIT_PATH = Path("models") / "split.json"
ALPHA = 0.05

# --------------------------------------------------------------------------- #
# Ground truth for the test split (same indices used everywhere).
# --------------------------------------------------------------------------- #
split = json.loads(SPLIT_PATH.read_text())
df = pd.read_parquet(DATA_PATH)
y_true = df.iloc[split["test"]]["label"].to_numpy()
n = len(y_true)
print(f"test items: {n}")

# --------------------------------------------------------------------------- #
# Collect every model's cached predictions.
# --------------------------------------------------------------------------- #
preds = {}
struct_path = OUT_DIR / "test_preds.npy"
if not struct_path.exists():
    raise SystemExit("test_preds.npy not found -- run evaluate_models.py first.")
preds.update(np.load(struct_path, allow_pickle=True).item())

lex_path = OUT_DIR / "lexical_preds.npy"
if lex_path.exists():
    preds.update(np.load(lex_path, allow_pickle=True).item())
    print("included lexical baseline in the comparison")

# correctness vector per model
correct = {name: (np.asarray(p) == y_true).astype(int) for name, p in preds.items()}
acc = {name: c.mean() for name, c in correct.items()}
names = sorted(correct, key=lambda k: acc[k], reverse=True)


def mcnemar(a, b):
    """Exact (binomial) McNemar test on two correctness vectors.
    Returns (b_only, c_only, p_value) where
        b_only = A right & B wrong,  c_only = A wrong & B right."""
    b_only = int(np.sum((a == 1) & (b == 0)))
    c_only = int(np.sum((a == 0) & (b == 1)))
    disc = b_only + c_only
    if disc == 0:
        return b_only, c_only, 1.0
    p = st.binomtest(min(b_only, c_only), disc, 0.5, alternative="two-sided").pvalue
    return b_only, c_only, p


# --------------------------------------------------------------------------- #
# Full pairwise p-value matrix.
# --------------------------------------------------------------------------- #
pmat = pd.DataFrame(np.nan, index=names, columns=names)
for x, z in combinations(names, 2):
    _, _, p = mcnemar(correct[x], correct[z])
    pmat.loc[x, z] = pmat.loc[z, x] = round(p, 4)
pmat.to_csv(OUT_DIR / "mcnemar_pvalues.csv")

print("\n==== McNemar pairwise p-values (two-sided exact) ====")
print(pmat.to_string())

# --------------------------------------------------------------------------- #
# Readable verdict: best model vs each of the rest.
# --------------------------------------------------------------------------- #
best = names[0]
lines = [f"Reference (highest test accuracy): {best}  (acc {acc[best]:.4f})",
         f"alpha = {ALPHA}   (a 'significant' result means the two models differ "
         f"beyond chance)\n"]
for other in names[1:]:
    b_only, c_only, p = mcnemar(correct[best], correct[other])
    verdict = "SIGNIFICANT" if p < ALPHA else "not significant"
    lines.append(
        f"{best} vs {other:<20} acc {acc[best]:.3f} vs {acc[other]:.3f}  "
        f"| {best} right/other wrong = {b_only}, other right/{best} wrong = {c_only}  "
        f"| p = {p:.4f}  -> {verdict}")

summary = "\n".join(lines)
(OUT_DIR / "mcnemar_summary.txt").write_text(summary)
print("\n==== Verdict: best model vs each other model ====")
print(summary)
print(f"\nWritten to {OUT_DIR.resolve()}")
