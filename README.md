# Pattern Recognition Project

Topic-agnostic fake-news detection from **structural sequences only** (POS, DEP,
SHAPE) — no lexical content. We test whether structure alone can match a
word-based model, and which structural channel carries the most signal.

## How to run

### 0. Python Environment Setup
Python Versions: 3.14.x
```bash
#venv creation (skip if venv is already created)
python3 -m venv .venv
source .venv/bin/activate

#dependencies installation (DONT SKIP)
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```
On Windows (PowerShell): `python -m venv .venv` then `.\.venv\Scripts\Activate.ps1`.

### 1. Clean the dataset
```bash
python prepare_dataset.py
```
Outputs `welfake_clean.csv` — 5,000 fake + 5,000 real news articles, no nulls.

### 2. Extract features
```bash
python feature_extraction.py
```
Outputs:
- `sequences.parquet` — ordered POS / DEP / SHAPE tag-index sequences per article
- `vocabs.json` — tag → integer mappings for each channel
- `parsed_docs.spacy` — spaCy parse cache (re-running skips parsing automatically)

### 3. Train the ablation models
Run `train_ablation.ipynb`. It trains a BiLSTM on each of the 7 channel
combinations (pos, dep, shape, and their unions), saves the checkpoints to
`models/bilstm_*.pt`, and writes the fixed train/val/test split to
`models/split.json`. The split file is required by everything below, so keep it
under version control.

## Evaluation & ablation

> Steps 4–7 only need the trained checkpoints + `models/split.json`. They run on
> **CPU** in seconds (steps 4–6); only the multi-seed sweep (step 7) is slow.

### 4. Test-set evaluation (all 7 models + baselines)
```bash
python evaluate_models.py
```
Scores every model on the held-out test split and writes to `eval_out/`:
accuracy, macro precision/recall/F1, ROC-AUC, confusion matrices, ROC curves,
plus majority-class and length-only baselines. Also caches `test_preds.npy` for
the significance test.

### 5. Lexical baseline
```bash
python lexical_baseline.py
```
TF-IDF + logistic regression on the raw words (same split) — the content-based
reference the structural models are compared against. Writes `lexical_results.csv`
and a combined `all_results.csv`.

### 6. Significance testing
```bash
python significance_tests.py
```
Pairwise McNemar exact tests over the cached predictions. Writes
`mcnemar_pvalues.csv` and a readable `mcnemar_summary.txt`.

### 7. Multi-seed robustness (optional, slower)
```bash
python train_multiseed.py --seeds 0 1 2 3 4
```
Retrains all 7 combos over multiple seeds with the split held fixed, and reports
test mean +/- std per model (`multiseed_summary.csv`). GPU recommended but not
required; on CPU expect ~30–60 min for 5 seeds (use `--seeds 0` for a quick check).

## Outputs
- `eval_out/` — all evaluation tables and figures
- `Results_and_Ablation.docx` — written-up results section
