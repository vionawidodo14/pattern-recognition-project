# Pattern Recognition Project

## How to run

### 0. Python Environment Setup
Python Versions: 3.14.x
```bash
#venv creation
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

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
