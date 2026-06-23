# Pattern Recognition Project

## How to run

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
