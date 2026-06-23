"""
Parses each WELFake article with spaCy and produces:

  sequences.parquet — ordered per-token tag indices (pos_ids, dep_ids, shape_ids)
  vocabs.json       — per-channel vocab  (<pad>=0, <unk>=1, sorted tags from 2)

Parsed docs are cached via DocBin so reparsing never happens.
Run with --dry-run to sanity-check one article without writing any files.
"""

import argparse
import json
from pathlib import Path

import pandas as pd
import spacy
from spacy.tokens import DocBin
from tqdm import tqdm

SEED = 404
INPUT_CSV = "welfake_clean.csv"
CACHE_PATH = "parsed_docs.spacy"
SEQ_PARQUET = "sequences.parquet"
VOCAB_JSON = "vocabs.json"


# ── shape helper ──────────────────────────────────────────────────────────────

def token_shape(token) -> str:
    if token.is_punct:
        return token.text
    text = token.text
    if token.like_num:
        return "dddd"
    if text.isupper() and len(text) > 1:
        return "XXXX"
    if len(text) > 1 and text[0].isupper() and text[1:].islower():
        return "Xxxx"
    return "xxxx"


# ── parse / cache ─────────────────────────────────────────────────────────────

def load_or_parse(texts: list[str], nlp) -> list:
    if Path(CACHE_PATH).exists():
        print(f"Loading cached docs from {CACHE_PATH}")
        doc_bin = DocBin().from_disk(CACHE_PATH)
        return list(doc_bin.get_docs(nlp.vocab))

    print(f"Parsing {len(texts):,} articles with spaCy")
    docs = []
    for doc in tqdm(
        nlp.pipe(texts, batch_size=256, disable=["ner", "lemmatizer"]),
        total=len(texts),
    ):
        docs.append(doc)

    doc_bin = DocBin(attrs=["ORTH", "POS", "DEP"], docs=docs)
    doc_bin.to_disk(CACHE_PATH)
    print(f"Cached to {CACHE_PATH}")
    return docs


# ── sequence extraction ───────────────────────────────────────────────────────

def extract_sequences(docs):
    pos_seqs, dep_seqs, shape_seqs = [], [], []
    for doc in tqdm(docs, desc="Extracting sequences"):
        pos_seqs.append([t.pos_ for t in doc])
        dep_seqs.append([t.dep_ for t in doc])
        shape_seqs.append([token_shape(t) for t in doc])
    return pos_seqs, dep_seqs, shape_seqs


# ── vocab ─────────────────────────────────────────────────────────────────────

def build_vocab(sequences: list[list[str]]) -> dict[str, int]:
    """<pad>=0, <unk>=1, then sorted unique tags from index 2 onwards."""
    tags = sorted({tok for seq in sequences for tok in seq})
    vocab: dict[str, int] = {"<pad>": 0, "<unk>": 1}
    for tag in tags:
        vocab[tag] = len(vocab)
    return vocab


def encode(seq: list[str], vocab: dict[str, int]) -> list[int]:
    unk = vocab["<unk>"]
    return [vocab.get(tag, unk) for tag in seq]



# ── main ──────────────────────────────────────────────────────────────────────

def main():
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df):,} articles")

    texts = (df["title"].fillna("") + " " + df["text"].fillna("")).tolist()

    nlp = spacy.load("en_core_web_sm")
    docs = load_or_parse(texts, nlp)
    print("Extracting sequences")
    pos_seqs, dep_seqs, shape_seqs = extract_sequences(docs)

    vocabs = {
        "pos":   build_vocab(pos_seqs),
        "dep":   build_vocab(dep_seqs),
        "shape": build_vocab(shape_seqs),
    }

    print("Encoding sequences")
    pos_ids   = [encode(s, vocabs["pos"])   for s in tqdm(pos_seqs,   desc="POS")]
    dep_ids   = [encode(s, vocabs["dep"])   for s in tqdm(dep_seqs,   desc="DEP")]
    shape_ids = [encode(s, vocabs["shape"]) for s in tqdm(shape_seqs, desc="SHAPE")]

    Path(VOCAB_JSON).write_text(json.dumps(vocabs, indent=2))
    print(f"Saved {VOCAB_JSON}  (pos={len(vocabs['pos'])}, dep={len(vocabs['dep'])}, shape={len(vocabs['shape'])})")

    seq_df = pd.DataFrame({
        "pos_ids":   pos_ids,
        "dep_ids":   dep_ids,
        "shape_ids": shape_ids,
        "n_tokens":  [len(s) for s in pos_seqs],
        "label":     df["label"].values,
    })
    seq_df.to_parquet(SEQ_PARQUET, index=False)
    print(f"Saved {SEQ_PARQUET}  —  {len(seq_df):,} rows × {len(seq_df.columns)} columns")


if __name__ == "__main__":
    main()
