"""
Downloads and cleans the WELFake dataset from Hugging Face.
Outputs 5000 real news + 5000 fake news with no null values.
"""

import pandas as pd
from datasets import load_dataset

SEED = 404


def prepare_welfake(output_path: str = "welfake_clean.csv") -> pd.DataFrame:
    dataset = load_dataset("davanstrien/WELFake", split="train")
    df = dataset.to_pandas()

    df_clean = df.dropna()
    text_cols = [c for c in df_clean.columns if df_clean[c].dtype == object]
    for col in text_cols:
        df_clean = df_clean[df_clean[col].str.strip() != ""]
    fake = df_clean[df_clean["label"] == 0]
    real = df_clean[df_clean["label"] == 1]

    if len(fake) < 5000 or len(real) < 5000:
        raise ValueError(
            f"Not enough clean samples — fake: {len(fake)}, real: {len(real)}"
        )

    fake_sample = fake.sample(n=5000, random_state=SEED)
    real_sample = real.sample(n=5000, random_state=SEED)

    result = pd.concat([fake_sample, real_sample]).sample(frac=1, random_state=SEED).reset_index(drop=True)

    print(f"\nFinal dataset: {len(result)} rows")
    print(f"Label distribution:\n{result['label'].value_counts()}")
    print(f"Null values:\n{result.isnull().sum()}")

    result.to_csv(output_path, index=False)
    print(f"\nSaved to {output_path}")
    return result

#Raw data: 72,134 rows → after dropping nulls: 71,537
if __name__ == "__main__":
    prepare_welfake()
