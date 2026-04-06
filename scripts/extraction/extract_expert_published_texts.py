"""
Extract published_texts that have expert translations.
Combines train.csv and Sentences_Oare expert pairs.
"""

import os
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_DIR = SCRIPT_DIR.parent.parent  # scripts/extraction/ → repo root
DATA_DIR = Path(os.environ.get("DATA_DIR", str(REPO_DIR / "data")))


def main():
    # Load data
    pub = pd.read_csv(f"{DATA_DIR}/published_texts.csv")
    train = pd.read_csv(f"{DATA_DIR}/train.csv")
    sentences = pd.read_csv(f"{DATA_DIR}/sentences_oare_expert_pairs.csv")

    # Get train expert pairs (use train's translation)
    train_expert = train[['oare_id', 'transliteration', 'translation']].copy()
    train_expert['source'] = 'train'

    # Get Sentences_Oare pairs NOT in train
    train_ids = set(train['oare_id'].dropna())
    sentences_new = sentences[~sentences['oare_id'].isin(train_ids)].copy()
    sentences_new = sentences_new.drop_duplicates(subset=['oare_id'], keep='first')
    sentences_new['source'] = 'sentences_oare'

    # Combine
    output = pd.concat([train_expert, sentences_new], ignore_index=True)

    # Save
    output.to_csv(f"{DATA_DIR}/published_texts_expert.csv", index=False)

    print(f"Total expert pairs: {len(output):,}")
    print(f"  - From train.csv: {len(train_expert):,}")
    print(f"  - From Sentences_Oare (new): {len(sentences_new):,}")
    print(f"Saved to: {DATA_DIR}/published_texts_expert.csv")


if __name__ == "__main__":
    main()
