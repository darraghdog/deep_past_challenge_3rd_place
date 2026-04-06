"""
Name Lookup Injection for DPC Akkadian→English translation.

Scans transliteration for name spellings, looks them up in the onomasticon,
and returns a hint prefix string prepended to the model input.

Training:  match variant against ground truth, pick the exact match.
Inference: use highest-prob variant.

Polysemic tokens (same bytes = name or word) are loaded from a CSV
and exposed via find_names so the dataset can format ambiguity hints.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Onomasticon:
    def __init__(self, path, polysemic_path=None):
        df = pd.read_parquet(path)

        # lowercased key -> [{prob, translation}, ...]
        # Case collisions are merged: union variants, sum probs
        merged = {}
        for _, row in df.iterrows():
            key = row["transliteration"]
            v = row["translation"]
            if isinstance(v, np.ndarray):
                v = v.tolist()
            if len(key) <= 5 or key.startswith("-"):
                continue
            lk = key.lower()
            if lk not in merged:
                merged[lk] = {d["translation"]: d["prob"] for d in v}
            else:
                for d in v:
                    merged[lk][d["translation"]] = merged[lk].get(d["translation"], 0) + d["prob"]

        self.variants = {lk: [{"translation": t, "prob": p} for t, p in probs.items()] for lk, probs in merged.items()}

        # Polysemic tokens: lowercased token -> word meaning
        self.polysemic = {}
        if polysemic_path and Path(polysemic_path).exists():
            if polysemic_path.endswith(".parquet"):
                pdf = pd.read_parquet(polysemic_path)
            else:
                pdf = pd.read_csv(polysemic_path)
            self.polysemic = dict(zip(pdf["token"].str.lower(), pdf["word_meaning"]))

        logger.info(f"Onomasticon: {len(self.variants)} entries, {len(self.polysemic)} polysemic")

    def find_names(self, transliteration):
        """Match each token against the onomasticon (case-insensitive).

        Returns [(token, variants, word_meaning), ...] in input order.
        word_meaning is None for unambiguous names, a string for polysemic tokens.
        """
        results = []
        for token in transliteration.split():
            lk = token.lower()
            v = self.variants.get(lk)
            if (not v) and token.endswith("-ma") and (len(token) > 6):
                v = self.variants.get(token[:-3].lower())
            if v:
                word_meaning = self.polysemic.get(lk)
                results.append((token, v, word_meaning))
        return results

    def pick(self, variants, translation=None):
        """Pick a variant.

        Training (translation provided): find variant that appears as a whole token in GT.
        Inference (translation=None): return highest-prob variant.
        Returns None if no variant matches during training.
        """
        if translation is None:
            return max(variants, key=lambda v: v["prob"])["translation"]

        gt_tokens = set()
        for token in translation.split():
            gt_tokens.add(token.rstrip(".,;:!?\"'").rstrip("'s"))

        matching = [v for v in variants if v["translation"] in gt_tokens]
        if matching:
            return matching[0]["translation"]
        return None
