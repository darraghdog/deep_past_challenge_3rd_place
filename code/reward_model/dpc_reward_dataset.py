import random

import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer

SYS_PROMPT = """You are an expert judge comparing two Akkadian-to-English translations of the same source text. The transliteration is the ground truth — a syllable-by-syllable representation of an Old Assyrian cuneiform tablet (circa 1950-1750 BCE).

Your task: determine which translation is more faithful to the transliteration, or whether they are effectively equal.

Key evaluation criteria (in order of importance):
1. SEMANTIC FIDELITY: Who did what to whom, under what conditions, with what consequences. Reversing a debt relationship or dropping a negation is catastrophic.
2. NAME ACCURACY: Personal names must be derivable from the syllables in the transliteration. Minor spelling variants (macrons, gemination) are acceptable.
3. NUMBERS AND QUANTITIES: Weights, counts, and amounts must be preserved exactly. 1 mina = 60 shekels, 1 talent = 60 minas.
4. COMPLETENESS: A translation covering all content is better than one that stops midway. But fabrication is worse than omission.
5. ALIGNMENT: Every element in the transliteration should have a corresponding element in the translation.

Pick A if Translation A is more faithful. Pick B if Translation B is more faithful. Pick EQUAL if both are substantively equivalent."""

USER_TEMPLATE = """Transliteration: {transliteration}

Translation A: {translation_a}

Translation B: {translation_b}

Which translation is more faithful to the transliteration? Respond with only: A, B, or EQUAL."""


def get_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.model.backbone_path,
        use_fast=cfg.model.tokenizer.use_fast,
        add_eos_token=False,
        truncation_side=cfg.model.tokenizer.truncation_side,
    )

    tokenizer.padding_side = "left"

    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
            tokenizer.pad_token_id = tokenizer.unk_token_id
        elif tokenizer.eod_id is not None:
            tokenizer.pad_token = tokenizer.eod
            tokenizer.pad_token_id = tokenizer.eod_id
            tokenizer.bos_token = tokenizer.im_start
            tokenizer.bos_token_id = tokenizer.im_start_id
            tokenizer.eos_token = tokenizer.im_end
            tokenizer.eos_token_id = tokenizer.im_end_id
        else:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

    return tokenizer


def prepare_reward_data(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Swap-augment to eliminate position bias. No oversampling — use WeightedRandomSampler instead.

    Creates swapped copies (A<->B with flipped labels) -> 2x data.
    After swap, A and B counts are perfectly balanced.
    """
    rows_orig = df.to_dict("records")

    rows_swapped = []
    for row in rows_orig:
        new_row = row.copy()
        new_row["translation_a"], new_row["translation_b"] = row["translation_b"], row["translation_a"]
        if row["pick"] == "A":
            new_row["pick"] = "B"
        elif row["pick"] == "B":
            new_row["pick"] = "A"
        rows_swapped.append(new_row)

    all_rows = rows_orig + rows_swapped
    augmented = pd.DataFrame(all_rows).sample(frac=1, random_state=seed).reset_index(drop=True)
    return augmented


def get_class_weights(df: pd.DataFrame):
    """Compute per-sample weights for WeightedRandomSampler to balance A/B/EQUAL."""
    class_counts = df["pick"].value_counts()
    max_count = class_counts.max()
    class_weight = {label: max_count / count for label, count in class_counts.items()}
    sample_weights = df["pick"].map(class_weight).values
    return sample_weights


class DPCRewardDataset(Dataset):
    """Dataset for pairwise translation quality judgment."""

    def __init__(self, cfg, df):
        self.cfg = cfg
        self.tokenizer = get_tokenizer(cfg)
        self.df = df
        self.label_map = {"A": 0, "B": 1, "EQUAL": 2}

    def __len__(self):
        return len(self.df)

    def _tokenize_function(self, texts):
        tx = self.tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=self.cfg.model.max_length,
            return_length=True,
            add_special_tokens=True,
        )
        return tx

    def _build_input(self, row):
        user_message = USER_TEMPLATE.format(
            transliteration=row["transliteration"],
            translation_a=row["translation_a"],
            translation_b=row["translation_b"],
        )
        conversation = [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": user_message},
        ]
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return [text]

    def __getitem__(self, idx):
        data = self.df.iloc[idx].to_dict()
        formatted_texts = self._build_input(data)

        pick = data["pick"]
        label_idx = self.label_map[pick]
        label = [0.0, 0.0, 0.0]
        label[label_idx] = 1.0
        labels = [label] * len(formatted_texts)

        tx = self._tokenize_function(formatted_texts)

        return dict(
            input_ids=tx["input_ids"],
            attention_mask=tx["attention_mask"],
            length=tx["length"],
            labels=labels,
        )
