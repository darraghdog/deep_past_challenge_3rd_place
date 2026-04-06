import logging
import os
import random

import kagglehub
import pandas as pd
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from utils.onomasticon import Onomasticon

logger = logging.getLogger(__name__)


def get_tokenizer(cfg):
    return AutoTokenizer.from_pretrained(cfg.model.backbone_path)


class DPCDataset(Dataset):
    """
    Dataset class for Deep Past Challenge - Translate Akkadian to English
    """

    def __init__(self, cfg, df, id_col="oare_id", is_train=False, lookup_df=None, ono_df=None, onomasticon=None):
        self.cfg = cfg
        self.df = df
        self.id_col = id_col

        self.tokenizer = get_tokenizer(cfg)
        self.rng = random.Random(cfg.seed)

        self.is_train = is_train

        # Augmentation config
        self.name_swap_prob = cfg.augmentation.name_swap_prob

        # Onomasticon-based name swap (single-variant only = safe swaps)
        self.ono = None
        self.swap_pool = []  # list of (transliteration_key, translation) for replacement

        ono_dataset = getattr(cfg, "onomasticon_dataset", None)
        if is_train and ono_dataset and self.name_swap_prob > 0:
            ono_dir = kagglehub.dataset_download(ono_dataset)
            ono_path = os.path.join(ono_dir, cfg.onomasticon_file)
            poly_path = os.path.join(ono_dir, cfg.polysemic_file)
            single_path = os.path.join(ono_dir, cfg.single_variant_file)

            self.ono = Onomasticon(ono_path, polysemic_path=poly_path)

            # Build swap pool from single-variant entries only (safe, unambiguous)
            sv_df = pd.read_parquet(single_path)
            for _, row in sv_df.iterrows():
                tl = row["transliteration"]
                variants = row["translation"]
                if isinstance(variants, list) and len(variants) == 1:
                    self.swap_pool.append((tl.lower(), variants[0]["translation"]))
            logger.info(f"Name-swap: {len(self.swap_pool)} single-variant entries in swap pool")

    def _name_swap(self, transliteration, translation):
        """Swap names using single-variant onomasticon entries (safe, unambiguous)."""
        matches = self.ono.find_names(transliteration)
        if not matches:
            return transliteration, translation

        for token, variants, word_meaning in matches:
            if word_meaning:
                continue  # skip polysemic tokens
            original = self.ono.pick(variants, translation)
            if not original:
                continue

            repl_tl, repl_tr = self.rng.choice(self.swap_pool)

            # Replace in transliteration (exact token match)
            tl_tokens = transliteration.split()
            if token not in tl_tokens:
                continue
            new_tl = [repl_tl if t == token else t for t in tl_tokens]

            # Replace in translation (whole-token, first occurrence only)
            tr_tokens = translation.split()
            found = False
            new_tr = []
            for t in tr_tokens:
                stripped = t.rstrip(".,;:!?\"'").rstrip("'s")
                if stripped == original and not found:
                    new_tr.append(t.replace(original, repl_tr))
                    found = True
                else:
                    new_tr.append(t)

            if found:
                transliteration = " ".join(new_tl)
                translation = " ".join(new_tr)

        return transliteration, translation

    def __len__(self):
        return len(self.df)

    def tokenize(self, texts):
        return self.tokenizer(texts, padding=False, truncation=True, max_length=self.cfg.model.max_length, return_length=True, add_special_tokens=True)

    def pre_process(self, row):
        lang_map = {
            "en": "English",
            "translated_en": "English",
            "augmented_en": "English",
            "de": "German",
            "tr": "Turkish",
            "fr": "French",
        }
        lang = lang_map.get(row["language"], "English")

        prompt_template = "Translate Akkadian to {lang}: {source}"

        if not self.is_train:
            return [prompt_template.format(lang=lang, source=row["transliteration"])], [None]

        sources = []
        targets = []

        akkadian, english = row["transliteration"], row["translation"]

        # Name-swap augmentation (train only, onomasticon-based)
        if self.ono and self.swap_pool and self.rng.random() < self.name_swap_prob:
            akkadian, english = self._name_swap(akkadian, english)
        sources.append(prompt_template.format(lang=lang, source=akkadian))
        targets.append(english)

        return sources, targets

    def __getitem__(self, idx):
        data = self.df.iloc[idx].to_dict()
        this_id = data[self.id_col]
        sources, targets = self.pre_process(data)
        assert len(sources) == len(targets), f"len(sources) != len(targets): {len(sources)} != {len(targets)}"
        tx_src = self.tokenize(sources)

        if not self.is_train:
            return dict(id=this_id, input_ids=tx_src["input_ids"], attention_mask=tx_src["attention_mask"])

        tx_tgt = self.tokenize(targets)
        return dict(id=this_id, input_ids=tx_src["input_ids"], attention_mask=tx_src["attention_mask"], labels=tx_tgt["input_ids"])
