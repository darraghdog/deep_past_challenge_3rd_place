from dataclasses import dataclass

import torch
from transformers import DataCollatorWithPadding


@dataclass
class DPCCollator(DataCollatorWithPadding):
    """
    Data collector for DPC dataset
    """

    tokenizer = None
    padding = True
    max_length = None
    pad_to_multiple_of = None

    def __call__(self, features):
        encoder_features = {"input_ids": [], "attention_mask": []}
        labels = []

        for feature in features:
            encoder_features["input_ids"].extend(feature["input_ids"])
            encoder_features["attention_mask"].extend(feature["attention_mask"])

            if "labels" in feature:
                labels.extend(feature["labels"])

        if len(labels) > 0:
            assert len(labels) == len(encoder_features["input_ids"])

        batch = self.tokenizer.pad(
            encoder_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # labels
        if len(labels) > 0:
            labels_batch = self.tokenizer.pad({"input_ids": labels}, padding=self.padding, max_length=self.max_length, pad_to_multiple_of=self.pad_to_multiple_of, return_tensors=None)
            labels = torch.tensor(labels_batch["input_ids"], dtype=torch.int64)
            # Mask padding with -100 (ignored in loss)
            labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels
        return batch


def show_batch(batch, tokenizer, print_fn=print, **kwargs):
    bs = batch["input_ids"].size(0)
    print_fn(f"batch size: {bs}")

    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")
    print_fn(f"shape of attention_mask: {batch['attention_mask'].shape}")
    if "labels" in batch.keys():
        print_fn(f"shape of labels: {batch['labels'].shape}")

    print_fn("\n\n")
    max_n = kwargs.get("n", 5)

    for idx in range(bs):
        print_fn(f"\n\n===== Example {idx + 1} =====\n\n")
        print_fn(f"## Input:\n\n{tokenizer.decode(batch['input_ids'][idx], skip_special_tokens=False)}\n\n")

        if "labels" in batch.keys():
            labels = batch["labels"][idx]
            labels[labels == -100] = tokenizer.pad_token_id
            print_fn(f"## Output:\n\n{tokenizer.decode(labels, skip_special_tokens=False)}")

        if idx >= max_n:
            break
