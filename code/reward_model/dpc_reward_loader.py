from dataclasses import dataclass

import torch


@dataclass
class DPCRewardCollator:
    """Collator for reward model: concatenates mini-batch into single long sequence.

    Used with flash attention — no padding within the sequence, only right-side
    padding to reach a multiple of pad_to_multiple_of.
    """

    def __init__(self, tokenizer, pad_to_multiple_of=32):
        self.pad_token_id = tokenizer.pad_token_id
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        ret = {"input_ids": [], "position_ids": [], "labels": [], "end_idxs": []}

        for feature in features:
            for current_ids in feature["input_ids"]:
                ret["input_ids"] += current_ids
                ret["position_ids"] += list(range(len(current_ids)))
                ret["end_idxs"].append(len(ret["input_ids"]) - 1)
            ret["labels"] += feature["labels"]

        # Pad to multiple
        n_tokens = len(ret["input_ids"])
        n_pad = (n_tokens // self.pad_to_multiple_of + 1) * self.pad_to_multiple_of - n_tokens
        last_position_id = ret["position_ids"][-1]

        ret["input_ids"] += [self.pad_token_id] * n_pad
        ret["position_ids"] += list(range(last_position_id + 1, last_position_id + n_pad + 1))

        batch = dict()
        batch["input_ids"] = torch.tensor(ret["input_ids"], dtype=torch.int64).unsqueeze(0)
        batch["position_ids"] = torch.tensor(ret["position_ids"], dtype=torch.int64).unsqueeze(0)
        batch["end_idxs"] = torch.tensor(ret["end_idxs"], dtype=torch.int64)
        batch["labels"] = torch.tensor(ret["labels"], dtype=torch.float32)

        return batch


def show_batch(batch, tokenizer, print_fn=print, n=3, **kwargs):
    bs = batch["end_idxs"].size(0)
    print_fn(f"batch size (num examples): {bs}")
    print_fn(f"shape of input_ids: {batch['input_ids'].shape}")
    print_fn(f"shape of position_ids: {batch['position_ids'].shape}")
    print_fn(f"shape of end_idxs: {batch['end_idxs'].shape}")

    if "labels" in batch:
        print_fn(f"shape of labels: {batch['labels'].shape}")
        print_fn(f"labels:\n{batch['labels']}")

    # Decode individual examples by splitting at end_idxs
    end_idxs = batch["end_idxs"].tolist()
    input_ids = batch["input_ids"][0].tolist()

    prev = 0
    for idx, end in enumerate(end_idxs[:n]):
        print_fn(f"\n=== Example {idx} ===")
        example_ids = input_ids[prev : end + 1]
        print_fn(f"Tokens: {len(example_ids)}")
        text = tokenizer.decode(example_ids, skip_special_tokens=False)
        print_fn(f"Input:\n{text}")
        if "labels" in batch:
            print_fn(f"Label: {batch['labels'][idx].tolist()}")
        print_fn("~~" * 40)
        prev = end + 1
