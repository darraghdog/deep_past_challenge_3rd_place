import math

import sacrebleu
import torch
from tqdm.auto import tqdm
from transformers import GenerationConfig


@torch.no_grad()
def generate_predictions(model, tokenizer, eval_dataloader, accelerator, generation_config):
    model.eval()
    accelerator.print("Generating predictions...")
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)

    all_predictions = []

    progress_bar = tqdm(eval_dataloader, desc="Evaluating", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        # Generate predictions
        generated_ids = unwrapped_model.generate(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], generation_config=generation_config)

        # Pad and gather across processes
        generated_ids = accelerator.pad_across_processes(generated_ids, dim=1, pad_index=tokenizer.pad_token_id or 0)
        generated_ids = accelerator.gather_for_metrics(generated_ids)

        # Decode on main process
        predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
        all_predictions.extend(predictions)

    accelerator.wait_for_everyone()

    return all_predictions


# ── MBR Decoding ──────────────────────────────────────────────────────


def comp_metric_utility(hypothesis, reference):
    """Sentence-level sqrt(BLEU * chrF++) — same as competition metric."""
    bleu = sacrebleu.sentence_bleu(hypothesis, [reference]).score
    chrf = sacrebleu.sentence_chrf(hypothesis, [reference], word_order=2).score
    return math.sqrt(max(bleu, 0.0) * max(chrf, 0.0))


def mbr_select(candidates, utility_fn=None):
    """Return index of candidate with highest average pairwise utility."""
    if utility_fn is None:
        utility_fn = comp_metric_utility
    n = len(candidates)
    if n == 1:
        return 0
    scores = []
    for i in range(n):
        total = sum(utility_fn(candidates[i], candidates[j]) for j in range(n) if j != i)
        scores.append(total / (n - 1))
    return max(range(n), key=lambda i: scores[i])


@torch.no_grad()
def generate_mbr_predictions(model, tokenizer, eval_dataloader, accelerator, generation_config, mbr_n=16):
    """Generate predictions using MBR decoding.

    For each input, generates mbr_n candidates via sampling, then selects
    the candidate with highest average pairwise sqrt(BLEU * chrF++) against
    all others. MBR selection happens locally on each GPU before gather.
    """
    model.eval()
    accelerator.print(f"MBR decoding: N={mbr_n} candidates per input")
    accelerator.wait_for_everyone()

    unwrapped_model = accelerator.unwrap_model(model)
    pad_id = tokenizer.pad_token_id or 0

    all_predictions = []

    progress_bar = tqdm(eval_dataloader, desc="MBR Decoding", disable=not accelerator.is_local_main_process)

    for batch in progress_bar:
        local_batch_size = batch["input_ids"].shape[0]

        # Generate N candidates per input: (local_batch * mbr_n, seq_len)
        generated_ids = unwrapped_model.generate(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            generation_config=generation_config,
        )

        # Decode all candidates locally on this GPU
        all_candidates = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

        # MBR select best candidate per input
        best_token_ids = []
        for i in range(local_batch_size):
            group = all_candidates[i * mbr_n : (i + 1) * mbr_n]
            best_idx = mbr_select(group)
            best_token_ids.append(generated_ids[i * mbr_n + best_idx])

        # Stack selected: (local_batch, seq_len) — back to 1 per input
        selected = torch.stack(best_token_ids, dim=0)

        # Standard gather (identical to generate_predictions)
        selected = accelerator.pad_across_processes(selected, dim=1, pad_index=pad_id)
        selected = accelerator.gather_for_metrics(selected)

        predictions = tokenizer.batch_decode(selected, skip_special_tokens=True)
        all_predictions.extend(predictions)

    accelerator.wait_for_everyone()

    return all_predictions


def get_generation_config(cfg, tokenizer):
    return GenerationConfig(
        max_new_tokens=cfg.generation.max_new_tokens,
        do_sample=cfg.generation.do_sample,
        top_k=cfg.generation.top_k,
        top_p=cfg.generation.top_p,
        temperature=cfg.generation.temperature,
    )
