"""Train a pairwise reward model for Akkadian translation quality judgment.

Uses Qwen3-8B + LoRA, fine-tuned on curated pairwise preference data.
The model picks A, B, or EQUAL given (transliteration, translation_a, translation_b).

Usage:
    accelerate launch code/train_reward.py
    accelerate launch --num_processes 8 code/train_reward.py
"""

import json
import os
import time

import hydra
import kagglehub
import pandas as pd
import torch
from accelerate.logging import get_logger
from omegaconf import OmegaConf
from reward_model.dpc_reward_dataset import DPCRewardDataset, get_class_weights, prepare_reward_data
from reward_model.dpc_reward_loader import DPCRewardCollator, show_batch
from reward_model.dpc_reward_model import DPCRewardModel, get_base_model
from reward_model.dpc_reward_optim import get_optimizer
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm.auto import tqdm
from utils.train_utils import EMA, get_custom_cosine_schedule_with_warmup, get_lr, setup_training_run

logger = get_logger(__name__)
torch._dynamo.config.optimize_ddp = False


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate(model, valid_dl, accelerator):
    """Run validation and return accuracy, per-class metrics, and B precision/recall at thresholds."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in valid_dl:
            outputs = model(**batch)
            logits = outputs.logits  # [bs, 3]
            probs = torch.softmax(logits, dim=-1)
            preds = torch.argmax(logits, dim=-1)
            labels = torch.argmax(batch["labels"], dim=-1)

            gathered_preds = accelerator.gather_for_metrics(preds)
            gathered_labels = accelerator.gather_for_metrics(labels)
            gathered_probs = accelerator.gather_for_metrics(probs)

            all_preds.extend(gathered_preds.cpu().tolist())
            all_labels.extend(gathered_labels.cpu().tolist())
            all_probs.extend(gathered_probs.cpu().tolist())

    all_preds = torch.tensor(all_preds)
    all_labels = torch.tensor(all_labels)
    all_probs = torch.tensor(all_probs)

    acc = (all_preds == all_labels).float().mean().item()

    label_names = ["A", "B", "EQUAL"]
    per_class = {}
    for i, name in enumerate(label_names):
        mask = all_labels == i
        if mask.sum() > 0:
            per_class[name] = (all_preds[mask] == i).float().mean().item()
        else:
            per_class[name] = 0.0

    # B precision/recall at various probability thresholds
    b_idx = 1  # B is index 1 in [A, B, EQUAL]
    b_probs = all_probs[:, b_idx]
    b_true = all_labels == b_idx

    b_pr = {}
    for th in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        predicted_b = b_probs >= th
        tp = (predicted_b & b_true).sum().item()
        fp = (predicted_b & ~b_true).sum().item()
        fn = (~predicted_b & b_true).sum().item()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        n_picked = predicted_b.sum().item()
        b_pr[th] = {"precision": precision, "recall": recall, "n": n_picked, "tp": tp}

    return acc, per_class, b_pr


@hydra.main(version_base=None, config_path="../conf/reward_model", config_name="conf_reward")
def run_training(cfg):
    # ------- Accelerator ---------------------------------------------------------------#
    accelerator = setup_training_run(cfg)
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    cfg.local_rank = accelerator.process_index

    def print_line():
        prefix, unit, suffix = "#", "~~", "#"
        accelerator.print(prefix + unit * 50 + suffix)

    print_line()
    accelerator.print(json.dumps(cfg_dict, indent=4))

    # ------- load data -----------------------------------------------------------------#
    print_line()

    with accelerator.main_process_first():
        input_dir = kagglehub.dataset_download(cfg.dataset.input_dataset)

    input_path = os.path.join(input_dir, cfg.dataset.input_file)
    raw_df = pd.read_parquet(input_path)
    accelerator.print(f"Raw data: {len(raw_df):,} rows")
    accelerator.print(f"Label distribution:\n{raw_df['pick'].value_counts().to_string()}")

    # Train/valid split
    import numpy as np

    full_fit = cfg.dataset.get("full_fit", False)

    if full_fit:
        accelerator.print("FULL FIT MODE: training on all data, no validation")
        train_df = raw_df.copy()
        valid_df = raw_df.sample(frac=0.02, random_state=cfg.seed).reset_index(drop=True)
    else:
        valid_frac = cfg.dataset.valid_frac
        unique_tls = raw_df["transliteration"].unique()
        rng = np.random.default_rng(cfg.seed)
        rng.shuffle(unique_tls)
        n_valid_tls = max(1, int(len(unique_tls) * valid_frac))
        valid_tls = set(unique_tls[:n_valid_tls])
        valid_df = raw_df[raw_df["transliteration"].isin(valid_tls)].reset_index(drop=True)
        train_df = raw_df[~raw_df["transliteration"].isin(valid_tls)].reset_index(drop=True)

    accelerator.print(f"Train (raw): {len(train_df):,} | Valid (raw): {len(valid_df):,}")

    # Prepare training data: swap augment (no oversampling — use weighted sampler)
    train_df = prepare_reward_data(train_df, seed=cfg.seed)
    accelerator.print(f"Train (after swap augment): {len(train_df):,}")
    accelerator.print(f"Train label distribution:\n{train_df['pick'].value_counts().to_string()}")

    # Compute class-balanced sampling weights
    sample_weights = get_class_weights(train_df)
    class_counts = train_df["pick"].value_counts()
    max_count = class_counts.max()
    accelerator.print(f"Sampling weights: { {l: f'{max_count / c:.2f}' for l, c in class_counts.items()} }")

    # Valid stays as-is (no augmentation)
    accelerator.print(f"Valid label distribution:\n{valid_df['pick'].value_counts().to_string()}")
    print_line()

    # Dataset
    train_ds = DPCRewardDataset(cfg, train_df)
    valid_ds = DPCRewardDataset(cfg, valid_df)

    tokenizer = train_ds.tokenizer

    data_collator = DPCRewardCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),
        replacement=True,
    )

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        sampler=train_sampler,
        collate_fn=data_collator,
    )

    valid_dl = DataLoader(
        valid_ds,
        batch_size=cfg.train_params.per_device_eval_batch_size,
        shuffle=False,
        collate_fn=data_collator,
    )

    accelerator.print("data preparation done...")
    print_line()

    # --- show batch -------------------------------------------------------------------#
    print_line()
    for idx, b in enumerate(train_dl):
        accelerator.print(f"TRAINING BATCH {idx}:")
        show_batch(b, tokenizer, print_fn=accelerator.print, n=2)
        if idx > 0:
            break

    # --- model -------------------------------------------------------------------------#
    print_line()
    accelerator.print("Loading model....")
    with accelerator.main_process_first():
        base_model = get_base_model(cfg)
        model = DPCRewardModel(cfg, base_model, tokenizer)

    if cfg.model.use_gradient_checkpointing:
        accelerator.print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    accelerator.wait_for_everyone()

    if cfg.model.compile_model:
        accelerator.print("Compiling model...")
        model = torch.compile(model)

    # --- optimizer ---------------------------------------------------------------------#
    print_line()
    optimizer = get_optimizer(cfg, model, print_fn=accelerator.print)

    # ------- Prepare -------------------------------------------------------------------#
    time_start = time.time()
    model, optimizer, train_dl, valid_dl = accelerator.prepare(model, optimizer, train_dl, valid_dl)
    time_end = time.time()
    accelerator.print(f"Time taken for HF accelerator prepare: {time_end - time_start:.1f} seconds")

    # ------- EMA -----------------------------------------------------------------------#
    ema = None
    if cfg.train_params.use_ema:
        ema = EMA(model, decay=cfg.train_params.ema_decay)
        ema.register()
        accelerator.print(f"EMA enabled with decay {cfg.train_params.ema_decay}")

    # ------- Scheduler -----------------------------------------------------------------#
    print_line()
    num_epochs = cfg.train_params.num_train_epochs
    grad_accumulation_steps = cfg.train_params.gradient_accumulation_steps
    warmup_pct = cfg.train_params.warmup_pct

    num_update_steps_per_epoch = len(train_dl) // grad_accumulation_steps
    num_training_steps = num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(warmup_pct * num_training_steps)

    accelerator.print(f"# training updates per epoch: {num_update_steps_per_epoch}")
    accelerator.print(f"# training steps: {num_training_steps}")
    accelerator.print(f"# warmup steps: {num_warmup_steps}")

    scheduler = get_custom_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        decay_target=cfg.train_params.decay_target,
    )

    # ------- training  -----------------------------------------------------------------#
    accelerator.wait_for_everyone()
    current_iteration = 0
    best_acc = 0.0
    progress_bar = None

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)
        loss_meter = AverageMeter()

        model.train()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if ema is not None:
                        ema.update()

                loss_meter.update(loss.item())

            if accelerator.sync_gradients:
                progress_bar.set_description(f"STEP: {current_iteration + 1:5}/{num_training_steps:5}. LR: {get_lr(optimizer):.4f}. Loss: {loss_meter.avg:.4f}.")
                progress_bar.update(1)
                current_iteration += 1

                if cfg.use_wandb:
                    accelerator.log(
                        {"train_loss": round(loss_meter.avg, 5), "lr": get_lr(optimizer)},
                        step=current_iteration,
                    )

                # --- Evaluation ---
                eval_frequency = cfg.train_params.eval_frequency
                if current_iteration % eval_frequency == 0:
                    accelerator.print(f"\n--- Evaluation at step {current_iteration} ---")

                    if ema is not None:
                        ema.apply_shadow()

                    acc, per_class, b_pr = evaluate(model, valid_dl, accelerator)

                    print_line()
                    accelerator.print(f"Acc={acc:.4f} | A={per_class['A']:.3f} B={per_class['B']:.3f} EQUAL={per_class['EQUAL']:.3f}")
                    accelerator.print("  B precision/recall at thresholds:")
                    accelerator.print(f"  {'Th':>4s}  {'Prec':>6s}  {'Rec':>6s}  {'N':>5s}  {'TP':>4s}")
                    for th, m in b_pr.items():
                        accelerator.print(f"  {th:4.1f}  {m['precision']:6.3f}  {m['recall']:6.3f}  {m['n']:5d}  {m['tp']:4d}")
                    print_line()

                    if cfg.use_wandb:
                        log_dict = {
                            "val/acc": acc,
                            "val/acc_A": per_class["A"],
                            "val/acc_B": per_class["B"],
                            "val/acc_EQUAL": per_class["EQUAL"],
                        }
                        for th, m in b_pr.items():
                            log_dict[f"val/B_prec_{th}"] = m["precision"]
                            log_dict[f"val/B_rec_{th}"] = m["recall"]
                        accelerator.log(log_dict, step=current_iteration)

                    # Checkpoint if best
                    if acc > best_acc and cfg.save_model and accelerator.is_main_process:
                        best_acc = acc
                        ckpt_dir = os.path.join(cfg.outputs.model_dir, "best")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        accelerator.print(f"Best model saved (acc={acc:.4f}) -> {ckpt_dir}")

                    if ema is not None:
                        ema.restore()

                    model.train()

    # --- end training ---
    accelerator.wait_for_everyone()

    if ema is not None:
        accelerator.print("Applying EMA shadow weights for final model...")
        ema.apply_shadow()

    if cfg.save_model:
        model.eval()

        if accelerator.is_main_process:
            final_dir = os.path.join(cfg.outputs.model_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save(final_dir)
            tokenizer.save_pretrained(final_dir)
            accelerator.print(f"Final model saved to {final_dir}")

            # Save merged model
            if cfg.model.use_lora and cfg.get("save_merged_model", False):
                accelerator.print("Saving merged model...")
                merged_model = unwrapped_model.model.merge_and_unload(safe_merge=True)
                merged_dir = os.path.join(cfg.outputs.model_dir, "merged")
                os.makedirs(merged_dir, exist_ok=True)
                merged_model.save_pretrained(merged_dir)
                tokenizer.save_pretrained(merged_dir)
                accelerator.print(f"Merged model saved to {merged_dir}")

    accelerator.print(f"\nTraining complete. Best validation accuracy: {best_acc:.4f}")
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
