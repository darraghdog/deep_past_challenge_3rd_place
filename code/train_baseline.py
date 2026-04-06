import json
import os
import time

import hydra
import kagglehub
import pandas as pd
import torch
from accelerate.logging import get_logger
from baseline.dpc_dataset import DPCDataset
from baseline.dpc_loader import DPCCollator, show_batch
from baseline.dpc_model import get_dpc_model
from baseline.dpc_optim import get_optimizer
from omegaconf import OmegaConf
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.generation_utils import generate_predictions, get_generation_config
from utils.metric_utils import compute_score
from utils.monitoring import TrainingMonitor
from utils.train_utils import AWP, EMA, get_custom_cosine_schedule_with_warmup, get_lr, setup_training_run

logger = get_logger(__name__)
torch._dynamo.config.optimize_ddp = False


@hydra.main(version_base=None, config_path="../conf/baseline", config_name="conf_baseline")
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

    train_df = pd.read_parquet(os.path.join(input_dir, "train.parquet"))
    valid_df = pd.read_parquet(os.path.join(input_dir, "valid.parquet"))

    lookup_df = None  # pd.read_parquet(os.path.join(input_dir, "lookup.parquet"))
    ono_df = None  # pd.read_parquet(os.path.join(input_dir, "ono.parquet"))

    train_df = train_df.reset_index(drop=True)
    valid_df = valid_df.reset_index(drop=True)
    valid_df = valid_df.drop_duplicates(subset=["oare_id", "sentence_id"]).reset_index(drop=True)
    solution_df = valid_df[["oare_id", "sentence_id", "translation"]].copy()

    accelerator.print(f"shape of train data: {train_df.shape}")
    accelerator.print(f"shape of validation data: {valid_df.shape}")
    print_line()

    # dataset ----
    train_ds = DPCDataset(cfg, train_df, is_train=True, lookup_df=lookup_df, ono_df=ono_df)
    valid_ds = DPCDataset(cfg, valid_df, is_train=False)

    tokenizer = train_ds.tokenizer

    data_collator = DPCCollator(tokenizer=tokenizer, pad_to_multiple_of=16)

    train_dl = DataLoader(
        train_ds,
        batch_size=cfg.train_params.per_device_train_batch_size,
        shuffle=True,
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
        show_batch(b, tokenizer, task="training", print_fn=accelerator.print, n=5)
        break

    # --- model -------------------------------------------------------------------------#
    print_line()
    accelerator.print("Loading model....")
    with accelerator.main_process_first():
        model = get_dpc_model(cfg)

    if cfg.model.use_gradient_checkpointing:
        accelerator.print("enabling gradient checkpointing")
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    accelerator.wait_for_everyone()

    if cfg.model.compile_model:
        accelerator.print("Compiling model...")
        model = torch.compile(model)

    # model.to(accelerator.device)

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

    # ------- Label Smoothing -----------------------------------------------------------#
    label_smoothing = cfg.train_params.label_smoothing
    loss_fct = CrossEntropyLoss(ignore_index=-100, label_smoothing=label_smoothing) if label_smoothing > 0 else None
    if loss_fct is not None:
        accelerator.print(f"Label smoothing enabled with factor {label_smoothing}")

    # ------- AWP -----------------------------------------------------------------------#
    awp = None
    awp_named_params = None
    if cfg.train_params.use_awp:
        awp_model = accelerator.unwrap_model(model)
        awp = AWP(awp_model, adv_lr=cfg.train_params.awp_adv_lr, adv_eps=cfg.train_params.awp_adv_eps)
        awp_named_params = [(n, p) for n, p in awp_model.named_parameters() if p.requires_grad]
        accelerator.print(f"AWP enabled: adv_lr={cfg.train_params.awp_adv_lr}, adv_eps={cfg.train_params.awp_adv_eps}, trigger_epoch={cfg.train_params.awp_trigger_epoch}, params={len(awp_named_params)}")

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
        decay_target=cfg.train_params.lr_decay_target,
    )

    # ------- Generation Config ------------------------------------------------------------#
    generation_config = get_generation_config(cfg, tokenizer)
    accelerator.print(f"Generation: max_new_tokens={generation_config.max_new_tokens}, top_k={generation_config.top_k}")

    # ------- Training Monitor -------------------------------------------------------------#
    monitor = TrainingMonitor(
        accelerator=accelerator,
        model=model,
        model_dir=cfg.outputs.model_dir,
        max_grad_norm=cfg.optimizer.max_grad_norm,
        optimizer=optimizer,
        grad_accumulation_steps=grad_accumulation_steps,
    )

    # ------- training setup ---------------------------------------------------------------#
    current_iteration = 0

    # ------- training  --------------------------------------------------------------------#
    accelerator.wait_for_everyone()
    progress_bar = None

    for epoch in range(num_epochs):
        if epoch != 0:
            progress_bar.close()

        progress_bar = tqdm(range(num_update_steps_per_epoch), disable=not accelerator.is_local_main_process)

        # Training ------
        model.train()

        for step, batch in enumerate(train_dl):
            with accelerator.accumulate(model):
                if awp is not None and epoch >= cfg.train_params.awp_trigger_epoch:
                    # clean forward → grads for perturbation direction
                    outputs_clean = model(**batch)
                    if loss_fct is not None:
                        loss_clean = loss_fct(outputs_clean.logits.view(-1, outputs_clean.logits.size(-1)), batch["labels"].view(-1))
                    else:
                        loss_clean = outputs_clean.loss
                    grads = torch.autograd.grad(loss_clean, [p for _, p in awp_named_params], create_graph=False)
                    grads_by_name = {n: g for (n, _), g in zip(awp_named_params, grads) if g is not None}

                    # perturb → adversarial forward → backward → restore
                    awp.save()
                    awp.perturb_from_grads(grads_by_name)
                    outputs_adv = model(**batch)
                    if loss_fct is not None:
                        loss = loss_fct(outputs_adv.logits.view(-1, outputs_adv.logits.size(-1)), batch["labels"].view(-1))
                    else:
                        loss = outputs_adv.loss
                    accelerator.backward(loss)
                    awp.restore()
                else:
                    outputs = model(**batch)
                    if loss_fct is not None:
                        loss = loss_fct(outputs.logits.view(-1, outputs.logits.size(-1)), batch["labels"].view(-1))
                    else:
                        loss = outputs.loss
                    accelerator.backward(loss)

                monitor.accumulate_loss(loss.detach().item())

                if accelerator.sync_gradients:
                    monitor.capture_gradients()
                    monitor.periodic(current_iteration + 1)
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), cfg.optimizer.max_grad_norm)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    if ema is not None:
                        ema.update()

            if accelerator.sync_gradients:
                current_iteration += 1

                step_metrics = monitor.step(
                    step_num=current_iteration,
                    grad_norm=grad_norm.item(),
                    lr=get_lr(optimizer),
                    batch_size=cfg.train_params.per_device_train_batch_size,
                    num_processes=accelerator.num_processes,
                )

                progress_bar.set_description(
                    f"STEP: {current_iteration:5}/{num_training_steps:5}. LR: {step_metrics['lr']:.4f}. Loss: {step_metrics['train_loss_avg']:.4f}. |Grad|: {step_metrics['grad_norm']:.4f}. samp/s: {step_metrics['samples_per_sec']:.0f}"
                )
                progress_bar.update(1)

                # --- Evaluation ---
                eval_frequency = cfg.train_params.eval_frequency
                if current_iteration % eval_frequency == 0:
                    accelerator.print(f"\n--- Evaluation at step {current_iteration} ---")

                    if ema is not None:
                        ema.apply_shadow()

                    predictions = generate_predictions(
                        model=model,
                        tokenizer=tokenizer,
                        eval_dataloader=valid_dl,
                        accelerator=accelerator,
                        generation_config=generation_config,
                    )
                    predictions_df = pd.DataFrame()  # {"oare_id": valid_df["oare_id"], "translation": predictions})
                    predictions_df["oare_id"] = valid_df["oare_id"].values.tolist()
                    predictions_df["sentence_id"] = valid_df["sentence_id"].values.tolist()
                    predictions_df["translation"] = predictions

                    out_df = pd.merge(valid_df, predictions_df, on=["oare_id", "sentence_id"], how="left", suffixes=("_true", "_pred"))
                    out_dir = os.path.join(cfg.outputs.model_dir, "predictions")
                    os.makedirs(out_dir, exist_ok=True)
                    out_df.to_csv(os.path.join(out_dir, f"predictions_{current_iteration}.csv"), index=False)

                    eval_metrics = compute_score(solution_df, predictions_df, "oare_id", "translation")

                    print_line()
                    accelerator.print(f"BLEU={eval_metrics['bleu']:.2f} | chrF++={eval_metrics['chrf']:.2f} | Score={eval_metrics['score']:.2f}")
                    print_line()

                    if cfg.use_wandb:
                        accelerator.log(
                            {
                                "val/bleu": eval_metrics["bleu"],
                                "val/chrf": eval_metrics["chrf"],
                                "val/score": eval_metrics["score"],
                            },
                            step=current_iteration,
                        )

                    # --- Checkpoint (EMA weights are active if enabled) ---
                    if cfg.save_model and accelerator.is_main_process:
                        ckpt_dir = os.path.join(cfg.outputs.model_dir, "checkpoints", f"step_{current_iteration}")
                        os.makedirs(ckpt_dir, exist_ok=True)
                        unwrapped_model = accelerator.unwrap_model(model)
                        unwrapped_model.save_pretrained(ckpt_dir)
                        tokenizer.save_pretrained(ckpt_dir)
                        accelerator.print(f"Checkpoint saved to {ckpt_dir}")

                    if ema is not None:
                        ema.restore()

                    model.train()  # Switch back to training mode

    # --- end training
    accelerator.wait_for_everyone()

    if ema is not None:
        accelerator.print("Applying EMA shadow weights for final model...")
        ema.apply_shadow()

    # evaluate final model
    predictions = generate_predictions(
        model=model,
        tokenizer=tokenizer,
        eval_dataloader=valid_dl,
        accelerator=accelerator,
        generation_config=generation_config,
    )
    predictions_df = pd.DataFrame()
    predictions_df["oare_id"] = valid_df["oare_id"].values.tolist()
    predictions_df["sentence_id"] = valid_df["sentence_id"].values.tolist()
    predictions_df["translation"] = predictions
    out_df = pd.merge(valid_df, predictions_df, on=["oare_id", "sentence_id"], how="left", suffixes=("_true", "_pred"))
    out_dir = os.path.join(cfg.outputs.model_dir, "predictions")
    os.makedirs(out_dir, exist_ok=True)
    out_df.to_csv(os.path.join(out_dir, "predictions_final.csv"), index=False)
    eval_metrics = compute_score(solution_df, predictions_df, "oare_id", "translation")
    accelerator.print(f"BLEU={eval_metrics['bleu']:.2f} | chrF++={eval_metrics['chrf']:.2f} | Score={eval_metrics['score']:.2f}")

    if cfg.save_model:
        model.eval()

        if accelerator.is_main_process:
            final_dir = os.path.join(cfg.outputs.model_dir, "model")
            os.makedirs(final_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(final_dir)
            tokenizer.save_pretrained(final_dir)
            accelerator.print(f"Final model saved to {final_dir}")

    # --- end training
    monitor.close()
    accelerator.end_training()


if __name__ == "__main__":
    run_training()
