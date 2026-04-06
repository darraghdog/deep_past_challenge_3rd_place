"""Training monitor for logging diagnostics to CSV and wandb."""

import csv
import math
import os
import time

import torch


# Encoder blocks to hook: first, 1/3, 2/3, last of 36
ENCODER_HOOK_LAYERS = [0, 11, 23, 35]
# Decoder blocks to hook: first, mid, last of 12
DECODER_HOOK_LAYERS = [0, 5, 11]


class TrainingMonitor:
    """Lightweight training diagnostics. Logs to CSV always, wandb when enabled."""

    def __init__(self, accelerator, model, model_dir, max_grad_norm, optimizer, grad_accumulation_steps=1, periodic_interval=100):
        self.accelerator = accelerator
        self.model = model
        self.unwrapped_model = accelerator.unwrap_model(model)
        self.max_grad_norm = max_grad_norm
        self.optimizer = optimizer
        self.grad_accumulation_steps = grad_accumulation_steps
        self.periodic_interval = periodic_interval
        self.is_main = accelerator.is_main_process

        # Loss accumulation
        self._loss_sum = 0.0
        self._loss_count = 0

        # Grad clip fraction (EMA)
        self._clip_frac_ema = 0.0

        # Parameter grouping (built once from model structure)
        self._param_groups = self._build_param_groups()
        self._group_names = sorted(set(self._param_groups.values()))

        # Log group counts for verification
        group_counts = {g: sum(1 for v in self._param_groups.values() if v == g) for g in self._group_names}
        accelerator.print(f"Monitor param groups: {group_counts}")

        # Per-group grad norms (populated by capture_gradients)
        self._grad_norms = {}

        # Activation hooks — store norm and max as CUDA tensors, .item() only at log time
        self._activation_stats = {}  # name -> (norm_tensor, max_tensor)
        self._hook_handles = []
        self._register_activation_hooks()

        # Throughput
        self._step_start = time.time()

        # CSV logging (main process only)
        self._csv_file = None
        self._csv_writer = None
        self._periodic_csv_file = None
        self._periodic_csv_writer = None

        # Build field list dynamically from hook points
        self._activation_names = sorted(self._activation_stats.keys()) if self._activation_stats else self._expected_activation_names()
        act_fields = []
        for name in self._activation_names:
            act_fields.append(f"act_norm/{name}")
            act_fields.append(f"act_max/{name}")

        self._step_fields = [
            "step", "train_loss_avg", "lr", "grad_norm",
            "grad_clip_frac",
            "grad_norm_encoder", "grad_norm_decoder", "grad_norm_embed", "grad_norm_lm_head",
        ] + act_fields + [
            "samples_per_sec",
        ]

        if self.is_main:
            self._metrics_dir = os.path.join(model_dir, "metrics")
            os.makedirs(self._metrics_dir, exist_ok=True)
            self._csv_file = open(os.path.join(self._metrics_dir, "step_metrics.csv"), "w", newline="")
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._step_fields)
            self._csv_writer.writeheader()
            # Periodic CSV — headers written on first periodic() call (fields depend on model structure)

    def _expected_activation_names(self):
        """Return expected activation hook names (used before first forward pass populates _activation_stats)."""
        names = []
        for i in ENCODER_HOOK_LAYERS:
            names.append(f"enc_{i}")
        for i in DECODER_HOOK_LAYERS:
            names.append(f"dec_{i}")
        return sorted(names)

    def _build_param_groups(self):
        """Build param name -> group mapping from actual model structure. Call once at init."""
        mapping = {}
        for name, _ in self.unwrapped_model.named_parameters():
            top = name.split(".")[0]
            if top == "shared" or top == "embed_tokens":
                mapping[name] = "embed"
            elif top == "lm_head":
                mapping[name] = "lm_head"
            elif top == "encoder":
                mapping[name] = "encoder"
            elif top == "decoder":
                mapping[name] = "decoder"
            else:
                mapping[name] = "other"
        return mapping

    def _register_activation_hooks(self):
        """Register forward hooks on sampled encoder/decoder blocks.

        Hooks are on T5Block modules — safe with gradient checkpointing (use_reentrant=False)
        because HF passes block.forward (bound method) to checkpoint, so __call__ (which
        dispatches hooks) only fires once during the original forward, not during recomputation.
        """
        registered = []

        # Encoder blocks
        if hasattr(self.unwrapped_model, "encoder") and hasattr(self.unwrapped_model.encoder, "block"):
            num_enc = len(self.unwrapped_model.encoder.block)
            for i in ENCODER_HOOK_LAYERS:
                if i < num_enc:
                    name = f"enc_{i}"
                    handle = self.unwrapped_model.encoder.block[i].register_forward_hook(self._make_hook(name))
                    self._hook_handles.append(handle)
                    registered.append(name)

        # Decoder blocks
        if hasattr(self.unwrapped_model, "decoder") and hasattr(self.unwrapped_model.decoder, "block"):
            num_dec = len(self.unwrapped_model.decoder.block)
            for i in DECODER_HOOK_LAYERS:
                if i < num_dec:
                    name = f"dec_{i}"
                    handle = self.unwrapped_model.decoder.block[i].register_forward_hook(self._make_hook(name))
                    self._hook_handles.append(handle)
                    registered.append(name)

        self.accelerator.print(f"Monitor activation hooks registered: {registered}")

    def _make_hook(self, name):
        """Create a forward hook that captures activation norm and max. No .item() — stays on GPU."""
        stats = self._activation_stats

        def hook(module, input, output):
            hidden = output[0].detach().float()
            stats[name] = (hidden.norm(), hidden.abs().max())

        return hook

    def accumulate_loss(self, loss_value):
        """Call every microbatch with loss.item()."""
        self._loss_sum += loss_value
        self._loss_count += 1

    @torch.no_grad()
    def capture_gradients(self):
        """Compute per-group gradient norms. Call after backward on final microbatch, BEFORE clip_grad_norm_."""
        group_sq = {g: 0.0 for g in self._group_names}

        for name, p in self.unwrapped_model.named_parameters():
            if p.grad is None:
                continue
            group = self._param_groups[name]
            group_sq[group] += p.grad.data.float().norm().item() ** 2

        self._grad_norms = {k: math.sqrt(v) for k, v in group_sq.items()}

    @torch.no_grad()
    def periodic(self, step_num):
        """Compute expensive diagnostics. Call before clip_grad_norm_, at periodic_interval."""
        if step_num % self.periodic_interval != 0:
            return None

        # Build param -> optimizer LR mapping
        # Accelerate wraps optimizer in AcceleratedOptimizer; .optimizer is the inner torch optimizer
        param_to_lr = {}
        inner_opt = getattr(self.optimizer, "optimizer", self.optimizer)
        for group in inner_opt.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                param_to_lr[id(p)] = lr

        # Update-to-weight ratio and weight norms per group
        group_update_ratios = {g: [] for g in self._group_names}
        group_weight_sq = {g: 0.0 for g in self._group_names}

        for name, p in self.unwrapped_model.named_parameters():
            if p.grad is None:
                continue
            group = self._param_groups[name]
            w = p.data.float()
            g = p.grad.data.float()

            w_std = w.std().item()
            if w_std > 1e-10:
                lr_p = param_to_lr.get(id(p), 0.0)
                ratio = (lr_p * g.std().item()) / w_std
                group_update_ratios[group].append(ratio)

            group_weight_sq[group] += w.norm().item() ** 2

        # Average update-to-weight ratio per group
        update_ratios = {}
        for g in self._group_names:
            vals = group_update_ratios[g]
            update_ratios[g] = sum(vals) / len(vals) if vals else 0.0

        weight_norms = {g: math.sqrt(v) for g, v in group_weight_sq.items()}

        # Per-block gradient norms (all 36 enc + 12 dec)
        block_grad_norms = {}
        if hasattr(self.unwrapped_model, "encoder") and hasattr(self.unwrapped_model.encoder, "block"):
            for i, block in enumerate(self.unwrapped_model.encoder.block):
                sq = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        sq += p.grad.data.float().norm().item() ** 2
                block_grad_norms[f"enc_{i}"] = math.sqrt(sq)

        if hasattr(self.unwrapped_model, "decoder") and hasattr(self.unwrapped_model.decoder, "block"):
            for i, block in enumerate(self.unwrapped_model.decoder.block):
                sq = 0.0
                for p in block.parameters():
                    if p.grad is not None:
                        sq += p.grad.data.float().norm().item() ** 2
                block_grad_norms[f"dec_{i}"] = math.sqrt(sq)

        # Build metrics dict
        metrics = {"step": step_num}
        for g in self._group_names:
            metrics[f"update_ratio/{g}"] = round(update_ratios.get(g, 0.0), 8)
            metrics[f"weight_norm/{g}"] = round(weight_norms.get(g, 0.0), 2)
        for block_name, gnorm in sorted(block_grad_norms.items()):
            metrics[f"grad_flow/{block_name}"] = round(gnorm, 6)

        # Write periodic CSV (create on first call — fields depend on model structure)
        if self.is_main:
            if self._periodic_csv_writer is None and hasattr(self, "_metrics_dir"):
                self._periodic_csv_file = open(os.path.join(self._metrics_dir, "periodic_metrics.csv"), "w", newline="")
                self._periodic_csv_writer = csv.DictWriter(self._periodic_csv_file, fieldnames=list(metrics.keys()))
                self._periodic_csv_writer.writeheader()

            if self._periodic_csv_writer is not None:
                self._periodic_csv_writer.writerow(metrics)
                self._periodic_csv_file.flush()

        # Log to wandb
        wandb_metrics = {k: v for k, v in metrics.items() if k != "step"}
        self.accelerator.log(wandb_metrics, step=step_num)

        return metrics

    def step(self, step_num, grad_norm, lr, batch_size, num_processes):
        """Call once per optimizer step (inside sync_gradients block).

        Returns dict of metrics for this step.
        """
        # Average loss across microbatches
        avg_loss = self._loss_sum / max(self._loss_count, 1)
        self._loss_sum = 0.0
        self._loss_count = 0

        # Grad clip fraction (EMA with alpha=0.01)
        clipped = 1.0 if grad_norm > self.max_grad_norm else 0.0
        self._clip_frac_ema = 0.99 * self._clip_frac_ema + 0.01 * clipped

        # Throughput
        now = time.time()
        elapsed = now - self._step_start
        samples_per_sec = (batch_size * num_processes * self.grad_accumulation_steps) / max(elapsed, 1e-6)
        self._step_start = now

        metrics = {
            "step": step_num,
            "train_loss_avg": round(avg_loss, 5),
            "lr": lr,
            "grad_norm": round(grad_norm, 5),
            "grad_clip_frac": round(self._clip_frac_ema, 4),
            "grad_norm_encoder": round(self._grad_norms.get("encoder", 0.0), 5),
            "grad_norm_decoder": round(self._grad_norms.get("decoder", 0.0), 5),
            "grad_norm_embed": round(self._grad_norms.get("embed", 0.0), 5),
            "grad_norm_lm_head": round(self._grad_norms.get("lm_head", 0.0), 5),
        }

        # Activation stats — .item() here (single GPU sync for all hooks)
        for name in self._activation_names:
            if name in self._activation_stats:
                norm_t, max_t = self._activation_stats[name]
                metrics[f"act_norm/{name}"] = round(norm_t.item(), 2)
                metrics[f"act_max/{name}"] = round(max_t.item(), 2)
            else:
                metrics[f"act_norm/{name}"] = 0.0
                metrics[f"act_max/{name}"] = 0.0

        metrics["samples_per_sec"] = round(samples_per_sec, 1)

        # Write CSV (main process only)
        if self.is_main and self._csv_writer is not None:
            self._csv_writer.writerow(metrics)
            self._csv_file.flush()

        # Log to wandb via accelerator
        wandb_metrics = {k: v for k, v in metrics.items() if k != "step"}
        self.accelerator.log(wandb_metrics, step=step_num)

        return metrics

    def close(self):
        """Remove hooks and close CSV files."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

        if self._periodic_csv_file is not None:
            self._periodic_csv_file.close()
            self._periodic_csv_file = None
