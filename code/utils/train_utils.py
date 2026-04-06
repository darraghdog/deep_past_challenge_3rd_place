import logging
import math
import os

import datasets
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from omegaconf import OmegaConf

logger = get_logger(__name__)


def print_line(logger=None):
    prefix, unit, suffix = "#", "~~", "#"
    if logger is None:
        print(prefix + unit * 50 + suffix)
    else:
        logger.print(prefix + unit * 50 + suffix)


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm%ds" % (m, s)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"] * 1e6


class EMA:
    """
    credit: https://www.kaggle.com/competitions/us-patent-phrase-to-phrase-matching/discussion/332567
    """

    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


class AWP:
    def __init__(self, model, adv_lr=1.0, adv_eps=1e-4, param_selector=None):
        self.model = model
        self.adv_lr = adv_lr
        self.adv_eps = adv_eps
        self.param_selector = param_selector or (lambda n, p: True)
        self.backup = {}
        self.lower = {}
        self.upper = {}

    def _selected(self, name, p):
        return p.requires_grad and self.param_selector(name, p)

    def iter_named_params(self):
        for name, p in self.model.named_parameters():
            if self._selected(name, p):
                yield name, p

    @torch.no_grad()
    def save(self):
        self.backup.clear()
        self.lower.clear()
        self.upper.clear()
        for name, p in self.iter_named_params():
            w0 = p.data.clone()
            self.backup[name] = w0
            eps = self.adv_eps * p.abs()
            self.lower[name] = w0 - eps
            self.upper[name] = w0 + eps

    @torch.no_grad()
    def perturb_from_grads(self, grads_by_name):
        e = 1e-6
        for name, p in self.iter_named_params():
            g = grads_by_name.get(name, None)
            if g is None:
                continue
            g_norm = torch.norm(g)
            if g_norm == 0 or torch.isnan(g_norm):
                continue
            w_norm = torch.norm(p.data)
            r_at = self.adv_lr * g / (g_norm + e) * (w_norm + e)
            p.add_(r_at)
            p.data.clamp_(self.lower[name], self.upper[name])

    @torch.no_grad()
    def restore(self):
        for name, p in self.iter_named_params():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup.clear()
        self.lower.clear()
        self.upper.clear()


def enable_cuda_optimizations():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.fp32_precision = "tf32"
    torch.backends.cudnn.conv.fp32_precision = "tf32"


def setup_training_run(cfg):
    """set up training run

    Args:
        cfg: config for the training run
    """

    mixed_precision = getattr(cfg, "mixed_precision", None)

    if cfg.use_wandb:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            log_with="wandb",
        )

        accelerator.init_trackers(
            cfg.wandb.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": {"name": cfg.wandb.run_name}},
        )

    else:
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.train_params.gradient_accumulation_steps,
            mixed_precision=mixed_precision,
        )
    accelerator.print(f"using wandb: {cfg.use_wandb}")
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    accelerator.print(f"setting seed: {cfg.seed}")
    set_seed(cfg.seed)

    if accelerator.is_main_process:
        os.makedirs(cfg.outputs.model_dir, exist_ok=True)

    if cfg.enable_cuda_optimizations:
        enable_cuda_optimizations()
    return accelerator


def get_custom_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    num_cycles=0.5,
    last_epoch=-1,
    decay_target=0.1,
):
    """
    Create a schedule with a learning rate that decreases from the initial lr set in the optimizer to 10% of it,
    following a cosine curve, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer: The optimizer for which to schedule the learning rate.
        num_warmup_steps: The number of steps for the warmup phase.
        num_training_steps: The total number of training steps.
        num_cycles: The number of times the learning rate will decay to 10% of the maximum learning rate. Default: 0.5 (half a cycle).
        last_epoch: The index of the last epoch when resuming training.

    Returns:
        A PyTorch learning rate scheduler.
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))

        # Progress after warmup
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1 + math.cos(math.pi * num_cycles * 2.0 * progress))

        # Scale to decay to 10% of the max lr
        # decay_target = 0.1  # Decay to 10% of the max lr
        decay_factor = (1 - decay_target) * cosine_decay + decay_target

        return decay_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)


def is_nan(x):
    return x != x
