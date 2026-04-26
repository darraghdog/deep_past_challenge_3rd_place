---
title: Training Stack
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [training]
sources: [README.md, code/train_baseline.py, code/train_reward.py, conf/baseline, conf/reward_model, brain/raw/transcripts/codebase-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: conf/baseline/conf_baseline_pretrain_large.yaml
contradictions: []
---

# Training Stack

The training stack has two main model paths: ByT5 sequence-to-sequence translation and a Qwen-based reward model. It consumes the datasets prepared by [[extraction-and-preparation-pipeline]] and relies on [[normalization]] choices already baked into data preparation.

## ByT5 Baseline

`code/train_baseline.py` uses Hydra, Accelerate, Transformers, Kaggle Hub datasets, custom dataset/collator code, and utility modules for metrics, generation, monitoring, scheduling, EMA, and adversarial weight perturbation.

Relevant modules:

- `code/baseline/dpc_dataset.py`: tokenization, prompt formatting, optional onomasticon name-swap augmentation.
- `code/baseline/dpc_loader.py`: collation and batch display.
- `code/baseline/dpc_model.py`: model construction.
- `code/baseline/dpc_optim.py`: optimizer parameter grouping.
- `code/utils/generation_utils.py`: prediction generation, MBR selection, generation config.
- `code/utils/metric_utils.py`: competition metric computation.
- `code/utils/monitoring.py`: checkpoint and training monitoring.
- `code/utils/train_utils.py`: Accelerate setup, EMA, AWP, scheduler helpers.

Hydra configs define continued pretraining and fine-tuning variants for ByT5-Large and ByT5-XL under `conf/baseline/`.

## Reward Model

`code/train_reward.py` trains a three-class preference model using `code/reward_model/` modules. `conf/reward_model/conf_reward.yaml` uses a Qwen3-8B backbone with LoRA enabled by default in the observed config.

The reward model path is separate from the ByT5 path, but both share the reproducibility concern that hard numeric config details should be verified from `conf/` before citation. See [[reproducibility-caveats]].

## Data Access

Training configs use Kaggle Hub dataset identifiers. Large training data is external to git; a clean checkout needs the Kaggle datasets described in `README.md`.
