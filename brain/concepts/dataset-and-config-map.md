---
title: Dataset and Config Map
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [data, training, repo]
sources: [README.md, conf/baseline, conf/reward_model, pyproject.toml, brain/raw/transcripts/codebase-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: conf/baseline/conf_baseline_pretrain_large.yaml
contradictions: []
---

# Dataset and Config Map

This page maps the repo's public training configuration surface. It supports [[training-stack]] and clarifies how data moves from extraction to model runs.

## Data Families

The README describes three broad data setup paths:

- competition data from Kaggle;
- pre-extracted training data from a Kaggle dataset;
- source PDFs/reference material from a separate Kaggle dataset for full extraction reruns.

The source collection summarized in [[missing-283-transliterations]] is historical working-repo provenance for some Round 2/4 files, while the reproducibility path should use the staged `DATA_DIR` layout or the Kaggle datasets documented in `README.md`.

## Training Configs

Current config files:

- `conf/baseline/conf_baseline_pretrain_large.yaml`
- `conf/baseline/conf_baseline_pretrain_xl.yaml`
- `conf/baseline/conf_baseline_continue_large.yaml`
- `conf/baseline/conf_baseline_continue_xl.yaml`
- `conf/reward_model/conf_reward.yaml`

The baseline configs cover continued pretraining and fine-tuning for ByT5-Large and ByT5-XL. The reward config covers the preference/reward model path.

## Practical Rule

Before citing a dataset ID, checkpoint path, batch size, learning rate, or epoch count, re-read the current YAML. These are perishable facts. This is especially important because the wiki is public and intended for a demo, while configs may change during cleanup.

Related pages: [[reproducibility-caveats]], [[evaluation-and-decoding]].
