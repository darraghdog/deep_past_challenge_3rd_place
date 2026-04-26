---
title: How Does This Repo Reproduce the Solution?
created: 2026-04-26
updated: 2026-04-26
type: query
tags: [repo, pipeline, training]
sources: [brain/concepts/codebase-overview.md, brain/concepts/extraction-and-preparation-pipeline.md, brain/concepts/dataset-and-config-map.md, brain/concepts/training-stack.md, brain/concepts/reproducibility-caveats.md]
verified: 2026-04-26
verified_source: brain/concepts/codebase-overview.md
contradictions: []
---

# How Does This Repo Reproduce the Solution?

This repo is the public reproduction artifact for the 3rd place Deep Past Initiative Machine Translation solution. It is not the original exploratory working tree; that role belongs to the adjacent historical `../akk/` repo. The clean reproduction path is summarized by [[codebase-overview]] and [[dataset-and-config-map]].

## Short Answer

The repo reproduces the solution by packaging:

- extraction and repair scripts for turning competition CSVs and source documents into aligned sentence pairs;
- prompt templates for source-specific document extraction;
- normalization and deduplication scripts;
- final dataset assembly logic;
- ByT5 training code and Hydra configs;
- reward-model training code;
- synthetic-data generation code for grammar, CAD, and template drills.

The main executable path is [[extraction-and-preparation-pipeline]] followed by [[training-stack]].

## Practical Walkthrough

1. Start with `README.md` for data downloads and environment setup.
2. Use `run_pipeline.sh` for the end-to-end extraction path, or use the pre-extracted Kaggle dataset shortcut.
3. Use `scripts/preparation/prepare_sentence_data_23.py` to assemble final augmented data.
4. Train baseline models with `code/train_baseline.py` and configs under `conf/baseline/`.
5. Train the reward model with `code/train_reward.py` and `conf/reward_model/conf_reward.yaml`.
6. Check [[evaluation-and-decoding]] for how predictions are scored and decoded.

## Caveat

Before running from scratch, read [[reproducibility-caveats]]. In particular, the current checkout has a known missing-helper-module issue in `prepare_sentence_data_23.py` that should be resolved or bypassed via pre-extracted data.
