---
title: Codebase Overview
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [repo]
sources: [README.md, AGENTS.md, pyproject.toml, run_pipeline.sh, brain/raw/transcripts/codebase-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: brain/raw/transcripts/codebase-inventory-2026-04-26.md
contradictions: []
---

# Codebase Overview

This repository is the public reproduction artifact for the 3rd place Deep Past Initiative Machine Translation solution. It packages code for extraction, data preparation, synthetic data generation, ByT5 training, reward modeling, and prompt-driven data processing.

The adjacent `../akk/` repository is the original working tree where investigations and intermediate datasets were developed. This repo should be treated as the cleaner reproduction path; historical notes from `../akk/` are useful for provenance but should not override observed current code. See [[missing-283-transliterations]] and [[reproducibility-caveats]].

## Structure

- `code/`: training code for ByT5 and reward models.
- `conf/`: Hydra configs for baseline continued pretraining, fine-tuning, and reward modeling.
- `scripts/extraction/`: expert-data repair, sentence splitting, PDF extraction, CAD extraction, Hecker scraping, and cross-reference scripts.
- `scripts/preparation/`: deduplication and final training-set assembly.
- `scripts/normalization.py`: shared normalization utilities.
- `sdg/`: synthetic grammar, CAD, and template-drill generation.
- `prompts/`: extraction, repair, sentence splitting, and source-layout prompts.
- `run_pipeline.sh`: sequential extraction and preparation workflow.

For a demo-oriented walkthrough, start with [[prompt-system]], [[synthetic-data-generation]], [[dataset-and-config-map]], and [[evaluation-and-decoding]]. These pages show the repository's distinctive pieces more clearly than a flat file listing.

## Main Reproduction Flow

The documented high-level path is:

1. Download competition/source/training data as described in `README.md`.
2. Run extraction and preparation through `run_pipeline.sh`, or skip extraction by using pre-extracted data.
3. Build final training data with `scripts/preparation/prepare_sentence_data_23.py`.
4. Train ByT5 and reward models using `code/` with `conf/` configs.

Detailed flow is in [[extraction-and-preparation-pipeline]], while model-side behavior is in [[training-stack]].
