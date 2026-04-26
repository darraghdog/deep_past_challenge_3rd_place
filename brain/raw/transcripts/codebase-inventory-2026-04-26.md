# Codebase Inventory — 2026-04-26

This raw note records the observed structure of the current checkout at `/Users/dhanley/Documents/akk-third`. It is a reproducibility repo, not the original exploratory working tree.

## Top-Level Files

- `README.md`: public overview, data setup, extraction pipeline, training pipeline, synthetic data generation.
- `AGENTS.md`: agent guidance for this repository.
- `pyproject.toml`: Python `>=3.11`, dependencies, optional dev tools.
- `run_pipeline.sh`: sequential extraction and preparation pipeline.
- `.env.example`: documents required API environment variables.

## Main Directories

- `code/`: model training code.
- `conf/`: Hydra configs for baseline and reward model training.
- `prompts/`: extraction, repair, and splitting prompts.
- `scripts/extraction/`: extraction, repair, sentence splitting, cross-reference, and normalization stages.
- `scripts/preparation/`: deduplication and final training data assembly.
- `sdg/`: synthetic data generation.

## Observed Entry Points

- `bash run_pipeline.sh`
- `accelerate launch code/train_baseline.py --config-name=conf_baseline_pretrain_large`
- `accelerate launch code/train_baseline.py --config-name=conf_baseline_continue_large`
- `accelerate launch code/train_baseline.py --config-name=conf_baseline_pretrain_xl`
- `accelerate launch code/train_baseline.py --config-name=conf_baseline_continue_xl`
- `accelerate launch code/train_reward.py`
- `python3 scripts/preparation/prepare_sentence_data_23.py --hecker --round4`

## Verification Notes

- `prepare_sentence_data_23.py` imports `akt_matching` and `consolidate_akt_v20`; those files were not present in this checkout when searched on 2026-04-26.
- Large data files are not expected in a clean checkout. `README.md` points users to Kaggle-hosted competition, source, and training data.
