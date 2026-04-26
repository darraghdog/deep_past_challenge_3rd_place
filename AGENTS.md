# Agent Guidelines

This repository contains a 3rd place solution for the Kaggle Deep Past Initiative Machine Translation competition. The task is Old Assyrian Akkadian transliteration to English translation, using extracted scholarly data, synthetic data generation, ByT5 training, and reward-model components.

## Privacy and Security

- Do not commit, print, log, or upload API keys, secrets, tokens, credentials, private hostnames, account names, or internal infrastructure details.
- Keep local credentials in `.env` or the user's shell environment. Use `.env.example` only as a template.
- Do not pass secrets inline in shell commands where they may appear in process listings or logs.
- When documenting workflows, keep them portable and public: describe generic local, Kaggle, or cluster execution without private machine names or organization-specific paths.
- Before adding copied notes from another project, remove internal provider details, usernames, absolute scratch paths, job IDs, and private monitoring commands.

## Data Safety

- Ask before deleting generated data, checkpoints, model outputs, or extracted JSONL files.
- Use distinct output and checkpoint paths. Verify they differ before any cleanup step.
- Prefer additive outputs with versioned filenames for extraction, deduplication, and dataset preparation.
- Large datasets are expected to live under `data/` or another ignored local data directory, not in git.

## Repository Map

- `README.md`: public project overview, setup, data download, extraction, and training commands.
- `run_pipeline.sh`: end-to-end extraction and preparation pipeline.
- `scripts/extraction/`: PDF, HTML, expert-data, repair, sentence-splitting, and cross-reference scripts.
- `scripts/preparation/`: deduplication and final training-set assembly scripts.
- `scripts/normalization.py`: shared transliteration and translation normalization logic.
- `code/`: ByT5 baseline, reward model, loaders, metrics, generation utilities, and training entry points.
- `conf/`: Hydra configs for continued pretraining, fine-tuning, and reward modeling.
- `sdg/`: synthetic data generation for grammar transforms, CAD drills, and template slot filling.
- `prompts/`: extraction, repair, sentence-splitting, and translation prompt templates.

## Environment

- Python requirement: `>=3.11`.
- Preferred setup:

```bash
uv sync
```

- If using optional development tools:

```bash
uv sync --extra dev
```

- API-backed extraction and synthetic-data scripts require provider credentials configured through `.env` or environment variables. Never hardcode them.

## Common Commands

Run the full extraction and preparation pipeline:

```bash
bash run_pipeline.sh
```

Prepare the final training dataset from pre-extracted data:

```bash
python3 scripts/preparation/prepare_sentence_data_23.py --hecker --round4
```

Train ByT5 models:

```bash
accelerate launch code/train_baseline.py --config-name=conf_baseline_pretrain_large
accelerate launch code/train_baseline.py --config-name=conf_baseline_continue_large
accelerate launch code/train_baseline.py --config-name=conf_baseline_pretrain_xl
accelerate launch code/train_baseline.py --config-name=conf_baseline_continue_xl
```

Train the reward model:

```bash
accelerate launch code/train_reward.py
```

Run checks when relevant:

```bash
ruff check .
pytest
```

## Pipeline Notes

- Competition data should be downloaded from Kaggle and placed according to `README.md`.
- Source PDFs and pre-extracted training data are distributed separately from code. Do not assume large data files are present in a clean checkout.
- Most extraction scripts support sharding via `--shard K/N`; use this for parallel API-backed runs.
- Use `--flatten-only` modes, where available, to rebuild flattened outputs from checkpoints without re-running API calls.
- Keep prompts and extraction modes aligned. PDF layout-specific prompts live in `prompts/` and are selected by the extraction scripts.
- Prefer `rapidfuzz` for fuzzy matching and deduplication; avoid Python-only fuzzy loops for large candidate sets.

## Competition-Specific Guidance

- Evaluation uses a geometric mean of corpus-level BLEU and chrF++ via SacreBLEU.
- The useful data families are expert translations, extracted publication pairs, CAD/eSAD-style dictionary attestations, Hecker/HPM transliterations, journal article extractions, and synthetic drills.
- Deduplication matters: avoid training leakage and near-duplicate overcounting across expert, synthetic, and extracted publication sources.
- Preserve domain-specific transliteration conventions unless a script explicitly normalizes them for compatibility.
- Validate character sets and gap markers before submission or public dataset release.

## Normalization Guidance

Use `scripts/normalization.py` rather than reimplementing normalization ad hoc. Important conventions include:

- Normalize subscripts, fractions, determinatives, and gap markers consistently.
- Keep transliteration and translation normalization decisions explicit in code and output names.
- Be careful with diacritics: do not blindly ASCII-fold Akkadian transliteration unless the relevant pipeline step requires it.
- Treat `<gap>` and `<big_gap>` markers as semantically meaningful, not formatting noise.

## Coding Guidelines

- Follow existing patterns in the nearest module before adding new abstractions.
- Keep scripts runnable from the repository root.
- For new CLI flags, use clear defaults and preserve existing behavior.
- Add focused validation around data-path, checkpoint, and output-file handling.
- Keep generated outputs out of git unless they are small, intentional fixtures.
- Update `README.md` or this file when workflow commands change.

## Sanitization Checklist

Before publishing docs, prompts, configs, logs, or notebooks:

- Remove secrets, private endpoint URLs, user names, host names, account names, internal project names, and absolute private paths.
- Replace organization-specific execution instructions with generic local, Kaggle, or cluster examples.
- Check that examples do not reveal unreleased models, private datasets, private dashboards, or non-public competition analysis.
- Prefer concise, reproducible instructions over exhaustive historical notes.
