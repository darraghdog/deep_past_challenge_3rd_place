# 3rd Place Solution — Deep Past Initiative Machine Translation

**Competition**: [Deep Past Initiative Machine Translation](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation)  
**Team**: Darragh Hanley & Raja Biswas  
**Result**: 3rd / 2,673 teams  
**Prize**: $8,000  
**Solution Writeup**: [Synthetic Data to Teach OA Fundamentals — Deep Past Challenge - Translate Akkadian to English](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/writeups/3rd-synthetic-data-to-teach-oa-fundamentals)  

## Competition Overview

Four thousand years ago, Assyrian merchants left behind one of the world's richest archives of everyday commercial life — tens of thousands of clay tablets recording debts, caravans, and family matters. Nearly 23,000 tablets survive documenting Old Assyrian trading networks connecting Mesopotamia to Anatolia. Only half have been translated, and fewer than a dozen scholars worldwide can read the rest.

The task: build neural machine-translation models that convert transliterated Old Assyrian Akkadian into English. The challenge is that Akkadian is a low-resource, morphologically complex language where a single word can encode what takes multiple words in English.

**Evaluation**: Geometric mean of corpus-level BLEU and chrF++ scores (micro-averaged via [SacreBLEU](https://github.com/mjpost/sacrebleu)).

**Training data**: ~1,500 document-level transliteration/translation pairs + ~8,000 untranslated transliterations + raw OCR from ~880 scholarly publications.  
**Test data**: ~4,000 sentence-level translations from ~400 documents.

```bibtex
@misc{deep-past-initiative-machine-translation,
    author = {Abdulla, F. and Agarwal, R. and Anderson, A. and Barjamovic, G. and Lassen, A. and Ryan Holbrook and Mar\'{i}a Cruz},
    title = {Deep Past Challenge - Translate Akkadian to English},
    year = {2025},
    howpublished = {\url{https://kaggle.com/competitions/deep-past-initiative-machine-translation}},
    note = {Kaggle}
}
```

## Data Setup

### 1. Competition Data (hosted by Kaggle)

Download from the [competition data page](https://www.kaggle.com/competitions/deep-past-initiative-machine-translation/data) and place in `data/competition/`.

### 2. Training Data (hosted on Kaggle Datasets)

All extracted JSONL files and preparation scripts needed to build the training dataset. Drop-in replacement for the original `synth-claude-v23-aug3-68e4ba` dataset.

```bash
kaggle datasets download darraghdog/akkadian-3rd-place-training-data -p data/ --unzip
```

Run `scripts/prepare_sentence_data_23.py` to produce the final augmented training set.

### 3. Source Data (hosted on Kaggle Datasets)

Download from [akkadian-3rd-place-source-data-darragh](https://www.kaggle.com/datasets/darraghdog/akkadian-3rd-place-source-data-darragh):

```bash
kaggle datasets download darraghdog/akkadian-3rd-place-source-data-darragh -p data/ --unzip
```

Only needed if re-running extraction from scratch. Contains all PDF and reference sources (~1.4 GB):

| Directory | Contents | Files | Size |
|-----------|----------|-------|------|
| `akt_subsets/` | AKT volume page-range subsets (side-by-side, top-bottom, OCR layouts) | 18 PDFs | 373M |
| `oaa_pihans/` | Reference publications (ICK4, Kouwenberg, Larsen, PIHANS96) | 4 PDFs | 183M |
| `cad/` | Chicago Assyrian Dictionary open-access volumes | 26 PDFs | 750M |
| `journal_articles/` | Turkish, English, German journal articles across 4 collection rounds | 61 PDFs | 173M |
| `hecker_hpm/` | Hecker aATU/HPM transliteration corpus (PDFs + web-scraped HTML) | 86 files | 36M |
| Root files | `holdout_oare_ids.txt`, `Dataset_Instructions.txt`, `Dataset_Instructions_v2.txt` | 3 files | 27K |

## Extraction Pipeline

The pipeline extracts transliteration/translation pairs from source PDFs and competition data, then assembles them into the final augmented training set. All Gemini API calls go through the NVIDIA inference endpoint.

### Prerequisites

```bash
pip install pandas requests PyMuPDF rapidfuzz
```

### API Key Setup

```bash
cp .env.example .env
# Edit .env and add your NVIDIA API key
# AKKADIAN_KEY — used by extraction pipeline (scripts/)
# INFERENCE_NVIDIA_API_KEY — used by synthetic data generation (sdg/)
```

### Running the Full Pipeline

```bash
bash run_pipeline.sh
```

This runs all stages sequentially. Individual scripts support `--shard K/N` for parallelization.

### Pipeline Stages

| Stage | Script | API | Output |
|-------|--------|-----|--------|
| 1a | `extract_expert_published_texts.py` | - | Combined expert pairs |
| 1b | `repair_expert_translations_v16.py` | Gemini Pro | Repaired translations |
| 1c | `split_expert_sentences_v16.py` | Gemini Pro | Sentence-split expert pairs |
| 1d | `dedup_expert_v19.py` | - | `expert_v19_dedup.jsonl` |
| 2a-c | `extract_akt_pairs_v24.py` (3 modes) | Gemini Pro 3.1 | AKT v24 JSONLs |
| 3a | `extract_cad_pairs_v20.py` | Gemini Pro 3.1 | CAD raw pairs |
| 3b | `normalize_cad_v20.py` | - | `cad_normalized.jsonl` |
| 4a-g | `extract_akt_pairs_v24.py` (journals) | Gemini Pro 3.1 | Journal article JSONLs |
| 5a | `extract_akt_pairs_v24.py` (hecker) | Gemini Pro 3.1 | Hecker transliterations |
| 5b | `crossref_hecker.py` | - | Cross-referenced pairs |
| 5c | `split_published_texts_v22.py` (hecker) | Gemini Pro 3.1 | `hecker_v22.jsonl` |
| 6a | `split_published_texts_v22.py` (synthetic) | Gemini Pro 3.1 | `synthetic_v22.jsonl` |
| 7 | `prepare_sentence_data_23.py` | - | Final training set |

### Shortcut: Using Pre-extracted Data

To skip extraction and go straight to training, download the training data dataset (step 2 above) which contains all pre-extracted JSONLs. Then run only Stage 7:

```bash
python3 scripts/preparation/prepare_sentence_data_23.py --hecker --round4
```

## Training Pipeline

### Environment Setup

```bash
# Requires Python >= 3.11 and uv (https://docs.astral.sh/uv/)
uv sync
```

### Training Datasets (hosted on Kaggle Datasets)

The training configs pull datasets from Kaggle Hub automatically via `kagglehub`. The following datasets are required for model training:

| Dataset | Used By | Purpose |
|---------|---------|---------|
| [conjuring92/dpc-mix-pretrain-a05](https://www.kaggle.com/datasets/conjuring92/dpc-mix-pretrain-a05) | CPT configs | Continued pre-training datamix (synthetic drills + real data) |
| [conjuring92/dpc-mix-gold-b2](https://www.kaggle.com/datasets/conjuring92/dpc-mix-gold-b2) | `conf_baseline_continue_large` | Fine-tuning datamix (ByT5-Large) |
| [conjuring92/dpc-mix-gold-m2](https://www.kaggle.com/datasets/conjuring92/dpc-mix-gold-m2) | `conf_baseline_continue_xl` | Fine-tuning datamix (ByT5-XL) |
| [conjuring92/dpc-reward-mix-v3](https://www.kaggle.com/datasets/conjuring92/dpc-reward-mix-v3) | `conf_reward` | Reward model preference data |
| [conjuring92/dpc-onomasticon-polysemy](https://www.kaggle.com/datasets/conjuring92/dpc-onomasticon-polysemy) | FT configs | Name-swap augmentation lookup |

### Running Training

Training uses hydra for configuration and HF Accelerate for distributed training. All commands are run from the repo root. Training was run on a node with 8xH100 GPUs.

**Stage 1 — Continued Pre-training (CPT):**

```bash
# ByT5-Large
accelerate launch code/train_baseline.py --config-name=conf_baseline_pretrain_large

# ByT5-XL
accelerate launch code/train_baseline.py --config-name=conf_baseline_pretrain_xl
```

**Stage 2 — Fine-tuning (FT):**

Update `model.backbone_path` in the config to point to your CPT checkpoint, then:

```bash
# ByT5-Large
accelerate launch code/train_baseline.py --config-name=conf_baseline_continue_large

# ByT5-XL
accelerate launch code/train_baseline.py --config-name=conf_baseline_continue_xl
```

**Reward Model:**

```bash
accelerate launch code/train_reward.py
```

## Synthetic Data Generation

Scripts for generating synthetic training data to teach OA fundamentals.

| Drill Type | Script | Description |
|------------|--------|-------------|
| Grammar | `sdg/grammar_transform.py` | Apply grammar rules to seed sentences via LLM |
| Slot-Fill Templates | `sdg/fill_engine.py` | Programmatic template filling (no API needed) |
| CAD Vocab  | `sdg/generate_cad_drills.py` | Generate sentence pairs per (lemma, gloss) via LLM |

### Running

```bash
# Grammar drills
python sdg/grammar_transform.py --config sdg/conf/conf_transform.yaml

# CAD drills
python sdg/generate_cad_drills.py --config sdg/conf/conf_cad_drill.yaml

# Template slot-fill
python -c "from sdg.fill_engine import generate; generate()"
```
