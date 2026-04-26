---
title: Extraction and Preparation Pipeline
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [pipeline, data]
sources: [README.md, run_pipeline.sh, scripts/extraction, scripts/preparation/prepare_sentence_data_23.py, brain/raw/transcripts/codebase-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: run_pipeline.sh
contradictions: []
---

# Extraction and Preparation Pipeline

The pipeline converts competition CSVs, source PDFs, HTML corpora, dictionary attestations, and synthetic translations into augmented sentence-level training data. It depends heavily on [[normalization]] and feeds [[training-stack]].

## Orchestration

`run_pipeline.sh` is the top-level sequential pipeline. It sets `DATA_DIR`, loads `.env` if present, verifies that the required API key environment variable is available, and runs seven stages:

1. Expert data: combine competition CSVs, repair translations, split into sentence pairs, deduplicate.
2. AKT extraction: side-by-side, top-bottom, and OCR extraction modes.
3. CAD extraction and normalization.
4. Journal article extraction for Dergipark, Michel, and Round 2/4 sources.
5. Hecker pipeline: PDF extraction, optional HTML scraping, cross-reference, synthetic translation generation.
6. Synthetic V22 generation from published transliterations.
7. Final preparation with `prepare_sentence_data_23.py --hecker --round4`.

Most extraction scripts expose checkpointing, sharding, or flattening behavior. Use existing flags rather than adding ad hoc parallel wrappers.

## Extraction Scripts

- `extract_akt_pairs_v24.py`: multimodal PDF extraction with modes such as `side_by_side`, `top_bottom`, `ocr`, `dergipark`, `michel`, `hecker`, and `gelb`.
- `extract_cad_pairs_v20.py`: CAD PDF extraction into structured entries and flattened pairs.
- `normalize_cad_v20.py`: filters and normalizes CAD/eSAD-style output.
- `repair_expert_translations_v16.py`: LLM-backed repair of expert translations.
- `split_expert_sentences_v16.py`: LLM-backed expert sentence splitting.
- `split_published_texts_v22.py`: synthetic translation and sentence splitting for published transliterations or Hecker inputs.
- `crossref_hecker.py`: matches Hecker transliterations to OARE, competition, expert, or synthetic records.
- `scrape_hpm_html.py`: scrapes HPM HTML transliteration pages.

## Preparation Scripts

- `dedup_expert_v19.py`: deduplicates expert data by OARE ID and near-exact similarity.
- `dedup_synthetic_v19.py`: removes synthetic records overlapping expert or AKT transliterations.
- `prepare_sentence_data_23.py`: assembles the final augmented training set with source-specific copy counts and sliding-window merging.

The final preparation script references the `missing_283_transliterations` directory under `DATA_DIR` for Round 2 and Round 4 source files. The historical source collection behind that path is summarized in [[missing-283-transliterations]].

## Output

The pipeline message says final training data is written to `DATA_DIR/synth_claude_v23_aug1/`. The public README also documents Kaggle-hosted pre-extracted training data as the shortcut path.
