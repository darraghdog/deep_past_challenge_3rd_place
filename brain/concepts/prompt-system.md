---
title: Prompt System
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [pipeline, data]
sources: [prompts, scripts/extraction/extract_akt_pairs_v24.py, scripts/extraction/extract_cad_pairs_v20.py, scripts/extraction/repair_expert_translations_v16.py, scripts/extraction/split_published_texts_v22.py, brain/raw/transcripts/prompt-and-sdg-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: brain/raw/transcripts/prompt-and-sdg-inventory-2026-04-26.md
contradictions: []
---

# Prompt System

The prompt system is one of the clearest demo surfaces in the repository. It shows how the solution turned heterogeneous scholarly sources into structured training data for [[extraction-and-preparation-pipeline]].

The prompts are stored in `prompts/` and are selected by extraction and splitting scripts according to document layout, source language, and data type.

## Prompt Families

- AKT extraction prompts handle side-by-side, top-bottom, OCR, Turkish, German, and Kouwenberg/Larsen-style alignment cases.
- Journal prompts handle Turkish Dergipark papers and English Michel-style academic chapters.
- Hecker prompts extract transliteration-only tablet entries from born-digital PDFs.
- CAD prompts extract Old Assyrian attestations from dictionary-style OCR scans into dual raw/MT-normalized tracks.
- Repair and sentence-splitting prompts turn document-level expert or synthetic outputs into sentence-aligned examples.

## Design Pattern

Most extraction prompts combine:

- a domain role, usually Assyriologist plus data engineer;
- layout-specific instructions;
- cleaning rules;
- atomic chunking rules for witnesses, seals, goods, and itemized lists;
- quality or confidence fields;
- strict JSON output delimiters.

This is worth showing in a demo because it makes the data pipeline inspectable. The model-training code in [[training-stack]] is conventional enough; the prompt system explains much of how the non-standard data was converted into usable supervision.

## Public-Wiki Handling

Do not copy full prompts into the wiki unless there is a specific reason. Summaries are preferred because the prompt files are already in the repo and can be read directly.

Related pages: [[synthetic-data-generation]], [[normalization]], [[missing-283-transliterations]].
