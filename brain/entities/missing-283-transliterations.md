---
title: Missing 283 Transliterations
created: 2026-04-26
updated: 2026-04-26
type: entity
tags: [data, source-collection]
sources: [../akk/datamount/missing_283_transliterations/README.md, ../akk/datamount/missing_283_transliterations/dataset-metadata.json, brain/raw/transcripts/missing-283-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: ../akk/datamount/missing_283_transliterations/README.md
contradictions: []
---

# Missing 283 Transliterations

`../akk/datamount/missing_283_transliterations/` is a historical source-collection directory from the original working repository. It is not part of the clean reproduction repo by default, but it explains where several Round 2/4 data sources used by [[extraction-and-preparation-pipeline]] came from.

The collection investigated 283 OARE tablets that had expert translations but no transliterations in the available `published_texts.csv` data. The README states that none overlapped `train.csv`, all had complete English translations, and transliterations had been found for 53 of 283 at the time the notes were written.

## Role in Reproduction

The current reproduction repo's `prepare_sentence_data_23.py` defines `R4_BASE = DATA_DIR / "missing_283_transliterations"` and references Round 2 and Round 4 extracted JSONL files under that path. This means reproduction runs need the corresponding data staged under `DATA_DIR/missing_283_transliterations/`, or they need to use the pre-extracted Kaggle training data path documented in `README.md`.

This is a data provenance note for [[codebase-overview]] and a practical input requirement for [[reproducibility-caveats]].

## Contents

Observed directory-level structure:

- `round_1/`: initial manual collection, including the 283-tablet reference JSON and early source PDFs/HTML.
- `round_2/`: Michel and DergiPark sources, with English/Turkish/table/no-transliteration subfolders and extraction outputs.
- `round_3_hecker_hpm/`: Hecker/HPM bulk PDF collection and extraction outputs.
- `round_4/`: broad Turkish, English, German, Gelb/OIP27, Hecker web, and related sources.
- `round_5_cdli/`: CDLI Old Assyrian corpus clone and cross-reference output.
- `round_6/`, `round_7/`, `round_8/`: later source rounds.
- `extra_1003/`: DergiPark and Michel source batches.

Observed inventory on 2026-04-26: 558 files, including 231 PDFs and 170 JSONL files. See `brain/raw/transcripts/missing-283-inventory-2026-04-26.md` for the raw inventory note.

## Training-Data Usage Claimed by Source README

The source README states that selected DergiPark, Michel, Round 2, Round 4, and Hecker HPM translated data were included in the best training data variant, while Round 3 Hecker transliteration-only, Gelb OIP27, HPM HTML scrape, and CDLI were not included.

Treat those claims as historical notes unless re-verified against the current `DATA_DIR` and `prepare_sentence_data_23.py`.
