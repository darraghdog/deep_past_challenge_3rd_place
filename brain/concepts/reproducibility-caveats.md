---
title: Reproducibility Caveats
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [repo, caveat]
sources: [brain/raw/transcripts/codebase-inventory-2026-04-26.md, scripts/preparation/prepare_sentence_data_23.py, README.md]
verified: 2026-04-26
verified_source: brain/raw/transcripts/codebase-inventory-2026-04-26.md
contradictions: []
---

# Reproducibility Caveats

This page records observed issues and assumptions that matter when reproducing the solution from this checkout. It should be checked before running [[extraction-and-preparation-pipeline]] or [[training-stack]].

## Current Checkout Gap

On 2026-04-26, `scripts/preparation/prepare_sentence_data_23.py` imports:

- `akt_matching`
- `consolidate_akt_v20`

Those modules were not present in the current checkout when searched with `find`. Any reproduction run that exercises those imports may fail unless those files are restored, vendored, or the code path is adjusted.

## Data Location Assumption

The current repo is the reproduction artifact. The original `../akk/` repo contains historical investigations and source collections. The page [[missing-283-transliterations]] summarizes one such directory, but reproduction should prefer data downloaded as documented in `README.md` or explicitly staged into this repo's `DATA_DIR`.

## External Data

Large data files are intentionally not committed. Training configs and scripts expect datasets from Kaggle or local `DATA_DIR` paths.

Before reporting numbers from configs, data inventories, or historical notes, re-read the corresponding source file. The wiki is a guide, not the authority.
