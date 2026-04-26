---
title: Normalization
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [normalization, data]
sources: [scripts/normalization.py, scripts/preparation/prepare_sentence_data_23.py, scripts/extraction/normalize_cad_v20.py, AGENTS.md]
verified: 2026-04-26
verified_source: scripts/normalization.py
contradictions: []
---

# Normalization

Normalization is a central reproducibility surface. Shared logic lives in `scripts/normalization.py`; pipeline-specific wrappers are used in `prepare_sentence_data_23.py` and `normalize_cad_v20.py`.

This page connects directly to [[extraction-and-preparation-pipeline]] and [[training-stack]] because normalization affects deduplication, training targets, gap markers, and final model behavior.

## Covered Transform Families

`scripts/normalization.py` includes functions for:

- fractions and slash fractions
- subscripts
- h-dot handling
- gap markers
- determinatives
- brackets and unmatched brackets
- whitespace and punctuation spacing
- scribal insertions
- special characters
- line dividers
- ceiling brackets
- figure dash
- circumflex-to-macron conversion
- CDLI-to-target conversion
- Hecker transliteration normalization
- final translation postprocessing
- character cleaning for transliteration and translation fields

## Important Practice

Do not reimplement normalization ad hoc in a new script. Reuse the shared functions or add a named pipeline-specific wrapper when behavior needs to differ.

`prepare_sentence_data_23.py` explicitly overrides subscript handling with V15-era behavior, mapping subscript x to plain `x` rather than removing it. This is a deliberate variant and should be preserved unless a reproduction run intentionally changes it.

Gap markers such as `<gap>` and `<big_gap>` are semantic. Treat them as model-facing data, not generic markup.
