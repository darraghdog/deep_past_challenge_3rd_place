---
title: Evaluation and Decoding
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [evaluation, training]
sources: [code/utils/metric_utils.py, code/utils/generation_utils.py, README.md]
verified: 2026-04-26
verified_source: code/utils/metric_utils.py
contradictions: []
---

# Evaluation and Decoding

Evaluation logic lives under `code/utils/` and is used by [[training-stack]]. The public README states that the competition metric is the geometric mean of corpus-level BLEU and chrF++ via SacreBLEU.

## Metric

`code/utils/metric_utils.py` computes:

- corpus BLEU with SacreBLEU;
- corpus chrF++ with `word_order=2`;
- final score as the square root of BLEU multiplied by chrF++.

The function also returns the component BLEU and chrF values, which is useful for debugging whether a model is improving through exact lexical overlap, character-level similarity, or both.

## Decoding

`code/utils/generation_utils.py` includes:

- standard batched generation;
- generation config construction from Hydra config;
- minimum Bayes risk style candidate selection using pairwise sentence-level utility.

The MBR helper samples multiple candidates per input, scores candidates against each other using the same BLEU/chrF++-style utility, and selects the candidate with the highest average pairwise utility.

This page is a useful demo companion to [[dataset-and-config-map]] because it connects training configs to the score reported by the competition.
