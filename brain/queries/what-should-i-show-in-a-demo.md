---
title: What Should I Show in a Demo?
created: 2026-04-26
updated: 2026-04-26
type: query
tags: [repo, data, evaluation]
sources: [brain/concepts/codebase-overview.md, brain/concepts/prompt-system.md, brain/concepts/synthetic-data-generation.md, brain/concepts/dataset-and-config-map.md, brain/concepts/evaluation-and-decoding.md, brain/concepts/brain-publishing.md]
verified: 2026-04-26
verified_source: brain/concepts/codebase-overview.md
contradictions: []
---

# What Should I Show in a Demo?

Use this as a short demo path through the published brain.

## Suggested Flow

1. Start at [[codebase-overview]] to establish that this is a reproduction repo, not the original exploratory workspace.
2. Open [[prompt-system]] to show the source-specific extraction design.
3. Open [[synthetic-data-generation]] to show how the repo creates additional structured supervision.
4. Open [[dataset-and-config-map]] to show where data and training configs live.
5. Open [[evaluation-and-decoding]] to connect the repository to the competition metric.
6. Open [[missing-283-transliterations]] if the audience wants provenance from the original working repo.
7. Close with [[brain-publishing]] to show that the wiki itself is generated from `brain/` and published by Quartz/GitHub Pages.

## Demo Framing

The strongest framing is that the solution is reproducible because the repo separates:

- source extraction;
- data normalization;
- synthetic data generation;
- final dataset assembly;
- model training;
- evaluation and decoding;
- provenance notes.

This makes the brain useful as a guided map rather than a replacement for the code.
