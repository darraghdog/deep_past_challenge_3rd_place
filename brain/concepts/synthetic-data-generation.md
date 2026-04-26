---
title: Synthetic Data Generation
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [synthetic-data, data, pipeline]
sources: [sdg, README.md, brain/raw/transcripts/prompt-and-sdg-inventory-2026-04-26.md]
verified: 2026-04-26
verified_source: brain/raw/transcripts/prompt-and-sdg-inventory-2026-04-26.md
contradictions: []
---

# Synthetic Data Generation

The `sdg/` directory contains synthetic-data generation workflows used to teach Old Assyrian fundamentals and broaden training coverage. It complements source extraction from [[prompt-system]] and feeds the final data assembly in [[extraction-and-preparation-pipeline]].

## Workflows

- `grammar_transform.py`: generates grammar transformations from seed examples and grammar/context resources.
- `generate_cad_drills.py`: generates deliberate-practice examples from CAD/eSAD senses, examples, and a generation plan.
- `fill_engine.py`: produces deterministic slot-filled examples from JSON templates without requiring an API call.

## Template System

Template files under `sdg/templates/` cover:

- debts and loans;
- legal and seal formulas;
- letter openings and correspondence patterns;
- accounting, memoranda, and trade examples.

The template engine draws from slot pools in `template_pools.py`, including names, commodities, amounts, places, months, eponyms, deadlines, penalties, occupations, kinship terms, and containers. Constraint helpers live in `template_constraints.py`.

## Demo Value

This section is useful in the public brain because it shows the solution was not only model training. It combined:

- mined scholarly data;
- LLM-assisted extraction;
- LLM-assisted transformations;
- deterministic template generation;
- source-specific normalization in [[normalization]].

Provider-specific endpoint details in SDG config files should stay out of public prose. The reusable point is the data-generation architecture, not the API vendor.
