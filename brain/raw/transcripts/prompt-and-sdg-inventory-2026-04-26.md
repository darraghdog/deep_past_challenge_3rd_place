# Prompt and SDG Inventory — 2026-04-26

This raw note records an inventory of prompt and synthetic-data generation files in the current checkout.

## Prompt Files

Prompt directory: `prompts/`.

- `akt_ocr_turkish_align_v24.txt`: 181 lines
- `akt_side_by_side_kouwenberg_parsing_v24.txt`: 129 lines
- `akt_side_by_side_parsing_v24.txt`: 118 lines
- `akt_top_bottom_and_align_parsing_v24.txt`: 145 lines
- `akt_top_bottom_turkish_align_v24.txt`: 205 lines
- `cad_side_by_side_parsing.txt`: 144 lines
- `dergipark_inline_parsing_v24.txt`: 173 lines
- `hecker_translit_parsing_v24.txt`: 118 lines
- `michel_en_parsing_v24.txt`: 101 lines
- `prompt_repair_translations_a.txt`: 50 lines
- `prompt_v08_sentence_split.txt`: 136 lines
- `prompt_v13_published_texts_sentence_split.txt`: 262 lines

## Prompt Families

- AKT side-by-side extraction.
- AKT top-bottom extraction and alignment.
- OCR/Turkish alignment extraction.
- Dergipark Turkish inline article extraction.
- Michel English article/chapter extraction.
- Hecker transliteration-only extraction.
- CAD side-by-side dictionary-attestation extraction.
- Expert translation repair.
- Expert/synthetic sentence splitting and translation.

## Synthetic Data Generation Files

Directory: `sdg/`.

- `grammar_transform.py`: LLM-backed grammar transformation from seed examples.
- `generate_cad_drills.py`: LLM-backed CAD/eSAD deliberate-practice drill generation.
- `fill_engine.py`: deterministic slot-fill template generation.
- `seed.py`: seed selection from training/OARE data.
- `template_pools.py`: names, commodities, amounts, dates, professions, and related slot pools.
- `template_constraints.py`: constraints and formatting helpers.
- `templates/*.json`: domain templates for debt, legal, letter, accounting, and trade examples.
- `prompts/*.j2`: Jinja prompt templates for SDG workflows.
- `conf/*.yaml`: SDG run configuration. Provider-specific endpoint details should not be copied into public notes.
