# Wiki Log

> Chronological record of wiki actions.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete

## [2026-04-26] create | Wiki initialized

- Domain: public reproduction repository for the Deep Past Initiative Machine Translation solution.
- Created `brain/SCHEMA.md`, `brain/index.md`, `brain/log.md`.
- Created initial concept/entity pages for codebase overview, pipeline, training stack, normalization, reproducibility caveats, and the missing 283 transliteration source collection.
- Created raw inventory notes under `brain/raw/transcripts/`.

## [2026-04-26] ingest | Current codebase and missing 283 source collection

- Sources inspected:
  - `README.md`
  - `AGENTS.md`
  - `run_pipeline.sh`
  - `pyproject.toml`
  - `code/`
  - `conf/`
  - `scripts/`
  - `sdg/`
  - `../akk/datamount/missing_283_transliterations/README.md`
  - `../akk/datamount/missing_283_transliterations/dataset-metadata.json`
- Files created:
  - `brain/raw/transcripts/codebase-inventory-2026-04-26.md`
  - `brain/raw/transcripts/missing-283-inventory-2026-04-26.md`
  - `brain/concepts/codebase-overview.md`
  - `brain/concepts/extraction-and-preparation-pipeline.md`
  - `brain/concepts/training-stack.md`
  - `brain/concepts/normalization.md`
  - `brain/concepts/reproducibility-caveats.md`
  - `brain/entities/missing-283-transliterations.md`

## [2026-04-26] update | Brain publishing workflow

- Added GitHub Actions workflow to build `brain/` with Quartz and deploy it to GitHub Pages.
- Removed one absolute local path from the raw codebase inventory note before publishing.
- Files created or updated:
  - `.github/workflows/deploy-brain.yml`
  - `brain/concepts/brain-publishing.md`
  - `brain/index.md`
  - `brain/log.md`
  - `brain/raw/transcripts/codebase-inventory-2026-04-26.md`

## [2026-04-26] update | Pages deployment enabled

- GitHub Pages source was set to GitHub Actions.
- Touched the wiki log to trigger the Quartz Pages workflow after enabling the setting.

## [2026-04-26] ingest | Demo expansion for prompts and repo sections

- Added pages that make the public brain more useful as a demo of the current repository.
- Sources inspected:
  - `prompts/`
  - `sdg/`
  - `conf/`
  - `code/utils/metric_utils.py`
  - `code/utils/generation_utils.py`
  - `code/utils/onomasticon.py`
- Files created or updated:
  - `brain/raw/transcripts/prompt-and-sdg-inventory-2026-04-26.md`
  - `brain/concepts/prompt-system.md`
  - `brain/concepts/synthetic-data-generation.md`
  - `brain/concepts/dataset-and-config-map.md`
  - `brain/concepts/evaluation-and-decoding.md`
  - `brain/index.md`
  - `brain/log.md`

## [2026-04-26] query | Demo query pages

- Added filed query pages to make the published brain easier to navigate during a demo.
- Files created or updated:
  - `brain/queries/how-does-this-repo-reproduce-the-solution.md`
  - `brain/queries/what-makes-the-data-pipeline-distinctive.md`
  - `brain/queries/what-should-i-show-in-a-demo.md`
  - `brain/index.md`
  - `brain/log.md`
