# Wiki Schema

## Domain

This wiki covers the public reproduction repository for the 3rd place Deep Past Initiative Machine Translation solution, plus selected notes from the original working repository when they explain data provenance or reproducibility.

The current repository is the reproducibility artifact. The adjacent `../akk/` repository is treated as historical working context: development notes, investigations, intermediate extracted data, and source-collection records may be summarized here, but private infrastructure details should not be copied.

## Conventions

- File names are lowercase with hyphens and no spaces.
- Every wiki page starts with YAML frontmatter.
- Use `[[wikilinks]]` for internal links.
- When updating a page, bump the `updated` date.
- Every new page must be listed in `index.md`.
- Every wiki action must be appended to `log.md`.
- Keep private infrastructure names, private paths, secrets, and user/account identifiers out of wiki pages unless the path is necessary local provenance and already referenced by the user.

## Frontmatter

```yaml
---
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entity | concept | comparison | query | summary
tags: [from taxonomy below]
sources: [path/or/source]
verified: YYYY-MM-DD
verified_source: path/to/file
contradictions: []
---
```

## Verification Policy

Pages containing hard numbers, paths, configs, scores, file counts, or command behavior are perishable.

- Before citing hard numbers, re-read the source file or rerun the relevant inventory command.
- When verifying a fact, update `verified` and `verified_source`.
- When a fact changes, note the old and new value with dates. Do not silently overwrite.
- Treat code observed in the current checkout as more authoritative than older notes from `../akk/`.

## Tag Taxonomy

- `repo`: repository structure, ownership, or reproducibility notes
- `pipeline`: extraction, preparation, or orchestration workflow
- `data`: datasets, source collections, extracted records, provenance
- `training`: model training, reward modeling, configs, generation
- `normalization`: transliteration or translation normalization
- `evaluation`: metrics, validation, scoring, or submission checks
- `synthetic-data`: generated drills or LLM-produced training data
- `source-collection`: external PDFs, HTML corpora, or historical acquisition notes
- `caveat`: known gap, missing dependency, stale assumption, or risk
- `security`: secrets, privacy, or publishing sanitization

## Page Thresholds

- Create a page when an entity, dataset, concept, or workflow is central to reproducibility.
- Add to an existing page for minor updates to a covered concept.
- Do not create pages for passing mentions or one-off implementation details.
- Split pages over roughly 200 lines.
- Archive superseded pages under `brain/_archive/` and remove them from `index.md`.

## Update Policy

When current code conflicts with historical notes:

1. Re-read the current code or data file.
2. Check dates and provenance.
3. Record both positions if the conflict matters.
4. Mark the page with `caveat` if it affects reproduction.
5. Ask before mass-updating 10 or more pages.
