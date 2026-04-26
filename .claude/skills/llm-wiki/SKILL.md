---
name: llm-wiki
description: "Karpathy's LLM Wiki — build and maintain a persistent, interlinked markdown knowledge base in brain/. Ingest sources, query compiled knowledge, and lint for consistency."
version: 2.0.0
---

# Karpathy's LLM Wiki

Build and maintain a persistent, compounding knowledge base as interlinked markdown files.
Based on Andrej Karpathy's LLM Wiki pattern.

Unlike traditional RAG, which rediscovers knowledge from scratch per query, the wiki
compiles knowledge once and keeps it current. Cross-references are already there.
Contradictions have already been flagged. Synthesis reflects everything ingested.

Division of labor: the human curates sources and directs analysis. The agent
summarizes, cross-references, files, and maintains consistency.

## When This Skill Activates

Use this skill when the user:

- Asks to create, build, or start a wiki or knowledge base.
- Asks to ingest, add, or process a source into their wiki.
- Asks a question and the `brain/` directory exists in the project root.
- Asks to lint, audit, or health-check their wiki.
- References their wiki, knowledge base, brain, or notes in a research context.

## Wiki Location

The wiki lives at `brain/` in the project root. This is a project-local knowledge
base, not a global one.

The wiki is just a directory of markdown files. Open it in Obsidian, VS Code, or
any editor. No database or special tooling is required.

## Architecture

```text
brain/
├── SCHEMA.md           # Conventions, structure rules, domain config
├── index.md            # Sectioned content catalog with one-line summaries
├── log.md              # Chronological action log
├── raw/                # Layer 1: immutable source material
│   ├── articles/       # Web articles, clippings
│   ├── papers/         # PDFs, arxiv papers
│   ├── transcripts/    # Meeting notes, interviews
│   └── assets/         # Images, diagrams referenced by sources
├── entities/           # Layer 2: entity pages
├── concepts/           # Layer 2: concept/topic pages
├── comparisons/        # Layer 2: side-by-side analyses
├── queries/            # Layer 2: filed query results worth keeping
└── proposals/          # Layer 2: dated proposals for future work
```

Layer 1 is raw sources. The agent reads but never modifies these.
Layer 2 is the wiki. The agent creates, updates, and cross-references these files.
Layer 3 is the schema. `SCHEMA.md` defines structure, conventions, and tag taxonomy.

### Proposals

Proposals live in `brain/proposals/` and are date-prefixed:
`YYYY-MM-DD-short-title.md`.

The date is the creation date and signals freshness. When reading a proposal, check
the date against the current date. Proposals older than 30 days should be
re-evaluated before acting on them.

Proposals contain speculative or forward-looking content that does not belong in
entity or concept pages. They should clearly separate verified facts from open
questions.

## Resuming an Existing Wiki

When the user has an existing wiki, always orient yourself before doing anything:

1. Read `brain/SCHEMA.md` to understand the domain, conventions, and tag taxonomy.
2. Read `brain/index.md` to learn what pages exist and their summaries.
3. Scan recent `brain/log.md` entries to understand recent activity.

Only after orientation should you ingest, query, or lint. This prevents duplicate
pages, missed cross-references, schema violations, and repeated work.

For large wikis, also search for the topic at hand before creating anything new.

## Initializing a New Wiki

When the user asks to create or start a wiki:

1. Create the `brain/` directory structure above.
2. Ask the user what domain the wiki covers.
3. Write `brain/SCHEMA.md` customized to the domain.
4. Write initial `brain/index.md` with sectioned headers.
5. Write initial `brain/log.md` with a creation entry.
6. Confirm the wiki is ready and suggest first sources to ingest.

### `SCHEMA.md` Template

Adapt this to the user's domain. The schema constrains agent behavior and ensures
consistency:

```markdown
# Wiki Schema

## Domain
[What this wiki covers, such as "AI/ML research" or "Akkadian translation research".]

## Conventions
- File names: lowercase, hyphens, no spaces, e.g. `transformer-architecture.md`.
- Every wiki page starts with YAML frontmatter.
- Use `[[wikilinks]]` to link between pages.
- When updating a page, always bump the `updated` date.
- Every new page must be added to `index.md` under the correct section.
- Every action must be appended to `log.md`.

## Frontmatter

```yaml
---
title: Page Title
created: YYYY-MM-DD
updated: YYYY-MM-DD
type: entity | concept | comparison | query | summary
tags: [from taxonomy below]
sources: [raw/articles/source-name.md]
verified: YYYY-MM-DD
verified_source: path/to/file
contradictions: [page-name]
---
```

## Verification Policy

Pages containing hard numbers, configs, benchmarks, or results are perishable.
They can go stale when code, data, or external sources change.

- Before citing hard numbers from a wiki page, re-read the source file if it is
  available in the project.
- When verifying a fact, bump `verified` and `verified_source` in frontmatter.
- When a fact changes, update the page with the new value, note the old value
  with its date, and bump `updated`.
- Do not silently overwrite contradictory information.
- Lint flags pages where `verified` is more than 30 days old and the page
  contains tables or numeric values.

## Tag Taxonomy

[Define 10-20 top-level tags for the domain. Add new tags here before using them.]

Example for AI/ML:
- Models: model, architecture, benchmark, training
- People/Orgs: person, company, lab, open-source
- Techniques: optimization, fine-tuning, inference, alignment, data
- Meta: comparison, timeline, controversy, prediction

Rule: every tag on a page must appear in this taxonomy. If a new tag is needed,
add it here first, then use it.

## Page Thresholds

- Create a page when an entity or concept appears in 2+ sources or is central to
  one source.
- Add to an existing page when a source mentions something already covered.
- Do not create a page for passing mentions, minor details, or material outside
  the domain.
- Split a page when it exceeds about 200 lines.
- Archive a page when its content is fully superseded.

## Entity Pages

One page per notable entity. Include:
- Overview
- Key facts and dates
- Relationships to other entities using `[[wikilinks]]`
- Source references

## Concept Pages

One page per concept or topic. Include:
- Definition
- Current state of knowledge
- Open questions or debates
- Related concepts using `[[wikilinks]]`

## Comparison Pages

Side-by-side analyses. Include:
- What is being compared and why
- Dimensions of comparison
- Verdict or synthesis
- Sources

## Update Policy

When new information conflicts with existing content:

1. Re-read the source.
2. Check the dates.
3. If genuinely contradictory, note both positions with dates and sources.
4. Mark the contradiction in frontmatter.
5. Flag for user review in the lint report.
6. Never silently overwrite.
```

### `index.md` Template

The index is sectioned by type. Each entry is one line: wikilink plus summary.

```markdown
# Wiki Index

> Content catalog. Every wiki page is listed under its type with a one-line summary.
> Read this first to find relevant pages for any query.
> Last updated: YYYY-MM-DD | Total pages: N

## Entities
<!-- Alphabetical within section -->

## Concepts

## Comparisons

## Queries
```

When any section exceeds 50 entries, split it into subsections by first letter or
sub-domain. When the index exceeds 200 entries total, create a
`_meta/topic-map.md` that groups pages by theme.

### `log.md` Template

```markdown
# Wiki Log

> Chronological record of all wiki actions.
> Format: `## [YYYY-MM-DD] action | subject`
> Actions: ingest, update, query, lint, create, archive, delete

## [YYYY-MM-DD] create | Wiki initialized
- Domain: [domain]
- Structure created with SCHEMA.md, index.md, log.md
```

## Core Operations

### 1. Ingest

When the user provides a source, integrate it into the wiki:

1. Capture the raw source:
   - URL: fetch content and save to `brain/raw/articles/`.
   - PDF: save to `brain/raw/papers/`.
   - Pasted text: save to the appropriate `brain/raw/` subdirectory.
   - Name files descriptively, e.g. `brain/raw/articles/source-title-YYYY-MM-DD.md`.
2. Discuss takeaways with the user when the source requires interpretation.
3. Check what already exists by reading `brain/index.md` and searching wiki files.
4. Write or update wiki pages:
   - Create new entity or concept pages only if they meet `SCHEMA.md` thresholds.
   - Add new information to existing pages and bump `updated`.
   - When new information contradicts existing content, follow the update policy.
   - Cross-reference with `[[wikilinks]]`.
   - Use only tags from the taxonomy in `SCHEMA.md`.
5. Update navigation:
   - Add new pages to `brain/index.md`.
   - Update the total page count and last-updated date.
   - Append to `brain/log.md`.
   - List every file created or updated in the log entry.
6. Report what changed to the user.

A single source can trigger updates across multiple wiki pages. That is expected
when the source connects to existing entities and concepts.

### 2. Query

When the user asks a question about the wiki's domain:

1. Read `brain/index.md` to identify relevant pages.
2. For larger wikis, search across all `.md` files for key terms.
3. Read relevant pages. If a page contains hard numbers, configs, scores, or
   benchmarks, check the `verified` date. If stale or absent, re-read the source
   file before answering.
4. Synthesize an answer from the compiled knowledge and cite the wiki pages used.
5. File valuable answers back into `brain/queries/` or `brain/comparisons/`.
   Do not file trivial lookups.
6. Update `brain/log.md` with the query and whether it was filed.

### 3. Lint

When the user asks to lint, health-check, or audit the wiki:

1. Find orphan pages with no inbound `[[wikilinks]]`.
2. Find broken wikilinks that point to pages that do not exist.
3. Check index completeness against the filesystem.
4. Validate frontmatter fields and tags.
5. Flag stale content.
6. Identify contradictions or pages with conflicting claims.
7. Flag pages over about 200 lines.
8. Audit tags against the `SCHEMA.md` taxonomy.
9. Rotate `brain/log.md` if it has grown too large.
10. Flag stale verification on pages with numeric values or tables.
11. Report findings with file paths and suggested actions, grouped by severity.
12. Append a lint entry to `brain/log.md`.

## Working with the Wiki

### Searching

```bash
rg "transformer" brain/ -g "*.md"
rg "tags:.*alignment" brain/ -g "*.md"
tail -30 brain/log.md
```

### Bulk Ingest

When ingesting multiple sources at once:

1. Read all sources first.
2. Identify all entities and concepts across all sources.
3. Check existing pages for all of them in one search pass.
4. Create or update pages in one pass.
5. Update `brain/index.md` once at the end.
6. Write a single log entry covering the batch.

### Archiving

When content is fully superseded or the domain scope changes:

1. Create `brain/_archive/` if needed.
2. Move the page to `brain/_archive/` with its original path preserved where useful.
3. Remove it from `brain/index.md`.
4. Update pages that linked to it.
5. Log the archive action.

### Obsidian Integration

The `brain/` directory works as an Obsidian vault:

- `[[wikilinks]]` render as clickable links.
- Graph View visualizes the knowledge network.
- YAML frontmatter can power Dataview queries.
- `brain/raw/assets/` holds images referenced by notes.

For best results:

- Set Obsidian's attachment folder to `brain/raw/assets/`.
- Enable Wikilinks.
- Install Dataview if table-style queries are useful.

## Pitfalls

- Record findings, not assertions. Only write what is verified or directly sourced.
- Do not add speculation, assumptions, guesses, or untested claims to entity or
  concept pages. Put proposals or recommendations in clearly labeled proposal or
  query pages with caveats.
- Never modify files in `brain/raw/`; sources are immutable.
- Always orient first by reading `SCHEMA.md`, `index.md`, and recent `log.md`.
- Always update `brain/index.md` and `brain/log.md`.
- Do not create pages for passing mentions.
- Do not create isolated pages without cross-references.
- Frontmatter is required.
- Tags must come from the taxonomy. Add new tags to `SCHEMA.md` before using them.
- Keep pages scannable. Split pages over about 200 lines.
- Ask before mass-updating if an ingest would touch 10 or more existing pages.
- Handle contradictions explicitly. Do not silently overwrite.
