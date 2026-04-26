---
title: Brain Publishing
created: 2026-04-26
updated: 2026-04-26
type: concept
tags: [repo, security]
sources: [.github/workflows/deploy-brain.yml, brain/SCHEMA.md]
verified: 2026-04-26
verified_source: .github/workflows/deploy-brain.yml
contradictions: []
---

# Brain Publishing

The `brain/` wiki is published with Quartz through GitHub Pages. The workflow lives at `.github/workflows/deploy-brain.yml`.

The workflow:

1. Runs on pushes to `main` that touch `brain/**` or the workflow file.
2. Can also be run manually with `workflow_dispatch`.
3. Builds `brain/` using Quartz.
4. Uploads the generated static site artifact.
5. Deploys it with GitHub Pages.

This publishing setup makes [[codebase-overview]], [[extraction-and-preparation-pipeline]], and [[missing-283-transliterations]] publicly readable. Before adding new raw notes or source summaries, follow the privacy rules in `brain/SCHEMA.md` and avoid private paths, credentials, internal infrastructure details, and unreleased/private analysis.
