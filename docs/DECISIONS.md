# Decisions

## ADR-001 — Legacy event records are candidates, not semantic truth

Decision: The measured Football Core remains the perception baseline and candidate generator. Its geometry-derived `pass` label cannot populate semantic pass statistics until a VLM adjudicates it.

Reason: The inherited benchmark reports only 43% pass accuracy even with ideal tracks, teams and ball annotations.

## ADR-002 — Structured SQLite is the source of truth

Decision: Semantic events and evidence live in typed SQLite tables. Vector retrieval may be added later, but cannot generate counts.

Reason: Exact filters and stable evidence IDs are reproducible and auditable.

