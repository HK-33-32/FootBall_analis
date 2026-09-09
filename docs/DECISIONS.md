# Decisions

## ADR-001 — Legacy event records are candidates, not semantic truth

Decision: The measured Football Core remains the perception baseline and candidate generator. Its geometry-derived `pass` label cannot populate semantic pass statistics until a VLM adjudicates it.

Reason: The inherited benchmark reports only 43% pass accuracy even with ideal tracks, teams and ball annotations.

## ADR-002 — Structured SQLite is the source of truth

Decision: Semantic events and evidence live in typed SQLite tables. Vector retrieval may be added later, but cannot generate counts.

Reason: Exact filters and stable evidence IDs are reproducible and auditable.

## ADR-003 — Evidence-aware provider-style reports are additive

Decision: Keep the existing match report backward compatible and add a versioned
`detailed_statistics` subtree. Reviewed/provider annotations replace geometric
candidates within that subtree; they never silently upgrade the original event
stream. Missing coverage and qualifiers remain null, not fabricated zeroes.

Reason: The requested RuStat-style breadth exceeds the measured capabilities of
single-camera geometry. A broad output schema must not imply semantic accuracy.

Consequences: xG/xA require identified models; identity namespaces remain separate;
each count retains evidence; a future semantic adapter has an explicit typed
boundary. Source-media milliseconds are not assumed to be the match clock.

## ADR-004 — Runtime optimizations must be reproducible image layers

Decision: Package measured legacy-core changes as code-only overlays with before,
after and patch hashes; fail clearly on an incompatible base. Compose selects the
combined performance image, not a modified container writable layer.

Reason: The old default image did not include optimizations present in external
source and a stopped container. Source-directory inspection alone did not establish
which code served inference. EXP-20260908-01 established the actual runtime A/B.

Consequences: Record immutable image IDs and model fingerprints with benchmarks.
Cache volumes hold reused models; external source/weights remain unchanged. The
legacy base historically includes weights; code-only overlays do not make that
base weight-free. Base replacement must pass compatibility checks.

## ADR-005 — Deterministic calibration is explicit, not an accuracy upgrade

Decision: Keep legacy circle sampling when FG_CALIB_SEED is unset. The isolated
circle-pcg64-v1 adapter is enabled only by an explicit seed/config.

Reason: EXP-20260908-02 produced byte-identical repeated outputs but raw GS-HOTA
50.527 versus 50.534 for the unseeded input. Reproducibility and accuracy are distinct.

Consequences: Never tune seeds against validation labels or silently change the
default. Record sampler/seed provenance. The separately validated exact-output
projection I/O cache can remain enabled regardless of the sampler choice.

## ADR-006 — Public clone contains the core source but no governed data assets

Decision: The default quick start launches the API, debugging UI and report viewer
with perception/VLM unset. The owner-confirmed Football Core source is integrated under
`services/perception`; full analysis is an explicit source-built profile after validating
CUDA and required checkpoint files. No mock predictions are used.

Reason: The application and core have the same owner and can be distributed together,
while model weights, SoccerNet data and raw match footage are large, separately governed
artifacts. A public clone must not assume the original machine's paths or Docker images.

Consequences: `.env.example` is portable and secret-free; host ports bind to
localhost. Quick-start scripts work on Windows and POSIX systems. The application
reports `not_configured` until a real backend exists. Publishing GPU overlay patches
and integrated source requires preserving upstream notices documented in
`THIRD_PARTY_NOTICES.md`; weights and raw datasets remain outside Git and images.

## ADR-007 — Model bundles are external, manifest-pinned artifacts

Decision: Keep model binaries out of Git and Docker images. Online installation uses
the upstream download paths; offline installation uses an uncompressed TAR containing
only paths declared in `configs/weights-manifest.json`, with exact sizes and SHA-256.

Reason: The verified perception set is 10.5 GiB, is already internally compressed and
has licenses independent from the application. Git/LFS or image layers would make
clones and upgrades expensive without solving redistribution rights.

Consequences: The same TAR works on Windows and Linux. Installers reject path traversal,
unknown members, revision drift and implicit replacement. A bundle can be hosted as a
private release/object-storage artifact only after its model licenses are reviewed.
