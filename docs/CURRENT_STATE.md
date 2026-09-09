# Current State

Updated: 2026-09-09 20:25 Europe/Moscow

## Objective

Improve match analysis speed without sacrificing measured correctness; export
evidence-aware statistics resembling the supplied RuStat report. Full-match/SOTA
acceptance is not achieved. Continue benchmark-first, without frozen-test tuning.

## Working

- Existing rosters, viewer, chunked analysis and recovery retained.
- Sep 7 CPU/video optimizations: exact outputs on SoccerNet valid 021/033/045/090;
  statistics stage 2.24–2.61x faster. See docs/PERFORMANCE_20260907.md.
- Native GPU ARG–FRA A/B: 450.3 -> 220.7 s, 2.04x on a 7.6 s clip.
  All 2,664 frame/track IDs, image boxes and attributes identical. Pitch values
  differ in 1,269 rows, max 0.303 m; unseeded calibration sampling diagnosed.
  This is not a claim of exact full-pipeline equality.
- Native full 30 s SNGS-021 validation rerun: 418.1 s, raw GS-HOTA 50.534,
  IDF1 55.175; historical raw GS-HOTA 50.529. No meaningful accuracy gain claimed.
- Optional seeded calibration replay: 750/750 matrices and final predictions
  byte-identical across two repeats. GS-HOTA 50.527, IDF1 unchanged. Seed remains
  UNSET BY DEFAULT; do not tune seeds against these validation labels.
- Projection cache: three real-data A/B repeats, median 25.723 -> 1.757 s,
  14.64x for this stage only. Matrix reads 12,260 -> 750, all outputs byte-identical.
  Actual final Docker image also produces the same output hash.
- Startup: full RF-DETR MD5 checks retained, read block 4 KiB -> 1 MiB, isolated
  hash benchmark 6.56x faster. Initial observed median init 84.600 -> 47.502 s;
  host service changes confound precise causal startup timing. All nine completed
  unprofiled tracking runs (including final release) have byte-identical CSVs and
  loaded weight hashes. No further startup retries; accepted narrow claim is faster
  complete hashing, not a precise guaranteed whole-startup speedup.
- Detailed JSON/API/viewer includes evidence, uncertainty and explicit nulls for
  unsupported metrics. Known source duration now supplies match/player coverage
  denominator: fresh clip 183/190 frames = 96.32%, not observed-span 100%.
- CLIP full-checkpoint loader accepted at both call sites. Component state/logits/
  preprocessing exact; native seeded 7.6 s pair 195.2 -> 173.4 s, all 2,664 final
  predictions exact, including pitch coordinates. Single short pair, not a robust
  full-match speed estimate. CLIP-process peak allocation 1,804.7 -> 1,056.4 MiB.
  See docs/CLIP_LOADING_20260909.md; no model/precision/threshold change.
- 283 tests passed in the weight-bundle publication run; Ruff passed; both
  compose configurations and both Dockerfiles validated. Earlier Docker API,
  viewer and video byte-range responses verified. Annotated video visually checked.
- GitHub publication package prepared: one-command Windows/POSIX standalone start,
  Apache-2.0 text and third-party boundary, portable `.env.example`, localhost-only
  ports, pinned Python base digest and Linux container dependency lock, install/
  contributing/security/publication docs, CI on Python 3.11/3.12 plus Docker smoke,
  Dependabot and a repository artifact/secret/path guard. Exact final container
  `football-intelligence:github-ready` (sha256:a87ff189c9afe026e96e0c83a82a84315419a33b97e6ab0b1e24dc41d6584a8f)
  passed health smoke and `pip check`; cached rebuild completed in 0.2 s.
- Owner-confirmed previous Football Core is integrated under `services/perception`:
  420 runtime files, 416 byte-identical to the accepted CLIP release image. The four
  declared changes make Qwen paths portable, remove two hidden stale-bytecode imports
  dependency and enforce exact SHA-256 checks for the three required checkpoints.
  `SOURCE_PROVENANCE.json` records every file digest.
  No weights, raw footage, SoccerNet archives, bytecode or compiled YOLOX binaries
  are included; YOLOX imports through its verified Python fallback.
- Source-built perception image `football-intelligence-perception:0.1.0`
  (sha256:2ca9de31729278991f45461a2194e6999a02e5b4af639bb527f79402c9dc5d8c)
  passed `pip check`, offline main-inference import, 420-file image/source hash
  comparison and an HTTP health smoke. Without GPU/weights it truthfully reports
  `degraded`; a fresh GPU inference was not possible because the Docker adapter vanished.

## Failing / unknown

- No new SOTA claim. Inherited refined macro GS-HOTA 55.998 on four sequences
  remains in ACCURACY.md; it was not rerun by the new native single-sequence test.
- No 90-minute GPU A/B. Fresh complete calibration/tracking/IDATR pair now exists
  for the CLIP release; do not multiply separately measured component speedups.
- No reviewed semantic-event gold set; five events in the fresh clip are estimates,
  not verified passes/shots. Unsupported PDF metrics remain null.
- Attribute refinement still takes 58.4 s on the native development clip; profile
  repeated crop preparation/encoding before selecting the next optimization.
- Batching raises analysis peak VRAM to 13,493 MiB; overall peak ~14,355 MiB on
  a 16 GB GPU. Do not co-start a large resident VLM without a memory budget.
- No Git remote is configured, so nothing has been pushed. The owner confirmed rights
  to combine the original app/core code. Upstream notices remain mandatory, and model
  weights, datasets and match footage remain governed external assets.

## Artifacts and reproducibility

- Canonical protocols: docs/PERFORMANCE_20260908.md and docs/STARTUP_20260908.md;
  EXP-20260908-01 through 05. Latest artifacts data/runtime/startup_20260908/
  {comparison.json,release_verification.json,baseline1,candidate1,candidate2,baseline2,
  releasecheck,profile,weight_io,launches}. All managed GPU benchmark jobs COMPLETE.
  observations.json references all nine run manifests by SHA-256. Audit JSONs show
  host service changes (infra restart/stop, RAG start/stop); supplemental_observation.json
  is observational, not a controlled pair. Do not chase further whole-startup
  timing pairs under changing workload or stop user services.
- GPU A/B: runs/gpu_20260908_{baseline,optimized}/ and
  runs/gpu_20260908_comparison.json. Jobs 7c29109a9578 / 81ed12f471ba.
- SoccerNet native job b4a3bb9b1cb1: runs/gpu_20260908_SNGS-021/;
  runs/gpu_20260908_SNGS-021_metrics.json. Official evaluator commit
  9c25232f6f2b56c9f203f1eb55784ff1e97df683.
- Seed replay: data/runtime/calibration_replay_20260908/SNGS-021/replay_manifest.json;
  runs/calibration_seed_20260908_metrics.json. COMPLETE, no active GPU jobs.
- Projection: data/runtime/projection_benchmark_20260908/SNGS-021/
  {comparison.json,packaged_verification.json}; source/config/output hashes saved.
- Fresh user artifacts: runs/gpu_20260908_optimized/
  {annotated.mp4,statistics.json,match_report.json,viewer.html,clip_web.mp4}.
  ARG–FRA short fragment, not Orenburg–Akron and not a full match.
- JSON contract: docs/REPORT_SCHEMA.md, configs/report_metrics.json;
  metric definition reference-report-v2, schema 1.0.0.
- Previous core / startup baseline: football-core:performance-20260908,
  sha256:f444a3fee87e9662c9585107f1cd9c95048651ca1aafbdc6e542f03eb9e94109.
- CLIP baseline: football-core:startup-release-20260908,
  sha256:ca763f2e495376aaffb53ec62399e048af0608485d35e0f895504f1ea77d97a6.
- Final core: football-core:clip-release-20260909,
  sha256:7a4900f1893fb6e71ea294c84beeeaaad5b0c6dcbc1bf7c128b29d4d0a36b4ad.
- Final app: football-intelligence:performance-20260908,
  sha256:e32d0b7f1d69e98f8ebd822c82ec98f15da83bd32def7991f6f9f9e10762dd9e.
- `services/perception/Dockerfile` now builds the optimized service directly from
  integrated source with pinned CUDA/Python dependencies and external `/opt/weights`.
  Historical `docker/Dockerfile.core-clip` and immutable image IDs remain only for
  benchmark provenance. Caches were reused without new model downloads at
  data/runtime/gpu_20260908/cache.
- Baseline git 7c485be97daa34892b1d9581426211e5ccc08bed; dirty source fingerprints
  and model weight hashes saved in run manifests. Do not reset the worktree.

## Next exact action

CLIP experiment COMPLETE: data/runtime/clip_loading_20260909/ contains two
component pairs, native_comparison.json, release_source_equivalence.json and
native_commands.json. Native runs: runs/clip_native_{baseline,candidate}_20260909.
Both owned native containers stopped; no managed GPU benchmark still running.
Production compose promoted to the tested complete release. FG_CALIB_SEED stays
unset by default (seed 0 was explicit in native comparison only).

GitHub-preparation milestone COMPLETE locally. Publication check passed for 606
candidate files; Git history has no large blobs (largest observed <70 KiB) and the
targeted history secret scan returned no findings. Wheel includes LICENSE, static
UI, roster JSON and VLM prompt. Clean standalone Docker health is truthful:
status=ok, perception=not_configured, semantic_vlm.configured=false. The owned
  quick-start/smoke containers were removed after testing; unrelated services untouched.

Offline perception-weight distribution is implemented and exercised end to end.
`configs/weights-manifest.json` pins nine files (10.5 GiB); pack/install verifies
path safety, embedded manifest, byte sizes and SHA-256. The local ignored artifact is
`.build/football-intelligence-weights-2026-09-09.tar` (11,277,219,840 bytes), SHA-256
`67ce4c4fefb197e5078e377af42bf5ad21883ef12a9fd87b132de2eac907503f`.
All nine files were installed to a separate directory and reverified; that temporary
copy was then deleted. The working ignored `weights/` source and bundle remain.

Before any public push: review all dirty/untracked project changes and upstream
notices, then follow docs/PUBLISHING.md. Do not add a remote, commit, or publish
without the owner's chosen GitHub URL and visibility.

1. Profile crop decoding/repeated feature preparation in gen_tracklets and
   tracklet_attributes on the existing development clip. Choose one measured
   hotspot; do not assume a cache/reuse path is safe without full input provenance.
2. Use current CLIP release as baseline, fresh output paths and the same config.
   Prove exact intermediate/final outputs and log compute; retain failed hypotheses.
   Do not introduce resident multi-model workers on the current VRAM budget.
3. Build reviewed semantic-event validation examples before expanding automatic
   PDF-style counts. Current JSON breadth does not establish semantic accuracy.

## Resume commands

~~~powershell
python -m pytest
python -m ruff check .
docker build -f services/perception/Dockerfile -t football-intelligence-perception:0.1.0 .
docker compose --profile full config --quiet
python scripts/verify_perception_source.py
python scripts/weights_bundle.py verify --weights-dir weights
~~~

Latest benchmark commands and limitations: docs/CLIP_LOADING_20260909.md.
Do not overwrite prior inference artifacts or start the stopped legacy benchmark
container to recover optimizations; the reproducible overlays now contain them.
