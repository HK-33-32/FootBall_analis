# Experiments

## EXP-20260826-01 — Executable architecture smoke suite

Hypothesis: The new contracts can preserve evidence, abstention and identity constraints before expensive model inference.
Change: Initial typed implementation, active re-analysis router, Match Memory, API/UI and Docker definition.
Dataset/split: Test-only synthetic fixtures; no football accuracy claim.
Config: `configs/default.yaml`.
Model revision: No model inference in this experiment.
Git commit: uncommitted initial repository.
Command: `pytest`.
Metrics: 14/14 tests passed; statement coverage 68%; Ruff passed with zero findings.
Compute: CPU only.
Conclusion: Contracts, evidence integrity, active re-analysis and caching are executable. This is not an accuracy result.
Artifacts: pytest output in session log.
Next: Benchmark pinned Qwen3-VL on a reviewed, non-frozen semantic gold set.

## EXP-20260826-02 — Legacy development artifact identity audit

Hypothesis: Team and jersey evidence can reconnect obvious non-overlapping fragments without unsafe identity collapse.
Change: Imported a completed 60-second ARG–FRA development artifact from the supplied legacy project.
Dataset/split: Supplied development artifact; not a frozen benchmark.
Config: `configs/default.yaml` identity weights.
Model revision: Precomputed external perception; no new model inference.
Git commit: uncommitted initial repository.
Command: `football-intelligence import-legacy ...`.
Metrics: 36 tracklets, 21 semantic candidates. Initial solver produced 23 players and exposed 13 merges, 12 of them based on team alone. After adding the evidence gate: 35 players, one merge (`10+39`) supported by team plus jersey 8, 143 team-only pairs rejected.
Compute: CPU, < 3 seconds excluding artifact generation.
Conclusion: Team alone is not identity evidence; the gate prevented aggressive false linking. Recall of the remaining linking requires ReID/GT evaluation.
Artifacts: `runs/legacy_arg_fra_import_v2.json` (ignored runtime artifact).
Next: Ingest ReID embeddings and score ID switches/track fragmentation on validation annotations.

## EXP-20260826-03 — Docker API/frontend smoke

Hypothesis: The low-compute control plane is reproducible in a clean Linux container.
Change: Built and started `football-intelligence:0.1.0`.
Dataset/split: None.
Config: `configs/local_16gb.yaml`.
Model revision: Models disabled for control-plane smoke.
Git commit: uncommitted initial repository.
Command: `docker compose build api && docker compose up -d api`.
Metrics: Container health `healthy`; `/api/v1/health` HTTP 200; `/` HTTP 200 with frontend title.
Compute: CPU.
Conclusion: Dockerized API/UI path is verified. The GPU `full` profile is configured but its large perception/VLM images were not pulled or benchmarked in this session.
Artifacts: Docker image `football-intelligence:0.1.0`.
Next: Build the legacy perception image and run a pinned Qwen3-VL validation subset.

## EXP-20260826-04 — Supplied ARG–FRA clip inference export

Hypothesis: The supplied Football Core run contains a real rendered result and sufficient provenance to serve as a reproducible development baseline.
Change: Verified the exact source SHA-256, inspected the completed GPU job, exported all artifacts, and visually checked a rendered frame.
Dataset/split: User-provided 33.58-second clip, Argentina–France 2022; unlabelled development media.
Config: RF-DETR large, 1080p at 12 FPS, Qwen2.5-VL-7B Q8 jersey reader, starting-XI roster, render and analytics enabled.
Model revision: GGUF filenames and effective config recorded; upstream weight commit unavailable.
Git commit: uncommitted initial repository.
Command: `python scripts/export_legacy_run.py ...`.
Metrics: 403 frames; player detections on 340; ball on 322; 52 heuristic events; 16 identified players, 14 anonymous fragments; on-ball attribution coverage 75.8%.
Compute: 1,604.5 seconds wall time on the supplied RTX 5080 Laptop run.
Conclusion: The annotated artifact is valid and useful for debugging. Event and possession values remain heuristic because the clip has no human event labels.
Artifacts: `runs/arg_fra_2022_user_clip/`.
Next: Use an annotated SoccerNet validation sequence for perception scoring.

## EXP-20260826-05 — SoccerNet GSR v1.3 validation audit

Hypothesis: Dataset-level visibility statistics can quantify why single-crop jersey recognition saturates.
Change: Downloaded and CRC-verified the official validation archive; audited every annotation without accessing the frozen test set.
Dataset/split: SoccerNet GSR v1.3 validation, 58 sequences.
Config: `SoccerNet==0.1.62`; audit script in `scripts/soccernet_validation_audit.py`.
Model revision: No model inference.
Git commit: uncommitted initial repository.
Command: `python scripts/soccernet_validation_audit.py --archive ... --output ...`.
Metrics: 43,500 frames, 792,166 annotations, 1,222 player/GK tracks; jersey frame coverage 83.60%; jersey track coverage 79.05%; median bbox height 8.80% of frame.
Compute: CPU, full archive CRC and JSON scan under two minutes.
Conclusion: Whole-track and cross-track evidence are necessary for coverage; a narrow active lineup is a useful configurable prior. This is not an accuracy result.
Artifacts: `runs/soccernet_valid_audit.json`; archive SHA-256 `d60e517e...3047a50`.
Next: Run the perception model on a deterministic validation sequence and compute official GS-HOTA.

## EXP-20260826-06 — Official GS-HOTA evaluator smoke

Hypothesis: The pinned official evaluator and local validation layout produce a trustworthy metric before model scoring.
Change: Cloned `SoccerNet/sn-trackeval` at `9c25232f6f2b56c9f203f1eb55784ff1e97df683` and evaluated an explicitly labelled oracle conversion of SNGS-033.
Dataset/split: SoccerNet GSR v1.3 validation, SNGS-033 only.
Config: Official pitch-space Gaussian GS-HOTA, 5 m tolerance, role/team/jersey enabled, ball ignored.
Model revision: Oracle harness only; no model.
Git commit: uncommitted initial repository.
Command: `python data/tools/sn-trackeval/scripts/run_soccernet_gs.py ... --SEQ_INFO SNGS-033`.
Metrics: GS-HOTA 100, DetA 100, AssA 100, IDF1 100; 7,707 detections and 25 identities matched exactly.
Compute: CPU, 0.93 seconds.
Conclusion: Evaluator wiring is valid. The 100 score is only an oracle smoke test and must never be reported as model performance.
Artifacts: `data/soccernet/eval/trackers/SoccerNetGS-valid/oracle/person_summary.txt`.
Next: Replace oracle predictions with Football Core SNGS-033 output.

## EXP-20260826-07 — Football Core on SoccerNet validation SNGS-033

Hypothesis: The supplied RF-DETR + IDATR + Qwen jersey pipeline can reproduce a
competitive official GS-HOTA result, and controlled attribute ablations can
identify its highest-leverage failure.
Change: Built the CUDA Docker image from local pinned weights, ran all 750 frames
at native 25 FPS, evaluated the untouched predictions with the pinned official
evaluator, and added team/jersey diagnostic ablations.
Dataset/split: SoccerNet GSR v1.3 validation, SNGS-033 only; frozen test untouched.
Config: RF-DETR large at 1080p/confidence 0.35, calibration stride 1, IDATR,
CLIP + Qwen2.5-VL-7B Q8 jersey reader at stride 3.
Model revision: Exact SHA-256 values in
`runs/soccernet_valid_sngs033_core/benchmark.json`.
Git commit: uncommitted initial repository.
Command: `python scripts/run_soccernet_core.py ...`; then
`python scripts/evaluate_soccernet_gs.py ...`.
Metrics: Official full GS-HOTA 38.617, DetA 20.859, AssA 71.508, IDF1 33.648;
6,998 detections/38 IDs vs 7,707 detections/25 IDs. Without team/jersey matching:
GS-HOTA 70.221, DetA 76.931, AssA 64.157, IDF1 73.472. Team-only GS-HOTA 45.539;
jersey-only 51.862.
Compute: RTX 5080 Laptop GPU; 1,554.7 s cold-start wall time (6.8 extract,
809.1 analyze, 736.5 IDATR); ~890 MB RF-DETR backbone fetched on first launch.
Conclusion: Geometry is substantially stronger than the full identity score.
Team assignment collapsed to 5,382/861 observations versus GT 2,641/4,112,
making TEAM_SWAP/TEAM_COLLAPSE the primary measured failure. Added a configurable
Player Memory quarantine for globally imbalanced team evidence; it must be
validated on more sequences before any improvement claim.
Artifacts: `runs/soccernet_valid_sngs033_core/benchmark.json`, predictions and
official summaries under `data/soccernet/eval/trackers/SoccerNetGS-valid/`.
Next: Validate team-evidence quarantine and team clustering on multiple validation
sequences, then address jersey errors without tuning the frozen test.

## EXP-20260907-01 — Exact-output CPU/video acceleration

Hypothesis: Repeated GOP seeks, scalar trajectory predecessor scans and unused
Spearman p-values waste time without contributing information.
Change: Sequential skip/grab video reads; vectorized bounded reach scan; reused
average ranks with SciPy-identical matrix element. No model/FPS/threshold change.
Dataset/split: SoccerNet GSR v1.3 valid 021/033/045/090; existing unlabelled ARG–FRA
clip; additional fixed native-seek window 350–381 on SNGS-021. Frozen test untouched.
Config: Three timed repeats plus separate untimed profile; resolved config/hashes
and exact input checksums in each run manifest. Full-sequence video equality uses
the explicitly labelled independent sequential-reference decoder.
Model revision: No model invocation; immutable perception outputs hashed in manifests.
Git commit: Baseline 7c485be97daa34892b1d9581426211e5ccc08bed; dirty current source hashes saved.
Command: scripts/benchmark_postprocessing.py and scripts/summarize_postprocessing.py;
full commands and methodology in docs/PERFORMANCE_20260907.md.
Metrics: All ten paired comparisons have identical output hashes. Calibration +
statistics improved 2.24–2.61x over four validation sequences. Native decoder
window 6.634 → 0.457 s; native user-clip refinement 11.120 → 0.567 s (v2 protocol).
Compute: CPU, Python 3.12.10, NumPy 2.2.6, SciPy 1.16.1, OpenCV 4.12.0.88;
per-run samples/hardware identifier saved. No end-to-end GPU timing claim.
Conclusion: Preserve outputs while reducing postprocessing cost. Differential
tests exposed and fixed a one-ULP correlation difference. Exhaustive repeated
native 1080p seek baseline was canceled as wasteful; incomplete timing not reported.
Artifacts: runs/performance_20260907/comparison.json, *_v2.json, *.prof.
Next: Measure GPU stage costs, then controlled persistent-worker A/B with VRAM checks.

## EXP-20260907-02 — Evidence-aware reference-style JSON

Hypothesis: A broad provider-style schema can preserve uncertainty rather than
fabricate unsupported metrics from single-camera geometry.
Change: Add typed annotations and detailed_statistics, team/player/period metrics,
passing networks, spatial maps, config/evidence provenance and API/UI export.
Dataset/split: Supplied PDF is a content reference only; existing ARG–FRA clip
demonstrates real output. Synthetic reviewed events exist only in unit tests.
Config: configs/report_metrics.json; report/annotation schemas 1.0.0.
Model revision: No new model; xG/xA require explicit external model identity.
Git commit: 7c485be plus working changes; no baseline accuracy values overwritten.
Command: scripts/build_match_report.py; pytest; docker build; Docker API and
browser download, independently compared as parsed JSON.
Metrics: 198 tests passed; Ruff passed. Real output has 4 estimated events and
33 non-referee identities. Unsupported metrics remain null. Unknown-team events
cannot manufacture verified team zeroes; unknown outcomes cannot produce 100% passes.
Compute: Expanded-report timing is stored separately in
runs/performance_20260907/SNGS-021_expanded_statistics.json; includes the new
builder but not GPU inference, disk output or JSON serialization time.
Conclusion: Schema/UI/evidence correctness verified; semantic accuracy not measured.
Artifacts: runs/reference_report_20260907/{statistics,match_report}.json;
docs/REPORT_SCHEMA.md; Docker verification in docs/PERFORMANCE_20260907.md.
Next: Obtain reviewed semantic-event validation data before automatic enrichment.

## EXP-20260908-01 — Package and measure the actual optimized GPU runtime

Hypothesis: Previously optimized code was absent from the legacy image actually
selected by compose, hiding useful batching/preparation improvements.
Change: Hash-checked seven-file overlay; checkpointed native profiling. No model,
FPS, precision or threshold change. External source and weights left untouched.
Dataset/split: Unlabelled ARG–FRA development clip, 190 frames/7.6 s; SoccerNet GSR
validation SNGS-021 milestone, 750 frames/30 s. Frozen test untouched.
Config: Same effective A/B config and weight hashes; complete manifests in run folders.
Model revision: RF-DETR/CLIP/jersey reader weight fingerprints and runtime source
hashes saved by scripts/core_runtime_manifest.py; image IDs in performance report.
Git commit: 7c485be97daa34892b1d9581426211e5ccc08bed plus fingerprinted dirty changes.
Command: scripts/profile_core.py, scripts/compare_core_profiles.py,
scripts/evaluate_core_milestone.py; commands in docs/PERFORMANCE_20260908.md.
Metrics: Total 450.3 -> 220.7 s (2.04x), identical 2,664 frame/track IDs, image
boxes and attributes. 1,269 pitch-only differing rows, max 0.303 m. Native valid
raw GS-HOTA 50.534 vs historical 50.529; IDF1 55.175 unchanged, no improvement claim.
Compute: RTX 5080 Laptop, overall peak 14,359 -> 14,355 MiB; analysis peak increases
8,053 -> 13,493 MiB. Native SNGS-021 job 418.1 s. Single job pair, no timing CI.
Conclusion: Useful measured short-clip speedup; full JSON equality not established.
Artifacts: runs/gpu_20260908_comparison.json, runs/gpu_20260908_SNGS-021_metrics.json,
associated run directories; docs/PERFORMANCE_20260908.md.
Next: Isolate calibration randomness and profile initialization overhead.

## EXP-20260908-02 — Optional deterministic circle sampling

Hypothesis: Unseeded circle-pixel sampling contributes to projection variation.
Change: Local PCG64 sampler circle-pcg64-v1; legacy global RNG retained when seed
unset. Seed 0 selected before label evaluation, never tuned. Two fixed-track replays.
Dataset/split: Native SNGS-021 validation tracks/matrices; separate synthetic RNG
diagnostic is explicitly a simulation, not match ground truth.
Config: configs/core_reproducible.env, workers=2; replay manifest stores hashes.
Model revision: Inherited fixed predictions/model provenance from EXP-20260908-01;
no detector, ReID or jersey inference rerun. Calibration runtime image pinned.
Git commit: 7c485be plus fingerprinted dirty changes.
Command: scripts/audit_core_calibration_randomness.py, scripts/replay_core_calibration.py,
scripts/evaluate_core_milestone.py --runtime-manifest (full args in run manifests).
Metrics: 750/750 matrix files and final prediction JSON identical between repeats.
Raw GS-HOTA 50.534 -> 50.527; IDF1 remains 55.175. Small negative accuracy result.
Compute: Calibration wall time 135.72 / 141.56 s, not full-job times.
Conclusion: Reproducibility demonstrated for this replay, not accuracy improvement.
Production seed stays unset; adapter is explicitly opt-in. Other nondeterminism
has not been excluded by this experiment.
Artifacts: data/runtime/calibration_replay_20260908/SNGS-021/;
runs/calibration_seed_20260908_metrics.json.
Next: Preserve the negative result; do not search seeds against validation labels.

## EXP-20260908-03 — Cache repeated homography reads

Hypothesis: Loading the same matrix once per player wastes projection I/O.
Change: One per-sequence, per-call matrix cache, no arithmetic or filtering change.
Dataset/split: Real fixed SNGS-021 validation tracks and matrices from EXP-02.
Config: Three alternating-order baseline/candidate repeats; exact CSV output hashes.
Model revision: No new model inference; source replay and model provenance hashed
in comparison.json. Installed Docker source hash verified independently.
Git commit: 7c485be plus fingerprinted dirty changes.
Command: scripts/benchmark_core_projection.py, then --verify-packaged against final
football-core:performance-20260908; full command/config in manifests.
Metrics: Median 25.723 -> 1.757 s (14.64x stage only); reads 12,260 -> 750.
All six outputs byte-identical; actual packaged module output also identical.
Compute: CPU projection; samples 25.723/56.815/23.666 vs 1.757/2.046/1.662 s.
Packaged verification 2.165 s. Outlier retained. No full-job speedup extrapolation.
Conclusion: Promote exact-output cache into final hash-checked Docker overlay.
Artifacts: data/runtime/projection_benchmark_20260908/SNGS-021/;
docker/projection-overlay; docker/Dockerfile.core-performance.
Next: Measure startup costs; new full-job A/B needed for any new total-time claim.

## EXP-20260908-04 — Source-duration coverage and refreshed deliverables

Hypothesis: Observed-span denominator overstates coverage when leading/trailing
source frames lack predictions.
Change: Known source duration supplies match/player denominators; unknown duration
is labeled. API, CLI and UI share metric definition reference-report-v2.
Dataset/split: Fresh ARG–FRA development output, 190 source frames; unit fixtures.
Config: CLI --duration-ms 7600, configs/report_metrics.json; report schema 1.0.0.
Model revision: Fixed EXP-01 optimized predictions, no inference change.
Git commit: 7c485be plus dirty changes.
Command: scripts/build_match_report.py; scripts/render_annotated_clip.py; pytest;
Docker API/viewer/video-range checks. See docs/PERFORMANCE_20260908.md.
Metrics: Correct 183/190 = 96.32% coverage, previously observed-span 100%; 225 tests
passed, Ruff passed. Five estimated events; unsupported semantic metrics stay null.
Compute: Report/render verification only; no semantic accuracy benchmark claimed.
Conclusion: More truthful reporting and a visually inspected 7.6 s annotated clip.
Artifacts: runs/gpu_20260908_optimized/{statistics.json,match_report.json,viewer.html,
annotated.mp4}; docs/REPORT_SCHEMA.md; application Docker image performance-20260908.
Next: Reviewed semantic-event validation data before asserting PDF-like accuracy.

## EXP-20260908-05 — Profile startup and retain integrity checks with larger reads

Hypothesis: Initialization I/O accounts for much of the gap between tracking loop
counters and process-stage wall time. Profile before introducing resident workers.
Change: Diagnostic constructor/import timings and init-only cProfile; then one
production factor: RF-DETR full-file MD5 reads 4 KiB -> 1 MiB. All checks retained,
no digest memoization, model/precision/FPS/threshold or ReID diagnostic changes.
Dataset/split: Same ARG–FRA 190-frame/7.6 s development clip; real cached weights
for component A/B. No validation/frozen labels used in this startup experiment.
Config: configs/core_io.json; identical effective tracking config, regenerated
frame hashes and actual loaded weight hashes. Isolated tracking, not concurrent jobs.
Model revision: Actual checkpoint SHA-256 inventories in baseline/candidate manifests;
unchanged between all four runs. Images and upstream source hashes pinned.
Git commit: 7c485be97daa34892b1d9581426211e5ccc08bed plus fingerprinted dirty changes.
Command: scripts/run_core_startup.py, scripts/profile_core_startup.py,
scripts/benchmark_weight_io.py, scripts/compare_core_startup.py. Canonical commands
and protocol in docs/STARTUP_20260908.md; exact launcher commands under launches/.
Metrics: Three hash pairs: median 7.609 -> 1.160 s, 6.56x, same MD5. Two unprofiled
tracking pairs in A/B/B/A order: median init 84.600 -> 47.502 s (1.781x); sum of
imports/init/tracking 129.033 -> 92.684 s (1.392x). All four CSV outputs byte-identical.
Tracking alone 25.804 -> 26.171 s, not improved. No new semantic/GS-HOTA claim.
Compute: RTX 5080 Laptop, existing disk caches, fresh processes, network disabled.
PyTorch peak allocated 7,345.5 MiB/reserved 13,232 MiB for all four runs; not total
device memory. Init cProfile is diagnostic only and excluded from these A/B numbers.
Conclusion: Accept the full-integrity I/O patch. Timing variation retained, two
pairs insufficient for robust CI; no full-job or long-match extrapolation.
Artifacts: data/runtime/startup_20260908/; docker/weight-io-overlay;
docker/Dockerfile.core-startup; release image ID in docs/STARTUP_20260908.md.
Release verification: COMPLETE; full tracking CSV, loaded weights and monitored
runtime sources identical to the experiment image. release_verification.json.
Tests: 252 passed in 13.50 s; Ruff passed; both compose configurations validated.
Environment qualification: Final audit found unrelated infra services started at
17:23 UTC during the original series. Pair 2 timing is confounded by host workload;
original samples/aggregate retained as exploratory. Supplemental controls also saw
host changes across Sep 8–9: infra restarted, then stopped; simply-rag-app-1 started
before candidate4. Saved audits reject equal environments. Do not claim a precise
causal whole-startup speedup or stop user services; preserve these observations.
Final continuation: Nine unprofiled runs complete, all tracking outputs and loaded
weight hashes identical. Baseline4/candidate4 observed init 64.928/42.117 s, not a
controlled host-load pair (RAG stopped again by final audit). observations.json
references all manifest hashes and labels timing observational. Final tests: 254
passed in 11.45 s, Ruff passed. No more runtime retries; retain the isolated 6.56x
hashing result and exact-output acceptance as the defensible claims.
Next: Inspect redundant CLIP base/fine-tuned loading with strict state/dtype
equivalence before another change.

## EXP-20260909-01 — direct full-checkpoint CLIP loading

Hypothesis: Base ViT-L/14 archive is redundant because the trained checkpoint
contains all backbone/head state; removing it reduces load time and peak memory.
Change: Strict from_checkpoint loader at tracker and tracklet-attribute sites;
same upstream model builder, mixed precision, preprocessing and freeze flags.
Dataset/split: ARG-FRA development, 16 fixed real Player crops plus three text
probes; full native 7.6 s / 190-frame development clip. No frozen test use.
Config: Same model/settings, component A/B then B/A; native seed 0 only for
calibration in both runs. Production seed remains unset.
Model revision: All actual checkpoint hashes in manifests; complete CLIP state
SHA-256 c068865c6ff28ff8b2dd7e90d2cfd37b72dfe4e117a3be79eb88a5f98e8afa1e.
Git commit: 7c485be97daa34892b1d9581426211e5ccc08bed plus dirty source fingerprints.
Command: benchmark_clip_loading.py / compare_clip_loading.py; profile_core.py /
compare_core_profiles.py. Exact native launch commands and canonical protocol in
data/runtime/clip_loading_20260909/native_commands.json and docs/CLIP_LOADING_20260909.md.
Metrics: Component load A/B 54.655/20.636 s, reversed pair 23.499/19.671 s. All
460 state entries, inputs, outputs and train/freeze flags exact. Native 195.2 ->
173.4 s (1.126x observed), 2,664/2,664 final predictions exact including pitch.
Compute: RTX 5080 Laptop; cached weights; component network disabled. Both component
pairs peak PyTorch allocation 1,804.728 -> 1,056.443 MiB, not total-device memory.
Shared host and cache/order affect timing; no robust CI/full-match extrapolation.
Conclusion: Accepted narrowly as exact-output startup/memory optimization. No
semantic accuracy gain claimed. Complete Docker release tested natively; 485 runtime
files match experiment image. Compose defaults promoted, previous images retained.
Artifacts: data/runtime/clip_loading_20260909/; runs/clip_native_{baseline,candidate}_20260909;
docker/clip-loading-overlay; docker/Dockerfile.core-clip. Immutable images in report.
Tests: 275 passed in 9.97 s; Ruff and both compose configurations passed. Owned
native containers stopped after collection; unrelated user services untouched.
Next: Profile remaining crop/feature preparation in tracklet refinement before
choosing a behavior-preserving reuse optimization; no new infrastructure.

## EXP-20260909-02 — integrate the accepted Football Core source

Hypothesis: The owner-confirmed earlier core can be shipped in the monorepo and
built without the machine-local legacy image while preserving the accepted runtime.
Change: Exported only runtime source/assets from the immutable CLIP release into
`services/perception`; removed weights, datasets, bytecode, compiled YOLOX extension,
nested Git metadata and demo media. Added source-built Docker stages, pinned direct
dependencies, portable Qwen paths, two source-level ReID import repairs and SHA-256
verification for required checkpoints.
Dataset/split: No dataset or frozen annotations used; packaging/integration test only.
Config: Production calibration seed remains unset; model/precision/threshold defaults
unchanged. `runtime` uses CLIP and `runtime-vlm` optionally compiles llama.cpp.
Model revision: Required checkpoint SHA-256 values are recorded in `fetch_weights.py`;
no model file is included in the repository or image.
Git commit: dirty worktree based on 7c485be97daa34892b1d9581426211e5ccc08bed.
Command: `python scripts/verify_perception_source.py`; `docker build --check -f
services/perception/Dockerfile .`; full Docker build and offline import/pip checks.
Metrics: 420 runtime files; 416 exact against source image
sha256:7a4900f1893fb6e71ea294c84beeeaaad5b0c6dcbc1bf7c128b29d4d0a36b4ad.
Four declared changes only: portable config, two source-level import repairs that
remove a stale-bytecode dependency, and checkpoint integrity verification.
Compute: Packaging/build validation; no GPU inference rerun because Docker GPU access
was unavailable during integration. Prior accepted GPU evidence is not relabelled.
Conclusion: Source integration accepted as a reproducibility/release improvement,
not an accuracy or speed improvement. Historical overlay images remain provenance.
Artifacts: `services/perception/SOURCE_PROVENANCE.json`, Dockerfile, requirements and
lock; `scripts/verify_perception_source.py`; Windows/POSIX GPU quick starts.
Tests: 280 tests; Ruff; 601-file publication guard; Compose and Dockerfile validation;
source-built image `pip check`, offline inference import, source hashes and HTTP health.
Next: Complete the source-built image smoke checks; rerun the same development clip on
GPU when the adapter is available before replacing the immutable benchmark reference.

## EXP-20260909-03 — portable full perception-weight bundle

Hypothesis: A separate manifest-pinned TAR gives a simpler and more reproducible offline
install than committing weights, Git LFS or baking them into the perception image.
Change: Added a nine-file/10.5-GiB weight manifest, standard-library pack/verify/install
tool and short Windows/POSIX installers. Extraction is staged and validated before move.
Dataset/split: No dataset or annotations used; packaging test only.
Config/model revision: Exact file sizes and SHA-256 in `configs/weights-manifest.json`.
Git commit: dirty worktree based on 7c485be97daa34892b1d9581426211e5ccc08bed.
Command: `weights_bundle.py pack`, then install into a fresh ignored directory and verify.
Metrics: TAR 11,277,219,840 bytes; archive SHA-256
67ce4c4fefb197e5078e377af42bf5ad21883ef12a9fd87b132de2eac907503f;
9/9 extracted model files passed size and digest verification.
Compute: Local disk packaging only; no inference. Temporary installed copy removed after
verification; source weights and external `.build` bundle retained and Git-ignored.
Conclusion: Accepted for offline installation. This does not grant redistribution rights
and does not include the separate optional Hugging Face semantic-VLM cache.
Artifacts: `.build/football-intelligence-weights-2026-09-09.tar[.sha256]` (ignored),
`configs/weights-manifest.json`, `scripts/weights_bundle.py`, `scripts/install-gpu.*`.
Tests: 283 passed; Ruff; 606-file publication guard; Compose validation.
Next: Run `install-gpu.ps1` on a CUDA-enabled target; current Docker reports WSL with no
adapters, so no new GPU inference claim was made.
