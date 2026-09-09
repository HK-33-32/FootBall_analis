# GPU perception reproducibility — 8 September 2026

## Scope

The default legacy image `football-core:1.0.0` did **not** contain the batching,
threaded crop preparation and shared ReID changes previously measured in
ACCURACY.md. Those changes were only in an external working directory and the
writable layer of stopped container `football-core-benchmark`. A new code-only
Docker overlay freezes those seven files, validates before/after SHA-256 hashes,
and refuses a different source revision. No weights or data are added to the layer.

Base image: `sha256:17f66f8619db3017c50dd5c031b4d1c9aaf2ccbab027ee802532acada5d52720`.
Optimized image: `football-core:optimized-20260908`.
Exact source hashes and reviewable patch: `docker/core-overlay/manifest.json`
and `optimizations.patch`. This is a compatibility layer over the locally
available legacy image, not a new redistributable weight-free base runtime.

## Protocol

- ARG–FRA `ARGFRA-chunk0000.mp4`, first 7.6 seconds, 190 frames, development data.
- RF-DETR large, 25 FPS, 1080p, calibration stride 1, jersey stride 3,
  CLIP plus explicitly selected Qwen2.5-VL reader. No model/threshold changes.
- RTX 5080 Laptop 16 GB, Docker Linux CUDA runtime; one inference job at a time.
- Both runs reuse identical local CLIP, Torch and RF-DETR caches. Model inventories
  are hashed before submission; effective checkpoint paths (including OSNet's
  `.pth.tar-60`) are hashed separately. Preparation/hashing is outside job time.
- Elapsed time includes extraction, analysis, stitching, VLM and projection;
  excludes daemon startup, report generation, rendering and profiling exports.
- Job ID is persisted immediately. `--resume` observes the existing job and never
  silently submits another. Raw telemetry, config, predictions and metadata remain
  under each run directory.
- Comparison ignores only job-specific `id`, `image_id`, `video_id`; explicit frame,
  track identity, image/pitch coordinates and every attribute remain checked.
- Legacy seed is uncontrolled and explicitly recorded as null. A single paired
  timing is not a confidence interval or an accuracy score. Background host
  services/cache order and brief development checks can affect wall time.
- GPU utilization/VRAM are sampled whole-device values, not CUDA kernel time.
  In-module timers are wall times and need not cover import/model initialization.

## Reproduction

Build from the known local baseline image; weights are reused, not downloaded:

```powershell
docker build -f docker/Dockerfile.core -t football-core:optimized-20260908 .
$env:CORE_PROFILE_IMAGE="football-core:optimized-20260908"
$env:CORE_PROFILE_CACHE="./data/runtime/gpu_20260908/cache"
docker compose -f docker-compose.benchmark.yml up -d
python scripts/profile_core.py --core-url http://localhost:8012 --container fi-core-profile --local-media data/perception_benchmark/media/ARGFRA-chunk0000/ARGFRA-chunk0000.mp4 --media-path /media/ARGFRA-chunk0000/ARGFRA-chunk0000.mp4 --output runs/core-profile-new
```

For a baseline run select `CORE_PROFILE_IMAGE=football-core:1.0.0`; use separate
output directories and do not run both jobs concurrently. For SoccerNet use
`--dataset SNGS-021 --split validation --duration 30`, local video
`data/soccernet/valid/SNGS-021/SNGS-021.mp4` and container path
`/soccernet/SNGS-021/SNGS-021.mp4`. No frozen-test tuning.

```powershell
python scripts/compare_core_profiles.py --before runs/gpu_20260908_baseline --after runs/gpu_20260908_optimized --output runs/gpu_20260908_comparison.json
```

Do not regenerate the overlay from an arbitrary newer source directory. The
packaging script deliberately refuses to overwrite an existing release.

## Results: ARG–FRA development A/B

| Stage | Baseline, s | Packaged optimizations, s |
|---|---:|---:|
| Extraction | 2.0 | 2.0 |
| Calibration + detection/tracking | 176.1 | 109.2 |
| Stitching + attributes + projection | 271.2 | 109.0 |
| Complete job elapsed | 450.3 | 220.7 |

Observed speedup: **2.04x**. One pair, not a full-match estimate. All 2,664 rows
retain exactly the same frame/track IDs, image boxes, roles, jerseys and teams.
**Full JSON equality fails**: 1,269 rows have different pitch projections. Across
all pitch-coordinate components, median absolute delta is 0 m, p90 0.02188 m,
p99 0.07344 m, maximum 0.30270 m. Do not describe this as bit-exact accuracy retention.

The untouched `engine/kpts.py` samples circle points using `np.random.choice`
without a fixed seed. Its source hash is identical in both images. Executing that
real sampler 20 times on one explicitly synthetic diagnostic heatmap produced
20 distinct outputs; resetting seed before each call produced one. This supports
pre-existing calibration randomness as a cause of projection drift, but is not
an annotated accuracy measurement or proof that it is the only source.
No production seed/threshold was changed or tuned to validation labels.

Peak sampled VRAM: baseline 14,359 MiB, optimized 14,355 MiB. During analysis,
however, batching raises the sampled peak from 8,053 to 13,493 MiB. Another large
resident VLM must not be co-launched on this 16 GB GPU. Analysis GPU utilization
sample mean rises from 9.49% to 20.96%; stitching from 8.55% to 16.73%.

Optimized in-loop tracking timers: detection 6.8 s, ReID 6.9 s, jersey 7.3 s,
tracker 0.2 s. They total 21.2 s of a 109.2 s process stage; the remaining time
includes imports, initialization and preparation, not yet separated by a profiler.
IDATR: tracklet generation 14.7 s, attributes 74.9 s, court projection 7.1 s.

Artifacts: `runs/gpu_20260908_{baseline,optimized}/`,
`runs/gpu_20260908_comparison.json`,
`runs/gpu_20260908_calibration_randomness.json`. The simulation is explicitly
labeled and never used as production predictions or match ground truth.

## SoccerNet validation

SNGS-021, 750 source frames: new full GPU job completed in **418.1 s**.
Pinned official evaluator `9c25232f6f2b56c9f203f1eb55784ff1e97df683`:

| Raw perception | GS-HOTA | IDF1 | Person detections | Identities |
|---|---:|---:|---:|---:|
| Historical saved baseline | 50.529 | 55.175 | 11,341 | 30 |
| Packaged optimized runtime | 50.534 | 55.175 | 11,341 | 30 |

11,894 person GT observations, 23 GT identities. Roles/teams/jerseys enabled,
ball excluded per the official benchmark. This is raw perception, not the
multi-component refined score or the previous four-sequence macro average.
The +0.005 GS-HOTA difference is **not** claimed as accuracy improvement;
uncontrolled calibration and historical-artifact comparison remain limitations.

Artifacts: `runs/gpu_20260908_SNGS-021/`,
`runs/gpu_20260908_SNGS-021_metrics.json`. `scripts/evaluate_core_milestone.py`
records prediction/annotation hashes, evaluator commit and the runtime manifest.

## Follow-up: deterministic calibration

An optional adapter isolates circle sampling using `circle-pcg64-v1` and the
explicit seed in `configs/core_reproducible.env`. Without `FG_CALIB_SEED` the
adapter preserves legacy global-RNG sampling exactly. Seed 0 was selected before
label evaluation; it is not a tuned accuracy threshold. Two real calibration /
projection replays with fixed native SNGS-021 tracks are complete.

Image `football-core:calibration-20260908` adds this adapter over the optimized
runtime. `scripts/replay_core_calibration.py` clones a completed single-segment
job into a new experiment directory, preserves per-repeat matrices/predictions,
and supports `--resume` with fingerprint checks. It does not rerun or fabricate
detections, jersey reads or tracks.

Both passes produced **750/750 byte-identical matrices and byte-identical final
prediction JSON**. Calibration wall times: 135.72 and 141.56 s, not end-to-end times.
Official GS-HOTA: 50.534 (unseeded input run) -> 50.527 (fixed seed), IDF1 unchanged
at 55.175. This is reproducibility, not an accuracy gain. Seed tuning on these
labels was not performed; the adapter remains opt-in and the default retains
the legacy RNG. Artifacts: `data/runtime/calibration_replay_20260908/SNGS-021/`,
`runs/calibration_seed_20260908_metrics.json`.

## Projection I/O cache: controlled A/B and packaged verification

The legacy projection loaded a frame's identical `.npy` once per player. A cache
local to one sequence projection now reads each matrix once; it does not persist
across calls, clips or calibration reruns. No arithmetic, coordinates, filtering
or identity logic changed.

Real fixed SNGS-021 matrices and tracks, three repeats with alternating order:

| Variant | Wall-time samples, s | Median, s | Matrix loads/run |
|---|---|---:|---:|
| Baseline | 25.723, 56.815, 23.666 | 25.723 | 12,260 |
| Cached | 1.757, 2.046, 1.662 | 1.757 | 750 |

**14.64x stage speedup**, all six output files byte-identical. The baseline's
second-run outlier is retained, not discarded. This is projection only, not a
new full-job speedup. Benchmark candidate source was then packaged unchanged:
the actual final Docker image produced the same output hash in 2.165 s.

Protocol/script: `scripts/benchmark_core_projection.py` (real data, no model
inference); results, complete provenance and output files:
`data/runtime/projection_benchmark_20260908/SNGS-021/`.
The patch is only exported when equality and speed acceptance both pass.

Final core image `football-core:performance-20260908`:
`sha256:f444a3fee87e9662c9585107f1cd9c95048651ca1aafbdc6e542f03eb9e94109`.
Build with `docker build -f docker/Dockerfile.core-performance -t football-core:performance-20260908 .`.
Compose now selects this image and correctly shares API media with perception.
`FG_CALIB_SEED` remains unset by default; `configs/core_reproducible.env` enables
the separately measured deterministic adapter explicitly.

## Reporting and application verification

- Coverage correction: 183/190 source frames = 96.32%, not 100% of only the
  observed span. Match/player percentages share the source denominator.
- Metric definition `reference-report-v2`; unknown source duration is labeled.
- 225 tests passed, Ruff passed, both compose files validated.
- Final application image `football-intelligence:performance-20260908`, ID
  `sha256:e32d0b7f1d69e98f8ebd822c82ec98f15da83bd32def7991f6f9f9e10762dd9e`.
- Docker statistics endpoint and viewer return 200; source video range returns
  206/1024 bytes. JSON reports 0.9632/source_duration/7600 ms.
- Fresh ARG–FRA outputs: `runs/gpu_20260908_optimized/statistics.json`,
  `viewer.html` (self-contained), `annotated.mp4` (190 frames, 7.6 s, 3 MB).
  Annotation frame visually checked; predictions explicitly labeled unverified.
  Five estimated events, not verified semantic match statistics or a full match.
