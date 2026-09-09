# CLIP direct-checkpoint experiment — 2026-09-09

## Hypothesis and scope

The production tracker and tracklet-attribute stage both first construct base
ViT-L/14 from its 932,768,134-byte archive, then overwrite the entire state from
the trained checkpoint. Remove only the redundant archive path. No threshold,
precision, preprocessing, model, prompt or sampling changes.

The trained checkpoint contains all 460 state entries, including the backbone
and heads. Its SHA-256 is
`c068865c6ff28ff8b2dd7e90d2cfd37b72dfe4e117a3be79eb88a5f98e8afa1e`.
The new loader reads it once with `weights_only=True`, validates ViT-L/14 shapes
and depth, constructs the original upstream CLIP model and performs strict full
state loading. Freeze flags and upstream preprocessing are retained. Unsupported
architectures/incomplete checkpoints fail; no download or silent fallback.
The explicitly invoked legacy constructor remains available.

## Component protocol and results

Development source: existing ARG–FRA startup baseline, eight predetermined frames,
two Player crops per frame (16 total), plus red/blue/white text probes. Both modes
run the same saved benchmark script in fresh GPU containers, network disabled,
using cached weights. Image/model/software/hardware/config and input hashes are
in every manifest. Timed load includes device transfer and CUDA synchronization;
fingerprints and checkpoint audit are outside that interval.

| Execution order | Baseline load, s | Candidate load, s | Ratio |
|---|---:|---:|---:|
| A then B | 54.655 | 20.636 | 2.649x |
| B then A | 23.499 | 19.671 | 1.195x |

Both comparisons are exact for all 460 state values/shapes/dtypes, preprocessed
input tensors, classifier logits, color and text embeddings, training and
requires-grad flags. The first baseline is substantially slower: cache/order and
host workload affect wall time. Two pairs do not establish a robust confidence
interval or a universal speedup. Unrelated user infrastructure was running during
this session and was not stopped. These are load timings, not new accuracy scores.

Both pairs also agree on CLIP-process PyTorch peak allocation: 1,804.728 MiB
baseline versus 1,056.443 MiB candidate (41.46% less); peak reserved memory
1,820 versus 1,162 MiB. This covers loading and the probe, not whole-pipeline or
whole-device VRAM. The candidate stages checkpoint tensors on CPU instead of
temporarily restoring an additional CUDA copy.

Evidence: `data/runtime/clip_loading_20260909/`, four run folders and
`pair1.json`, `pair2.json`. Each run snapshots the benchmark script.

## Images and reproducibility

- Baseline `football-core:startup-release-20260908`:
  `sha256:ca763f2e495376aaffb53ec62399e048af0608485d35e0f895504f1ea77d97a6`.
- Experiment `football-core:clip-direct-20260909`:
  `sha256:d0307f7950ab6936eead4e9d69b0cf3389cbc89f59a14d18fd53f727428de030`.
- Complete build `football-core:clip-release-20260909`:
  `sha256:7a4900f1893fb6e71ea294c84beeeaaad5b0c6dcbc1bf7c128b29d4d0a36b4ad`.

The complete build applies all checked code overlays over the existing legacy
base (see `docker/Dockerfile.core-clip`), without copying new weights/datasets.
All 485 Python/YAML runtime files in engine/core/RF-DETR match the experiment image
byte-for-byte; `release_source_equivalence.json` records both inventories.
The legacy base already includes some weights; external cache mounts are reused.
Git: `7c485be97daa34892b1d9581426211e5ccc08bed`, dirty source recorded in manifests.

```powershell
docker build -f docker/Dockerfile.core-clip -t football-core:clip-release-20260909 .
python scripts/compare_clip_loading.py --baseline data/runtime/clip_loading_20260909/baseline1/manifest.json --candidate data/runtime/clip_loading_20260909/candidate1/manifest.json --output data/runtime/clip_loading_20260909/pair1_recheck.json
```

Use new output paths; never overwrite original evidence. Following the native
acceptance below, the full build is now the default in both compose files.

## Native integration — accepted

Same 7.6 s / 190-frame ARG–FRA development source, RF-DETR large, CLIP, same cached
checkpoints and FPS/resolution. Explicit `FG_CALIB_SEED=0` for both runs controls
circle sampling only; no claim of a unified pipeline seed. Seed remains unset in
production. Benchmark harness: `scripts/profile_core.py`; comparison:
`scripts/compare_core_profiles.py`. Full native jobs include calibration, tracking
and attribute refinement. Frozen tests and SoccerNet splits are not used to tune
this behavior-preserving loader.

Baseline `runs/clip_native_baseline_20260909` (job `35a9ea2a2da1`), candidate
`runs/clip_native_candidate_20260909` (job `795100e6b25c`). The candidate run uses
the complete release image directly. Both completed successfully. All 2,664
predictions agree exactly, including image boxes, track identities, team/jersey
attributes and pitch coordinates (maximum delta 0 m). Only job-specific transport
IDs are ignored. Raw tracking CSVs are also byte-identical (SHA-256
`c5733f13bf71318b4c28d13400507f275afd7d1869e123d39a71e93f9fed8a12`).

| Wall time | Baseline, s | Candidate, s |
|---|---:|---:|
| Complete job | 195.2 | 173.4 |
| Calibration/tracking concurrent stage | 95.3 | 79.6 |
| Complete identity/attribute/projection stage | 97.0 | 91.2 |
| Attribute refinement substep | 72.8 | 58.4 |

Complete job observation: 1.126x, 11.17% less elapsed time. Nested/concurrent times
must not be added. Both load sites were exercised; downstream local Qwen jersey
inference was retained. No new remote fallback was introduced. Same media, config,
model inventories/effective checkpoint hashes and recorded runtime environment.
Single A/B pair on a shared host: no confidence interval or causal full-match claim.
Calibration warnings/limited source coverage remain baseline issues, not repaired
by this loader. Exact output equality does not establish correct football semantics.

Machine-readable result: `data/runtime/clip_loading_20260909/native_comparison.json`.
Exact launch commands: `native_commands.json` beside it. The two owned native
containers were stopped after collection; artifacts and unrelated services retained.

```powershell
python scripts/compare_core_profiles.py --before runs/clip_native_baseline_20260909 --after runs/clip_native_candidate_20260909 --output data/runtime/clip_loading_20260909/native_recheck.json
python -m pytest -ra
python -m ruff check .
docker compose --profile full config --quiet
docker compose -f docker-compose.benchmark.yml config --quiet
```

Verification: 275 tests passed; Ruff passed. Overlay checks cover helper drift and
both call sites; real GPU evidence covers full-state strict loading and inference.
Accepted as a behavior-preserving startup/memory optimization, not an accuracy gain.
