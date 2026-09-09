# Postprocessing performance — 7 September 2026

## Outcome and scope

Existing outputs are preserved exactly on four SoccerNet GSR validation sequences.
The **calibration + statistics stage** is 2.24–2.61 times faster. These are CPU/video
postprocessing measurements, **not** end-to-end match acceleration or new accuracy.
Detector, weights, precision, sample rate and semantic prompts were not changed.

| Validation sequence | Original stats, median s | Optimized stats, median s | Speedup | Output hash |
|---|---:|---:|---:|---|
| SNGS-021 | 1.13 | 0.50 | 2.24x | same |
| SNGS-033 | 1.38 | 0.53 | 2.61x | same |
| SNGS-045 | 2.08 | 0.80 | 2.61x | same |
| SNGS-090 | 1.45 | 0.59 | 2.47x | same |

Inputs are each sequence's immutable `runs/soccernet_valid_*_core/predictions.json`.
The hash includes rejected-calibration diagnostics, not just event counts. Every
repeat agrees. No ground-truth labels were inspected for tuning.

## Changes

- `VideoFrames`: `grab()` skipped nearby frames, then `read()` the requested one.
  Previously even stride two repeatedly sought and decoded the same compressed GOP.
  Backwards/large seeks remain supported.
- `longest_reachable_run`: vectorized bounded predecessor scan, preserving the
  exact reach test and earliest-predecessor tie rule.
- Calibration: reused image ranks and correlated average ranks without unused
  Spearman p-values. Preserve SciPy's `[1, 0]` element: the other triangle can differ
  by one floating-point ULP. The first differential test caught and fixed this.

Tests cover ties, duplicate frames, gaps, constant axes, sequential/backward/sparse
video reads, pixel equality and EOF recovery.

## Video/refinement protocol

Existing broadcast clip, native old-vs-new decoding/refinement, three repeats:
**11.120 → 0.567 s** median in the uninstrumented v2 run. All hashes equal
`a50b0ed3a883ce83a127634d7104a7ab6c40c538463e5fd54cd662dd9bdcca4d`.
This ratio depends strongly on codec/GOP/access pattern and host conditions.
An earlier exploratory, profiler-instrumented run measured 6.618 → 0.971 s.
Do not pool those protocols or present either as a full-match speedup.

Exhaustive repeated native seeking on 1080p SNGS-021 proved very expensive and
was canceled without publishing incomplete timing. For **full-sequence equality**,
the original refinement algorithm instead used an independent decoder reading
every frame sequentially, including skipped images. This verifies all selected
images/outputs without repeatedly paying the old seek cost. That baseline is
marked `sequential-reference`, not called the original production decoder.

| Sequence | Full sequential reference, s | Optimized native loader, s | Prediction hash |
|---|---:|---:|---|
| SNGS-021 | 3.97 | 2.39 | same |
| SNGS-033 | 3.97 | 2.04 | same |
| SNGS-045 | 6.15 | 3.24 | same |
| SNGS-090 | 4.40 | 2.40 | same |

These are **not native old-vs-new timing claims**. A separate bounded native-seek
check uses frames 350–381 of SNGS-021 identically in both variants; its measurements
are **6.634 → 0.457 s** median, with equal hashes (`seek_window_*_v2.json`).

## Reproduction

Baseline source loads from git `7c485be97daa34892b1d9581426211e5ccc08bed` without
changing/resetting files. Manifests store source/input/video hashes, resolved
configs, dataset/split, seed, processor, software, inference scope (none), timing
samples and output hashes. Python 3.12.10, NumPy 2.2.6, SciPy 1.16.1, OpenCV
4.12.0.88; Intel64 Family 6 Model 197 CPU. Three uninstrumented repeats; cProfile
is a separate untimed fourth call. This small timing sample is not a confidence interval.

```powershell
python scripts/benchmark_postprocessing.py --predictions runs/soccernet_valid_SNGS-021_core/predictions.json --reference-revision 7c485be --dataset SoccerNet-GSR-v1.3/SNGS-021 --split valid --output runs/performance_20260907/SNGS-021_stats_baseline_v2.json
python scripts/benchmark_postprocessing.py --predictions runs/soccernet_valid_SNGS-021_core/predictions.json --dataset SoccerNet-GSR-v1.3/SNGS-021 --split valid --output runs/performance_20260907/SNGS-021_stats_current_v2.json
python scripts/summarize_postprocessing.py --directory runs/performance_20260907 --output runs/performance_20260907/comparison.json
```

For refinement add `--video data/soccernet/valid/SNGS-021/SNGS-021.mp4`; for the
full sequential reference additionally use `--decoder sequential-reference`.
The summarizer rejects mismatched inputs/configs and unequal hashes. Do not
compare different frame intervals. Artifacts live under ignored
`runs/performance_20260907/`; the benchmark script can reproduce them.

## Remaining bottleneck

Perception/calibration/IDATR still use subprocesses per chunk. Batch/ReID
optimizations already existed in the supplied core. Existing GPU measurements
are in ACCURACY.md, not re-measured here. Changing FPS, detector size or calibration
stride would be an accuracy trade-off and was not used for these gains.

New analyses save `timings_<attempt>.json`: screening, transcoding, perception
round-trip, refinement, calibration, base/detailed statistics and web encoding.
Restored chunks do not count as new compute. These are wall times, not CUDA kernel
times, and exclude uninstrumented I/O. Next: controlled GPU profiling and an A/B
of persistent model workers vs per-chunk reloads, with unchanged model/config
revisions and measured VRAM. Multiple resident models may exceed 16 GB.

## Verification

- 198 pytest tests and Ruff passed.
- Base statistics plus the new detailed builder measured 0.482 s on SNGS-021
  in the final run (preliminary 0.563 s), still below the 1.13 s original stage.
  This includes neither JSON serialization nor disk writes. The preliminary
  timing was recovered from the tool log after an accidental same-path rerun;
  its separate manifest labels that recovery explicitly. Output hashes match.
- Real-clip detailed JSON: four estimated events, 33 non-referee identities.
- Docker image `football-intelligence:performance-20260907` built;
  final ID `sha256:359ef466cd96abc38f7e4a05257785d25919817874e8001a51abaa28a6171ef0`.
- Docker health/API passed. Browser-downloaded JSON equals the generated JSON as
  parsed data. The automation's download-event wait timed out, but the actual
  file was saved and independently verified.
- Localhost-only test container stopped; unrelated services untouched.
- Final rebuilt image independently passed health, statistics, viewer and the
  unattributed-event regression; its disposable test container was removed.
