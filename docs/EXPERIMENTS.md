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
