# Current State

Updated: 2026-08-26 04:30 Europe/Moscow

## Objective

Deliver an evidence-grounded local football intelligence system, an annotated
Argentina–France 2022 clip with honest statistics, roster input, a debugging
frontend, reproducible Docker execution and measured SoccerNet validation.

## Working

- Typed event/evidence, Player Memory, global identity and run metadata contracts.
- Strict OpenAI-compatible Qwen3-VL semantic backend and uncertainty-triggered
  second pass; production semantic calls are schema validated.
- Football Core perception adapter, content-addressed cache and SQLite Match Memory.
- FastAPI/debug UI supports video jobs, inline annotated playback, statistics,
  full roster JSON input, the bundled ARG–FRA 2022 preset and starting-XI mode.
- Exact user clip is exported under `runs/arg_fra_2022_user_clip/` with video,
  JSON/CSV statistics, events, predictions, config, hashes and report.
- Docker API image is healthy; CUDA Football Core image builds from local weights
  and passes its GPU/model check on an RTX 5080 Laptop GPU.
- SoccerNet GSR v1.3 validation archive is CRC/SHA-256 verified and fully audited;
  frozen test was not downloaded or used.
- Official `SoccerNet/sn-trackeval` evaluator is pinned and its SNGS-033 oracle
  smoke test returns 100 for all metrics.

## Failing / unknown

- No reviewed semantic event gold set exists; event SOTA/accuracy claims remain
  prohibited. ARG–FRA event and possession statistics are explicitly heuristic.
- Only one SoccerNet validation sequence has a model score; it is insufficient
  for a SOTA claim or threshold tuning.
- Qwen3-VL weights are not present locally; legacy perception uses the supplied
  Qwen2.5-VL-7B Q8 GGUF for jersey reading.
- Football Core cold start fetches an RF-DETR upstream backbone (~890 MB), so its
  cache must be persisted for reproducible steady-state timings.

## Last verified benchmark

- run_id: EXP-20260826-07
- dataset: SoccerNet GSR v1.3 validation, SNGS-033
- command: official `run_soccernet_gs.py`, pitch-space Gaussian GS-HOTA at 5 m
- result: model GS-HOTA 38.617, DetA 20.859, AssA 71.508, IDF1 33.648;
  diagnostic no-team/no-jersey GS-HOTA 70.221. See `benchmark.json` for provenance.

## Files changed

- Implementation under `src/football_intelligence/`, tests, schemas and configs.
- Roster preset and storage: `src/football_intelligence/rosters.py` and
  `src/football_intelligence/roster_data/wc2022_final_arg_fra.json`.
- Reproducible data/export scripts under `scripts/`.
- Benchmark and clip artifacts under ignored `runs/` and `data/soccernet/`.
- Docker dependency fix in the supplied external Football Core Dockerfile: install
  `git` before the pinned OpenAI CLIP VCS dependency.

## Decisions made

- Keep full-squad and starting-XI roster modes explicit. Starting XI is a narrow
  prior for early-match clips; full squad is required across substitutions.
- Treat team labels as constraints, not sufficient identity evidence.
- Report ARG–FRA event counts as heuristic outputs, never labelled accuracy.
- Use validation only for development; leave the SoccerNet frozen test untouched.

## Next action

1. Rebuild and smoke-test the updated API/frontend Docker image.
2. Validate team-evidence quarantine on multiple validation sequences.
3. Improve team clustering/alignment, then rerun the same validation subset.
4. Address jersey errors without tuning the frozen test.

## Resume command

```powershell
python scripts\evaluate_soccernet_gs.py `
  --predictions runs\soccernet_valid_sngs033_core\predictions.json `
  --sequence SNGS-033 --tracker-name football-core-full
```
