# Football Match Intelligence — VLM-Centric Full-Match Analysis

> **Goal:** build a local-first, evidence-grounded football match analysis system that beats the current project baseline on SoccerNet-style perception metrics and substantially improves semantic event understanding by using a strong open-weight VLM as the semantic reasoning core.

**Status:** executable research MVP; measured on four SoccerNet GSR validation sequences  
**Primary research direction:** hybrid perception + VLM reasoning + persistent match memory + active re-analysis  
**Primary open-weight VLM candidate:** Qwen3-VL  
**Primary football-specific foundation/perception reference:** SoccerMaster  
**Benchmark source of truth:** `ACCURACY.md`

---

## Executable implementation

The repository now includes typed Player Memory, an inspectable global identity solver, strict schema-valid semantic VLM calls, uncertainty-triggered second-pass analysis, content-addressed inference caching, SQLite Match Memory, a FastAPI control plane, a debugging frontend and Docker execution. The UI accepts a complete match roster as JSON, includes the Argentina–France 2022 final preset, can restrict early clips to the starting XI, and plays the rendered annotated result alongside measured/heuristic statistics with explicit labels.

Quick start for the verified low-compute control plane:

```powershell
python -m pip install -e ".[dev]"
pytest
docker compose up -d --build api
```

Open `http://localhost:8080`. The full profile expects the measured legacy Football Core image plus a local Qwen3-VL server:

```powershell
$env:FI_PERCEPTION_URL="http://perception:8000"
$env:FI_VLM_BASE_URL="http://vlm:8000"
docker compose --profile full up -d
```

The supplied legacy source uses a two-stage image so weights stay outside this
repository. Build the local weights layer first, then the CUDA 12.8 runtime:

```powershell
Set-Location C:\Users\Andrei\Documents\Football_grade\football_core
docker build -f Dockerfile.weights -t football-core-weights:1.0.0 .
docker build --build-arg WEIGHTS_IMAGE=football-core-weights:1.0.0 `
  --build-arg WITH_VLM=1 --build-arg WITH_YOLOX=0 `
  --build-arg CUDA_ARCHS=120 -t football-core:1.0.0 .
docker run --rm --gpus all football-core:1.0.0 check
```

Model weights and raw SoccerNet media are mounted/cached runtime data and are
not committed or baked into the application image. The first RF-DETR launch
may populate an upstream backbone cache; benchmark metadata must distinguish a
cold start from steady-state inference.

Existing development artifacts can be imported without rerunning perception:

```powershell
football-intelligence import-legacy --match-id dev-match --video match.mp4 `
  --events events.json --predictions predictions.json --end 60 `
  --output runs/dev-import.json
```

SoccerNet development uses the official validation split only. Raw media and
weights stay ignored. After obtaining access through the official downloader,
the auditable commands are:

```powershell
python scripts\soccernet_validation_audit.py `
  --archive data\soccernet\gamestate-2024\valid.zip `
  --output runs\soccernet_valid_audit.json
python scripts\run_soccernet_core.py --core-url http://localhost:8000 `
  --sequence SNGS-033 --media-path /media/SNGS-033/SNGS-033.mp4 `
  --output-dir runs\soccernet_valid_sngs033_core --fps 25 --duration 30
python scripts\evaluate_soccernet_gs.py `
  --predictions runs\soccernet_valid_sngs033_core\predictions.json `
  --sequence SNGS-033 --tracker-name football-core-full
```

### Game-state refinement

`football_intelligence.gamestate` repairs the identity attributes that the
perception backend decides frame by frame — the attributes GS-HOTA scores as
hard constraints. It checks the goalkeeper label against pitch geometry,
resolves teams from kit colour, chains the tracklets that are one player, has
the backend's own VLM re-read each chained identity's jersey number from crops
spread across the whole clip, and enforces one number per team at a time. No
retraining, no ground truth.

```powershell
python scripts\reread_jerseys.py --sequences SNGS-090
python scripts\refine_gamestate.py `
  --predictions runs\soccernet_valid_SNGS-090_core\predictions.json `
  --sequence SNGS-090 --link-tracklets `
  --jersey-reads runs\soccernet_valid_SNGS-090_core\jersey_reads.json `
  --output runs\soccernet_valid_SNGS-090_core\predictions_refined.json
python scripts\benchmark_gamestate.py `
  --sequences SNGS-021 SNGS-033 SNGS-045 SNGS-090 `
  --output runs\gamestate_benchmark.json
python scripts\attribute_report.py --sequence SNGS-033 `
  --predictions runs\soccernet_valid_sngs033_core\predictions_refined.json
```

To look at a result rather than a number, `scripts/render_annotated_clip.py`
draws both outputs side by side over a shared pitch minimap that carries the
annotation as hollow markers:

```powershell
python scripts\render_annotated_clip.py --sequence SNGS-033 `
  --predictions runs\soccernet_valid_sngs033_core\predictions.json `
               runs\soccernet_valid_sngs033_core\predictions_refined.json `
  --titles "perception as delivered" "refined game state" `
  --output runs\soccernet_valid_sngs033_core\annotated_comparison.mp4
```

`reread_jerseys.py` stages crops on the volume the perception container mounts
and drives the container's own Qwen2.5-VL over `docker exec`; everything else
is CPU work, about 7 s per 750-frame clip.

Measured over four SoccerNet GSR validation sequences, macro GS-HOTA goes from
**38.42 to 56.00**, improving on every sequence, with mean team accuracy rising
from 0.725 to 0.961 and mean jersey accuracy when emitted from 0.618 to 0.828.
Role is left almost alone on purpose: `ACCURACY.md` records the measurement
showing a perfect role classifier would be worth +0.6 GS-HOTA at most.
This implementation does not claim SOTA accuracy: the published challenge-set
state of the art is GS-HOTA 63.81 (2024) / 63.90 (2025) on a different split.
See `ACCURACY.md` for the full tables, ablations, tuning caveats and
provenance, and `runs/gamestate_benchmark.json` for raw output.

---

## 1. Why this project is changing direction

The previous architecture relied on a long chain of task-specific components:

```text
detector
  → tracker
  → OCR
  → team classifier
  → ball geometry
  → possession heuristics
  → event rules
  → statistics
```

That design is useful for geometry and deterministic measurements, but it has an important weakness: each stage loses information before the next stage sees it.

In experiments, a strong general-purpose multimodal model analyzing short match fragments can understand player numbers and football actions better than a complicated multi-stage rule pipeline. This suggests a different design principle:

> **Use specialist CV models as sensors and retrieval tools. Use the VLM as the semantic interpreter.**

The new system therefore does **not** replace every vision module with a VLM. Instead, it combines:

- specialist detection/tracking/calibration for dense, frame-level geometry;
- a strong VLM for jersey reasoning, identity aggregation and event semantics;
- persistent full-match memory;
- global constraints across the whole match;
- uncertainty-aware re-analysis of only difficult fragments.

---

## 2. Measured baseline that must be beaten

The current repository baseline is defined by `ACCURACY.md`. Do not replace these values with estimates.

### 2.1 Perception baseline

Measured on 2,250 SoccerNet Game State Reconstruction frames with IoU ≥ 0.5:

| Metric | Current baseline |
|---|---:|
| Detection recall | **77%** |
| Detection precision | **89%** |
| Role classification | **100%** |
| Team | **80%** |
| Jersey number | **59% accuracy at 86% coverage** |
| Ball found | **89%** of GT frames containing the ball |
| Pitch-position error | **0.57 m median**, **1.63 m p90** |
| Static-player projection noise | **0.07 m/frame** |
| Track fragments per player / 30 s | **1.94** |

### 2.2 Current semantic limitation

The existing pass inference reaches only **43% accuracy even when run on ideal ground-truth tracks, teams and ball annotations**.

That result is especially important: better player detection alone cannot fix the semantic event layer.

The current rule-based system also cannot reliably distinguish an intentional pass from:

- clearance;
- interception;
- tackle;
- deflection;
- loose-ball recovery;
- contested possession.

This is the main reason for introducing a temporal VLM reasoning layer.

### 2.3 Important metric warning

The old project's "pass accuracy" is not directly comparable with Opta/StatsBomb-style football statistics. The previous metric is based on possession transitions rather than an explicitly annotated intentional-pass ontology.

The new architecture must therefore maintain two separate concepts:

1. **legacy possession-transition metric** — retained only for regression compatibility;
2. **semantic event accuracy** — measured against a newly defined annotated event benchmark.

Never silently rename one into the other.

---

## 3. Research targets

These are **targets, not claimed results**.

### 3.1 Target A — beat the current repository baseline

| Metric | Baseline | Minimum target | Stretch target |
|---|---:|---:|---:|
| Detection recall | 77% | ≥ 90% | **≥ 94%** |
| Detection precision | 89% | ≥ 93% | **≥ 96%** |
| Role | 100% | ≥ 99% | **100%** |
| Team | 80% | ≥ 95% | **≥ 98%** |
| Jersey accuracy | 59% | ≥ 80% | **≥ 88%** |
| Jersey coverage | 86% | ≥ 90% | **≥ 95%** |
| Ball recall | 89% | ≥ 94% | **≥ 97%** |
| Position median error | 0.57 m | < 0.55 m | **≤ 0.40 m** |
| Position p90 | 1.63 m | < 1.40 m | **≤ 1.10 m** |
| Track fragments / player / 30 s | 1.94 | < 1.40 | **≤ 1.20** |

### 3.2 Target B — semantic understanding

Create a manually verified event subset before claiming these metrics.

Target macro-F1:

| Event family | Initial target | Stretch target |
|---|---:|---:|
| intentional pass vs non-pass | ≥ 0.75 | **≥ 0.88** |
| pass outcome | ≥ 0.80 | **≥ 0.92** |
| clearance | ≥ 0.70 | **≥ 0.85** |
| interception/recovery | ≥ 0.70 | **≥ 0.85** |
| tackle/duel | ≥ 0.65 | **≥ 0.80** |
| shot | ≥ 0.80 | **≥ 0.92** |
| set piece type | ≥ 0.80 | **≥ 0.95** |

All event metrics must include:

- per-class precision;
- recall;
- F1;
- confusion matrix;
- confidence calibration;
- coverage/abstention rate.

### 3.3 Target C — SoccerNet-GSR research benchmark

SoccerNet GSR evaluates localization and identity with **GS-HOTA**.

As of this specification, SoccerMaster reports:

- GS-HOTA: **64.1**
- GS-DetA: **51.5**
- GS-AssA: **79.9**

Research stretch goal:

```text
GS-HOTA ≥ 65.0
```

A result below 65 can still be valuable if the proposed full-match semantic system demonstrates capabilities not captured by GS-HOTA.

---

## 4. Core hypothesis

The system should exploit information at three different temporal scales.

### Frame scale

Best for:

- player detection;
- ball detection;
- pose/crop quality;
- pitch calibration;
- immediate geometric evidence.

### Tracklet scale

Best for:

- re-identification;
- jersey number;
- role;
- team;
- visual identity;
- short actions.

### Full-match scale

Best for:

- resolving jersey ambiguity;
- reconnecting fragmented tracklets;
- substitution reasoning;
- global player identity;
- repeated visual evidence;
- roster constraints;
- semantic history;
- event relationships;
- natural-language queries about the match.

The full-match system is therefore not a single giant prompt. It is a **persistent evidence memory** over the whole match.

---

# 5. High-level architecture

```text
                               FULL MATCH
                                   │
                   ┌───────────────┴────────────────┐
                   │                                │
                   ▼                                ▼
            MEDIA INGEST                     AUDIO / METADATA
                   │                                │
      shot boundaries / timestamps             roster / ASR
                   │                                │
                   ▼                                │
          PERCEPTION LAYER                         │
                   │                                │
      ┌────────────┼──────────────┐                 │
      ▼            ▼              ▼                 │
 detection      tracking      calibration           │
      │            │              │                 │
      ├───── SAM2 refinement ─────┤                 │
      │            │              │                 │
      ▼            ▼              ▼                 │
 players        tracklets      pitch coords         │
 ball             │                                  │
                  ▼                                  │
             PLAYER MEMORY ◄─────────────────────────┘
                  │
                  ├─ best crops / views
                  ├─ ReID embeddings
                  ├─ team hypotheses
                  ├─ jersey hypotheses
                  ├─ roster candidates
                  └─ temporal appearances
                  │
                  ▼
            GLOBAL IDENTITY SOLVER
                  │
                  ▼
        ┌─────────────────────────┐
        │   VLM SEMANTIC ENGINE   │
        │       Qwen3-VL          │
        └────────────┬────────────┘
                     │
           events + evidence + confidence
                     │
                     ▼
             MULTIMODAL MATCH MEMORY
      ┌──────────────┼─────────────────┐
      ▼              ▼                 ▼
 structured DB   vector retrieval   match graph
      │              │                 │
      └──────────────┼─────────────────┘
                     ▼
              UNCERTAINTY ROUTER
                     │
          confidence below threshold?
             ┌───────┴────────┐
             │ no             │ yes
             ▼                ▼
           accept      ACTIVE RE-ANALYSIS
                        original video
                       higher FPS / ROI
                       better crops
                       larger VLM optional
                             │
                             ▼
                         final event
                             │
                             ▼
                    STATS / API / UI
```

---

# 6. Design principles

## 6.1 VLM is the semantic brain, not the pixel-level tracker

Do not call the VLM on every frame to reproduce tasks that fast specialist models already solve well.

Use specialist CV for:

- dense detection;
- ball localization;
- tracking;
- segmentation;
- camera calibration;
- pitch coordinates.

Use the VLM for:

- jersey reasoning across multiple views;
- resolving ambiguous identity;
- interpreting temporal actions;
- intentional-pass classification;
- action relationships;
- high-level football semantics;
- natural-language queries;
- checking ambiguous results.

## 6.2 Evidence before answers

Every semantic prediction must point back to evidence.

Example:

```json
{
  "event_id": "evt_00428",
  "type": "intentional_pass",
  "start_ms": 2052400,
  "end_ms": 2058900,
  "actor_player_id": "P07",
  "target_player_id": "P12",
  "outcome": "complete",
  "confidence": 0.91,
  "evidence": [
    {"timestamp_ms": 2053100, "kind": "frame"},
    {"timestamp_ms": 2055700, "kind": "frame"},
    {"timestamp_ms": 2058100, "kind": "frame"}
  ],
  "model": "qwen3-vl",
  "reason_code": "visual_temporal_evidence"
}
```

A user-visible statistic must be traceable to:

```text
statistic
  → events
  → evidence timestamps
  → source video
  → model/config/version
```

## 6.3 Abstention is better than invented certainty

All semantic modules must support:

```text
unknown
uncertain
not_visible
insufficient_evidence
```

Do not force a jersey number, player identity or event type when evidence is insufficient.

## 6.4 Never use a single frame when temporal evidence exists

Especially for jersey recognition.

Instead of:

```text
crop → OCR → 18
```

use:

```text
tracklets
   → best-view selection
   → multi-image contact sheet
   → original crops
   → roster constraints
   → VLM posterior
   → global identity solver
```

---

# 7. Recommended model stack

The exact implementation must remain modular so each model can be replaced.

## 7.1 Primary semantic VLM

### Default

```text
Qwen3-VL-8B-Instruct
```

Use it for:

- clip understanding;
- event classification;
- multi-frame jersey reasoning;
- evidence selection;
- ambiguity resolution;
- structured JSON output.

Reasons for starting here:

- open weights;
- strong multimodal reasoning;
- video support;
- long-context support;
- controllable video frame sampling;
- practical enough to test locally in quantized form.

Qwen3-VL documents a native 256K context and optional extension to 1M. Long context is useful for summaries and match memory, but **must not be treated as a reason to send every raw match frame into one prompt**.

### Expert model

Make the semantic backend replaceable:

```python
SemanticVLMBackend
├── Qwen3VLBackend
├── LocalOpenAICompatibleBackend
└── RemoteEvaluationBackend   # optional, never required for production
```

A larger Qwen3-VL variant may be used as a second-pass expert when hardware permits.

Closed models may be used only as **evaluation references/teachers**. The production path must remain functional with open-weight models.

---

## 7.2 Soccer-specific visual backbone/reference

Evaluate **SoccerMaster** rather than rebuilding the entire soccer perception stack from scratch.

Its released GSR pipeline combines:

- detection/tracking;
- SAM2 segmentation refinement;
- camera calibration;
- ReID;
- jersey recognition;
- role classification;
- team assignment.

Its repository currently recommends Qwen2.5-VL-72B for best jersey-number results and Qwen2.5-VL-7B for an accelerated configuration. Our project should benchmark whether Qwen3-VL provides a better accuracy/compute trade-off.

Do not assume it does. Measure it.

---

## 7.3 Detection

Start with the strongest reproducible football-tuned detector available in the selected SoccerMaster/SoccerNet pipeline.

Interface:

```python
class PersonBallDetector:
    def predict(frame) -> list[Detection]:
        ...
```

Required classes:

```text
player
goalkeeper
referee
ball
other
```

The detector must return raw confidence and bounding boxes. It must never assign persistent player identity.

---

## 7.4 Tracking and segmentation refinement

Required components:

```text
detector
  → MOT tracker
  → SAM2-based/refinement-compatible temporal segmentation
  → ReID association
  → tracklet graph
```

Track IDs from a short segment are **not** global player IDs.

Always distinguish:

```text
detection_id
tracklet_id
global_player_id
jersey_number
roster_player_id
```

---

## 7.5 Ball tracking

Keep ball processing separate from player tracking.

Required output:

```json
{
  "timestamp_ms": 12345,
  "bbox": [x, y, w, h],
  "confidence": 0.93,
  "visible": true,
  "occluded": false
}
```

Do not project an airborne ball onto the pitch and treat that result as a real 2D ball position.

The existing baseline already demonstrates why planar homography is invalid for a flying ball.

For event reasoning, prefer:

- ball position in image space;
- relative position to players;
- temporal trajectory;
- visibility state;
- optical motion;
- VLM visual evidence.

---

# 8. Player Memory

`Player Memory` is one of the main innovations of the architecture.

Each global-player hypothesis accumulates evidence across the entire match.

```json
{
  "global_player_id": "P07",
  "team_id": "TEAM_A",
  "team_confidence": 0.992,

  "tracklets": [
    "T0017",
    "T0051",
    "T0087"
  ],

  "appearances": [
    {"start_ms": 221000, "end_ms": 228000},
    {"start_ms": 492000, "end_ms": 499000}
  ],

  "jersey_distribution": {
    "8": 0.06,
    "18": 0.89,
    "28": 0.05
  },

  "roster_candidates": [
    {"jersey": 18, "player_id": "R018", "score": 0.94}
  ],

  "best_views": {
    "back": ["obs_101", "obs_173"],
    "front": ["obs_201"],
    "side": ["obs_118"]
  }
}
```

## 8.1 Best-view mining

Do not send hundreds of almost identical crops to the VLM.

Rank crops using:

```text
crop resolution
jersey/back visibility
blur
occlusion
pose
bbox size
contrast
track confidence
frame uniqueness
```

Keep diversified top-K views:

```text
K_back   = 4
K_front  = 2
K_side   = 2
```

Values must be configurable.

## 8.2 Jersey reasoning

Input to VLM:

- top-K crops;
- contact sheet;
- team;
- roster number set;
- earlier hypotheses;
- explicit instruction to abstain if unreadable.

Output:

```json
{
  "jersey_number": 18,
  "confidence": 0.91,
  "alternatives": [
    {"number": 8, "confidence": 0.06},
    {"number": 28, "confidence": 0.03}
  ],
  "best_evidence_observations": [
    "obs_101",
    "obs_173"
  ]
}
```

The VLM output is a **local likelihood**, not the final identity.

---

# 9. Global Identity Solver

A match provides constraints that do not exist in isolated clips.

Build a global optimization/graph stage that combines:

- ReID similarity;
- time overlap;
- team;
- jersey posterior;
- roster;
- substitution windows;
- goalkeeper role;
- repeated appearances;
- tracklet continuity;
- VLM identity evidence.

Example objective:

```text
maximize Σ assignment_score(tracklet, player)

subject to:
    a physical player cannot occupy incompatible simultaneous tracklets
    team assignment must be consistent
    jersey candidate must exist in roster when roster is provided
    substitution/time constraints must be respected
```

Do not force global uniqueness across time when substitutions or jersey reuse rules make that assumption invalid for the supplied competition.

Competition rules must be configuration, not hard-coded folklore.

---

# 10. Semantic Event Engine

The semantic VLM must operate on **candidate windows**, not blindly on the whole match.

## 10.1 Candidate generation

Cheap modules propose windows using:

- possession transitions;
- ball acceleration;
- ball/player proximity in image space;
- shot/camera boundary;
- audio spikes;
- scoreboard changes;
- known SoccerNet action spotters;
- temporal motion;
- user queries.

Candidate generation should favor recall.

## 10.2 VLM event classification

Initial event ontology:

```text
touch
controlled_possession
intentional_pass
cross
through_ball
clearance
interception
recovery
tackle
duel
carry
dribble
shot
goal
save
foul
offside
corner
throw_in
free_kick
penalty
kickoff
substitution
unknown
```

Do not implement all classes at once.

### MVP event subset

Phase 1:

```text
touch
controlled_possession
intentional_pass
clearance
interception
recovery
tackle
duel
shot
unknown
```

Only expand once the confusion matrix on this subset is stable.

## 10.3 Structured VLM output

All VLM calls must request schema-valid structured output.

Example:

```json
{
  "primary_event": "intentional_pass",
  "confidence": 0.87,
  "actor": "P07",
  "target": "P12",
  "outcome": "complete",
  "start_ms": 2052400,
  "end_ms": 2058900,
  "evidence_ms": [2053100, 2055700, 2058100],
  "alternatives": [
    {"event": "clearance", "confidence": 0.09}
  ],
  "insufficient_evidence": false
}
```

Validate the JSON against a versioned schema.

Invalid model output is not silently repaired into a plausible event.

---

# 11. Active re-analysis

This is mandatory.

Do not spend maximum compute uniformly across a 90-minute match.

## Pass 1 — cheap

Suggested starting profile:

```text
VLM: Qwen3-VL-8B-Instruct
candidate clip FPS: 1–2
resolution: bounded
context: local event + relevant player memory
```

## Pass 2 — uncertainty-triggered

Trigger when, for example:

```text
event_confidence < τ_event
jersey_confidence < τ_jersey
top1 - top2 < τ_margin
identity conflicts with global solver
event conflicts with geometry
event materially affects final statistics
```

Re-run using:

- original source frames;
- higher FPS;
- higher-resolution player crops;
- longer temporal window;
- additional views;
- player memory;
- roster;
- optional larger local VLM.

Suggested initial windows:

```text
normal candidate:    -3 s ... +4 s
ambiguous event:     -6 s ... +8 s
identity/jersey:     best observations across whole match
```

All thresholds must be calibrated on validation data.

---

# 12. Match Memory — not just RAG

Do not implement the architecture as only:

```text
captions → embeddings → vector DB
```

That loses exact temporal and structured football information.

Use three memory layers.

## 12.1 Structured memory

Recommended first implementation:

```text
PostgreSQL
```

Tables:

```text
matches
teams
rosters
media_segments
shots
detections
tracklets
tracklet_observations
players
player_hypotheses
ball_observations
camera_states
events
event_evidence
model_runs
metrics
```

## 12.2 Vector memory

Use `pgvector`, Qdrant or an interchangeable adapter.

Index:

- event descriptions;
- clip embeddings;
- player visual embeddings;
- VLM summaries;
- commentary transcript chunks;
- user annotations.

Every vector result must retain a foreign key to structured evidence.

## 12.3 Match graph

Represent relationships such as:

```text
P07 --performed--> EVT128
EVT128 --target--> P12
EVT128 --preceded_by--> EVT127
P07 --same_identity_as--> T087
P07 --team--> TEAM_A
```

For the first implementation this may live in PostgreSQL tables or an in-memory graph. Do not add Neo4j unless graph queries demonstrably justify another service.

---

# 13. Retrieval strategy

Use hybrid retrieval.

Query example:

> Show every attack where #18 won the ball and the team created a shot within 15 seconds.

Retriever:

```text
1. resolve jersey #18 → global player
2. structured filter → recoveries/interceptions
3. graph traversal → following events within 15 s
4. structured filter → shot
5. retrieve original video windows
6. optionally re-run VLM for final verification
7. answer with timestamps and evidence
```

Vector similarity is supplementary, not authoritative.

---

# 14. Audio and commentary

Audio is optional but strongly recommended as an auxiliary signal.

Pipeline:

```text
audio
  → VAD
  → ASR
  → timestamped transcript
  → entity/name extraction
  → Match Memory
```

Useful for:

- player names;
- goals;
- shots;
- saves;
- fouls/cards;
- substitutions;
- set pieces.

Important:

> Commentary is evidence, not ground truth.

A commentator may refer to an earlier player/action, speculate, or speak after the visual event.

Semantic decisions that affect statistics should require visual or structured corroboration.

---

# 15. Statistics layer

Statistics are derived only from versioned accepted events and tracking data.

Every statistic declares its source:

```json
{
  "name": "completed_passes",
  "value": 47,
  "source": "semantic_events",
  "event_schema_version": "1.0.0"
}
```

or:

```json
{
  "name": "distance_visible_m",
  "value": 8214.2,
  "source": "tracking"
}
```

Never present "distance covered" without indicating visibility/coverage when derived from broadcast video.

The existing baseline already notes that a large fraction of broadcast frames may not show the main field camera.

---

# 16. Evaluation protocol

This section is non-negotiable.

## 16.1 Frozen test data

Maintain:

```text
data/
  splits/
    train.json
    val.json
    test_frozen.json
```

`test_frozen.json` must not be used for:

- prompt tuning;
- threshold selection;
- model selection;
- manual debugging.

## 16.2 Legacy regression suite

Re-run the exact evaluation that produced `ACCURACY.md`.

Store results in:

```text
runs/benchmark/<run_id>/
  config.yaml
  environment.json
  metrics.json
  per_clip.csv
  confusion_matrices/
  failures/
  report.md
```

## 16.3 New semantic gold set

Create an event benchmark with clear annotation rules.

Each event must include:

```text
event type
start/end
actor
target if applicable
outcome
ambiguity flag
annotator
review state
```

For ambiguous football situations, permit:

```text
accepted_labels: [duel, tackle]
```

rather than injecting label noise.

## 16.4 Statistical reporting

Report:

- point estimate;
- number of examples;
- per-match/per-clip spread;
- bootstrap confidence interval when feasible;
- coverage;
- abstention rate.

Never claim improvement from a tiny sample without showing sample size.

---

# 17. Required ablation study

A publication-quality implementation must isolate where improvements come from.

Minimum ablations:

```text
A0 current baseline

A1 stronger perception only

A2 + tracklet-level best-view jersey reasoning

A3 + Player Memory

A4 + global identity solver

A5 + VLM semantic event engine

A6 + active re-analysis

A7 + Match Memory retrieval

A8 + commentary/ASR

A9 full system
```

Report accuracy **and compute cost** for every stage.

Recommended compute metrics:

```text
processing FPS
seconds of compute / minute of video
peak VRAM
RAM
VLM tokens
VLM frames/images processed
disk cache size
```

---

# 18. Error taxonomy

Every benchmark failure should be assignable to a category.

```text
DETECTION_MISS
FALSE_DETECTION
BALL_MISS
TRACK_FRAGMENT
ID_SWITCH
REID_ERROR
TEAM_SWAP
JERSEY_UNREADABLE
JERSEY_VLM_ERROR
ROSTER_CONSTRAINT_ERROR
CALIBRATION_ERROR
EVENT_WINDOW_ERROR
EVENT_VLM_ERROR
EVENT_SCHEMA_ERROR
RETRIEVAL_ERROR
AUDIO_MISLEADING
INSUFFICIENT_EVIDENCE
GROUND_TRUTH_AMBIGUITY
```

A model improvement is not useful if it only moves failures into an unmeasured category.

---

# 19. Local-first execution

The project should run without requiring a proprietary API.

Recommended development profile:

```text
GPU: modern NVIDIA GPU with ~16 GB VRAM
RAM: 32 GB+
storage: SSD, 100 GB+ free for datasets/caches recommended
OS: Linux preferred; Windows + WSL2 supported where practical
Docker: required for reproducible services
```

For a ~16 GB GPU:

- start with an appropriately quantized Qwen3-VL-8B;
- keep visual-token budgets bounded;
- process candidate clips, not whole-match raw frames;
- cache perception outputs;
- batch crop inference;
- offload only when benchmarked;
- use active second-pass inference.

Do not hard-code one quantization library. Provide backend adapters.

Larger models are optional expert backends and may require multi-GPU/server hardware.

---

# 20. Proposed repository structure

```text
.
├── README.md
├── AGENTS.md
├── ACCURACY.md
├── pyproject.toml
├── docker-compose.yml
├── .env.example
│
├── configs/
│   ├── default.yaml
│   ├── local_16gb.yaml
│   ├── benchmark.yaml
│   └── models/
│
├── src/
│   └── football_intelligence/
│       ├── ingest/
│       ├── detection/
│       ├── tracking/
│       ├── segmentation/
│       ├── reid/
│       ├── ball/
│       ├── calibration/
│       ├── player_memory/
│       ├── identity_solver/
│       ├── vlm/
│       ├── events/
│       ├── audio/
│       ├── memory/
│       ├── retrieval/
│       ├── statistics/
│       ├── evaluation/
│       └── api/
│
├── schemas/
│   ├── event.schema.json
│   ├── player_memory.schema.json
│   └── model_run.schema.json
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── golden/
│   └── benchmark/
│
├── evaluation/
│   ├── legacy/
│   ├── gsr/
│   ├── semantic_events/
│   └── ablations/
│
├── scripts/
│   ├── bootstrap.py
│   ├── ingest_match.py
│   ├── run_match.py
│   ├── evaluate.py
│   └── make_report.py
│
├── docs/
│   ├── ARCHITECTURE.md
│   ├── EVENT_ONTOLOGY.md
│   ├── ANNOTATION_GUIDE.md
│   ├── CURRENT_STATE.md
│   ├── DECISIONS.md
│   └── EXPERIMENTS.md
│
├── runs/
│   └── .gitkeep
│
└── data/
    ├── README.md
    ├── splits/
    └── .gitignore
```

---

# 21. Stable interfaces

Avoid tightly coupling the code to one research model.

```python
class DetectorBackend: ...
class TrackerBackend: ...
class SegmenterBackend: ...
class ReIDBackend: ...
class CalibrationBackend: ...
class BallBackend: ...
class SemanticVLMBackend: ...
class ASRBackend: ...
class VectorStoreBackend: ...
```

Each backend must expose:

- version/model identifier;
- deterministic configuration when possible;
- raw confidence;
- runtime telemetry;
- structured output.

---

# 22. Cache strategy

Inference is expensive. Cache by content, not by filename.

Cache key should include:

```text
source video hash
time range
frame sampling config
crop coordinates/version
model identifier
model revision/hash
prompt version
schema version
generation parameters
```

Changing the prompt must invalidate semantic VLM cache entries.

Changing only the UI must not invalidate perception.

---

# 23. Version every learned/semantic decision

Store:

```text
model_name
model_revision
prompt_version
schema_version
config_hash
git_commit
timestamp
```

Without this, benchmark differences cannot be trusted.

---

# 24. Development phases

## Phase 0 — freeze baseline

Deliverables:

- reproduce `ACCURACY.md`;
- machine-readable baseline JSON;
- frozen test list;
- one-command benchmark;
- failure gallery.

Exit criterion:

```text
reproduced metrics within documented tolerance
```

## Phase 1 — modernize perception

Integrate/benchmark:

- SoccerMaster-compatible detection;
- tracking;
- SAM2 refinement;
- ReID;
- calibration;
- ball detector.

Exit criterion:

```text
detection recall ≥ 90%
track fragmentation < 1.5
no regression in calibration
```

## Phase 2 — Player Memory + jersey

Deliver:

- best-view mining;
- Qwen3-VL multi-view jersey inference;
- roster constraints;
- posterior aggregation;
- global identity solver.

Primary exit criterion:

```text
jersey accuracy ≥ 80%
with coverage ≥ 90%
```

## Phase 3 — semantic event MVP

Create manually verified gold set for:

```text
intentional_pass
clearance
interception
recovery
tackle
duel
shot
```

Exit criterion:

```text
intentional-pass vs non-pass macro-F1 ≥ 0.75
```

## Phase 4 — active re-analysis

Deliver confidence router and high-resolution second pass.

Exit criterion:

```text
measurable semantic gain
with bounded compute increase
```

## Phase 5 — full Match Memory

Deliver:

- structured DB;
- vector retrieval;
- match graph;
- evidence retrieval;
- natural-language queries.

Exit criterion:

```text
every returned answer links to source timestamps
```

## Phase 6 — audio

Add timestamped ASR/commentary evidence.

Must include an ablation proving whether it improves results.

## Phase 7 — research benchmark

Run:

- full SoccerNet GSR evaluation;
- semantic benchmark;
- ablation suite;
- compute profile;
- failure analysis.

Stretch target:

```text
GS-HOTA ≥ 65.0
```

---

# 25. Definition of Done

The project is not "done" when a demo looks convincing.

A release candidate must satisfy all of the following:

- [ ] full match can be ingested locally;
- [ ] every model can run through versioned adapters;
- [ ] player/ball detections are persisted;
- [ ] tracklets are distinct from global identities;
- [ ] Player Memory is persistent across the whole match;
- [ ] jersey inference uses multiple observations;
- [ ] roster constraints are optional and explicit;
- [ ] semantic events contain timestamped evidence;
- [ ] uncertain events can abstain;
- [ ] active second-pass inference works;
- [ ] statistics are generated from versioned evidence;
- [ ] legacy benchmark is reproducible;
- [ ] semantic gold-set benchmark exists;
- [ ] ablation report exists;
- [ ] no proprietary API is required for the normal production path;
- [ ] no unlabelled mock/stub is used in production;
- [ ] all benchmark configurations and model revisions are recorded.

---

# 26. What must NOT happen

Do not:

- send an entire 90-minute match at full FPS into one VLM prompt;
- use vector RAG as the only source of match truth;
- use OCR on a single crop as final jersey identity;
- treat a short-lived tracker ID as a player identity;
- infer airborne ball pitch coordinates using planar homography as if they were valid;
- rename possession transitions as Opta-like passes;
- force a semantic label when evidence is insufficient;
- tune thresholds on the frozen test set;
- report only aggregate accuracy without coverage and failure examples;
- hide expensive fallback calls when reporting runtime;
- silently call a closed API when local inference fails.

---

# 27. Research questions worth publishing

The architecture should make the following questions measurable:

1. How much does **full-match player memory** improve jersey identification over isolated tracklets?
2. Can a **global identity constraint solver** reduce VLM jersey/identity errors?
3. How much does temporal VLM reasoning improve intentional-pass classification over geometric heuristics?
4. How much compute can **active re-analysis** save relative to high-quality inference on every candidate?
5. Does commentary audio improve event recognition after visual corroboration?
6. Does multimodal retrieval preserve full-match reasoning better than brute-force long-context video prompting?
7. Can full-match context improve SoccerNet-style identity metrics even when the official benchmark is composed of short clips?

---

# 28. References

Primary external references used to define the initial architecture:

- Qwen3-VL official repository: https://github.com/QwenLM/Qwen3-VL
- SoccerMaster official repository: https://github.com/haolinyang-hlyang/SoccerMaster
- SoccerMaster, CVPR 2026: https://openaccess.thecvf.com/content/CVPR2026/papers/Yang_SoccerMaster_A_Vision_Foundation_Model_for_Soccer_Understanding_CVPR_2026_paper.pdf
- SoccerNet Game State Reconstruction: https://www.soccer-net.org/tasks/game-state-reconstruction
- SoccerNet tasks: https://www.soccer-net.org/tasks
- SoccerNet GSR paper: https://arxiv.org/abs/2404.11335

---

# 29. Immediate first experiment

Before building the entire platform, run one experiment capable of falsifying the main idea.

Use the same clips from `ACCURACY.md`.

Compare:

```text
A. current jersey pipeline
B. best single crop + Qwen3-VL
C. top-K crops from one tracklet + Qwen3-VL
D. top-K crops across globally linked tracklets + Qwen3-VL
E. D + roster + global constraint solver
```

Measure:

```text
accuracy
coverage
calibration
runtime
VRAM
number of VLM calls
```

Then run the equivalent semantic experiment:

```text
A. current geometric pass inference
B. geometry + short VLM clip
C. B + player memory
D. C + active re-analysis
```

If C/D do not provide a meaningful accuracy gain, stop and investigate before expanding the architecture.

That experiment should be the first milestone because it directly tests the project's central hypothesis.
