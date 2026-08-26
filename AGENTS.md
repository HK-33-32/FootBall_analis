# AGENTS.md — Football Match Intelligence

This file defines mandatory operating rules for coding agents working in this repository.

The project is research-oriented, benchmark-driven and compute-expensive. Agents must optimize for:

1. **correctness;**
2. **reproducibility;**
3. **measured improvement;**
4. **context/token efficiency;**
5. **safe interruption/restart;**
6. **working end-to-end software without hidden stubs.**

Read `README.md` and `ACCURACY.md` before making architectural changes.

---

# 1. Mission

Build a local-first football match intelligence system based on:

```text
specialist perception
+ VLM semantic reasoning
+ Player Memory
+ global identity constraints
+ multimodal Match Memory
+ uncertainty-triggered re-analysis
```

The system must beat the repository baseline honestly.

Do not optimize for a visually convincing demo at the expense of measured correctness.

---

# 2. Source of truth hierarchy

When sources disagree, use this precedence:

```text
1. executable tests and evaluation code
2. frozen benchmark annotations
3. ACCURACY.md measured baseline
4. versioned schemas/configuration
5. README.md architecture
6. docs/
7. comments
8. assumptions
```

Never overwrite measured values in `ACCURACY.md` because a new number "looks more reasonable."

If `ACCURACY.md` is outdated, create a new benchmark report first and then update it with explicit provenance.

---

# 3. Mandatory startup procedure

At the beginning of a work session:

1. Read this `AGENTS.md`.
2. Read the relevant sections of `README.md`.
3. Read `docs/CURRENT_STATE.md` if it exists.
4. Read only the most relevant recent entries in:
   - `docs/DECISIONS.md`
   - `docs/EXPERIMENTS.md`
5. Inspect `git status`.
6. Locate the code directly related to the requested task.
7. Run the smallest relevant test/benchmark before editing when practical.

Do **not** recursively read the whole repository unless the task genuinely requires it.

---

# 4. Context/token conservation rules

Context is a limited engineering resource.

## 4.1 Search before reading

Prefer:

```text
rg
git grep
targeted file reads
specific test names
specific symbols
```

over opening entire files.

Example:

```bash
rg "jersey_distribution|PlayerMemory|SemanticVLMBackend" src tests
```

## 4.2 Do not dump huge artifacts into context

Never paste:

- full video metadata dumps;
- large JSON benchmark files;
- full model logs;
- hundreds of detections;
- entire SQL tables;
- complete stack traces when the final lines identify the error.

Instead capture:

```text
error class
relevant command
first relevant cause
last relevant stack frames
artifact path
```

Keep raw logs on disk.

## 4.3 Reuse summaries

`docs/CURRENT_STATE.md` is the compact restart state.

It should summarize:

```text
current phase
what works
what is failing
last benchmark
current hypothesis
next exact action
important file paths
```

Do not reconstruct the entire project from chat/history when this file already exists.

## 4.4 Avoid duplicated documentation

A fact should have one canonical home.

Examples:

```text
architecture     → README.md / docs/ARCHITECTURE.md
baseline         → ACCURACY.md
current progress → docs/CURRENT_STATE.md
experiments      → docs/EXPERIMENTS.md
decisions        → docs/DECISIONS.md
```

Reference canonical docs rather than repeating pages of content.

---

# 5. Session checkpoint protocol

The repository must be resumable after:

- context exhaustion;
- network loss;
- rate limit;
- IDE crash;
- model switch;
- agent handoff.

## 5.1 Update checkpoint after each meaningful milestone

Maintain:

```text
docs/CURRENT_STATE.md
```

Recommended compact format:

```markdown
# Current State

Updated: YYYY-MM-DD HH:MM

## Objective
...

## Working
- ...

## Failing / unknown
- ...

## Last verified benchmark
- run_id:
- command:
- result:

## Files changed
- ...

## Decisions made
- ...

## Next action
1. ...

## Resume command
```bash
...
```
```

Keep this file concise. Prefer < 250 lines.

## 5.2 Experiment ledger

Append important experiments to:

```text
docs/EXPERIMENTS.md
```

Each entry:

```markdown
## EXP-YYYYMMDD-NN — short name

Hypothesis:
Change:
Dataset/split:
Config:
Model revision:
Git commit:
Command:
Metrics:
Compute:
Conclusion:
Artifacts:
Next:
```

Do not paste giant logs into the ledger.

## 5.3 Decision ledger

Architectural decisions belong in:

```text
docs/DECISIONS.md
```

Record only decisions with long-term consequences.

Example:

```markdown
## ADR-012 — Tracklet IDs are never player IDs

Decision:
Reason:
Alternatives:
Consequences:
```

---

# 6. No-placeholder rule

Production paths must contain real implementations.

Forbidden unless explicitly marked as experimental/test-only:

```python
pass
raise NotImplementedError
return fake_result
return hardcoded_example
TODO: implement later
```

Also forbidden:

- fake detections;
- fabricated VLM JSON;
- random confidence scores;
- silent fallback to a canned answer;
- UI showing metrics not produced by the backend.

If a dependency is unavailable, fail clearly with actionable information.

A mock is allowed only under:

```text
tests/
dev fixtures/
explicit benchmark simulation
```

and must be named as a mock.

---

# 7. Do not rewrite working code without evidence

Before replacing an existing subsystem:

1. reproduce its current benchmark;
2. define the metric expected to improve;
3. implement the smallest comparable alternative;
4. run A/B evaluation;
5. keep the new version only if the trade-off is documented.

Do not perform large "clean architecture" rewrites while accuracy is unverified.

---

# 8. Benchmark-first development

Every major feature must answer:

> Which measurable failure does this fix?

Examples:

```text
Player Memory
→ jersey accuracy / coverage / ID consistency

global identity solver
→ jersey accuracy / ID switches / track fragmentation

SAM2 refinement
→ association / track fragmentation

VLM event classifier
→ pass-vs-clearance macro-F1

active re-analysis
→ accuracy vs compute curve

audio
→ event F1 with/without commentary
```

A feature with no measurable acceptance criterion is not ready to merge.

---

# 9. Preserve baseline comparability

Do not change at the same time:

```text
dataset
split
evaluation metric
threshold
model
preprocessing
```

and then attribute the score difference to one component.

For comparison runs, change one conceptual factor at a time where practical.

Always save the effective config.

---

# 10. Frozen test set rules

The frozen test set is not for development.

Never use it for:

- prompt iteration;
- model selection;
- threshold tuning;
- crop-selection tuning;
- manual example cherry-picking;
- debugging.

Use train/validation or a development subset.

Only run frozen tests for milestone evaluation.

If accidental leakage occurs, document it immediately and create a new untouched test partition if possible.

---

# 11. Required run metadata

Every non-trivial benchmark run must record:

```json
{
  "run_id": "...",
  "timestamp": "...",
  "git_commit": "...",
  "dirty_worktree": false,
  "dataset": "...",
  "split": "...",
  "config_hash": "...",
  "models": {},
  "prompt_versions": {},
  "schema_versions": {},
  "hardware": {},
  "software": {},
  "seed": 0
}
```

Store with the run.

Never report a benchmark if the model revision cannot be identified.

---

# 12. Model download and version rules

Never depend on floating model revisions for benchmark results.

When possible, record:

```text
repository/model ID
revision/commit
quantization
dtype
backend
transformers/vLLM version
visual-token budget
max context
generation parameters
```

A change from:

```text
Qwen3-VL revision X
```

to:

```text
Qwen3-VL revision Y
```

is an experiment, not a transparent dependency update.

---

# 13. VLM engineering rules

## 13.1 Structured outputs only

Semantic production calls must return data validated against a schema.

Do not parse free-form prose with fragile regular expressions when a structured schema can be used.

## 13.2 Prompts are versioned code

Store prompts under, for example:

```text
src/football_intelligence/vlm/prompts/
```

Each prompt has a version.

Changing wording can change benchmark output, therefore prompt changes require evaluation.

## 13.3 Never hide model uncertainty

Preserve:

```text
top candidate
alternatives
confidence
insufficient_evidence
evidence timestamps
```

If the backend cannot produce calibrated confidence, label the value clearly as a model score/heuristic and calibrate it separately.

## 13.4 No implicit closed-model fallback

The local/open-weight production path must never silently call:

```text
Claude
GPT
Gemini
other closed API
```

Remote/closed models may exist behind an explicit evaluation adapter only.

The user must know when one is selected.

## 13.5 Minimize VLM calls

Before making a call, ask:

```text
Can cached inference answer this?
Can structured DB answer this?
Can the graph answer this?
Can a cheap CV module answer this?
Does the clip need re-analysis?
```

Use the VLM where semantic reasoning matters.

---

# 14. Video-context rules

Do not brute-force an entire full-resolution match into one prompt.

For candidate events:

```text
retrieve exact source window
sample frames intentionally
include only relevant crops
include Player Memory summary
include geometry as compact structured data
```

Use long context primarily for:

- event summaries;
- player history;
- retrieved evidence;
- transcript;
- match state.

Not for thousands of redundant raw frames.

---

# 15. Frame/crop selection rules

Avoid repeated near-identical images.

Before VLM jersey inference, score observations for:

```text
sharpness
resolution
occlusion
back visibility
front visibility
side visibility
bbox area
track confidence
diversity
```

Select diversified top-K.

All selection thresholds/configuration must be logged.

---

# 16. Identity semantics

The following identifiers are different concepts and must never be conflated:

```text
detection_id
tracklet_id
global_player_id
jersey_number
roster_player_id
```

A jersey number is an attribute/hypothesis, not an internal primary key.

A tracklet is a temporal observation, not a person.

A player may have many tracklets.

---

# 17. Global identity solver rules

The solver may use:

```text
ReID similarity
team
jersey posterior
roster
temporal overlap
substitution state
role
track continuity
VLM evidence
```

Every constraint must be configurable and inspectable.

Do not encode competition-specific assumptions as universal rules.

When constraints conflict, preserve the conflict and confidence rather than silently overwriting evidence.

---

# 18. Ball geometry rules

Never treat pitch homography of an airborne ball as physically correct.

For semantic decisions, preserve:

```text
image-space coordinates
player-relative position
motion
visibility
temporal path
```

Pitch coordinates for the ball require a model that explicitly handles height/3D uncertainty.

When such a model is absent, report the limitation.

---

# 19. Event semantics rules

Do not infer an intentional pass solely from:

```text
player A possession
→ ball movement
→ player B possession
```

Use the semantic event model.

Maintain `unknown` and ambiguous classes.

Recommended MVP ontology:

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

Do not expand the ontology until the current subset has a stable annotated confusion matrix.

---

# 20. Evidence rules

Every semantic event must contain:

```text
source match
time range
actor hypothesis
target hypothesis if relevant
evidence timestamps
model/config provenance
confidence/abstention
```

No event should exist only as text in a VLM response.

---

# 21. RAG / Match Memory rules

The vector database is **not** ground truth.

Query order should prefer:

```text
exact structured filter
→ identity resolution
→ graph relation
→ semantic/vector retrieval
→ source-video verification
```

Vector matches must point back to stable evidence IDs.

Do not generate statistics from embeddings or summaries.

---

# 22. Audio rules

Commentary is supplementary evidence.

Never convert a spoken player name directly into confirmed visual identity without corroboration.

Keep:

```text
ASR text
timestamps
ASR confidence if available
speaker/channel metadata if available
```

Ablate audio before claiming it improves the model.

---

# 23. Cache rules

Cache expensive deterministic/interpretable stages.

Cache keys must include all inputs that materially affect output.

For VLM:

```text
video hash
clip interval
frame sampler
image/crop hashes
model revision
prompt version
schema version
generation config
```

Do not reuse stale cache after a prompt/model change.

---

# 24. Performance rules

Accuracy is the primary goal, but compute must be measurable.

Track:

```text
wall time
GPU time when available
peak VRAM
RAM
disk cache
frames processed
VLM images/frames
input/output tokens
number of first-pass calls
number of second-pass calls
```

Do not call a pipeline "faster" because one module is faster while hiding additional fallback calls.

---

# 25. Active re-analysis policy

Second-pass analysis should be triggered by measurable uncertainty.

Initial candidate triggers may include:

```text
confidence < threshold
top1-top2 margin < threshold
identity solver conflict
geometry/VLM disagreement
important event
jersey evidence insufficient
```

Thresholds are learned/tuned on validation, never on frozen test.

Log why every second-pass call occurred.

---

# 26. Experiment discipline

Change one hypothesis at a time.

Bad experiment:

```text
new tracker + new VLM + new prompt + new thresholds
→ score improved
```

Useful experiment:

```text
same pipeline
+ multi-tracklet Player Memory
→ jersey 72.1% → 82.4%
```

For multi-component changes, require ablations afterwards.

---

# 27. Failure-driven development

After every meaningful benchmark:

1. sort failures by category;
2. inspect representative cases;
3. estimate category prevalence;
4. choose the highest-leverage failure;
5. implement one targeted fix;
6. rerun.

Do not spend days optimizing an attractive corner case that affects <0.5% of errors unless it blocks a research claim.

---

# 28. Error taxonomy

Use consistent categories:

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

Extend only when existing categories cannot represent the failure.

---

# 29. Testing hierarchy

Run the smallest sufficient test first.

```text
1. unit test
2. component golden test
3. short integration clip
4. development benchmark subset
5. validation benchmark
6. frozen test benchmark
```

Do not run a 90-minute VLM inference to verify a JSON parser.

---

# 30. Golden tests

Maintain a small set of deterministic/reviewed examples for:

```text
jersey clear
jersey ambiguous
team swap
track fragment
pass
clearance
interception
duel
shot
ball airborne
camera cut
```

Golden tests should include expected evidence, not only labels.

---

# 31. Dataset integrity

Never commit copyrighted/raw match video unless licensing explicitly permits it.

Repository files should contain:

```text
download instructions
dataset identifiers
checksums where appropriate
annotations allowed by license
```

Respect SoccerNet and model licenses.

Document any dataset with restrictions.

---

# 32. Dependency rules

Prefer stable libraries and lock versions.

Avoid adding a dependency for trivial utilities.

Before introducing another infrastructure service (e.g. Neo4j, Kafka, separate vector DB), prove that the existing stack cannot satisfy the requirement.

Default principle:

```text
fewer services
+ clean interfaces
> architecture for hypothetical scale
```

---

# 33. Docker rules

All core services must have reproducible Docker execution.

Do not bake model weights or private datasets into images.

Use mounted/cache volumes.

GPU containers must fail with a clear message if CUDA is unavailable.

Keep a low-compute development mode for tests.

---

# 34. Configuration rules

No important threshold should be buried in source code.

Put tunable parameters under `configs/`.

Examples:

```text
detector confidence
NMS
track association thresholds
best-view K
VLM FPS
VLM clip windows
visual-token budget
uncertainty thresholds
identity weights
retrieval top-K
```

Every benchmark stores its resolved configuration.

---

# 35. Database migration rules

Do not mutate schemas manually in production data stores.

Use migrations.

Semantic schemas should be versioned independently from DB migrations.

Never delete evidence needed to reproduce historical benchmark outputs.

---

# 36. API/UI truthfulness

Frontend/API labels must correspond to measured semantics.

Forbidden:

```text
"Pass accuracy"
```

when the backend only measures generic possession transitions.

Prefer explicit names:

```text
semantic completed passes
possession transitions to teammate
visible-field distance
identity confidence
```

Expose coverage/uncertainty when it materially affects interpretation.

---

# 37. Code quality

Prefer:

- typed interfaces;
- small composable modules;
- pure functions in evaluation;
- Pydantic/JSON Schema for boundaries;
- explicit units in field names (`timestamp_ms`, `distance_m`);
- deterministic IDs;
- structured logging.

Avoid:

- giant orchestration classes;
- magic global state;
- implicit units;
- positional dictionaries with undocumented fields.

---

# 38. Units and timestamps

Internally prefer:

```text
time: integer milliseconds from source media start
frame: explicit source frame index
distance: meters
image coords: pixels or normalized values with explicit schema
confidence: [0,1] only when semantically meaningful
```

Do not use formatted `MM:SS` strings as primary database keys.

---

# 39. Git safety

Before destructive commands, inspect status.

Do not:

```text
git reset --hard
git clean -fd
force push
delete user data
```

unless explicitly requested.

Do not rewrite unrelated user changes.

Keep commits logically scoped when commits are part of the workflow.

---

# 40. Secrets

Never commit:

```text
API keys
tokens
passwords
private dataset credentials
SSH keys
```

Use `.env` / secret stores.

`.env.example` must contain names and descriptions, not real secrets.

---

# 41. Logging

Use structured logs.

Recommended fields:

```text
run_id
match_id
segment_id
tracklet_id
player_id
event_id
model
stage
latency_ms
error_code
```

Avoid logging base64 images or full prompts containing huge visual payload metadata.

---

# 42. Progress communication

When an agent works for a long sequence:

- report concrete findings, not narration;
- surface a discovered blocker immediately;
- do not repeatedly restate the plan;
- finish with what changed, tests run and remaining risk.

Do not claim success before tests finish.

---

# 43. Stop conditions

Stop expanding a feature and investigate if:

- its benchmark does not improve;
- runtime rises sharply without accuracy benefit;
- the metric cannot distinguish improvement;
- test leakage is suspected;
- ground truth is too ambiguous;
- model output is not reproducible enough to compare.

Do not hide a failed hypothesis. Record it in `docs/EXPERIMENTS.md`.

Negative results are useful.

---

# 44. First-task priority order

Unless a user explicitly requests otherwise, prioritize:

```text
P0 reproduce baseline
P1 jersey multi-view experiment
P2 Player Memory
P3 global identity solver
P4 semantic pass-vs-nonpass gold set
P5 VLM event classifier
P6 active re-analysis
P7 Match Memory / retrieval
P8 audio
P9 full research benchmark
```

Do not build a large frontend before P0–P5 are demonstrated.

---

# 45. Mandatory first experiment

Before broad implementation, test the main hypothesis on existing annotated clips.

Jersey experiment:

```text
A current system
B single best crop + Qwen3-VL
C one-tracklet top-K + Qwen3-VL
D whole-match linked-tracklet top-K + Qwen3-VL
E D + roster/global solver
```

Event experiment:

```text
A existing geometric rule
B candidate clip + Qwen3-VL
C B + structured geometry/player context
D C + active re-analysis
```

If the proposed architecture cannot beat the baseline on a carefully controlled small experiment, do not scale it yet.

---

# 46. Acceptance gates by phase

## Gate 0 — baseline

Required:

```text
one-command reproduction
machine-readable metrics
frozen split
failure report
```

## Gate 1 — perception

Target:

```text
detection recall ≥ 90%
track fragments < 1.5/player/30 s
```

## Gate 2 — identity

Target:

```text
jersey accuracy ≥ 80%
coverage ≥ 90%
```

## Gate 3 — semantic pass classification

Target on annotated gold set:

```text
macro-F1 ≥ 0.75
```

## Gate 4 — active re-analysis

Required:

```text
positive accuracy/compute trade-off
```

## Gate 5 — evidence QA

Required:

```text
100% returned semantic events have resolvable video evidence
```

---

# 47. Definition of an honest improvement

An improvement is valid only if:

```text
same/declared dataset
same/declared metric
no test tuning
sample size reported
configuration saved
model revision saved
failure cases available
compute cost reported
```

Prefer:

```text
+6.2 pp jersey accuracy
95% CI [...]
at +18% compute
```

over:

```text
looks much better
```

---

# 48. Final agent checklist

Before declaring a task complete:

- [ ] Did I read the relevant current state first?
- [ ] Did I avoid unrelated rewrites?
- [ ] Is the production path real, not mocked?
- [ ] Are IDs/units/schemas explicit?
- [ ] Did I preserve evidence provenance?
- [ ] Did I version prompt/model/config changes?
- [ ] Did I run the smallest relevant tests?
- [ ] Did I run the requested benchmark if applicable?
- [ ] Did I record experimental results?
- [ ] Did I update `docs/CURRENT_STATE.md`?
- [ ] Can another agent resume from disk without chat history?
- [ ] Did I avoid claiming an unmeasured accuracy improvement?

If any answer is "no", either fix it or state the limitation explicitly.
