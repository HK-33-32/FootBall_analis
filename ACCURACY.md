# Accuracy and benchmark provenance

## Measured inherited baseline

The only measured baseline currently available comes from the supplied legacy project at `C:\Users\Andrei\Documents\Football_grade\football_core\docs\ACCURACY.md`. It reports 2,250 SoccerNet GSR frames, IoU ≥ 0.5:

| Metric | Baseline |
|---|---:|
| Detection recall / precision | 77% / 89% |
| Role classification | 100% |
| Team | 80% |
| Jersey number | 59% accuracy at 86% coverage |
| Ball found | 89% of annotated ball frames |
| Pitch position error | 0.57 m median, 1.63 m p90 |
| Track fragments | 1.94/player/30 s |
| Legacy geometry-derived pass classification | 43% on ideal GT tracks/teams/ball |

These values have not yet been reproduced from this repository because the frozen annotations/split are not present here. They are retained verbatim and must not be described as results of the new pipeline.

## New implementation

No accuracy claim is made yet. Unit and integration tests verify contracts, identity constraints, evidence integrity, metrics and active re-analysis control flow; they are not a football-accuracy benchmark. A semantic score becomes reportable only after a reviewed gold set with source video evidence is added and Qwen3-VL is run with a pinned revision.

## SoccerNet validation dataset audit

The official SoccerNet GSR v1.3 validation archive was downloaded with
`SoccerNet==0.1.62` and verified with a full ZIP CRC pass. This is a dataset
audit, not a model score and not a frozen-test run.

| Property | Measured value |
|---|---:|
| Sequences / frames | 58 / 43,500 |
| Object annotations | 792,166 |
| Player + goalkeeper tracks | 1,222 |
| Jersey-labelled player frames | 83.60% |
| Tracks with any jersey label | 79.05% |
| Team-labelled player frames/tracks | 100% / 100% |
| Median player bbox height | 8.80% of 1080p frame |
| p10 / p90 bbox height | 5.74% / 13.06% |

Archive SHA-256:
`d60e517e0bfbce83d46a2805842d73bc98bbb65e84ba76be7b55af9bb3047a50`.
Raw audit: `runs/soccernet_valid_audit.json` (runtime artifact, not committed).

The audit supports multi-view/linked-tracklet jersey aggregation: roughly one
in five ground-truth player tracks has no jersey-labelled observation at all,
and most player crops are small. It does not by itself prove that the proposed
aggregation improves accuracy.

## First reproduced model result

The supplied Football Core was evaluated on the deterministic SoccerNet GSR
v1.3 validation sequence `SNGS-033` (750 frames). This is a development result
on one sequence, not the frozen test set and not a SOTA claim.

| Configuration | GS-HOTA | DetA | AssA | IDF1 |
|---|---:|---:|---:|---:|
| Official full identity metric | 38.617 | 20.859 | 71.508 | 33.648 |
| Diagnostic: no team/jersey matching | 70.221 | 76.931 | 64.157 | 73.472 |
| Diagnostic: team, no jersey | 45.539 | 31.280 | 66.320 | 41.809 |
| Diagnostic: jersey, no team | 51.862 | 38.641 | 69.628 | 53.002 |

The full run produced 6,998 detections and 38 identities against 7,707 GT
detections and 25 GT identities. The large gap between geometry-only and full
GS-HOTA localizes the dominant failure to identity attributes, especially team
assignment: predicted player observations were split 5,382/861 between the two
teams, versus 2,641/4,112 in GT. Based on this failure, the new Player Memory
path now quarantines globally collapsed team evidence instead of treating it as
a hard identity constraint. This improves truthfulness but is not presented as
an official metric improvement until evaluated on additional validation clips.

The run took 1,554.7 s on an RTX 5080 Laptop GPU: 6.8 s extraction, 809.1 s
perception and 736.5 s identity/tracklet stitching. Exact config, model hashes,
Docker image digest, ablations and limitations are recorded in
`runs/soccernet_valid_sngs033_core/benchmark.json`.

## Game-state refinement on four validation sequences

Four SoccerNet GSR v1.3 validation sequences were run end to end through the
supplied Football Core (`SNGS-021`, `SNGS-033`, `SNGS-045`, `SNGS-090`; 750
frames each, 25 fps, 1080p, RF-DETR large at confidence 0.35, Qwen2.5-VL jersey
reader), then post-processed by `football_intelligence.gamestate` and scored
again with the same pinned official GS-HOTA runner. These are development
results on the validation split, not the frozen test or challenge set.

| Sequence | GS-HOTA base | GS-HOTA refined | Δ | DetA base → refined | AssA base → refined | IDs → GT IDs |
|---|---:|---:|---:|---:|---:|---:|
| SNGS-021 | 50.529 | **51.568** | +1.039 | 38.611 → 39.400 | 66.131 → 67.505 | 30 → 28 (GT 23) |
| SNGS-033 | 38.617 | **64.751** | +26.134 | 20.859 → 55.265 | 71.508 → 75.877 | 38 → 30 (GT 25) |
| SNGS-045 | 30.758 | **45.870** | +15.112 | 17.197 → 32.258 | 55.014 → 65.227 | 46 → 38 (GT 25) |
| SNGS-090 | 33.777 | **61.806** | +28.029 | 19.927 → 50.963 | 57.262 → 74.985 | 35 → 27 (GT 24) |
| **Macro average** | **38.420** | **55.999** | **+17.579** | 24.148 → 44.471 | 62.479 → 70.898 | |

Raw output: `runs/gamestate_benchmark.json` (runtime artifact, not committed).
The refinement changes no geometry — detection recall and precision are
bit-identical before and after. Everything except the jersey re-read is CPU
post-processing, about 7 s per 750-frame sequence; the re-read adds roughly
10 s of crop staging plus 40–80 s of GPU inference, against 1,555 s of GPU
perception for the same clip.

### What the pipeline does

1. **Roles** are checked against geometry: a tracklet the backend calls a
   goalkeeper while he spends the clip in midfield is demoted to outfield
   player, which also returns his kit to the team clustering.
2. **Teams** are decided once per tracklet by clustering torso colour in
   CIELAB, and mapped to the `left`/`right` convention by the goalkeeper's own
   half plus an offside-line vote. Lightness is excluded from the clustering
   distance: it mostly encodes sun versus shadow and has the widest spread of
   the three channels, so at full weight it splits each team by illumination.
   When the chroma-only split comes out barely wider than the scatter inside
   each cluster — separation ratio below 3 — the kits differ mainly in
   lightness and the clustering is retried with lightness at full weight. On
   the four validation sequences the chroma ratio is 4.7 to 11.1 and the
   fallback never fires; on broadcast footage of white stripes against navy it
   measured 2.06, and the fallback lifted it to 10.6.
3. **Tracklets** that are one player either side of a gap are chained by a
   Hungarian assignment over fragment endpoints, gated on pitch distance and,
   where the backend kept them, on re-identification embedding similarity.
4. **Jersey numbers** are re-read by the backend's own VLM, but asked about the
   whole chained identity at once, from up to 24 crops spread across the clip
   rather than the eight largest boxes of one fragment (`scripts/reread_jerseys.py`).
5. **Numbers are made exclusive** within a team: two identities on the pitch at
   the same moment cannot wear the same shirt, so the weaker claim is struck
   and falls back to its next-best number.

### Where the gain comes from

`scripts/attribute_report.py` matches predictions to ground truth on pitch
geometry alone (the evaluator's own Gaussian similarity, 5 m tolerance) and
then scores each attribute on the matched pairs, isolating identity errors from
detection errors.

| Sequence | Det recall | Det precision | Role base → refined | Team base → refined | Jersey when emitted, base → refined |
|---|---:|---:|---:|---:|---:|
| SNGS-021 | 0.877 | 0.918 | 0.984 → **0.996** | 0.915 → **0.971** | 0.667 → **0.791** |
| SNGS-033 | 0.866 | 0.956 | 0.994 → 0.994 | 0.455 → **0.994** | 0.565 → **0.848** |
| SNGS-045 | 0.890 | 0.982 | 0.904 → 0.904 | 0.530 → **0.894** | 0.764 → 0.730 |
| SNGS-090 | 0.880 | 0.976 | 0.996 → 0.996 | 0.998 → 0.998 | 0.475 → **0.942** |

Mean team accuracy 0.725 → 0.961; mean jersey accuracy when a number is emitted
0.618 → 0.828. Detection is untouched by design.

### Component ablation (macro GS-HOTA over the four sequences)

| Configuration | Macro GS-HOTA | DetA | AssA |
|---|---:|---:|---:|
| Perception as delivered | 38.420 | 24.148 | 62.479 |
| Teams + tracklet linking + per-tracklet jersey vote | 48.265 | 34.800 | 67.972 |
| + one number per team at a time | 49.336 | 35.919 | 68.752 |
| + identity-level VLM re-read, without exclusivity | 52.867 | 40.227 | 69.776 |
| **Full** | **55.999** | **44.471** | **70.898** |

Two settings of the re-read were measured. Asking about 24 crops at least
400 ms apart scores 55.999; asking about 36 crops at least 300 ms apart scores
54.752, so the extra questions do not pay for the loss of spacing between
views. The reader's mass is scaled by how often it agreed with itself across
its own questions, and a tally whose winner holds under a third of the
questions asked is discarded rather than blended in — without that floor the
macro average falls to 52.4 at the same weight.

### Roles: what is fixable and what is not

Role is the one attribute where the measured ceiling is close to zero.
Substituting ground-truth roles into the refined output scores GS-HOTA 46.4 on
`SNGS-045` against the achieved 45.9, and 51.5 on `SNGS-021` against the
achieved 51.6 — a perfect role classifier is worth **+0.6 and −0.1**. The rows
carrying a wrong role also carry a jersey number the reader gets wrong, so
correcting the role only moves the mismatch from one attribute to another.

The geometric demotion above is kept because it is deterministic and does
correct the attribute: `SNGS-021` role accuracy 0.984 → 0.996 and team accuracy
0.958 → 0.971, at unchanged GS-HOTA. Two further rules were built, measured and
rejected:

* **Promote a deep, kit-distinct player to goalkeeper.** Fires correctly on the
  one mislabelled keeper in `SNGS-045`, but also on two genuine outfield players
  in `SNGS-090` who defend near their own goal in a distinctive kit. Macro
  GS-HOTA 56.0 → 54.6.
* **Promote a referee the jersey reader gives a number to.** Referees wear no
  number, so the test is sound in principle, and on one pass it fired on
  exactly the two mislabelled players in `SNGS-045`. It is not stable: on a
  second reading pass a genuine `SNGS-045` referee answered "10" on three of
  six questions, landing exactly on the threshold and costing 7.1 GS-HOTA on
  that clip. What the rule actually measures is how often the model
  hallucinates "10", not whether a number is there.

The remaining role error is therefore left in place and reported: `SNGS-045`
stays at 0.904, where three tracks the backend calls referees are players
wearing 30, 99 and 23.

### Honest limits

`SNGS-021` is the sequence the refinement helps least (+1.0). Its backend team
classifier already worked (0.915), and the re-read raises jersey accuracy
(0.667 → 0.791) while emitting numbers on 1,889 detections that the annotation
leaves unlabelled, up from 1,366. Reading a number the annotator could not read
is scored as an error, and roughly half of `SNGS-021`'s player detections carry
no ground-truth number at all.

The `identity_read_weight`, `identity_read_min_consensus` and view-selection
settings were chosen by sweeping on these same four validation sequences. They
are validation-tuned, not held out.

### Relation to published state of the art

The SoccerNet GSR challenge leaderboard is not the validation split used here,
so these numbers are **not** a like-for-like comparison and no state-of-the-art
claim is made. For context, Constructor Tech won the 2024 GSR challenge with
GS-HOTA 63.81 (GS-DetA 49.52, GS-AssA 82.23) against a 23.36 baseline, and the
best 2025 challenge entry reported 63.90. The refined system reaches a macro
average of 56.0 across four validation clips, with individual clips at 64.8 and
61.8. Its weakest clip, at 45.9, is what the average rests on; a challenge-set
submission would be needed before any comparison is meaningful.
