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

1. **Roles** are checked against geometry. A tracklet the backend calls a
   goalkeeper while he spends the clip in midfield is demoted to outfield
   player, which also returns his kit to the team clustering. And because one
   referee runs inside the field of play while the assistants stay beyond the
   touchline, simultaneous in-field referees are a contradiction: the heaviest
   chain of referee tracklets that is continuous at a jogging pace is kept and
   the rest are demoted.
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
| SNGS-045 | 0.890 | 0.982 | 0.904 → **0.991** | 0.530 → **0.987** | 0.764 → 0.730 |
| SNGS-090 | 0.880 | 0.976 | 0.996 → 0.996 | 0.998 → 0.998 | 0.475 → **0.942** |

Mean team accuracy 0.725 → 0.988; mean jersey accuracy when a number is emitted
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

What finally fixed `SNGS-045` was not colour but count. The three tracks the
backend calls referees there are players in yellow shirts; the officials wear
black, and once illumination is factored out black and the white team are the
same colour, so no kit test separates them. The structural constraint does:
the genuine referee's fragments chain at 2.9 and 4.8 m/s while attaching the
impostor would need 15.0 m/s. Role accuracy on that clip goes 0.904 → 0.991
and team accuracy 0.894 → 0.987, because the demoted tracks rejoin the kit
clustering. GS-HOTA is unchanged at 45.87, exactly as the oracle above
predicted: those rows now fail on the jersey number instead.

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

## Trajectories, statistics and the viewer

The ball is excluded from GS-HOTA scoring, so none of this moves the
benchmark. It matters because a game state that cannot be read as motion is
not usable as one.

Read literally, the exported ball track teleports. On the broadcast clip in
`runs/user_arg_fra_clip` a third of its frame-to-frame steps implied speeds
above 40 m/s and the worst implied 2,075 m/s, against roughly 35 m/s for a
struck ball; 15 detections sat outside the pitch altogether.
`football_intelligence.trajectories` drops off-pitch points, keeps the
largest subsequence a ball moving at up to 30 m/s could have visited,
interpolates short gaps and smooths what is left:

| Clip | Ball steps before | Ball steps after | Points kept |
|---|---:|---:|---:|
| SNGS-033 | p90 21.3, max 1,019.7 m/s | p90 14.9, max 36.0 m/s | 580 of 645 |
| SNGS-045 | p90 47.7, max 667.5 m/s | p90 26.7, max 40.6 m/s | 678 of 710 |
| Broadcast clip | p90 431.9, max 2,074.8 m/s | p90 17.3, max 34.4 m/s | 251 of 288 |

The same treatment applies to players at a human top speed, and it has to:
before it, summing raw positions gave a referee 77 m of running in eight
seconds and top speeds above 90 km/h. Two details carry most of that fix.
Smoothing and speed windows are aligned to frame numbers rather than to
positions in a list, so a hole in a track is never averaged across; and speed
is measured over a baseline rather than between neighbouring frames, because
dividing a few centimetres of calibration noise by 40 ms reads as a sprint.
Residual noise still lifts the fastest readings for distant players, which is
stated on the page rather than hidden.

`scripts/build_match_report.py` derives per-identity distance, speeds,
sprints, thirds and time nearest the ball, each with the timecodes it came
from, and `scripts/build_match_viewer.py` renders a self-contained viewer
around it. Both report coverage prominently: on the broadcast clip the pitch
is only calibratable for 44% of the frames, the rest being close-ups the
broadcast cut to, so every figure describes what happened on camera.

## Frames the calibration cannot be trusted on

A homography estimated from pitch markings does not fail loudly on a camera
the estimator has not seen: it returns a mapping, and every player gets a
position that is simply wrong. On the broadcast clip in
`runs/user_arg_fra_clip` the footage cuts to a wide tactical camera at 23 s,
and after that cut the mapping is *mirrored*. The rank correlation between a
player's position across the image and his position along the pitch is +0.87
before the cut and reaches -0.97 after it, so the left winger lands on the
right touchline; one frame collapsed twelve players onto a single point, and
others projected them past the touchline.

`football_intelligence.calibration` rejects such frames rather than
publishing positions from them, on three ground-truth-free tests: the image
order must follow some pitch axis and agree in sign with the rest of the
clip, the people in a frame cannot occupy a few metres, and players standing
on the pitch cannot project outside it. The correlation floor of 0.7 is taken
from the footage itself -- no frame in the healthy part of that clip falls
below it.

| | Frames kept | Median order correlation | Below 0.7 |
|---|---:|---:|---:|
| Before the cut | 183 | 0.87 | 0% |
| After the cut, unfiltered | 142 | 0.52 | 45% |
| After the cut, filtered | 56 | 0.93 | 0% |

90 of 340 frames are dropped, all of them after the cut, and clip coverage
falls from 44% to 33%. That is the honest number: the earlier 44% included
frames whose statistics were not noisy but meaningless.

## Cost of a full match, and where it went

Perception started at forty-four to eighty-two times real time on this
hardware, split almost evenly between detection and the backend's own identity
stitching. That is where this section begins; by the end of it the same
predictions arrive at fourteen to twenty-four times real time.

| Clip | Frames | Extract | Analyse | Stitch | Total |
|---|---:|---:|---:|---:|---:|
| SNGS-021 | 750 | 14 s | 802 s | 1,054 s | 2,453 s |
| SNGS-045 | 750 | 17 s | 650 s | 1,133 s | 1,802 s |
| Broadcast clip | 772 | 6 s | 550 s | 807 s | 1,366 s |

The chunked run below came out at 65x, so a ninety-minute match was 98 hours of
wall time on this machine, with the spread putting it between 66 and 123. What
that time was spent on, and why a faster card would barely dent it, is measured
below.

Four things were done about it: screen the footage, bound the jobs, stop the
analysis being quadratic, and stop the pipeline waiting on itself.

### Screening before the GPU

Much of a broadcast cannot produce a game state at all: replays, dugout
shots, graphics, and the close-ups a director cuts to whenever the ball is
dead. `football_intelligence.playability` finds the pitch by the share of
the frame that is lit grass, sampling the decoded video at a stride.

| | Value |
|---|---:|
| Scan speed | 21x real time (1.5 s for a 31 s clip) |
| Footage kept | 58% |
| Frames that produced a game state and were kept | 249 of 250 (**100%**) |
| Of what it keeps, share that really yielded state | 60% |

Precision is deliberately the loose end: the pass is permissive, and the
calibration gate removes what survives it but still cannot be trusted. On
this clip it removes 42% of the footage before the GPU sees it, and a CPU scan
of a full match takes about four minutes. At the original 65x that saved forty
machine-hours of ninety-eight; at the current 24x it saves fifteen of
thirty-six, and a screened match costs about 21 hours.

### Where the time actually goes

The hours quoted above are wall-clock hours during which the pipeline holds the
machine -- measured on an RTX 5080 Laptop (16 GB, 90 W, driver 610.88, CUDA
12.8) with the backend in Docker. It is not a measure of GPU work, and the
distinction matters, because the GPU is mostly idle while it waits.

Sampled once a second from inside the container over a full job:

| Stage | Share of the job | GPU utilisation (mean) | Seconds at 10% or less | Peak VRAM |
|---|---:|---:|---:|---:|
| extract | 0.5% | ~0% | -- | -- |
| analyze -- detection, tracking, calibration | 37% | **6.3%** | 78% | 9.5 GB |
| idatr -- stitching, projection, jersey VLM | 62% | **10.1%** | 75% | **14.3 GB** |

Container CPU over the same job averaged 892% of 1,600% available during
analyze and 588% during idatr, peaking near 1,800%. So neither processor is
saturated: the pipeline is serial. Frames go through one at a time, models are
loaded and dropped between stages -- hence a mean VRAM of 3 GB against a peak
of 14.3 -- and the jersey VLM answers one crop per call.

Two consequences for anyone trying to make this faster. A quicker GPU buys
little on its own, since the one here is idle three seconds in four. What the
measurement does argue for is concurrency: chunks are already independent, so
several can run at once, and the binding constraint is the 14.3 GB VRAM peak
rather than compute -- a card with 24 to 48 GB would hold two to four workers
on a GPU that currently has capacity to spare.

### Making the serial pipeline less serial

The utilisation figures say the machine was waiting, not working, so the next
step was to find out on what. The backend runs each stage as its own process,
and both of the heavy ones were instrumented to record where their seconds
went. Measured on 7.6 s of broadcast footage:

| | Before | After |
|---|---:|---:|
| analyze -- tracking | 158 s | 86 s |
| ... of which the jersey head | 47.3 s | 7.3 s |
| ... of which re-ID | 28.5 s | 9.1 s |
| ... of which detection | 18.0 s | 8.0 s |
| idatr -- gen_tracklets | ~249 s | 16-22 s |
| idatr -- tracklet_attributes | 167 s | 52 s |
| **Whole job** | **624 s** | **180 s** |

Four changes, none of them to the models or their arithmetic.

*Re-ID was called once per frame.* ``gen_tracklets`` built a batch out of the
twenty or so crops one frame holds -- growing it by ``torch.cat`` one crop at a
time -- and sent it to the GPU. That is a GPU call per frame of the match, each
too small to fill the card. Crops now accumulate across frames and go in
batches of 256. In the analyze stage the same thing happened next to a detector
that was already working eight frames at a time; re-ID now uses that batch too.

*Crops were prepared on one core.* Decoding a frame, cutting a player out and
resizing him is CPU work that releases the GIL, so it spreads over cores; done
serially it left fifteen of sixteen idle while the GPU waited. Frame decoding,
the re-ID transform, the CLIP preprocess and the torso histograms now run on a
thread pool. The CLIP preprocess alone was the single largest item in tracking.

*The same network was loaded twice per job.* ``gen_tracklets`` and
``tracklet_attributes`` share a process and each built its own OSNet, reaching
the same weights file through two different import paths -- ``torchreid`` and
``reid.torchreid`` -- which Python treats as two modules. Ten seconds and a
second copy in VRAM, for nothing.

*Frames were held longer than needed.* Crops were numpy views into their frame,
which kept every decoded frame of the clip alive. They are copied now, and
frames are decoded a block at a time, so a minute of 1080p no longer implies
nine gigabytes of live frames.

The point of all four is that the predictions must not move, and they do not:

| Clip | Before | After | Speed-up | Detections | Attributes identical |
|---|---:|---:|---:|---:|---|
| Broadcast chunk, 7.6 s | 624 s | 180 s | **3.5x** | 2,664 | 2,664 of 2,664 |
| SNGS-021, 30 s | 2,453 s | 427 s | **5.7x** | 12,131 | 12,131 of 12,131 |

Byte-for-byte the same game state, so GS-HOTA is unchanged at 55.998 by
construction rather than by re-measurement. Perception now costs 14x to 24x
real time instead of 44x to 82x, which puts a screened ninety-minute match at
roughly 12 to 21 hours instead of 57.

What is left is better balanced than it was. On SNGS-021 the analyze stage is
now calibration at 213 s beside tracking at 236 s -- two parts that run
concurrently and finish together, so there is little to win in either without
the other. The changes live in the ``football-core`` source tree, with the
originals kept beside them in ``optim_backup/``.

### Events, and what a single camera can and cannot settle

The report now carries events, all of them derived from one sequence: the chain
of possession spells. A **pass** is the ball moving to a team-mate; an
**interception** is the same movement ending at an opponent who was nowhere
near the passer; a **tackle** is possession changing hands while the two
players were close enough to have contested it. That last distinction is a
distance, not a guess, and it is the only thing separating a defender who read
the play from one who won the ball in a challenge.

Per player the report counts touches, passes and passes completed, passes
received, progressive passes and passes into the final third, losses,
interceptions, tackles, times dispossessed, shots and shots on target, goals,
assists, carries and the metres carried. Per team, the same totals.

Two things the possession chain needed before any of it read as football.

*Ownership flickers.* Two players contesting a loose ball are each nearest to
it on alternating frames. Taken literally that is a tackle every other frame:
on the broadcast clip it produced seven events, six of which were the same two
duels. A touch now has to last a fifth of a second before it counts.

*A duel is one event.* Even debounced, players wrestling over a ball trade it
several times before it settles. Changes between the same pair inside a second
and a half are one contest. The clip went from 7 events to 4: a tackle, an
interception, a 12.6 m pass and an 11.1 m carry, which is what is actually on
the tape.

Possession flow is computed off that same chain, so the page cannot report nine
turnovers above a list holding one.

### Goals, and why the scoreboard decides

A goal is the one event a single camera cannot settle. A ball passing behind
the net looks identical to one crossing the line, and the tracker usually loses
the ball at exactly the moment it matters. The broadcast already carries the
answer in its scoreboard graphic, and the perception container already has a
model that can read it.

`scripts/read_scoreboard.py` samples frames, has Qwen2.5-VL read the score, and
makes the series monotonic -- a score cannot fall, so a misread frame is
dropped rather than inventing a goal, and a change has to be confirmed by a
second frame before it counts, dated to the frame it was first seen. On the
broadcast clip it read **8 of 8 frames correctly at 1-0**, which is the true
score at that point of the 2022 final.

With a timeline supplied the scoreboard decides, including when it says nothing
happened: the clip contains no goal, and the report says so while still
reporting the standing 1-0. Without one, the ball position is used and the
result is labelled as such.

### Fouls and cards: one works, one does not

A foul is not visible in tracking; its consequence is. Play stops, the ball
sits still, and the referee walks over. `find_stoppages` marks stretches where
the ball moved under 1 m/s for at least 1.6 s with an official within 12 m --
candidates for a human to look at, and honestly labelled as such, since a
throw-in, a substitution and a booking are indistinguishable from here.

Reading the card itself was tried and **does not work on this footage**. At
Qatar 2022 the officials wore yellow. Asked whether a referee was showing a
card, Qwen2.5-VL answered "yellow" on two of three frames of open play with no
card in them -- and still did after the prompt was tightened to describe a
raised rectangular card, and again on crops tightened to the referee himself.
The kit is the yellow it sees. The script is kept, with that measurement in its
docstring and a warning it prints at startup, and the supported path is feeding
a verified list to `build_match_report.py --cards`. Nothing about the stoppage
candidates depends on it.

The ceiling on all of this is the ball. Every event needs it, and on the
broadcast clip the ball is tracked in 83% of the frames that have a game state
at all.

### Bounded, resumable jobs

`football_intelligence.longmatch` cuts the playable stretches into chunks of
at most a minute, rebases each chunk's frame numbers onto the match clock and
gives each its own track-id range. A failure costs one chunk, and a rerun
skips what is already on disk. Identity is not carried across a chunk
boundary -- chunks are cut where the broadcast already cut away, so the
tracker had lost the player there anyway.

The chain was run end to end on the 31-second broadcast clip: 3 chunks,
1,149 s of GPU, merged into 4,669 detections over 48 identities, refined in
3.6 s and reported. Against a single pass over the same clip it produces the
same football:

| | Chunked | Single pass |
|---|---:|---:|
| Frames with a game state | 263 | 250 |
| Ball points kept | 218 of 239 | 207 of 215 |
| Distance, left / right | 238.6 / 115.5 m | 254.8 / 118.9 m |
| Shape width, left / right | 24.7 / 26.5 m | 25.2 / 26.7 m |
| Line height, left / right | -21.8 / +18.6 m | -22.4 / +18.6 m |
| Turnovers | 7 | 9 |

It reaches that having skipped 42% of the footage on the GPU, and with three
chunk boundaries deliberately breaking identity.

The run did expose a real defect, in kit clustering rather than in chunking.
The chroma-only split scored a separation ratio of 3.43 -- just above the 3.0
gate -- so the lightness fallback this matchup needs did not fire, and the
teams came out 26 against 11. Separation alone cannot catch that, but size
can: both teams are on the pitch throughout, so a split holding three
quarters of the observed mass is not a split by kit.

| Clip | Chroma mass share | Chroma ratio | Correct? |
|---|---:|---:|---|
| SNGS-021 | 0.536 | 9.29 | yes |
| SNGS-033 | 0.541 | 6.15 | yes |
| SNGS-045 | 0.554 | 4.69 | yes |
| SNGS-090 | 0.585 | 11.08 | yes |
| Broadcast clip, merged | **0.761** | 3.43 | **no** |

A lopsided split is now retried with lightness and the retry kept only if it
is more even. The gate sits at 0.62, between the worst correct split (0.585)
and the broken one. It does not fire on any benchmark sequence and macro
GS-HOTA is unchanged at 55.998; the merged clip goes to 18 against 19 and its
numbers fall in line with the single pass, as above.

### The analysis had to stop being quadratic

The trajectory cleaner compared every detection with every earlier one. That
is invisible on a 30-second clip and fatal on a match:

| Ball detections | Before | After |
|---:|---:|---:|
| 800 | 0.79 s | 0.08 s |
| 3,200 | 23.3 s | 0.31 s |
| 51,200 | (hours) | 7.9 s |
| 135,000 (90 min) | **~12 hours** | **~21 s** |

The fix is a lookback window -- a body can only have come from where it could
have reached -- plus processing each stretch of a track separately instead of
keeping one longest chain, which is also the more honest model for footage
that cuts away and comes back. The possession pass was likewise indexed by
frame rather than scanned per track. End to end, the analysis of the
broadcast clip went from 0.85 s to 0.52 s, and now scales linearly.

### Statistics added

Per player: distance and time in five speed zones (walk, jog, run,
high-intensity, sprint), accelerations and decelerations with a peak, the
timecode of top speed, the width and depth of the area he worked in, and the
number of separate episodes nearest the ball. Per team: those totals plus
shape -- width, depth, compactness and line height measured towards the goal
the team attacks -- and possession flow: spells, turnovers, and how long a
spell lasted. Acceleration is the shakiest of these, being a second
derivative of a noisy position, and is taken from the smoothed speed series
over a baseline rather than between frames.
