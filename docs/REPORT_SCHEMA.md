# Detailed match statistics, v1

The supplied 32-page RuStat Orenburg–Akron report (6 September 2026) is a
**content reference**, not a source of model labels or proprietary formulas.
This implementation does not reproduce the provider's index or claim provider
accuracy. The included demonstration JSON comes from the existing Argentina–France
clip, **not** from an inferred Orenburg–Akron video.

## Generate JSON

```powershell
python scripts/build_match_report.py `
  --predictions runs/user_arg_fra_clip/predictions_refined.json `
  --video-source runs/user_arg_fra_clip/clip_web.mp4 `
  --match-id user_arg_fra_clip --fps 25 `
  --title "Argentina–France 2022 — existing clip" `
  --report-config configs/report_metrics.json `
  --output runs/reference_report_20260907/match_report.json `
  --statistics-output runs/reference_report_20260907/statistics.json
```

The old report fields remain backward-compatible. The new subtree is
`detailed_statistics`; `--statistics-output` writes that subtree separately.
New full-match analyses produce it automatically. The viewer's **Статистика JSON**
button downloads the detailed subtree when present (legacy JSON otherwise).
`GET /api/v1/reports/{report_id}/statistics` also derives the subtree for old
reports, without modifying their files. Old reports must declare source FPS.
For API analysis, `FI_REPORT_METRICS` selects a JSON config; Docker Compose points
it to `/app/configs/report_metrics.json`. Every report saves its resolved values.

## Coverage compared with the reference

| Reference pages | JSON sections | Available evidence / limitation |
|---|---|---|
| 2–3: match, team totals | `match`, `teams.*.general`, `shots`, `passes`, `duels` | Roster/scoreboard when supplied; geometric candidates are estimates |
| 4–5: progression, recoveries/losses | `progression`, `recoveries_losses`, `pressing`, `attacks` | Spatial/semantic fields require corresponding event annotations |
| 6: dynamics | `periods[].teams` | Source-video intervals by default; supply actual half/clock boundaries |
| 7–8, 11–12: players | `players[].metrics`, `physical_visible_field` | Tracking distance/time/heatmap are visible-field quantities, not full-match totals |
| 9–10: passing pairs | `passing_network` | Counts plus stable event IDs; geometric candidates are not confirmed completions |
| 13–31: action maps | `event_maps` | Source coordinates and event IDs; unavailable coordinates remain null |
| 32: terminology | `resolved_config`, metric definition version | Explicit independent definitions; no reverse-engineered provider formulas |

Included categories cover goals/assists/chances/saves, fouls/cards/offsides/corners,
shot breakdowns, xG/xA from identified external models, pass lengths/directions/
progression/key passes/crosses/set pieces, duels/dribbles/tackles, interceptions/
recoveries/losses, attacks/flanks, pressure/build-ups/PPDA, entries/carries and
possession counts/durations. Merely having a schema field does **not** mean that
the video pipeline detects that event class accurately.

## Truthfulness contract

Count metrics have `value`, `observed_count`, `successful`, `success_pct`, `status`,
`unknown_qualifiers`, `evidence_ids` and an unavailable `reason`. Measurement fields
have a value, unit, status and evidence references. Status means:

- `verified`: event provider explicitly certified complete type coverage and
  all contributing events were marked verified.
- `partial`: observed verified events, but coverage is not certified complete.
- `estimated`: geometric/model estimates; not measured semantic accuracy.
- `unavailable`: insufficient coverage/qualifiers to calculate the field.

`null` is not zero. A verified zero is emitted only when a provider certifies
complete coverage for that event type over the requested interval. A zero within
partial observations cannot establish absence across the match. Missing outcomes
do not become failures. Geometric teammate transitions have unknown semantic
outcomes; they cannot create artificial 100% pass completion. A network exposes
`observed_candidates` separately from confirmed `completed` counts.

Stats evidence IDs point into `events`; each event retains `evidence_ids`, source,
millisecond timestamps, actor/target hypotheses, status and optional confidence
with its calibration/score kind. IDs supplied by an annotation provider are not
joined to internal track IDs by string coincidence. xG/xA values require an
`xg_model` / `xa_model` qualifier on every contributing shot/pass.

## Richer statistics from reviewed/semantic annotations

Pass `--event-annotations path.json` with a `ReportAnnotations` document.
Its authoritative Pydantic schema is in `football_intelligence.reporting`;
`ReportAnnotations.model_json_schema()` produces JSON Schema.
Annotations replace, rather than double-count, geometric candidates **only in
the detailed subtree**. Legacy viewer fields remain unchanged. `match_id` must
match `--match-id`. Explicit `teams` can use provider IDs rather than `left/right`.

The following is an explicitly synthetic, documentation-only format example:

```json
{
  "schema_version": "1.0.0",
  "match_id": "documentation-fixture",
  "source": "reviewed-event-provider/revision",
  "teams": ["A", "B"],
  "coverage_start_ms": 0,
  "coverage_end_ms": 60000,
  "complete_event_types": ["pass"],
  "periods": [{"name": "reviewed-window", "start_ms": 0, "end_ms": 60000}],
  "events": [{
    "event_id": "example-pass-1", "kind": "pass", "timestamp_ms": 10000,
    "team": "A", "player_id": "roster-a10", "target_id": "roster-a9",
    "outcome": true, "position_m": [-10, 5], "end_position_m": [12, 3],
    "attacking_sign": 1, "evidence_ids": ["source-video#t=9,12"],
    "source": "manual-review/revision", "status": "verified",
    "qualifiers": {"key": false, "cross": false, "open_play": true}
  }]
}
```

Do not certify complete coverage just because inference finished. For Boolean
qualifiers, `false` is negative evidence; omission is unknown. Types in the
metric catalog use `goal`, `assist`, `chance`, `save`, `foul`, `foul_suffered`,
`offside`, `corner`, `card`, `shot`, `pass`, `duel`, `dribble`, `tackle`, `loss`,
`recovery`, `interception`, `miscontrol`, `clearance`, `attack`, `pressure`,
`build_up`, `carry`, `entry`, `possession`. Other types remain in the event maps.

## Geometry and time definitions

- Pitch coordinates: metres, origin at centre. `attacking_sign` is **per event**,
  allowing teams to switch direction. No implicit match-long direction prior.
- Short passes: `<10 m`; medium: `10–40 m` inclusive; long: `>40 m`.
- Forward/backward: within a 120-degree cone, cosine at least ±0.5. Progressive:
  forward longitudinal gain at least 30 m ending in own half, 15 m crossing
  midfield, 10 m starting in opponent half. All thresholds are configurable.
- Penalty area uses configured pitch and area dimensions. PPDA requires complete
  passes, tackles, interceptions, duels and fouls with coordinates/direction;
  opponent completed passes in their 40-m defensive zone divided by own defensive
  actions there. Zero denominator returns null.
- Possession is supplied as non-overlapping spans with `end_ms`. Period durations
  split spans at boundaries; the spell count within a period counts overlapping
  spans, not only spells beginning there. Point events use half-open time bins.
- Existing inference frames are 1-based; detailed timestamps convert frame 1 to
  source time 0. Legacy formatted timecodes are preserved in the legacy report.
- Camera event maps optionally use the player's ground-contact location. They
  are labelled `player_foot_proxy`, not an airborne ball's true pitch position.
- No whole-match minutes, formations, pressure intensity or proprietary index is
  invented from absent observations. Tracking proxies are in `observed_tracking`.

Changing the ontology, geometry definitions or meaning of a field requires a
metric/schema version bump and tests. The exact config and annotation hash are
stored in the output; generated reports also retain the prediction-file hash.

## Coverage correction (metric definition `reference-report-v2`)

Known source duration includes leading/trailing frames without detections in the
coverage denominator. CLI: `--duration-ms 7600`; browser analyses use the probed
source interval. Match/player percentages use the same source-frame denominator.
`match.coverage_basis` is `source_duration` or `observed_prediction_span`, and
`source_duration_ms` is null when unknown. Unknown-duration reports must not label
their observed span as complete clip coverage. Event annotation coverage remains
the provider's separately declared interval.
