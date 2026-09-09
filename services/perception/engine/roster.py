"""Turning a match roster into constraints on jersey numbers.

Two failure modes the per-tracklet vote cannot fix on its own:

* the model returns a number nobody on the pitch wears (27, 74 — squads are
  numbered 1..26 at a World Cup);
* two players visible in the same frame end up with the same number.

A roster fixes both.  It gives the set of numbers that exist for each team, and
it makes "one number belongs to one player" a hard constraint: tracklets whose
time ranges overlap cannot claim the same number.  Tracklets that never appear
together may share one — that is the normal case of a single player whose track
was broken by an occlusion or a camera cut.
"""
from __future__ import annotations

import json
import os


class Roster:
    """Numbers and names for the two teams of one match."""

    def __init__(self, data: dict):
        self.name = data.get("name", "")
        self.teams = []
        for entry in data.get("teams", [])[:2]:
            players = {int(k): v for k, v in (entry.get("players") or {}).items()
                       if str(k).isdigit()}
            self.teams.append({
                "team": entry.get("team", ""),
                "short": entry.get("short", ""),
                "players": players,
                "numbers": set(players),
                "goalkeepers": {int(g) for g in entry.get("goalkeepers", [])
                                if str(g).isdigit()},
            })

    @classmethod
    def load(cls, source) -> "Roster | None":
        """source: path, JSON string or dict."""
        if not source:
            return None
        try:
            if isinstance(source, dict):
                data = source
            elif os.path.isfile(str(source)):
                with open(source, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
            else:
                data = json.loads(source)
        except Exception as exc:
            print("[roster] не удалось прочитать заявку: %s" % exc, flush=True)
            return None
        roster = cls(data)
        return roster if roster.teams else None

    def numbers(self, index: int, role: str | None = None) -> set:
        """Numbers a tracklet of this role may wear.

        A goalkeeper wears one of the keeper numbers, and an outfield player
        never wears one — that alone removes a whole class of mistakes.
        """
        if index >= len(self.teams):
            return set()
        team = self.teams[index]
        keepers = team["goalkeepers"]
        if role == "Goalkeeper" and keepers:
            return set(keepers)
        if role == "Player" and keepers:
            return team["numbers"] - keepers
        return team["numbers"]

    def name_of(self, index: int, number) -> str | None:
        if index >= len(self.teams) or number is None:
            return None
        return self.teams[index]["players"].get(int(number))

    def describe(self) -> str:
        return " / ".join("%s (%d номеров)" % (t["short"] or t["team"], len(t["numbers"]))
                          for t in self.teams)


def match_clusters_to_teams(scores_by_track, cluster_by_track, roster) -> dict:
    """Decide which appearance cluster is which team on the roster.

    Both mappings are scored by how much of the accumulated number evidence
    falls inside each squad's number set; the better one wins.
    """
    if roster is None or len(roster.teams) < 2:
        return {}
    clusters = sorted({c for c in cluster_by_track.values() if c in (0, 1)})
    if len(clusters) < 2:
        return {clusters[0]: 0} if clusters else {}

    def fit(mapping):
        total = 0.0
        for track_id, cluster in cluster_by_track.items():
            if cluster not in mapping:
                continue
            allowed = roster.numbers(mapping[cluster])
            for number, score in scores_by_track.get(track_id, {}).items():
                if number in allowed:
                    total += score
        return total

    direct = {clusters[0]: 0, clusters[1]: 1}
    swapped = {clusters[0]: 1, clusters[1]: 0}
    return direct if fit(direct) >= fit(swapped) else swapped


def correct_teams(scores_by_track, cluster_by_track, roster, team_by_cluster,
                  min_score=2.0):
    """A number that only one squad wears tells us which team the track is.

    Appearance clustering does fail on individual tracklets, and without this
    the roster makes such a failure worse: a France player read as "18" would
    have that reading thrown away (Argentina has no 18) and replaced by the
    best Argentinian number instead.  Reading the evidence the other way round
    fixes the team rather than corrupting the number.
    """
    if roster is None or len(roster.teams) < 2 or not team_by_cluster:
        return {}
    first, second = roster.numbers(0), roster.numbers(1)
    unique = {n: 0 for n in first - second}
    unique.update({n: 1 for n in second - first})

    cluster_of_team = {team: cluster for cluster, team in team_by_cluster.items()}
    corrections = {}
    for track_id, scores in scores_by_track.items():
        if not scores:
            continue
        number, score = max(scores.items(), key=lambda kv: kv[1])
        if score < min_score:
            continue
        team = unique.get(number)
        if team is None:
            continue
        current = team_by_cluster.get(cluster_by_track.get(track_id))
        if current is not None and current != team and team in cluster_of_team:
            corrections[track_id] = cluster_of_team[team]
    return corrections


def _same_player(boxes, first, second) -> float:
    """Median IoU of two tracklets over the frames they share."""
    if not boxes:
        return 0.0
    a, b = boxes.get(first) or {}, boxes.get(second) or {}
    common = a.keys() & b.keys()
    if not common:
        return 0.0
    values = []
    for frame in common:
        ax, ay, aw, ah = a[frame]
        bx, by, bw, bh = b[frame]
        ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
        iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
        inter = ix * iy
        union = aw * ah + bw * bh - inter
        values.append(inter / union if union > 0 else 0.0)
    values.sort()
    return values[len(values) // 2]


def assign_numbers(scores_by_track, unknown_by_track, intervals, cluster_by_track,
                   roster, team_by_cluster, roles=None, min_score=1.0,
                   unknown_ratio=0.15, margin=0.6, boxes=None, share_iou=0.5,
                   debug=False):
    """Give every tracklet a number, honouring the roster and uniqueness.

    Greedy over the strongest evidence first: a tracklet takes the best number
    that no overlapping tracklet of the same team has already claimed.

    ``margin`` keeps the roster honest.  A candidate is only accepted while it
    still carries at least this share of the tracklet's own strongest reading;
    otherwise the tracklet stays unknown.  Without it a short roster would hand
    every track some name, because the eleven allowed numbers soak up all the
    probability mass even when nothing was actually read.

    Uniqueness is *not* enforced here.  The tracker splits one player into
    several overlapping tracklets (73 of them for 22 players in a minute), so
    rejecting every tracklet whose number is "taken" silently drops readings
    that are plainly correct.  Instead each tracklet keeps its own best reading
    together with a confidence, and the frame-level writer shows a number on at
    most one box per team per frame — the most confident one.
    """
    reasons = {}
    candidates = []
    for track_id, scores in scores_by_track.items():
        # Rank by how sure the tracklet is, not by how much mass it piled up:
        # accumulated mass grows with track length and with the VLM boost, so a
        # short lucky close-up used to outrank a long, consistently read track
        # and steal its number.
        total = sum(scores.values()) + unknown_by_track.get(track_id, 0.0) + 1e-9
        cluster = cluster_by_track.get(track_id)
        allowed = None
        if roster is not None and cluster in team_by_cluster:
            allowed = roster.numbers(team_by_cluster[cluster],
                                     (roles or {}).get(track_id))
        kept = 0
        for number, score in scores.items():
            if allowed is not None and number not in allowed:
                continue
            kept += 1
            candidates.append((score / total, score, track_id, number))
        if not kept:
            reasons[track_id] = "нет допустимых номеров в заявке"
    candidates.sort(reverse=True)          # most confident reading first

    assigned: dict = {}
    confidence: dict = {}
    for conf, score, track_id, number in candidates:
        if track_id in assigned:
            continue
        unknown = unknown_by_track.get(track_id, 0.0)
        if score < min_score or score < unknown_ratio * unknown:
            reasons.setdefault(track_id, "слабое чтение (%.1f при %.1f 'не видно')"
                               % (score, unknown))
            continue
        best = max(scores_by_track[track_id].values(), default=0.0)
        if score < margin * best:
            reasons.setdefault(track_id, "лучшее чтение вне заявки (%.1f против %.1f)"
                               % (score, best))
            continue        # the roster would be inventing a number here
        assigned[track_id] = number
        confidence[track_id] = conf

    for track_id in scores_by_track:
        assigned.setdefault(track_id, 100)
    if debug:
        for track_id, why in sorted(reasons.items()):
            if assigned.get(track_id) == 100:
                print("  трек %s без номера: %s" % (track_id, why), flush=True)
    return assigned, confidence
