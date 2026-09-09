"""Event taxonomy for the ledger.

The vocabulary is deliberately small.  Every type here must be derivable from
what perception actually produces — tracked players in pitch coordinates plus a
ball trajectory — so that no entry in the ledger is a guess dressed up as an
observation.  Types that need a trained spotter (T-DEED and friends) are listed
under SPOTTED_TYPES and stay absent until such a model is wired in.
"""
from __future__ import annotations

# --- on-ball, derived from possession transitions ---------------------------
TOUCH = "touch"
CARRY = "carry"
PASS = "pass"
RECEPTION = "reception"
LOSS = "loss"
RECOVERY = "recovery"
INTERCEPTION = "interception"
CHALLENGE = "challenge"
SHOT = "shot"

# --- phase markers ----------------------------------------------------------
POSSESSION_PHASE = "possession_phase"
LOOSE_BALL = "loose_ball"
OUT_OF_VIEW = "out_of_view"

DERIVED_TYPES = (
    TOUCH, CARRY, PASS, RECEPTION, LOSS, RECOVERY,
    INTERCEPTION, CHALLENGE, SHOT,
    POSSESSION_PHASE, LOOSE_BALL, OUT_OF_VIEW,
)

# Reserved names for a learned action spotter.  Nothing writes these yet.
SPOTTED_TYPES = (
    "goal", "corner", "throw_in", "free_kick", "penalty",
    "yellow_card", "red_card", "substitution", "offside", "foul",
)

ALL_TYPES = DERIVED_TYPES + SPOTTED_TYPES

# Outcome vocabulary, kept separate from the type so a pass is one type with
# two outcomes rather than two types that statistics have to re-unify.
OUTCOME_COMPLETE = "complete"
OUTCOME_INCOMPLETE = "incomplete"
OUTCOME_UNKNOWN = "unknown"

# --- pitch geometry (SoccerNet convention: origin at the centre spot) --------
PITCH_LENGTH = 105.0
PITCH_WIDTH = 68.0
HALF_LENGTH = PITCH_LENGTH / 2.0
HALF_WIDTH = PITCH_WIDTH / 2.0
GOAL_HALF_WIDTH = 3.66
BOX_LENGTH = 16.5
BOX_HALF_WIDTH = 20.16


def third(x: float, attacking_right: bool) -> str:
    """Defensive / middle / attacking third for a team playing to +x or -x."""
    if not attacking_right:
        x = -x
    if x < -HALF_LENGTH / 3.0:
        return "defensive"
    if x > HALF_LENGTH / 3.0:
        return "attacking"
    return "middle"


def in_penalty_box(x: float, y: float, attacking_right: bool) -> bool:
    if not attacking_right:
        x = -x
    return x >= HALF_LENGTH - BOX_LENGTH and abs(y) <= BOX_HALF_WIDTH


def goal_centre(attacking_right: bool):
    return (HALF_LENGTH if attacking_right else -HALF_LENGTH, 0.0)


def zone(x: float, y: float, cols: int = 6, rows: int = 5):
    """Grid cell index used for heatmaps; column 0 is the -x touchline end."""
    cx = min(cols - 1, max(0, int((x + HALF_LENGTH) / PITCH_LENGTH * cols)))
    ry = min(rows - 1, max(0, int((y + HALF_WIDTH) / PITCH_WIDTH * rows)))
    return cx, ry
