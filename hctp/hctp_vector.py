"""
hctp.hctp_vector
~~~~~~~~~~~~~~~~
HCTP 2.0 K-vector progression logic.

What changed from v1.0
----------------------
| Parameter      | v1.0   | v2.0                          |
|----------------|--------|-------------------------------|
| Base gain      | 0.020  | 0.025  (+25%)                 |
| Max gain cap   | 0.060  | 0.080  (new hard ceiling)     |
| Spillover rate | 15%    | 10%   (-33%)                  |
| Scoring input  | text heuristics | EQS (execution-grounded) |
| Spillover mod  | fixed  | multiplied by MCS             |

Design rationale
----------------
Raising the base gain (0.025) rewards consistent effort even in low-EQS
sessions.  The tighter spillover (10 %) focuses gains on the actual focus
checkpoint so siblings don't inflate cheaply.  The MAX_GAIN hard cap (0.08)
prevents runaway jumps on very high-EQS sessions — mastery should be earned
gradually.

The MCS multiplier on spillover is the key novelty: a learner who has
demonstrated stable, on-helix progress (high MCS) has genuinely consolidated
prior knowledge and can propagate learning to siblings.  A learner who is
erratic or off-helix (low MCS) keeps gains siloed to the focus checkpoint
until they stabilise.
"""

from __future__ import annotations

from typing import Sequence

from hctp.core import CHECKPOINTS

__all__ = [
    "update_vector_v2",
    "BASE_GAIN",
    "MAX_GAIN",
    "SPILLOVER",
]

# K-vector progression constants (HCTP 2.0)
BASE_GAIN = 0.025   # base per-session gain on focus checkpoint (was 0.020)
MAX_GAIN  = 0.080   # hard cap on gain per session            (was 0.060)
SPILLOVER = 0.10    # fraction of focus gain to siblings      (was 0.15)


def update_vector_v2(
    K: Sequence[float],
    eqs_score: float,
    focus_checkpoint: str,
    mcs: float = 0.5,
) -> tuple[list[float], list[float]]:
    """Compute a new knowledge vector using execution-grounded EQS scoring.

    Focus checkpoint gain:
        delta_focus = min(BASE_GAIN + eqs_score × 0.055, MAX_GAIN)

    This gives a range of [0.025, 0.080] depending on EQS quality.

    Sibling checkpoint gain (MCS modulates spillover):
        delta_sibling = delta_focus × SPILLOVER × mcs

    High MCS → richer sibling gains (learner is genuinely consolidating).
    Low  MCS → narrow sibling gains (gains stay siloed until stable).

    Args:
        K:                 Current knowledge vector, length 3.
        eqs_score:         EQSResult.score ∈ [0.0, 1.0].
        focus_checkpoint:  "A", "B", or "C".
        mcs:               Mastery Confidence Score ∈ [0.0, 1.0]. Default 0.5.

    Returns:
        Tuple of (K_new, delta) — updated vector and per-checkpoint gains.

    Raises:
        ValueError: If focus_checkpoint is not "A", "B", or "C".
        ValueError: If eqs_score or mcs are outside [0, 1].
    """
    checkpoint_keys = list(CHECKPOINTS.keys())
    if focus_checkpoint not in checkpoint_keys:
        raise ValueError(
            f"focus_checkpoint must be one of {checkpoint_keys}, "
            f"got '{focus_checkpoint}'"
        )
    if not (0.0 <= eqs_score <= 1.0):
        raise ValueError(f"eqs_score must be in [0, 1], got {eqs_score}")
    if not (0.0 <= mcs <= 1.0):
        raise ValueError(f"mcs must be in [0, 1], got {mcs}")

    K_list = list(K)
    focus_idx = checkpoint_keys.index(focus_checkpoint)
    delta_focus = min(BASE_GAIN + eqs_score * 0.055, MAX_GAIN)

    delta = [
        delta_focus if i == focus_idx else delta_focus * SPILLOVER * mcs
        for i in range(len(K_list))
    ]

    K_new = [
        min(1.0, max(0.0, K_list[i] + delta[i]))
        for i in range(len(K_list))
    ]
    return K_new, delta
