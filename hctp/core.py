"""
hctp.core
~~~~~~~~~
Pure math of the Helix Calculus Training Protocol.

No LLM dependency. No I/O. Import anywhere.

The HCTP models a learner's knowledge as a 3D vector K = [k1, k2, k3]
spiralling toward an ideal helical path as mastery increases.
"""

from __future__ import annotations
import math
from typing import Sequence

__all__ = [
    "helix_radius",
    "ideal_point",
    "distance",
    "progress",
    "smoothed_velocity",
    "num_breadcrumbs",
    "determine_focus",
    "update_vector",
    "CHECKPOINTS",
    "NUM_SPIRALS",
    "TARGET_VELOCITY",
    "MASTERY_THRESHOLD",
    # HCTP 2.0 constants
    "BASE_GAIN_V2",
    "MAX_GAIN_PER_SESSION",
    "SPILLOVER_RATE_V2",
]

# ── Constants ──────────────────────────────────────────────────────────────────

NUM_SPIRALS       = 5      # Full rotations of the helix from σ=0 to σ=1
TARGET_VELOCITY   = 0.15   # Ideal learning pace (σ gain per session)
MASTERY_THRESHOLD = 0.95   # k3 ≥ this → badge awarded

# HCTP 2.0 K-vector constants (used by hctp.hctp_vector.update_vector_v2)
BASE_GAIN_V2         = 0.025  # base per-session gain on focus checkpoint (was 0.020)
MAX_GAIN_PER_SESSION = 0.080  # hard cap on gain per session            (new in 2.0)
SPILLOVER_RATE_V2    = 0.10   # fraction of focus gain to siblings      (was 0.15)

CHECKPOINTS: dict[str, dict] = {
    "A": {
        "name": "Closures",
        "vector_index": 0,
        "concepts": "first-class functions, enclosing scope, free variables, nonlocal, late-binding",
    },
    "B": {
        "name": "Decorators",
        "vector_index": 1,
        "concepts": "@ syntax, functools.wraps, decorator factories, stacking decorators",
    },
    "C": {
        "name": "Metaclasses",
        "vector_index": 2,
        "concepts": "__new__, __init_subclass__, type(), custom class creation, ORMs, DSLs, singletons",
    },
}


# ── Geometry ───────────────────────────────────────────────────────────────────

def helix_radius(sigma: float) -> float:
    """Tightening radius R(σ) = 0.5(1 − σ)² + 0.05.

    As progress σ → 1 the helix tightens toward the axis,
    reflecting converging mastery.

    Args:
        sigma: Overall progress in [0, 1].

    Returns:
        Helix radius at this progress level.
    """
    return 0.5 * (1.0 - sigma) ** 2 + 0.05


def ideal_point(sigma: float) -> list[float]:
    """3D point on the ideal helix at progress σ ∈ [0, 1].

    The ideal learner sits exactly on this helix. Real learners
    orbit around it — ``distance()`` measures how far they stray.

    Args:
        sigma: Overall progress in [0, 1].

    Returns:
        [x, y, z] coordinates on the ideal helix.
    """
    R = helix_radius(sigma)
    n = NUM_SPIRALS
    return [
        R * math.cos(2 * math.pi * n * sigma),
        R * math.sin(2 * math.pi * n * sigma),
        sigma,
    ]


def distance(K: Sequence[float], sigma: float) -> float:
    """Euclidean distance from knowledge vector K to the ideal helix at σ.

    Lower is better. A distance near 0 means the learner is tracking
    the ideal helix tightly — balanced growth across all checkpoints.

    Args:
        K: Knowledge vector [k1, k2, k3], values in [0, 1].
        sigma: Current progress (typically ``progress(K)``).

    Returns:
        Scalar distance from the ideal helix.
    """
    r = ideal_point(sigma)
    return math.sqrt(sum((ki - ri) ** 2 for ki, ri in zip(K, r)))


# ── Progress & Velocity ────────────────────────────────────────────────────────

def progress(K: Sequence[float]) -> float:
    """Overall progress σ(t) = mean of knowledge vector.

    Args:
        K: Knowledge vector [k1, k2, k3].

    Returns:
        Scalar progress in [0, 1].
    """
    return sum(K) / len(K)


def smoothed_velocity(sigma_history: Sequence[float]) -> float:
    """Smoothed velocity over the last 3 σ steps.

    Takes up to the last 4 values so we always have at least 2 points
    to compute a velocity from (avoiding zero-division on short histories).

    Args:
        sigma_history: Ordered list of past σ values.

    Returns:
        Smoothed velocity (σ gain per session). Returns 0.0 if fewer
        than 2 history values are available.
    """
    if len(sigma_history) < 2:
        return 0.0
    recent = list(sigma_history[max(0, len(sigma_history) - 4):])
    velocities = [recent[i] - recent[i - 1] for i in range(1, len(recent))]
    return sum(velocities) / len(velocities) if velocities else 0.0


# ── Adaptive Scheduling ────────────────────────────────────────────────────────

def num_breadcrumbs(v_smooth: float) -> int:
    """Adaptive breadcrumb count: more when slow, fewer when fast.

    Slow learners receive more focused micro-tasks to build momentum.
    Fast learners receive fewer, broader tasks to sustain flow.

    Args:
        v_smooth: Smoothed velocity from ``smoothed_velocity()``.

    Returns:
        Number of breadcrumbs to generate this session (min 1).
    """
    return max(1, math.ceil(2 + 4 * max(0.0, TARGET_VELOCITY - v_smooth)))


def determine_focus(K: Sequence[float]) -> str:
    """Which checkpoint to focus on (the weakest one).

    Args:
        K: Knowledge vector [k1, k2, k3].

    Returns:
        Checkpoint label: "A", "B", or "C".
    """
    labels = ["A", "B", "C"]
    return labels[list(K).index(min(K))]


# ── Vector Update ──────────────────────────────────────────────────────────────

def update_vector(
    K: Sequence[float],
    karpathy_response: str,
    focus_checkpoint: str,
) -> tuple[list[float], list[float]]:
    """Compute new knowledge vector from a Karpathy loop response.

    Heuristic scoring: quality markers in the response text contribute
    micro-gains to the focus checkpoint, with 15% spillover to siblings.

    Args:
        K: Current knowledge vector.
        karpathy_response: Full text of the learner's Karpathy loop.
        focus_checkpoint: Which checkpoint was active ("A", "B", or "C").

    Returns:
        Tuple of (new_K, deltas) — updated vector and the per-component gains.
    """
    resp = karpathy_response.lower()
    focus_idx = CHECKPOINTS[focus_checkpoint]["vector_index"]
    delta = [0.0, 0.0, 0.0]
    base_gain = 0.02

    quality_markers = [
        ("error" in resp and "fix" in resp,                           0.010),
        ("production" in resp or "refactor" in resp,                  0.010),
        ("lesson" in resp or "connection" in resp,                    0.010),
        ("self-propose" in resp or "next question" in resp,           0.005),
        (len(karpathy_response) > 800,                                0.005),
        ("```" in karpathy_response,                                  0.005),
    ]
    bonus = sum(gain for condition, gain in quality_markers if condition)
    delta[focus_idx] = min(base_gain + bonus, 0.06)

    for i in range(3):
        if i != focus_idx:
            delta[i] = delta[focus_idx] * 0.15

    new_K = [min(1.0, max(0.0, K[i] + delta[i])) for i in range(3)]
    return new_K, delta
