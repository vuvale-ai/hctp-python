"""
hctp.scoring
~~~~~~~~~~~~
Execution-Based Quality Score (EQS) for HCTP 2.0.

EQS replaces the heuristic marker-scanning from v1.0 with a structured,
multi-signal score grounded in actual code execution outcomes.

Four components and their weights:

    Execution result (sandbox pass/fail):            25%
    Test pass rate (automated test suite):           30%
    Dual-Teacher score (harmonic mean Grok+Claude):  30%
    Karpathy Loop depth (steps 1-6 completed):       15%

The harmonic mean for Dual-Teacher punishes large inter-teacher disagreement:
if Grok gives 0.9 and Claude gives 0.1, the harmonic mean is 0.18, not 0.50.
This forces consensus before a high score is awarded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

# Component weights — must sum to 1.0
EXEC_WEIGHT    = 0.25
TEST_WEIGHT    = 0.30
TEACHER_WEIGHT = 0.30
DEPTH_WEIGHT   = 0.15

KARPATHY_MAX_DEPTH = 6  # Full Karpathy Loop has 6 steps

__all__ = [
    "EQSComponents",
    "EQSResult",
    "mastery_confidence_score",
    "EXEC_WEIGHT",
    "TEST_WEIGHT",
    "TEACHER_WEIGHT",
    "DEPTH_WEIGHT",
    "KARPATHY_MAX_DEPTH",
]


@dataclass
class EQSComponents:
    """Raw inputs for computing an Execution-Based Quality Score.

    All scores are normalised to [0, 1]. Validation is performed in
    __post_init__ to catch data-entry errors early.

    Args:
        sandbox_pass:    Whether the learner's submitted code ran without errors.
        test_pass_rate:  Fraction of automated tests that passed [0.0, 1.0].
        grok_score:      Grok holistic quality score [0.0, 1.0].
        claude_score:    Claude holistic quality score [0.0, 1.0].
        karpathy_depth:  Number of Karpathy Loop steps completed [0, 6].
    """

    sandbox_pass: bool = False
    test_pass_rate: float = 0.0
    grok_score: float = 0.0
    claude_score: float = 0.0
    karpathy_depth: int = 0

    def __post_init__(self) -> None:
        if not (0.0 <= self.test_pass_rate <= 1.0):
            raise ValueError(
                f"test_pass_rate must be in [0, 1], got {self.test_pass_rate}"
            )
        if not (0.0 <= self.grok_score <= 1.0):
            raise ValueError(
                f"grok_score must be in [0, 1], got {self.grok_score}"
            )
        if not (0.0 <= self.claude_score <= 1.0):
            raise ValueError(
                f"claude_score must be in [0, 1], got {self.claude_score}"
            )
        if not (0 <= self.karpathy_depth <= KARPATHY_MAX_DEPTH):
            raise ValueError(
                f"karpathy_depth must be in [0, {KARPATHY_MAX_DEPTH}], "
                f"got {self.karpathy_depth}"
            )

    @property
    def dual_teacher_score(self) -> float:
        """Harmonic mean of Grok and Claude scores.

        The harmonic mean penalises large teacher disagreements more harshly
        than the arithmetic mean.  A learner cannot game a high score by
        impressing one teacher while writing garbage for the other.
        """
        if self.grok_score == 0.0 and self.claude_score == 0.0:
            return 0.0
        return (
            2 * self.grok_score * self.claude_score
            / (self.grok_score + self.claude_score)
        )

    def compute(self) -> "EQSResult":
        """Compute the final EQS from all four components.

        Returns:
            EQSResult with score and per-component breakdown.
        """
        exec_component    = 1.0 if self.sandbox_pass else 0.0
        test_component    = self.test_pass_rate
        teacher_component = self.dual_teacher_score
        depth_component   = self.karpathy_depth / KARPATHY_MAX_DEPTH

        score = (
            EXEC_WEIGHT    * exec_component
            + TEST_WEIGHT    * test_component
            + TEACHER_WEIGHT * teacher_component
            + DEPTH_WEIGHT   * depth_component
        )

        return EQSResult(
            components=self,
            exec_component=exec_component,
            test_component=test_component,
            teacher_component=teacher_component,
            depth_component=depth_component,
            score=round(score, 4),
        )


@dataclass
class EQSResult:
    """Fully computed EQS with per-component breakdown for inspection.

    Do not construct directly — use ``EQSComponents.compute()``.
    """

    components: EQSComponents
    exec_component: float
    test_component: float
    teacher_component: float
    depth_component: float
    score: float  # Final EQS ∈ [0.0, 1.0]

    def summary(self) -> str:
        """Human-readable breakdown for logging and dashboards."""
        c = self.components
        return "\n".join([
            f"EQS Score: {self.score:.3f}",
            f"  Execution  (×{EXEC_WEIGHT:.2f}):   "
            f"{'PASS' if c.sandbox_pass else 'FAIL'} → {self.exec_component:.2f}",
            f"  Tests      (×{TEST_WEIGHT:.2f}):   "
            f"{c.test_pass_rate:.0%} → {self.test_component:.2f}",
            f"  Teacher    (×{TEACHER_WEIGHT:.2f}):  "
            f"Grok={c.grok_score:.2f} Claude={c.claude_score:.2f} "
            f"→ {self.teacher_component:.2f}",
            f"  K-Loop     (×{DEPTH_WEIGHT:.2f}):  "
            f"{c.karpathy_depth}/{KARPATHY_MAX_DEPTH} → {self.depth_component:.2f}",
        ])


def mastery_confidence_score(
    eqs_history: Sequence[float],
    sigma_history: Sequence[float],
    K: Sequence[float],
) -> float:
    """Compute the Mastery Confidence Score (MCS) ∈ [0, 1].

    MCS answers: "How confident are we that this learner has genuinely
    consolidated the knowledge at their current σ level?"

    Three equally-important signals are combined:

    1. EQS Consistency (weight 0.40)
       Low variance across recent EQS scores → stable, repeatable mastery.
       High variance → surface-skimming or lucky streaks.

    2. Velocity Stability (weight 0.35)
       Velocity close to TARGET_VELOCITY (0.15) scores highest.
       Too fast (skipping) or too slow (stuck) both reduce MCS.

    3. K-vector Helix Alignment (weight 0.25)
       How closely the current K tracks the ideal helix at σ=mean(K).
       On-helix = balanced growth; off-helix = one checkpoint neglected.

    MCS modulates sibling spillover in ``update_vector_v2``: consolidating
    learners (high MCS) spread knowledge to neighbours more than learners
    who are merely lucky this session (low MCS).

    Args:
        eqs_history:   EQS scores from recent sessions, most recent last.
        sigma_history: σ values across sessions, chronological order.
        K:             Current knowledge vector [k_A, k_B, k_C].

    Returns:
        MCS ∈ [0.0, 1.0], rounded to 4 decimal places.
    """
    from hctp.core import distance, progress, TARGET_VELOCITY  # local to avoid circular

    # --- Signal 1: EQS consistency ---
    if len(eqs_history) < 2:
        consistency = 0.5  # neutral prior with insufficient history
    else:
        recent = list(eqs_history[-5:])
        mean_eqs = sum(recent) / len(recent)
        variance = sum((e - mean_eqs) ** 2 for e in recent) / len(recent)
        # Max plausible variance is 0.25 (0 vs 1 alternating).
        consistency = max(0.0, 1.0 - variance / 0.25)

    # --- Signal 2: Velocity stability ---
    if len(sigma_history) < 3:
        velocity_score = 0.5
    else:
        deltas = [
            sigma_history[i + 1] - sigma_history[i]
            for i in range(len(sigma_history) - 1)
        ]
        recent_v = deltas[-4:]
        avg_v = sum(recent_v) / len(recent_v)
        deviation = abs(avg_v - TARGET_VELOCITY)
        velocity_score = max(0.0, 1.0 - deviation / TARGET_VELOCITY)

    # --- Signal 3: Helix alignment ---
    sigma = progress(K)
    d = distance(K, sigma)
    alignment = max(0.0, 1.0 - d / 0.6)  # normalised; max helix distance ≈ 0.6

    mcs = 0.40 * consistency + 0.35 * velocity_score + 0.25 * alignment
    return round(min(1.0, max(0.0, mcs)), 4)
