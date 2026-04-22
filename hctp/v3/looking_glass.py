"""
hctp.v3.looking_glass
~~~~~~~~~~~~~~~~~~~~~
External **Looking Glass Evaluator** for HCTP 3.0.

The Looking Glass is a post-hoc evaluator — it reads a session snapshot
and emits objective signals without ever sitting inside the Karpathy
loop. That separation is the whole point: v2 mixed evaluation into
training, which muddied flow and introduced feedback noise.

Design guarantees:
    * **Non-interfering**: evaluate *after* a session, never during.
    * **Lightweight**: pure Python, no I/O in the hot path.
    * **Deterministic**: same input → same output (seeded heuristics).
    * **Pluggable**: ``evaluate()`` is the only public contract.

The default implementation uses rule-based scoring layered over the v2
EQS + MCS machinery. Swap in a learned evaluator by subclassing and
overriding :meth:`LookingGlassEvaluator.evaluate`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from hctp.scoring import mastery_confidence_score
from .manifold import TunnelManifold, TunnelMetrics, DRIFT_THRESHOLD
from .tunnel import TunnelState

__all__ = [
    "LookingGlassReport",
    "LookingGlassEvaluator",
]


# ── Thresholds (tuneable) ──────────────────────────────────────────────────────

_MASTERY_ADVANCE_THRESHOLD = 0.80  # K_local mean must exceed this to advance
_COMPILATION_THRESHOLD     = 0.90  # depth above this → compile the sub-tunnel


# ── Reporting dataclass ────────────────────────────────────────────────────────

@dataclass
class LookingGlassReport:
    """Objective evaluation produced after a session.

    The agent only receives high-level fields (position, drift, focus,
    recommended_action, notes) — the rest stays inside the framework for
    logging and dashboards.

    Attributes:
        metrics:            Raw geometric metrics from the manifold.
        mcs:                Mastery Confidence Score at snapshot time.
        recommended_action: One of ``"continue"``, ``"re_center"``,
                            ``"advance"``, ``"compile"``, ``"revisit"``.
        notes:              Human-readable narrative appended across rules.
        next_focus:         Suggested ``(macro_key, sub_key)`` to work on
                            next — may equal the current one.
    """

    metrics: TunnelMetrics
    mcs: float
    recommended_action: str
    notes: list[str] = field(default_factory=list)
    next_focus: tuple[str, str] | None = None

    def summary(self) -> str:
        lines = [
            "── Looking Glass Report ──",
            self.metrics.summary(),
            f"MCS: {self.mcs:.3f}",
            f"Action: {self.recommended_action}",
        ]
        if self.next_focus is not None:
            lines.append(f"Next focus: {self.next_focus[0]} / {self.next_focus[1]}")
        if self.notes:
            lines.append("Notes:")
            lines.extend(f"  • {n}" for n in self.notes)
        return "\n".join(lines)


# ── Evaluator ──────────────────────────────────────────────────────────────────

class LookingGlassEvaluator:
    """Stateless external evaluator.

    Args:
        manifold: Tunnel manifold used to compute geometric signals.
                  If omitted, a default manifold is built.

    Example::

        evaluator = LookingGlassEvaluator()
        report = evaluator.evaluate(state, eqs_history, sigma_history)
        print(report.summary())
    """

    def __init__(self, manifold: TunnelManifold | None = None) -> None:
        self.manifold = manifold or TunnelManifold()

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(
        self,
        state: TunnelState,
        eqs_history: Sequence[float] = (),
        sigma_history: Sequence[float] = (),
    ) -> LookingGlassReport:
        """Produce a full evaluation report for the given tunnel state.

        Args:
            state:          Post-session learner state.
            eqs_history:    EQS scores across prior sessions, chronological.
            sigma_history:  Global σ values across prior sessions.

        Returns:
            A ``LookingGlassReport`` with objective signals + a recommended
            next action.
        """
        metrics = self.manifold.measure(state)
        mcs = mastery_confidence_score(eqs_history, sigma_history, state.K_local)

        notes: list[str] = []
        action = "continue"
        next_focus: tuple[str, str] | None = (state.macro_tunnel, state.sub_tunnel)

        # Rule 1 — drift dominates
        if metrics.drift_score >= DRIFT_THRESHOLD:
            action = "re_center"
            notes.append(
                f"Drift {metrics.drift_score:.2f} ≥ {DRIFT_THRESHOLD:.2f}; "
                "schedule re-centering breadcrumbs."
            )
        else:
            # Rule 2 — mastery to advance
            mean_local = sum(state.K_local) / len(state.K_local)
            if mean_local >= _MASTERY_ADVANCE_THRESHOLD:
                action = "advance"
                nxt = self.manifold.next_sub_tunnel(state)
                next_focus = nxt
                if nxt is None:
                    notes.append("Curriculum complete — no further sub-tunnels.")
                else:
                    notes.append(
                        f"Sub-tunnel mastered (K̄={mean_local:.2f}); "
                        f"advance to {nxt[0]}/{nxt[1]}."
                    )
                # Rule 3 — compile into a reusable primitive if depth is deep
                if state.depth >= _COMPILATION_THRESHOLD:
                    action = "compile"
                    notes.append(
                        f"Depth {state.depth:.2f} ≥ {_COMPILATION_THRESHOLD:.2f}; "
                        "compile sub-tunnel before advancing."
                    )

        # Rule 4 — stall / wobble detection
        if len(eqs_history) >= 3 and max(eqs_history[-3:]) < 0.35:
            notes.append("EQS has stalled below 0.35 for three sessions — consider revisit.")
            if action == "continue":
                action = "revisit"

        return LookingGlassReport(
            metrics=metrics,
            mcs=mcs,
            recommended_action=action,
            notes=notes,
            next_focus=next_focus,
        )
