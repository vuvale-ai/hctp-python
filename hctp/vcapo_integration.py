"""
hctp.vcapo_integration
~~~~~~~~~~~~~~~~~~~~~~
HCTP 2.0 → VCAPO fine-tuning integration.

VCAPO (Value-Calibrated Adaptive Policy Optimization) uses HCTP learner
trajectories as fine-tuning signal.  High-quality, on-helix learning paths
should influence training proportionally more than erratic or off-helix ones.

Trajectory weight formula
--------------------------
    w(τ) = α × EQS_mean(τ)  +  β × alignment(τ)  +  γ × kloop_completion(τ)

    α = 0.40  — execution-grounded quality signal
    β = 0.35  — helix-tracking signal (balanced checkpoint growth)
    γ = 0.25  — process completeness signal (full Karpathy loops)

All three terms are in [0, 1], so w ∈ [0, 1].

Usage
-----
    from hctp.vcapo_integration import TrajectoryRecord, VCAPOExporter

    exporter = VCAPOExporter()
    exporter.add(ren_trajectory)
    exporter.add(ner_trajectory)

    n = exporter.export("training_data.jsonl", min_weight=0.30)
    print(exporter.weight_summary())
    # VCAPO Trajectory Summary (2 trajectories)
    #   Mean weight:   0.612
    #   Min weight:    0.581
    #   Max weight:    0.643
    #   Badge earners: 2
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from hctp.core import distance, progress

__all__ = [
    "TrajectoryRecord",
    "VCAPOExporter",
    "EQS_ALPHA",
    "ALIGNMENT_BETA",
    "COMPLETION_GAMMA",
]

EQS_ALPHA        = 0.40  # weight on mean EQS across trajectory
ALIGNMENT_BETA   = 0.35  # weight on mean helix alignment
COMPLETION_GAMMA = 0.25  # weight on Karpathy loop completion rate


@dataclass
class TrajectoryRecord:
    """Single learner trajectory prepared for VCAPO training.

    A trajectory is one complete HCTP session sequence from K=[0,0,0]
    to badge or end of available data.  ``weight`` is populated by
    calling ``compute_weight()`` (done automatically by ``VCAPOExporter.add``).

    Attributes:
        learner_id:    Unique identifier for the learner.
        sessions:      List of serialised SessionResult dicts (one per session).
        eqs_scores:    EQS score per session, same length as ``sessions``.
        sigma_history: σ values across sessions (one per session).
        K_history:     Knowledge vector at the end of each session.
        role:          Learner role key (see ``difficulty_engine.ROLES``).
        badge_earned:  True if learner reached k₃ ≥ MASTERY_THRESHOLD.
        weight:        Computed training weight ∈ [0, 1].
    """

    learner_id: str
    sessions: list[dict]
    eqs_scores: list[float]
    sigma_history: list[float]
    K_history: list[list[float]]
    role: str = "generic"
    badge_earned: bool = False
    weight: float = field(init=False, default=0.0)

    def compute_weight(self) -> float:
        """Compute and store the VCAPO training weight for this trajectory.

        Three signals are averaged with their respective weights (α, β, γ):

        1. EQS mean — average execution-grounded quality across all sessions.
        2. Helix alignment mean — how well K tracked the ideal helix on average.
        3. Karpathy loop completion — fraction of sessions with karpathy_depth
           recorded in the session dict (normalised to [0, 1]).

        Returns:
            weight ∈ [0.0, 1.0]
        """
        # --- Signal 1: mean EQS ---
        eqs_mean = (
            sum(self.eqs_scores) / len(self.eqs_scores)
            if self.eqs_scores
            else 0.0
        )

        # --- Signal 2: mean helix alignment ---
        alignments: list[float] = []
        for K in self.K_history:
            sigma = progress(K)
            d = distance(K, sigma)
            alignments.append(max(0.0, 1.0 - d / 0.6))
        alignment_mean = (
            sum(alignments) / len(alignments) if alignments else 0.0
        )

        # --- Signal 3: Karpathy loop completion ---
        depth_scores = [
            s.get("karpathy_depth", 0) / 6.0
            for s in self.sessions
            if "karpathy_depth" in s
        ]
        completion_mean = (
            sum(depth_scores) / len(depth_scores) if depth_scores else 0.0
        )

        w = (
            EQS_ALPHA        * eqs_mean
            + ALIGNMENT_BETA   * alignment_mean
            + COMPLETION_GAMMA * completion_mean
        )
        self.weight = round(min(1.0, max(0.0, w)), 4)
        return self.weight

    def to_training_example(self) -> dict:
        """Serialise this trajectory for VCAPO training pipeline consumption.

        The format is intentionally flat and minimal.  Downstream tooling
        handles tokenisation, batching, and loss weighting using ``weight``.
        """
        return {
            "learner_id":    self.learner_id,
            "role":          self.role,
            "badge_earned":  self.badge_earned,
            "weight":        self.weight,
            "session_count": len(self.sessions),
            "eqs_mean":      (
                sum(self.eqs_scores) / len(self.eqs_scores)
                if self.eqs_scores else 0.0
            ),
            "sigma_final":   self.sigma_history[-1] if self.sigma_history else 0.0,
            "sessions":      self.sessions,
        }


class VCAPOExporter:
    """Collect, weight, and export HCTP trajectories for VCAPO fine-tuning.

    Each trajectory is weighted on addition so ``export`` can immediately
    apply a ``min_weight`` filter without re-computing anything.

    Example::

        exporter = VCAPOExporter()
        exporter.add(ren_trajectory)
        exporter.add(ner_trajectory)

        # Write only trajectories with weight ≥ 0.30
        n = exporter.export("training_data.jsonl", min_weight=0.30)
        print(f"Wrote {n} trajectories")
        print(exporter.weight_summary())
    """

    def __init__(self) -> None:
        self._trajectories: list[TrajectoryRecord] = []

    def __len__(self) -> int:
        return len(self._trajectories)

    def __repr__(self) -> str:
        return f"VCAPOExporter(trajectories={len(self._trajectories)})"

    def add(self, trajectory: TrajectoryRecord) -> None:
        """Add a trajectory, computing its weight automatically.

        Args:
            trajectory: A ``TrajectoryRecord`` to include in the export.
        """
        trajectory.compute_weight()
        self._trajectories.append(trajectory)

    def export(self, path: str | Path, min_weight: float = 0.0) -> int:
        """Write weighted trajectories to a JSONL file.

        Each line is a JSON object produced by
        ``TrajectoryRecord.to_training_example()``.

        Args:
            path:        Output file path (.jsonl).
            min_weight:  Skip trajectories with weight below this threshold.

        Returns:
            Number of trajectories written to disk.
        """
        path = Path(path)
        written = 0
        with path.open("w", encoding="utf-8") as fh:
            for traj in self._trajectories:
                if traj.weight < min_weight:
                    continue
                fh.write(json.dumps(traj.to_training_example()) + "\n")
                written += 1
        return written

    def weight_summary(self) -> str:
        """Human-readable summary of trajectory weights."""
        if not self._trajectories:
            return "No trajectories loaded."
        weights = [t.weight for t in self._trajectories]
        return "\n".join([
            f"VCAPO Trajectory Summary ({len(weights)} trajectories)",
            f"  Mean weight:   {sum(weights) / len(weights):.3f}",
            f"  Min weight:    {min(weights):.3f}",
            f"  Max weight:    {max(weights):.3f}",
            f"  Badge earners: {sum(1 for t in self._trajectories if t.badge_earned)}",
        ])
