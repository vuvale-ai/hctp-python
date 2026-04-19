"""
hctp.difficulty_engine
~~~~~~~~~~~~~~~~~~~~~~
Adaptive difficulty for HCTP 2.0.

What changed from v1.0
----------------------
v1.0 used only velocity to determine breadcrumb count (``num_breadcrumbs``).
v2.0 adds helix alignment as a second axis: a learner can be fast *and*
off-helix (rushing one checkpoint, neglecting others), which should produce
corrective "foundation"-level content, not harder material.

Difficulty score D ∈ [0, 1]:

    D = 0.5 × velocity_factor + 0.5 × helix_alignment

    velocity_factor = min(velocity / TARGET_VELOCITY, 1.5) / 1.5
    helix_alignment = max(0, 1 − distance(K, σ) / 0.6)

    D < 0.25  → foundation  (more hints, fewer breadcrumbs)
    D < 0.50  → standard
    D < 0.75  → advanced
    D ≥ 0.75  → expert      (fewer hints, broader tasks)

Role-specific adjustments
-------------------------
Each role has per-checkpoint concept weights that scale the breadcrumb count
and inject domain-relevant emphasis into generated prompts.

Example: a Security Engineer (role="security") at Checkpoint C (Metaclasses)
gets weight 1.5 × the standard breadcrumb count because metaclasses have
significant security implications (proxy interception, sandbox escape, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from hctp.core import (
    CHECKPOINTS,
    TARGET_VELOCITY,
    determine_focus,
    distance,
    num_breadcrumbs,
    progress,
)

__all__ = [
    "compute_difficulty",
    "DifficultyProfile",
    "ROLES",
    "DIFFICULTY_LEVELS",
]

ROLES: dict[str, dict] = {
    "generic": {
        "label": "Generic Engineer",
        "concept_weights": {
            "closures": 1.0,
            "decorators": 1.0,
            "metaclasses": 1.0,
        },
        "extra_emphasis": [],
    },
    "security": {
        "label": "Security Engineer",
        "concept_weights": {
            "closures": 1.1,
            "decorators": 1.3,
            "metaclasses": 1.5,
        },
        "extra_emphasis": [
            "injection attacks via closures",
            "monkey-patching vulnerabilities in decorators",
            "metaclass-based security sandboxing",
        ],
    },
    "data_scientist": {
        "label": "Data Scientist",
        "concept_weights": {
            "closures": 1.0,
            "decorators": 1.2,
            "metaclasses": 1.1,
        },
        "extra_emphasis": [
            "functional pipelines with closures",
            "caching and memoization decorators",
            "schema validation via metaclasses",
        ],
    },
    "devops": {
        "label": "DevOps Engineer",
        "concept_weights": {
            "closures": 1.1,
            "decorators": 1.4,
            "metaclasses": 1.0,
        },
        "extra_emphasis": [
            "retry and timeout decorators",
            "config injection via closures",
            "plugin registration with metaclasses",
        ],
    },
}

DIFFICULTY_LEVELS: tuple[str, ...] = ("foundation", "standard", "advanced", "expert")


@dataclass
class DifficultyProfile:
    """Complete difficulty profile for a learner's session.

    Attributes:
        level:             One of DIFFICULTY_LEVELS.
        breadcrumb_count:  Number of micro-tasks to generate this session.
        focus_checkpoint:  "A", "B", or "C".
        role:              Learner role key.
        role_emphasis:     Role-specific concept strings for prompt injection.
        helix_alignment:   0.0 (off-helix) → 1.0 (on-helix).
        velocity_ratio:    actual_velocity / TARGET_VELOCITY, capped at 2.0.
        hint_density:      Fraction of breadcrumbs that include scaffolding hints.
    """

    level: str
    breadcrumb_count: int
    focus_checkpoint: str
    role: str
    role_emphasis: list[str]
    helix_alignment: float
    velocity_ratio: float
    hint_density: float

    def summary(self) -> str:
        """Human-readable profile for logging and prompt injection."""
        cp_name    = CHECKPOINTS[self.focus_checkpoint]["name"]
        role_label = ROLES[self.role]["label"]
        lines = [
            "Difficulty Profile:",
            f"  Level:           {self.level}",
            f"  Breadcrumbs:     {self.breadcrumb_count}",
            f"  Focus:           {self.focus_checkpoint} — {cp_name}",
            f"  Role:            {role_label}",
            f"  Helix alignment: {self.helix_alignment:.2f}",
            f"  Velocity ratio:  {self.velocity_ratio:.2f}×",
            f"  Hint density:    {self.hint_density:.0%}",
        ]
        if self.role_emphasis:
            lines.append(
                f"  Role emphasis:   {', '.join(self.role_emphasis[:2])}"
            )
        return "\n".join(lines)


def compute_difficulty(
    K: Sequence[float],
    velocity: float,
    role: str = "generic",
    focus_checkpoint: str | None = None,
) -> DifficultyProfile:
    """Compute an adaptive difficulty profile for the current session.

    The difficulty score D combines velocity and helix alignment equally:

        D = 0.5 × velocity_factor + 0.5 × helix_alignment

    High D (fast + on-helix) → expert-level content, fewer hints.
    Low D (slow or off-helix) → foundation content, many hints.

    Args:
        K:                 Current knowledge vector.
        velocity:          Smoothed σ-gain per session (from ``smoothed_velocity``).
        role:              Learner role key (see ``ROLES``). Default "generic".
        focus_checkpoint:  Override focus checkpoint. If None, uses weakest k.

    Returns:
        DifficultyProfile describing this session's parameters.

    Raises:
        ValueError: If ``role`` is not in ``ROLES``.
    """
    if role not in ROLES:
        raise ValueError(
            f"Unknown role '{role}'. Valid roles: {list(ROLES.keys())}"
        )

    sigma     = progress(K)
    d         = distance(K, sigma)
    alignment = max(0.0, 1.0 - d / 0.6)  # max helix distance ≈ 0.6

    velocity_ratio  = min(velocity / TARGET_VELOCITY, 2.0) if TARGET_VELOCITY > 0 else 1.0
    velocity_factor = min(velocity_ratio, 1.5) / 1.5

    D = min(1.0, max(0.0, 0.5 * velocity_factor + 0.5 * alignment))

    if D < 0.25:
        level = "foundation"
    elif D < 0.50:
        level = "standard"
    elif D < 0.75:
        level = "advanced"
    else:
        level = "expert"

    focus   = focus_checkpoint or determine_focus(K)
    cp_name = CHECKPOINTS[focus]["name"].lower()

    role_data = ROLES[role]
    weight    = role_data["concept_weights"].get(cp_name, 1.0)
    breadcrumb_count = max(1, round(num_breadcrumbs(velocity) * weight))
    hint_density     = round(max(0.0, 1.0 - D), 2)

    return DifficultyProfile(
        level=level,
        breadcrumb_count=breadcrumb_count,
        focus_checkpoint=focus,
        role=role,
        role_emphasis=role_data["extra_emphasis"],
        helix_alignment=round(alignment, 3),
        velocity_ratio=round(velocity_ratio, 3),
        hint_density=hint_density,
    )
