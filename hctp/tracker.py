"""
hctp.tracker
~~~~~~~~~~~~
Stateful session manager for a single HCTP learner.

LearnerSession wraps core math + curriculum generation into a clean,
framework-agnostic object you can drop into any training pipeline.

Usage::

    from hctp import LearnerSession

    session = LearnerSession("Alice")
    prompts = session.start_session()          # generates breadcrumb prompts
    for prompt in prompts:
        response = your_llm(prompt)            # your LLM call
        session.submit_response(response)      # updates K-vector

    print(session.sigma)       # current progress
    print(session.badge)       # True once k3 >= 0.95
"""

from __future__ import annotations
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from .core import (
    progress, smoothed_velocity, distance, determine_focus,
    num_breadcrumbs, update_vector, MASTERY_THRESHOLD,
)
from .curriculum import breadcrumb_prompt, karpathy_loop_prompt, session_header

__all__ = ["LearnerSession", "SessionResult"]


@dataclass
class SessionResult:
    """Result of a completed HCTP training session."""
    session_number: int
    K_before: list[float]
    K_after: list[float]
    sigma_before: float
    sigma_after: float
    velocity: float
    focus_checkpoint: str
    breadcrumbs_completed: int
    badge_earned: bool = False
    deltas: list[list[float]] = field(default_factory=list)

    @property
    def sigma_gain(self) -> float:
        return self.sigma_after - self.sigma_before

    def __str__(self) -> str:
        gain = f"+{self.sigma_gain:.4f}"
        badge = " 🏆 BADGE EARNED" if self.badge_earned else ""
        return (
            f"Session {self.session_number} | "
            f"σ {self.sigma_before:.3f} → {self.sigma_after:.3f} ({gain}) | "
            f"K=[{self.K_after[0]:.3f}, {self.K_after[1]:.3f}, {self.K_after[2]:.3f}]"
            f"{badge}"
        )


class LearnerSession:
    """Manages HCTP state and generates prompts for one learner.

    Args:
        name: Learner identifier (used in prompts).
        K: Initial knowledge vector. Defaults to [0.0, 0.0, 0.0].
        sigma_history: Prior σ values for velocity calculation.
        mastery_threshold: k3 ≥ this value awards the badge.

    Example::

        session = LearnerSession("Ren", K=[0.3, 0.1, 0.0])

        # Generate prompts for this session
        prompts = session.start_session()

        # For each breadcrumb:
        for bc_prompt in prompts["breadcrumb_prompts"]:
            bc_response = llm(bc_prompt)
            kl_prompt   = session.karpathy_prompt_for(bc_prompt, bc_response)
            kl_response = llm(kl_prompt)
            session.submit_karpathy(kl_response)

        result = session.finish_session()
        print(result)
    """

    def __init__(
        self,
        name: str,
        K: Sequence[float] | None = None,
        sigma_history: Sequence[float] | None = None,
        mastery_threshold: float = MASTERY_THRESHOLD,
    ) -> None:
        self.name = name
        self.K: list[float] = list(K) if K is not None else [0.0, 0.0, 0.0]
        self.sigma_history: list[float] = list(sigma_history) if sigma_history else [0.0]
        self.mastery_threshold = mastery_threshold
        self.sessions_completed: int = 0
        self.badge: bool = self.K[2] >= mastery_threshold
        self._session_deltas: list[list[float]] = []
        self._session_breadcrumb_prompts: list[str] = []
        self._session_karpathy_pending: list[tuple[str, str]] = []  # (bc_text, bc_response)
        self._in_session: bool = False

    # ── Properties ─────────────────────────────────────────────────────────────

    @property
    def sigma(self) -> float:
        """Current overall progress σ ∈ [0, 1]."""
        return progress(self.K)

    @property
    def velocity(self) -> float:
        """Smoothed learning velocity."""
        return smoothed_velocity(self.sigma_history)

    @property
    def focus(self) -> str:
        """Current focus checkpoint label."""
        return determine_focus(self.K)

    @property
    def helix_distance(self) -> float:
        """Distance from ideal helix."""
        return distance(self.K, self.sigma)

    # ── Session lifecycle ───────────────────────────────────────────────────────

    def start_session(self) -> dict:
        """Begin a new training session.

        Returns a dict containing:
        - ``header``: Markdown session summary string
        - ``breadcrumb_prompts``: List of prompts to send to your LLM
        - ``n_breadcrumbs``: How many breadcrumbs this session
        - ``focus``: Active checkpoint label

        You must call ``submit_karpathy()`` for each breadcrumb response,
        then ``finish_session()`` to update state.
        """
        if self._in_session:
            raise RuntimeError("Already in a session. Call finish_session() first.")

        self._in_session = True
        self._session_deltas = []
        self._session_breadcrumb_prompts = []
        self._session_karpathy_pending = []

        v      = self.velocity
        d      = self.helix_distance
        f      = self.focus
        n_bc   = num_breadcrumbs(v)
        header = session_header(
            self.name, self.K, self.sigma, v, d, f,
            n_bc, self.sessions_completed + 1,
        )

        prompts = [
            breadcrumb_prompt(self.name, self.K, self.sigma, v, d, i + 1, f)
            for i in range(n_bc)
        ]
        self._session_breadcrumb_prompts = prompts

        return {"header": header, "breadcrumb_prompts": prompts,
                "n_breadcrumbs": n_bc, "focus": f}

    def karpathy_prompt_for(self, breadcrumb_text: str, breadcrumb_response: str) -> str:
        """Get the Karpathy loop prompt for a given breadcrumb + response pair.

        Args:
            breadcrumb_text: The original breadcrumb prompt text.
            breadcrumb_response: The learner's answer to the breadcrumb.

        Returns:
            Karpathy loop prompt string.
        """
        self._session_karpathy_pending.append((breadcrumb_text, breadcrumb_response))
        return karpathy_loop_prompt(
            self.name, breadcrumb_text, breadcrumb_response, self.K, self.focus
        )

    def submit_karpathy(self, karpathy_response: str) -> list[float]:
        """Submit a Karpathy loop response to update the knowledge vector.

        Args:
            karpathy_response: The full text of the learner's Karpathy loop.

        Returns:
            Delta vector [Δk1, Δk2, Δk3] applied this breadcrumb.
        """
        if not self._in_session:
            raise RuntimeError("Not in a session. Call start_session() first.")
        new_K, delta = update_vector(self.K, karpathy_response, self.focus)
        self.K = new_K
        self._session_deltas.append(delta)
        return delta

    def finish_session(self) -> SessionResult:
        """Finalise the session, update history, check badge.

        Returns:
            SessionResult with full session statistics.
        """
        if not self._in_session:
            raise RuntimeError("Not in a session. Call start_session() first.")

        sigma_before = self.sigma_history[-1] if self.sigma_history else 0.0
        sigma_after  = self.sigma

        self.sigma_history.append(sigma_after)
        if len(self.sigma_history) > 20:
            self.sigma_history = self.sigma_history[-20:]

        self.sessions_completed += 1
        self._in_session = False

        badge_earned = False
        if self.K[2] >= self.mastery_threshold and not self.badge:
            self.badge = True
            badge_earned = True

        return SessionResult(
            session_number=self.sessions_completed,
            K_before=[sigma_before] * 3,   # approximate
            K_after=list(self.K),
            sigma_before=sigma_before,
            sigma_after=sigma_after,
            velocity=self.velocity,
            focus_checkpoint=self.focus,
            breadcrumbs_completed=len(self._session_deltas),
            badge_earned=badge_earned,
            deltas=list(self._session_deltas),
        )

    # ── Persistence ─────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialise learner state to a JSON-safe dict."""
        return {
            "name": self.name,
            "K": self.K,
            "sigma_history": self.sigma_history,
            "sessions_completed": self.sessions_completed,
            "badge": self.badge,
            "mastery_threshold": self.mastery_threshold,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LearnerSession":
        """Restore a LearnerSession from a serialised dict."""
        session = cls(
            name=data["name"],
            K=data.get("K", [0.0, 0.0, 0.0]),
            sigma_history=data.get("sigma_history", [0.0]),
            mastery_threshold=data.get("mastery_threshold", MASTERY_THRESHOLD),
        )
        session.sessions_completed = data.get("sessions_completed", 0)
        session.badge = data.get("badge", False)
        return session

    def save(self, path: str | Path) -> None:
        """Save learner state to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LearnerSession":
        """Load learner state from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text()))

    def __repr__(self) -> str:
        return (
            f"LearnerSession(name={self.name!r}, "
            f"K={[round(k, 3) for k in self.K]}, "
            f"sigma={self.sigma:.3f}, "
            f"badge={self.badge})"
        )
