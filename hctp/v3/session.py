"""
hctp.v3.session
~~~~~~~~~~~~~~~
Stateful session manager for HCTP 3.0.

``TunnelLearnerSession`` is the Phase-1 entry point. It wraps the
manifold + Looking Glass + breadcrumb machinery into a single object
that drop-in replaces :class:`hctp.tracker.LearnerSession` for tunnel-
aware training.

Backward compatibility
----------------------
The v1/v2 :class:`hctp.tracker.LearnerSession` API is unchanged — this
class lives alongside it. A :meth:`to_v2_session` adapter is provided
to fall back to the flat checkpoint API when desired.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

from hctp.core import num_breadcrumbs, smoothed_velocity
from hctp.hctp_vector import update_vector_v2
from hctp.scoring import mastery_confidence_score

from .breadcrumbs import (
    exploration_breadcrumb_prompt,
    re_centering_breadcrumb_prompt,
    tunnel_session_header,
)
from .looking_glass import LookingGlassEvaluator, LookingGlassReport
from .manifold import TunnelManifold, DRIFT_THRESHOLD
from .tunnel import TunnelState, DEFAULT_TUNNEL_SYSTEM, DEFAULT_WIDTH

__all__ = ["TunnelLearnerSession", "TunnelSessionResult"]


# ── Session-result dataclass ───────────────────────────────────────────────────

@dataclass
class TunnelSessionResult:
    """Outcome of a single HCTP 3.0 session."""

    session_number: int
    macro_tunnel: str
    sub_tunnel: str
    mode: str                            # "exploration" or "re_centering"
    K_before: list[float]
    K_after: list[float]
    drift_before: float
    drift_after: float
    eqs_score: float
    mcs: float
    breadcrumbs_completed: int
    report: LookingGlassReport
    advanced: bool = False
    deltas: list[list[float]] = field(default_factory=list)

    def __str__(self) -> str:
        return (
            f"Session {self.session_number} [{self.mode}] "
            f"{self.macro_tunnel}/{self.sub_tunnel} | "
            f"drift {self.drift_before:.3f} → {self.drift_after:.3f} | "
            f"EQS={self.eqs_score:.2f} MCS={self.mcs:.2f} | "
            f"action={self.report.recommended_action}"
        )


# ── The main session class ─────────────────────────────────────────────────────

class TunnelLearnerSession:
    """Tunnel-aware training session for a single learner.

    The class wraps one complete round of the protocol:

    1. ``start_session()``           → emit breadcrumb prompts (exploration or
                                       re-centering, chosen automatically from
                                       the current drift score).
    2. ``submit_response(eqs_score)``→ update K_local using v2 math, per breadcrumb.
    3. ``finish_session()``          → run the Looking Glass, persist history,
                                       optionally auto-advance to the next sub-tunnel.

    Args:
        name:           Learner identifier.
        manifold:       Optional tunnel manifold. Default = built-in curriculum.
        state:          Optional starting ``TunnelState``. Default = start of M1/F1.
        evaluator:      Optional Looking Glass. Default = built-in rule-based.
        auto_advance:   If True, ``finish_session`` applies the evaluator's
                        ``advance`` recommendation automatically.

    Example::

        session = TunnelLearnerSession("Ren")
        start = session.start_session()
        for prompt in start["breadcrumb_prompts"]:
            # ... run LLM, compute EQS externally ...
            session.submit_response(eqs_score=0.72)
        result = session.finish_session()
        print(result.report.summary())
    """

    def __init__(
        self,
        name: str,
        manifold: TunnelManifold | None = None,
        state: TunnelState | None = None,
        evaluator: LookingGlassEvaluator | None = None,
        auto_advance: bool = True,
    ) -> None:
        self.name = name
        self.manifold = manifold or TunnelManifold()
        self.evaluator = evaluator or LookingGlassEvaluator(self.manifold)
        self.auto_advance = auto_advance

        if state is None:
            first_macro = self.manifold.macro_tunnels[0]
            state = TunnelState(
                macro_tunnel=first_macro.key,
                sub_tunnel=first_macro.sub_tunnels[0].key,
                width=DEFAULT_WIDTH,
            )
        self.state: TunnelState = state

        self.sigma_history: list[float] = [self.manifold.global_sigma(self.state)]
        self.eqs_history: list[float] = []
        self.sessions_completed: int = 0

        # transient per-session bookkeeping
        self._in_session: bool = False
        self._session_mode: str = "exploration"
        self._session_prompts: list[str] = []
        self._session_eqs: list[float] = []
        self._session_deltas: list[list[float]] = []
        self._drift_before: float = 0.0
        self._K_before: list[float] = list(self.state.K_local)

    # ── Properties ─────────────────────────────────────────────────────────

    @property
    def global_sigma(self) -> float:
        """Overall progress along the whole tunnel system."""
        return self.manifold.global_sigma(self.state)

    @property
    def velocity(self) -> float:
        """Smoothed σ_global velocity across recent sessions."""
        return smoothed_velocity(self.sigma_history)

    @property
    def drift_score(self) -> float:
        return self.manifold.drift_score(self.state)

    @property
    def mcs(self) -> float:
        return mastery_confidence_score(
            self.eqs_history, self.sigma_history, self.state.K_local
        )

    # ── Lifecycle: start ───────────────────────────────────────────────────

    def start_session(self) -> dict:
        """Open a new session and emit breadcrumb prompts.

        The drift score at the *start* of the session chooses the mode:

        * drift ≥ ``DRIFT_THRESHOLD`` → re-centering breadcrumbs.
        * otherwise                  → exploration breadcrumbs.

        Returns:
            Dict with ``header``, ``breadcrumb_prompts``, ``n_breadcrumbs``,
            ``mode``, ``macro_tunnel`` and ``sub_tunnel`` keys.
        """
        if self._in_session:
            raise RuntimeError(
                "Already in a session. Call finish_session() first."
            )

        metrics = self.manifold.measure(self.state)
        macro = self.manifold.macro_tunnel(self.state.macro_tunnel)
        sub = macro.sub_tunnel(self.state.sub_tunnel)

        mode = "re_centering" if metrics.drift_score >= DRIFT_THRESHOLD else "exploration"
        n_bc = num_breadcrumbs(self.velocity)
        # Re-centering sessions stay short and intense.
        if mode == "re_centering":
            n_bc = max(1, min(n_bc, 3))

        prompts: list[str] = []
        for i in range(n_bc):
            if mode == "re_centering":
                prompts.append(
                    re_centering_breadcrumb_prompt(
                        self.name, self.state, macro, sub, metrics, i + 1
                    )
                )
            else:
                prompts.append(
                    exploration_breadcrumb_prompt(
                        self.name, self.state, macro, sub, metrics,
                        i + 1, self.velocity,
                    )
                )

        header = tunnel_session_header(
            self.name, self.state, self.manifold, metrics,
            n_bc, self.sessions_completed + 1, mode,
        )

        # bookkeeping
        self._in_session = True
        self._session_mode = mode
        self._session_prompts = prompts
        self._session_eqs = []
        self._session_deltas = []
        self._drift_before = metrics.drift_score
        self._K_before = list(self.state.K_local)

        return {
            "header": header,
            "breadcrumb_prompts": prompts,
            "n_breadcrumbs": n_bc,
            "mode": mode,
            "macro_tunnel": self.state.macro_tunnel,
            "sub_tunnel": self.state.sub_tunnel,
        }

    # ── Lifecycle: per-breadcrumb response ─────────────────────────────────

    def submit_response(self, eqs_score: float) -> list[float]:
        """Update ``K_local`` with an EQS score for one breadcrumb.

        The focus checkpoint for a sub-tunnel is simply its weakest axis
        in ``K_local`` (same convention as v2). This keeps the v2
        update_vector_v2 math fully re-usable.

        Args:
            eqs_score: Computed externally, in [0, 1].

        Returns:
            The applied delta vector.
        """
        if not self._in_session:
            raise RuntimeError("Not in a session. Call start_session() first.")

        # Weakest local axis becomes the focus checkpoint.
        focus_idx = self.state.K_local.index(min(self.state.K_local))
        focus_cp = ["A", "B", "C"][focus_idx]

        K_new, delta = update_vector_v2(
            self.state.K_local, eqs_score, focus_cp, mcs=self.mcs,
        )
        self.state.K_local = K_new
        # A productive session also nudges depth along with mastery.
        self.state.depth = max(0.0, min(1.0, self.state.depth + eqs_score * 0.05))
        self.state.position = max(0.0, min(1.0,
            0.5 * self.state.position + 0.5 * sum(K_new) / len(K_new)
        ))

        self._session_eqs.append(eqs_score)
        self._session_deltas.append(delta)
        return delta

    # ── Lifecycle: finish ──────────────────────────────────────────────────

    def finish_session(self) -> TunnelSessionResult:
        """Close the session, run the Looking Glass, and optionally advance.

        Returns:
            A fully populated ``TunnelSessionResult``.
        """
        if not self._in_session:
            raise RuntimeError("Not in a session. Call start_session() first.")

        session_eqs = (
            sum(self._session_eqs) / len(self._session_eqs)
            if self._session_eqs else 0.0
        )
        self.eqs_history.append(session_eqs)
        self.sigma_history.append(self.manifold.global_sigma(self.state))
        self.sessions_completed += 1
        self._in_session = False

        # Refresh drift now that K_local may have shifted during the session.
        self.state.drift_score = self.manifold.drift_score(self.state)

        report = self.evaluator.evaluate(
            self.state, self.eqs_history, self.sigma_history
        )

        advanced = False
        if self.auto_advance and report.recommended_action in ("advance", "compile"):
            nxt = self.manifold.next_sub_tunnel(self.state)
            if nxt is not None:
                if report.recommended_action == "compile":
                    self.state.compilation_level += 1
                self.state.macro_tunnel, self.state.sub_tunnel = nxt
                self.state.K_local = [0.0, 0.0, 0.0]
                self.state.position = 0.0
                self.state.depth = 0.0
                self.state.drift_score = 0.0
                advanced = True

        return TunnelSessionResult(
            session_number=self.sessions_completed,
            macro_tunnel=report.next_focus[0] if (advanced and report.next_focus) else self.state.macro_tunnel,
            sub_tunnel=report.next_focus[1] if (advanced and report.next_focus) else self.state.sub_tunnel,
            mode=self._session_mode,
            K_before=list(self._K_before),
            K_after=list(self.state.K_local),
            drift_before=self._drift_before,
            drift_after=self.state.drift_score,
            eqs_score=session_eqs,
            mcs=self.mcs,
            breadcrumbs_completed=len(self._session_eqs),
            report=report,
            advanced=advanced,
            deltas=list(self._session_deltas),
        )

    # ── Persistence ────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """JSON-safe serialisation of this session's learner state."""
        return {
            "name": self.name,
            "state": self.state.to_dict(),
            "sigma_history": self.sigma_history,
            "eqs_history": self.eqs_history,
            "sessions_completed": self.sessions_completed,
            "auto_advance": self.auto_advance,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        manifold: TunnelManifold | None = None,
        evaluator: LookingGlassEvaluator | None = None,
    ) -> "TunnelLearnerSession":
        session = cls(
            name=data["name"],
            manifold=manifold,
            state=TunnelState.from_dict(data["state"]),
            evaluator=evaluator,
            auto_advance=data.get("auto_advance", True),
        )
        session.sigma_history = list(data.get("sigma_history", []))
        session.eqs_history = list(data.get("eqs_history", []))
        session.sessions_completed = data.get("sessions_completed", 0)
        return session

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "TunnelLearnerSession":
        return cls.from_dict(json.loads(Path(path).read_text()))

    # ── Back-compat bridge ─────────────────────────────────────────────────

    def to_v2_session(self):
        """Return a v2 :class:`LearnerSession` seeded from this tunnel's K_local.

        Useful for code that hasn't migrated yet — hand it the flat session
        and keep going.
        """
        from hctp.tracker import LearnerSession
        return LearnerSession(
            name=self.name,
            K=list(self.state.K_local),
            sigma_history=list(self.sigma_history),
        )

    def __repr__(self) -> str:
        return (
            f"TunnelLearnerSession(name={self.name!r}, "
            f"macro={self.state.macro_tunnel!r}, "
            f"sub={self.state.sub_tunnel!r}, "
            f"K_local={[round(k, 3) for k in self.state.K_local]}, "
            f"drift={self.state.drift_score:.3f})"
        )
