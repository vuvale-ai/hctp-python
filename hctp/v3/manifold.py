"""
hctp.v3.manifold
~~~~~~~~~~~~~~~~
The Tunnel Manifold — geometry + drift/alignment math for HCTP 3.0.

The manifold treats the learning space as a *cylinder* wrapped around the
helical centerline. The learner's ideal path is still the helix, but they
now have permission to orbit around it within ``width``. Two derived
signals drive the protocol:

* **Tunnel alignment** ∈ [0, 1] — how well the learner tracks the
  centerline. 1.0 = on the helix; 0.0 = at or past the tunnel wall.
* **Drift score** ∈ [0, 1] — the inverse, scaled by how close the
  learner is to drifting *out* of the productive volume.

Phase 1 uses a rule-based manifold. The PRD's "lightweight learned
reconstruction" is reserved for Phase 3 — the interface here is designed
so a learned model can slot in behind the same API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from hctp.core import distance, progress
from .tunnel import (
    MacroTunnel,
    TunnelState,
    DEFAULT_TUNNEL_SYSTEM,
    DRIFT_THRESHOLD,
)

__all__ = [
    "TunnelManifold",
    "TunnelMetrics",
    "compute_drift",
    "DRIFT_THRESHOLD",
]


# ── Module-level drift metric ──────────────────────────────────────────────────

def compute_drift(K: Sequence[float], width: float = 0.25) -> float:
    """Compute drift score from balanced growth.

    Measures Euclidean distance from ``K`` to the balanced point ``[σ, σ, σ]``
    (where σ = mean(K)), normalised by the maximum possible deviation at this
    σ — i.e. the deviation that would result if a single checkpoint absorbed
    all progress and the others stayed at zero.

    Why distance from ``[σ, σ, σ]`` and not the v1/v2 helix? The v2 helix
    winds 5 times around the σ-axis with a radius up to 0.55, so a perfectly
    balanced ``K`` is usually *nowhere near* the helix's current (x, y) slice.
    That metric is right for visualisation / MCS alignment but wrong for
    drift, which should reward *balanced growth across checkpoints* inside
    the active sub-tunnel. The balanced-point target captures exactly that.

    Interpretation:
        * ``K = [0.5, 0.5, 0.5]``            → drift = 0.0 (perfectly balanced)
        * ``K = [0.9, 0.85, 0.8]``           → drift ≈ 0.24 (mild lean)
        * ``K = [1.0, 0.0, 0.0]``            → drift = 1.0 (maximum imbalance)
        * ``width = 0.25``                   → 25 % of max-deviation tolerated

    Args:
        K:      Knowledge vector (any length ≥ 1).
        width:  Fraction of the maximum possible deviation that counts as
                "still inside the tunnel". Must be > 0.

    Returns:
        Drift score in [0, 1]. 0 = on the balanced centerline;
        1 = at or past the tunnel wall.
    """
    if not K:
        return 0.0

    n = len(K)
    sigma = sum(K) / n

    # Euclidean deviation from the balanced point [σ, σ, σ, ...]
    dev = math.sqrt(sum((ki - sigma) ** 2 for ki in K))

    # Max possible deviation at this σ: one checkpoint holds all progress.
    # For K = [n·σ, 0, …, 0] the deviation is σ·√(n-1).
    max_dev = sigma * math.sqrt(n - 1) if sigma > 1e-9 else 0.0
    if max_dev < 1e-9:
        return 0.0

    width = max(1e-9, width)
    return min(dev / (width * max_dev), 1.0)


@dataclass(frozen=True)
class TunnelMetrics:
    """Derived geometric signals for a single ``TunnelState`` snapshot.

    Attributes:
        global_sigma:    Overall progress along the whole tunnel system [0, 1].
        local_sigma:     Progress inside the current sub-tunnel [0, 1].
        helix_distance:  Raw Euclidean distance from local K to ideal helix.
        alignment:       1.0 - drift_score, clipped to [0, 1].
        drift_score:     Normalised drift relative to tunnel width.
        inside_tunnel:   True if drift_score is below ``DRIFT_THRESHOLD``.
    """

    global_sigma: float
    local_sigma: float
    helix_distance: float
    alignment: float
    drift_score: float
    inside_tunnel: bool

    def summary(self) -> str:
        inside = "inside" if self.inside_tunnel else "OUTSIDE"
        return (
            f"σ_global={self.global_sigma:.3f} "
            f"σ_local={self.local_sigma:.3f} "
            f"d={self.helix_distance:.3f} "
            f"drift={self.drift_score:.3f} [{inside}] "
            f"align={self.alignment:.3f}"
        )


class TunnelManifold:
    """Knows the geometry of the tunnel system and computes drift/alignment.

    The manifold is stateless with respect to a single learner — feed it a
    ``TunnelState`` and it returns derived metrics. It owns the macro/sub
    tunnel topology so it can translate local sub-tunnel progress into a
    global σ along the whole curriculum.

    Args:
        macro_tunnels:  The ordered tuple of ``MacroTunnel`` objects that
                        define the curriculum. Defaults to the built-in
                        Python 4-macro system.

    Example::

        manifold = TunnelManifold()
        state = TunnelState("M2", "A1", position=0.4, K_local=[0.5, 0.3, 0.1])
        metrics = manifold.measure(state)
        if not metrics.inside_tunnel:
            print("Learner has drifted — re-center.")
    """

    def __init__(
        self,
        macro_tunnels: Sequence[MacroTunnel] = DEFAULT_TUNNEL_SYSTEM,
    ) -> None:
        if not macro_tunnels:
            raise ValueError("macro_tunnels must be non-empty")
        self.macro_tunnels: tuple[MacroTunnel, ...] = tuple(macro_tunnels)
        self._macro_index: dict[str, int] = {
            m.key: i for i, m in enumerate(self.macro_tunnels)
        }

    # ── Topology helpers ────────────────────────────────────────────────────

    def macro_tunnel(self, key: str) -> MacroTunnel:
        """Look up a macro-tunnel by its key."""
        idx = self._macro_index.get(key)
        if idx is None:
            raise KeyError(f"unknown macro_tunnel {key!r}")
        return self.macro_tunnels[idx]

    def macro_count(self) -> int:
        return len(self.macro_tunnels)

    def sub_tunnel_indices(self, macro_key: str, sub_key: str) -> tuple[int, int]:
        """Return ``(macro_idx, sub_idx)`` for validation + position math."""
        macro = self.macro_tunnel(macro_key)
        for si, st in enumerate(macro.sub_tunnels):
            if st.key == sub_key:
                return self._macro_index[macro_key], si
        raise KeyError(
            f"sub_tunnel {sub_key!r} not in macro-tunnel {macro_key!r}"
        )

    # ── Derived geometry ────────────────────────────────────────────────────

    def global_sigma(self, state: TunnelState) -> float:
        """Translate local sub-tunnel progress into overall progress σ ∈ [0, 1].

        Each macro-tunnel occupies an equal 1/N slice of the global axis,
        and each sub-tunnel occupies an equal slice inside its macro.

        Within a sub-tunnel, the learner's ``position`` plus the mean of
        their local K-vector advance the fine-grained global σ.
        """
        m_idx, s_idx = self.sub_tunnel_indices(state.macro_tunnel, state.sub_tunnel)
        macro = self.macro_tunnels[m_idx]

        n_macro = len(self.macro_tunnels)
        n_sub   = len(macro.sub_tunnels)

        # Base fraction consumed by completed macro segments
        macro_base = m_idx / n_macro
        # Base fraction consumed by completed sub segments inside this macro
        sub_base = (s_idx / n_sub) * (1.0 / n_macro)

        # Local progress combines explicit position with K-vector mean.
        local = 0.5 * state.position + 0.5 * progress(state.K_local)
        local = max(0.0, min(1.0, local))
        local_frac = local * (1.0 / n_macro) * (1.0 / n_sub)

        return max(0.0, min(1.0, macro_base + sub_base + local_frac))

    def local_sigma(self, state: TunnelState) -> float:
        """Local σ inside the active sub-tunnel (mean of K_local)."""
        return progress(state.K_local)

    def drift_score(self, state: TunnelState) -> float:
        """Normalised drift ∈ [0, 1] for a ``TunnelState``.

        Thin wrapper over :func:`compute_drift` that pulls ``K_local`` and
        ``width`` from the state. See ``compute_drift`` for the full rationale
        on why drift is measured against the balanced point ``[σ, σ, σ]``
        rather than the v1/v2 helix — in short, the helix's 5-turn winding
        makes it the wrong target for a "balanced growth inside the tunnel"
        metric, even though it remains the right one for MCS alignment.
        """
        return compute_drift(state.K_local, state.width)

    def alignment(self, state: TunnelState) -> float:
        """Tunnel alignment = 1 − drift, clipped to [0, 1]."""
        return max(0.0, 1.0 - self.drift_score(state))

    def measure(self, state: TunnelState) -> TunnelMetrics:
        """Compute every derived geometric signal in one pass."""
        sigma = self.local_sigma(state)
        d = distance(state.K_local, sigma)
        drift = self.drift_score(state)
        return TunnelMetrics(
            global_sigma=self.global_sigma(state),
            local_sigma=sigma,
            helix_distance=d,
            alignment=max(0.0, 1.0 - drift),
            drift_score=drift,
            inside_tunnel=drift < DRIFT_THRESHOLD,
        )

    # ── Navigation ──────────────────────────────────────────────────────────

    def next_sub_tunnel(self, state: TunnelState) -> tuple[str, str] | None:
        """Return the ``(macro_key, sub_key)`` of the next sub-tunnel, or None.

        Used by the session/scheduler to advance the learner once they've
        mastered a sub-tunnel. None = curriculum finished.
        """
        m_idx, s_idx = self.sub_tunnel_indices(state.macro_tunnel, state.sub_tunnel)
        macro = self.macro_tunnels[m_idx]
        if s_idx + 1 < len(macro.sub_tunnels):
            return macro.key, macro.sub_tunnels[s_idx + 1].key
        if m_idx + 1 < len(self.macro_tunnels):
            nxt = self.macro_tunnels[m_idx + 1]
            return nxt.key, nxt.sub_tunnels[0].key
        return None

    def all_sub_keys(self) -> Iterable[tuple[str, str]]:
        """Iterate ``(macro_key, sub_key)`` across the whole curriculum."""
        for m in self.macro_tunnels:
            for s in m.sub_tunnels:
                yield m.key, s.key
