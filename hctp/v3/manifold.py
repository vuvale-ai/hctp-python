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
    "DRIFT_THRESHOLD",
]


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
        """Normalised drift ∈ [0, 1].

        Drift = imbalance across the local K-vector (max − min), normalised
        to the learner's tunnel ``width``. The v1/v2 helical centerline is
        preserved for reporting (helix_distance) but **not** used for drift
        because the 5-spiral winding makes that metric non-monotonic with
        respect to the intuitive "balanced progress" ideal of a tunnel.

        Interpretation:
            * ``K=[0.5, 0.5, 0.5]``   → perfectly balanced → drift 0.
            * ``K=[1.0, 0.0, 0.0]``   → maximally imbalanced → drift 1.
            * ``width`` = 0.25        → 25% imbalance tolerated before
                                        the tunnel wall is reached.
        """
        if not state.K_local:
            return 0.0
        imbalance = max(state.K_local) - min(state.K_local)
        width = max(1e-6, state.width)
        return max(0.0, min(1.0, imbalance / width))

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
