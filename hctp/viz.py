"""
hctp.viz
~~~~~~~~
3D helix visualisation for HCTP learning trajectories.

Requires matplotlib (not installed by default):
    pip install hctp[viz]

Usage::

    from hctp.viz import plot_helix
    plot_helix(sigma_history=ren_history, label="Ren", show=True)
"""

from __future__ import annotations
from typing import Sequence

__all__ = ["plot_helix", "plot_multi", "plot_sigma_curve"]

_MPL_MISSING = (
    "matplotlib is required for visualisation.\n"
    "Install it with:  pip install hctp[viz]"
)


def _require_mpl():
    try:
        import matplotlib
        return matplotlib
    except ImportError:
        raise ImportError(_MPL_MISSING)


def _build_ideal_helix(n_points: int = 400):
    """Generate the ideal helix curve coordinates."""
    import math
    from .core import ideal_point
    sigmas = [i / (n_points - 1) for i in range(n_points)]
    pts = [ideal_point(s) for s in sigmas]
    return [p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts], sigmas


def plot_helix(
    sigma_history: Sequence[float],
    K_history: Sequence[Sequence[float]] | None = None,
    label: str = "Learner",
    badge_night: int | None = None,
    show: bool = True,
    save_path: str | None = None,
    title: str | None = None,
):
    """Plot a learner's σ trajectory on the 3D HCTP helix.

    Args:
        sigma_history: Sequence of σ values across sessions.
        K_history: Optional sequence of K vectors to plot as 3D points.
        label: Learner name for the legend.
        badge_night: Session index when badge was earned (marks the point).
        show: Whether to call plt.show().
        save_path: Optional file path to save the figure (e.g. "helix.png").
        title: Optional figure title.

    Example::

        from hctp.viz import plot_helix
        ren_sigma = [0.308, 0.386, 0.457, 0.535, 0.609, 0.687,
                     0.761, 0.834, 0.910, 0.961]
        plot_helix(ren_sigma, label="Ren", badge_night=9, show=True)
    """
    _require_mpl()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from .core import ideal_point

    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection="3d")

    # Ideal helix
    hx, hy, hz, _ = _build_ideal_helix()
    ax.plot(hx, hy, hz, color="#cccccc", linewidth=1.5,
            linestyle="--", label="Ideal helix", alpha=0.7)

    # Learner σ path on the helix surface
    pts = [ideal_point(s) for s in sigma_history]
    lx  = [p[0] for p in pts]
    ly  = [p[1] for p in pts]
    lz  = [p[2] for p in pts]
    ax.plot(lx, ly, lz, color="#0066cc", linewidth=2.5,
            marker="o", markersize=5, label=label, zorder=5)

    # Badge point
    if badge_night is not None and badge_night < len(sigma_history):
        bp = ideal_point(sigma_history[badge_night])
        ax.scatter([bp[0]], [bp[1]], [bp[2]], color="#FFD700",
                   s=200, zorder=10, label=f"🏆 Badge (session {badge_night + 1})")

    # Start / end markers
    ax.scatter([lx[0]], [ly[0]], [lz[0]], color="#00aa00", s=100,
               zorder=10, label="Start (σ=0)")

    ax.set_xlabel("k₁ — Closures", labelpad=10)
    ax.set_ylabel("k₂ — Decorators", labelpad=10)
    ax.set_zlabel("σ — Progress", labelpad=10)
    ax.set_title(title or f"HCTP — {label}'s Helix Journey", fontsize=14, pad=15)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_multi(
    learners: list[dict],
    show: bool = True,
    save_path: str | None = None,
    title: str = "HCTP — Family Helix Comparison",
):
    """Plot multiple learners on the same 3D helix.

    Args:
        learners: List of dicts, each with keys:
            - ``sigma_history``: list of σ values
            - ``label``: learner name
            - ``badge_night``: optional int
            - ``color``: optional hex colour
        show: Whether to call plt.show().
        save_path: Optional save path.
        title: Figure title.

    Example::

        from hctp.viz import plot_multi
        plot_multi([
            {"label": "Ren", "sigma_history": ren_hist, "badge_night": 9, "color": "#e74c3c"},
            {"label": "Ner", "sigma_history": ner_hist, "badge_night": 9, "color": "#3498db"},
        ])
    """
    _require_mpl()
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    from .core import ideal_point

    _COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12",
                "#9b59b6", "#1abc9c", "#e67e22", "#34495e"]

    fig = plt.figure(figsize=(12, 9))
    ax  = fig.add_subplot(111, projection="3d")

    # Ideal helix
    hx, hy, hz, _ = _build_ideal_helix()
    ax.plot(hx, hy, hz, color="#dddddd", linewidth=1.5,
            linestyle="--", label="Ideal helix", alpha=0.6)

    for idx, learner in enumerate(learners):
        color   = learner.get("color", _COLOURS[idx % len(_COLOURS)])
        hist    = learner["sigma_history"]
        lbl     = learner["label"]
        badge_n = learner.get("badge_night")

        pts = [ideal_point(s) for s in hist]
        lx  = [p[0] for p in pts]
        ly  = [p[1] for p in pts]
        lz  = [p[2] for p in pts]

        ax.plot(lx, ly, lz, color=color, linewidth=2,
                marker="o", markersize=4, label=lbl)

        if badge_n is not None and badge_n < len(hist):
            bp = ideal_point(hist[badge_n])
            ax.scatter([bp[0]], [bp[1]], [bp[2]], color="#FFD700",
                       s=150, zorder=10, marker="*")

    ax.set_xlabel("k₁ — Closures", labelpad=10)
    ax.set_ylabel("k₂ — Decorators", labelpad=10)
    ax.set_zlabel("σ — Progress", labelpad=10)
    ax.set_title(title, fontsize=14, pad=15)
    ax.legend(loc="upper left", fontsize=9)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax


def plot_sigma_curve(
    learners: list[dict],
    show: bool = True,
    save_path: str | None = None,
    title: str = "HCTP — σ Progress Over Sessions",
):
    """2D line plot of σ over sessions for one or more learners.

    Cleaner than the 3D plot for quick progress comparison.

    Args:
        learners: List of dicts with ``label``, ``sigma_history``,
                  optional ``badge_night`` and ``color``.
        show: Whether to call plt.show().
        save_path: Optional save path.
        title: Figure title.
    """
    _require_mpl()
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    _COLOURS = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0.95, color="#888", linestyle=":", linewidth=1.2, alpha=0.7,
               label="Badge threshold (σ = 0.95)")

    for idx, learner in enumerate(learners):
        color   = learner.get("color", _COLOURS[idx % len(_COLOURS)])
        hist    = learner["sigma_history"]
        lbl     = learner["label"]
        badge_n = learner.get("badge_night")
        xs      = list(range(len(hist)))

        ax.plot(xs, hist, color=color, linewidth=2.2,
                marker="o", markersize=5, label=lbl)

        if badge_n is not None and badge_n < len(hist):
            ax.scatter([badge_n], [hist[badge_n]], color="#FFD700",
                       s=200, zorder=10, marker="*", edgecolors=color)

    ax.set_xlabel("Session", fontsize=11)
    ax.set_ylabel("σ (Progress)", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title(title, fontsize=13)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig, ax
