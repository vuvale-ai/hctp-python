"""
hctp.v3.breadcrumbs
~~~~~~~~~~~~~~~~~~~
Prompt generation for the Helical Tunnel Mastery Protocol.

HCTP 3.0 has **two** breadcrumb modes:

* **Exploration breadcrumbs** — the default. They are *inside* the
  tunnel and deliberately give the learner width: they can wander into
  neighbouring concepts inside the same sub-tunnel.

* **Re-centering breadcrumbs** — kick in automatically when the Looking
  Glass reports high drift. These prompts *explicitly* reference the
  tunnel boundary the learner is leaning on and pull the agent back to
  the centerline of the active sub-tunnel.

All prompts are LLM-agnostic strings — pass them to any model.
"""

from __future__ import annotations

from typing import Sequence

from hctp.core import TARGET_VELOCITY
from .manifold import TunnelManifold, TunnelMetrics
from .tunnel import MacroTunnel, SubTunnel, TunnelState

__all__ = [
    "exploration_breadcrumb_prompt",
    "re_centering_breadcrumb_prompt",
    "tunnel_session_header",
]


# ── Internal helpers ───────────────────────────────────────────────────────────

def _pace(velocity: float) -> str:
    if velocity >= TARGET_VELOCITY:
        return "fast"
    if velocity < 0.10:
        return "slow"
    return "average"


def _format_K(K: Sequence[float]) -> str:
    return "[" + ", ".join(f"{k:.3f}" for k in K) + "]"


# ── Public prompt builders ─────────────────────────────────────────────────────

def exploration_breadcrumb_prompt(
    learner_name: str,
    state: TunnelState,
    macro: MacroTunnel,
    sub: SubTunnel,
    metrics: TunnelMetrics,
    breadcrumb_idx: int,
    velocity: float,
) -> str:
    """Build an *inside-the-tunnel* Socratic breadcrumb prompt.

    Exploration breadcrumbs are permissive: they encourage the learner
    to connect the active sub-tunnel to neighbouring concepts *inside*
    its macro-tunnel. Drift is tolerated up to the tunnel width.
    """
    pace = _pace(velocity)
    return (
        f"HELICAL TUNNEL MASTERY PROTOCOL — Exploration Breadcrumb #{breadcrumb_idx}\n"
        f"Learner: {learner_name}\n"
        f"Macro-Tunnel: {macro.key} — {macro.name}\n"
        f"Sub-Tunnel:   {sub.key} — {sub.name}\n"
        f"Local K: {_format_K(state.K_local)}  "
        f"σ_local={metrics.local_sigma:.3f}  "
        f"σ_global={metrics.global_sigma:.3f}\n"
        f"Drift={metrics.drift_score:.3f} (inside)  "
        f"Alignment={metrics.alignment:.3f}  Pace={pace}\n"
        f"Concepts in focus: {sub.concepts}\n\n"
        f"RULES (STRICT):\n"
        f"1. Generate ONE Socratic micro-task about \"{sub.name}\".\n"
        f"2. {'Open the field — the learner is flowing; invite wide connections to the other sub-tunnels of this macro.' if pace == 'fast' else 'Stay focused and concrete — the learner needs traction, not breadth.'}\n"
        f"3. Include ONE plausible red-herring approach.\n"
        f"4. Include ONE Python-specific practical twist.\n"
        f"5. NEVER provide code, solutions, or direct answers.\n"
        f"6. Encourage the learner to name a connection to ONE OTHER sub-tunnel "
        f"inside the same macro-tunnel (\"{macro.name}\") — not outside it.\n"
        f"7. End with EXACTLY: 'Name the next concept inside this tunnel you "
        f"expect to need — and why.'\n\n"
        f"Generate the breadcrumb now."
    )


def re_centering_breadcrumb_prompt(
    learner_name: str,
    state: TunnelState,
    macro: MacroTunnel,
    sub: SubTunnel,
    metrics: TunnelMetrics,
    breadcrumb_idx: int,
) -> str:
    """Build a re-centering breadcrumb — used when the agent has drifted.

    These prompts name the tunnel explicitly, remind the learner of its
    centerline, and ask for a minimal, load-bearing exercise that
    reconnects them to the active sub-tunnel.
    """
    return (
        f"HELICAL TUNNEL MASTERY PROTOCOL — Re-Centering Breadcrumb #{breadcrumb_idx}\n"
        f"Learner: {learner_name}\n"
        f"Macro-Tunnel: {macro.key} — {macro.name}\n"
        f"Sub-Tunnel:   {sub.key} — {sub.name}\n"
        f"Local K: {_format_K(state.K_local)}  "
        f"Drift={metrics.drift_score:.3f}  "
        f"Alignment={metrics.alignment:.3f}\n\n"
        f"DIAGNOSIS: The learner has drifted outside the productive volume of the\n"
        f"current sub-tunnel (drift ≥ tunnel width). Do NOT introduce new concepts.\n\n"
        f"RULES (STRICT):\n"
        f"1. Name the tunnel explicitly in the first sentence: \"You are re-centering "
        f"in {macro.name} → {sub.name}.\"\n"
        f"2. Ask ONE minimal exercise whose ONLY subject is: {sub.concepts}.\n"
        f"3. Explicitly forbid tangents to other macro-tunnels in the prompt.\n"
        f"4. Require the learner to restate the centerline idea in ONE sentence "
        f"before attempting any code.\n"
        f"5. NEVER provide the solution.\n"
        f"6. End with EXACTLY: 'State the single invariant that defines "
        f"{sub.name} — then solve the exercise.'\n\n"
        f"Generate the re-centering breadcrumb now."
    )


def tunnel_session_header(
    learner_name: str,
    state: TunnelState,
    manifold: TunnelManifold,
    metrics: TunnelMetrics,
    n_breadcrumbs: int,
    session_number: int,
    mode: str,
) -> str:
    """Markdown session header shown to the learner / logged to dashboards.

    Args:
        mode: "exploration" or "re_centering".
    """
    macro = manifold.macro_tunnel(state.macro_tunnel)
    sub = macro.sub_tunnel(state.sub_tunnel)
    inside = "inside tunnel" if metrics.inside_tunnel else "OUTSIDE TUNNEL"
    return (
        f"## {learner_name} — HCTP 3.0 Session {session_number}\n\n"
        f"**Macro-Tunnel:** {macro.key} — {macro.name}  \n"
        f"**Sub-Tunnel:**   {sub.key} — {sub.name}  \n"
        f"**K_local:** {_format_K(state.K_local)}  \n"
        f"**σ_local:** {metrics.local_sigma:.3f}  "
        f"**σ_global:** {metrics.global_sigma:.3f}  \n"
        f"**Drift:** {metrics.drift_score:.3f} ({inside})  "
        f"**Alignment:** {metrics.alignment:.3f}  \n"
        f"**Mode:** {mode} | **Breadcrumbs:** {n_breadcrumbs}\n"
    )
