"""
hctp.v3 — Helical Tunnel Mastery Protocol (HCTP 3.0, Phase 1)
=============================================================

HCTP 3.0 upgrades the flat 3-checkpoint line of v2 into a rich,
multi-level **tunnel manifold**. The learner now navigates inside a
cylindrical knowledge volume around the original helical centerline,
with two new structural ideas:

* **Macro-Tunnels → Sub-Tunnels.** A hierarchical curriculum instead of
  three scalar checkpoints. Ships with a 4-macro Python curriculum.
* **Tunnel width + drift.** Productive deviation is now explicit. When
  drift exceeds the tunnel width the protocol switches from
  *exploration* to *re-centering* breadcrumbs automatically.

Plus:

* **External Looking Glass Evaluator** — post-hoc, non-interfering,
  returns an objective ``recommended_action``.
* **TunnelLearnerSession** — the new entry point. Drop-in replacement
  for the v2 ``LearnerSession`` for tunnel-aware training.

Phase-1 scope (this release):
    * Multi-level tunnel data model (tunnel.py)
    * TunnelManifold + drift/alignment math (manifold.py)
    * External Looking Glass (looking_glass.py)
    * Exploration + Re-Centering breadcrumbs (breadcrumbs.py)
    * TunnelLearnerSession (session.py, backward-compat to v2)

Phase-2 and beyond will add the Anticipatory Hypothesis Engine, Layered
Process Supervision, Multi-Perspective Reflection, and Skill Compilation.
The ``Hypothesis`` dataclass + ``compilation_level`` field are already in
place so upstream code can begin wiring against them.

Quick start::

    from hctp.v3 import TunnelLearnerSession

    session = TunnelLearnerSession("Ren")
    start = session.start_session()
    for prompt in start["breadcrumb_prompts"]:
        # response = your_llm(prompt)
        # eqs_score = your_eqs_compute(response)
        session.submit_response(eqs_score=0.72)
    result = session.finish_session()

    print(result)
    print(result.report.summary())
"""

from .tunnel import (
    Hypothesis,
    SubTunnel,
    MacroTunnel,
    TunnelState,
    DEFAULT_TUNNEL_SYSTEM,
    DEFAULT_WIDTH,
    DRIFT_THRESHOLD,
)
from .manifold import TunnelManifold, TunnelMetrics, compute_drift
from .looking_glass import LookingGlassEvaluator, LookingGlassReport
from .breadcrumbs import (
    exploration_breadcrumb_prompt,
    re_centering_breadcrumb_prompt,
    tunnel_session_header,
)
from .session import TunnelLearnerSession, TunnelSessionResult

__all__ = [
    # Data model
    "Hypothesis",
    "SubTunnel",
    "MacroTunnel",
    "TunnelState",
    "DEFAULT_TUNNEL_SYSTEM",
    "DEFAULT_WIDTH",
    "DRIFT_THRESHOLD",
    # Geometry
    "TunnelManifold",
    "TunnelMetrics",
    "compute_drift",
    # Evaluation
    "LookingGlassEvaluator",
    "LookingGlassReport",
    # Prompts
    "exploration_breadcrumb_prompt",
    "re_centering_breadcrumb_prompt",
    "tunnel_session_header",
    # Session
    "TunnelLearnerSession",
    "TunnelSessionResult",
]
