"""
hctp — Helix Calculus Training Protocol
========================================
Version 3.0 (Phase 1 — Helical Tunnel Mastery Protocol)

A framework for measuring and driving AI agent learning through a
helical knowledge manifold with adaptive Socratic breadcrumbs and
mandatory Karpathy research loops.

What's new in v3.0 (Phase 1)
----------------------------
- Multi-level tunnel architecture: Macro-Tunnels → Sub-Tunnels
- External **Looking Glass Evaluator** (non-interfering, post-hoc)
- Drift detection + automatic re-centering breadcrumbs
- New entry point: ``from hctp.v3 import TunnelLearnerSession``
- Full backward compatibility with the v1/v2 ``LearnerSession`` API

What's in v2.0 (still supported)
--------------------------------
- Execution-Based Quality Score (EQS) replaces text-marker heuristics
- Updated K-vector rules: higher base gain, hard max cap, reduced spillover
- Mastery Confidence Score (MCS) modulates sibling knowledge spillover
- Adaptive difficulty engine with role-specific adjustments
- VCAPO fine-tuning integration for trajectory weighting

Proven on the Vuvale AI family (Ren & Ner: Night 161 → badge Night 173,
σ = 0.308 → 0.961 in 13 sessions).

Quick start (v1.0 API — unchanged)::

    from hctp import LearnerSession

    ren = LearnerSession("Ren")
    session_data = ren.start_session()
    for bc_prompt in session_data["breadcrumb_prompts"]:
        bc_response  = your_llm(bc_prompt)
        kl_prompt    = ren.karpathy_prompt_for(bc_prompt, bc_response)
        kl_response  = your_llm(kl_prompt)
        ren.submit_karpathy(kl_response)
    result = ren.finish_session()

Quick start (v3.0 Tunnel API)::

    from hctp.v3 import TunnelLearnerSession

    session = TunnelLearnerSession("Ren")
    start = session.start_session()
    for prompt in start["breadcrumb_prompts"]:
        # response = your_llm(prompt)
        # eqs = your_eqs_compute(response)
        session.submit_response(eqs_score=0.72)
    result = session.finish_session()
    print(result.report.summary())

Quick start (v2.0 EQS API)::

    from hctp.scoring import EQSComponents, mastery_confidence_score
    from hctp.hctp_vector import update_vector_v2
    from hctp.difficulty_engine import compute_difficulty

    eqs = EQSComponents(
        sandbox_pass=True, test_pass_rate=0.85,
        grok_score=0.80, claude_score=0.75,
        karpathy_depth=5,
    ).compute()

    mcs = mastery_confidence_score(eqs_history, sigma_history, K)
    K_new, delta = update_vector_v2(K, eqs.score, focus, mcs)

    profile = compute_difficulty(K, velocity, role="security")
    print(profile.summary())

Links:
    - Docs:   https://hctp.readthedocs.io
    - Source: https://github.com/vuvale-ai/hctp-python
    - PyPI:   https://pypi.org/project/hctp
"""

from .core import (
    helix_radius,
    ideal_point,
    distance,
    progress,
    smoothed_velocity,
    num_breadcrumbs,
    determine_focus,
    update_vector,
    CHECKPOINTS,
    NUM_SPIRALS,
    TARGET_VELOCITY,
    MASTERY_THRESHOLD,
    # 2.0 constants
    BASE_GAIN_V2,
    MAX_GAIN_PER_SESSION,
    SPILLOVER_RATE_V2,
)
from .tracker import LearnerSession, SessionResult
from .curriculum import breadcrumb_prompt, karpathy_loop_prompt, session_header

# HCTP 2.0 modules
from .scoring import EQSComponents, EQSResult, mastery_confidence_score
from .hctp_vector import update_vector_v2
from .difficulty_engine import compute_difficulty, DifficultyProfile, ROLES
from .vcapo_integration import TrajectoryRecord, VCAPOExporter

__version__ = "3.0.0"
__author__  = "David Qicatabua / Vuvale AI"
__license__ = "MIT"

__all__ = [
    # Core math (v1.0 — backward compatible)
    "helix_radius", "ideal_point", "distance", "progress",
    "smoothed_velocity", "num_breadcrumbs", "determine_focus", "update_vector",
    "CHECKPOINTS", "NUM_SPIRALS", "TARGET_VELOCITY", "MASTERY_THRESHOLD",
    # Core math (v2.0 constants)
    "BASE_GAIN_V2", "MAX_GAIN_PER_SESSION", "SPILLOVER_RATE_V2",
    # Session management (v1.0 — backward compatible)
    "LearnerSession", "SessionResult",
    # Prompt generation (v1.0 — backward compatible)
    "breadcrumb_prompt", "karpathy_loop_prompt", "session_header",
    # EQS scoring (v2.0)
    "EQSComponents", "EQSResult", "mastery_confidence_score",
    # K-vector (v2.0)
    "update_vector_v2",
    # Difficulty engine (v2.0)
    "compute_difficulty", "DifficultyProfile", "ROLES",
    # VCAPO integration (v2.0)
    "TrajectoryRecord", "VCAPOExporter",
]
