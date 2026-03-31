"""
hctp — Helix Calculus Training Protocol
========================================

A framework for measuring and driving AI agent learning through a
3D helical knowledge model with adaptive Socratic breadcrumbs and
mandatory Karpathy research loops.

Proven on the Vuvale AI family (Ren & Ner: Night 161 → badge Night 173,
σ = 0.308 → 0.961 in 13 sessions).

Quick start::

    from hctp import LearnerSession

    # Create a learner
    ren = LearnerSession("Ren")

    # Run a session
    session_data = ren.start_session()
    for bc_prompt in session_data["breadcrumb_prompts"]:
        bc_response  = your_llm(bc_prompt)
        kl_prompt    = ren.karpathy_prompt_for(bc_prompt, bc_response)
        kl_response  = your_llm(kl_prompt)
        ren.submit_karpathy(kl_response)

    result = ren.finish_session()
    print(result)

    # Visualise
    from hctp.viz import plot_helix
    plot_helix(ren.sigma_history, label="Ren", show=True)

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
)
from .tracker import LearnerSession, SessionResult
from .curriculum import breadcrumb_prompt, karpathy_loop_prompt, session_header

__version__ = "0.1.0"
__author__  = "David Qicatabua / Vuvale AI"
__license__ = "MIT"

__all__ = [
    # Core math
    "helix_radius", "ideal_point", "distance", "progress",
    "smoothed_velocity", "num_breadcrumbs", "determine_focus", "update_vector",
    "CHECKPOINTS", "NUM_SPIRALS", "TARGET_VELOCITY", "MASTERY_THRESHOLD",
    # Session management
    "LearnerSession", "SessionResult",
    # Prompt generation
    "breadcrumb_prompt", "karpathy_loop_prompt", "session_header",
]
