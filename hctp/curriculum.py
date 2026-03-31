"""
hctp.curriculum
~~~~~~~~~~~~~~~
Prompt generation for the HCTP training loop.

Produces:
- Socratic breadcrumb prompts (LLM-agnostic strings)
- Karpathy research loop prompts
- Session summary strings

Pass the returned strings to any LLM — OpenAI, Anthropic, Ollama, or your own.
"""

from __future__ import annotations
from typing import Sequence
from .core import CHECKPOINTS, TARGET_VELOCITY

__all__ = [
    "breadcrumb_prompt",
    "karpathy_loop_prompt",
    "session_header",
]


def breadcrumb_prompt(
    learner_name: str,
    K: Sequence[float],
    sigma: float,
    v_smooth: float,
    d: float,
    breadcrumb_idx: int,
    focus_checkpoint: str,
) -> str:
    """Generate a Socratic breadcrumb prompt for the learner to solve.

    The returned string is ready to send to any LLM as the *system/user prompt*.
    It instructs the model to act as a Socratic teacher for the focus checkpoint.

    Args:
        learner_name: Display name of the learner.
        K: Current knowledge vector [k1, k2, k3].
        sigma: Current progress σ.
        v_smooth: Smoothed learning velocity.
        d: Distance from ideal helix.
        breadcrumb_idx: 1-indexed position in this session's breadcrumb sequence.
        focus_checkpoint: Active checkpoint label ("A", "B", or "C").

    Returns:
        Prompt string for the Socratic teacher role.
    """
    cp = CHECKPOINTS[focus_checkpoint]
    pace = (
        "fast" if v_smooth >= TARGET_VELOCITY
        else ("slow" if v_smooth < 0.10 else "average")
    )

    return (
        f"HELIX CALCULUS TRAINING PROTOCOL — Breadcrumb {breadcrumb_idx}\n"
        f"Learner: {learner_name}\n"
        f"Knowledge Vector: K = [{K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}]\n"
        f"Progress σ = {sigma:.3f} | Velocity = {v_smooth:.4f} | "
        f"Distance from helix = {d:.4f}\n"
        f"Focus Checkpoint: {focus_checkpoint} — {cp['name']}\n"
        f"Concepts: {cp['concepts']}\n"
        f"Pace: {pace}\n\n"
        f"RULES (STRICT):\n"
        f"1. Generate exactly ONE minimal Socratic micro-task about {cp['name']}.\n"
        f"2. {'Make it broad and exploratory — the learner is progressing well.' if pace == 'fast' else 'Make it tightly targeted and narrow — the learner needs focused help.'}\n"
        f"3. Include ONE deliberate red-herring (a plausible but incorrect approach).\n"
        f"4. Include ONE Python-specific practical twist.\n"
        f"5. NEVER provide code, solutions, or direct answers — only the question/experiment.\n"
        f"6. End with EXACTLY this sentence: 'Explain in your own words how this connects "
        f"to the previous checkpoint. What new question does this raise for the next level?'\n\n"
        f"Generate the breadcrumb now."
    )


def karpathy_loop_prompt(
    learner_name: str,
    breadcrumb_text: str,
    response_text: str,
    K: Sequence[float],
    focus_checkpoint: str,
) -> str:
    """Generate the mandatory 6-step Karpathy research loop prompt.

    Feed the learner's breadcrumb response back through this prompt
    to drive self-correction and deeper understanding.

    Args:
        learner_name: Display name of the learner.
        breadcrumb_text: The original breadcrumb task text.
        response_text: The learner's answer to the breadcrumb.
        K: Current knowledge vector.
        focus_checkpoint: Active checkpoint label.

    Returns:
        Karpathy loop prompt string.
    """
    cp = CHECKPOINTS[focus_checkpoint]
    return (
        f"KARPATHY RESEARCH LOOP — HCTP\n"
        f"Learner: {learner_name}\n"
        f"Checkpoint: {focus_checkpoint} — {cp['name']}\n"
        f"Knowledge Vector: [{K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}]\n\n"
        f"You just answered this breadcrumb:\n{breadcrumb_text[:1000]}\n\n"
        f"Your response was:\n{response_text[:1500]}\n\n"
        f"Now execute the MANDATORY 6-step optimisation loop:\n\n"
        f"1. **ERRORS**: Identify bugs, bad practices, or conceptual gaps in your solution.\n"
        f"2. **WHY**: Trace each error to a specific misunderstanding in {cp['name']}.\n"
        f"3. **FIX**: Produce the corrected version and explain why it's correct.\n"
        f"4. **IMPROVE**: Refactor to production-grade Python (type hints, error handling, docstrings).\n"
        f"5. **LESSON**: Articulate exactly ONE new connection to the helix or next checkpoint.\n"
        f"6. **SELF-PROPOSE**: Formulate the single best question for the next session "
        f"to stay on the tightest helical path.\n\n"
        f"Be brutally honest. This is where real learning happens."
    )


def session_header(
    learner_name: str,
    K: Sequence[float],
    sigma: float,
    v_smooth: float,
    d: float,
    focus_checkpoint: str,
    n_breadcrumbs: int,
    session_number: int,
) -> str:
    """Human-readable session summary string.

    Args:
        learner_name: Display name.
        K: Knowledge vector.
        sigma: Progress σ.
        v_smooth: Smoothed velocity.
        d: Helix distance.
        focus_checkpoint: Active checkpoint.
        n_breadcrumbs: Breadcrumbs planned this session.
        session_number: 1-indexed session counter.

    Returns:
        Formatted markdown string.
    """
    cp = CHECKPOINTS[focus_checkpoint]
    return (
        f"## {learner_name} — HCTP Session {session_number}\n\n"
        f"**Vector:** [{K[0]:.3f}, {K[1]:.3f}, {K[2]:.3f}]  \n"
        f"**σ:** {sigma:.3f} | **v:** {v_smooth:.4f} | "
        f"**d:** {d:.4f}  \n"
        f"**Focus:** {focus_checkpoint} — {cp['name']}  \n"
        f"**Breadcrumbs this session:** {n_breadcrumbs}\n"
    )
