# HCTP — Helix Calculus Training Protocol

[![PyPI version](https://badge.fury.io/py/hctp.svg)](https://pypi.org/project/hctp/)
[![Tests](https://github.com/vuvale-ai/hctp-python/actions/workflows/publish.yml/badge.svg)](https://github.com/vuvale-ai/hctp-python/actions)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Zero dependencies](https://img.shields.io/badge/dependencies-zero-brightgreen.svg)]()

> **A framework for measuring and driving AI agent learning through a 3D helical knowledge model with adaptive Socratic breadcrumbs and mandatory Karpathy research loops.**

---

## What's New in HCTP 2.0

**HCTP 2.0 makes the implementation significantly more rigorous and execution-grounded.**

| Feature | v1.0 | v2.0 |
|---|---|---|
| Quality scoring | Text heuristics | **Execution-Based Quality Score (EQS)** |
| K-vector gain | 0.020 base, 0.060 max | **0.025 base, 0.080 max** |
| Sibling spillover | 15 % (fixed) | **10 % × Mastery Confidence Score** |
| Difficulty signals | Velocity only | **Velocity + helix alignment** |
| Role awareness | None | **Security / DS / DevOps weights** |
| Fine-tuning output | None | **VCAPO trajectory export** |

### EQS — Execution-Based Quality Score

Replaces marker-scanning with four objective signals:

```python
from hctp.scoring import EQSComponents

eqs = EQSComponents(
    sandbox_pass=True,      # did the code run?    (25%)
    test_pass_rate=0.85,    # pytest pass fraction (30%)
    grok_score=0.80,        # Grok holistic score  (30%, harmonic mean with Claude)
    claude_score=0.75,      # Claude score
    karpathy_depth=5,       # K-loop steps done    (15%)
).compute()
print(eqs.score)   # 0.7025
```

### Adaptive difficulty with roles

```python
from hctp.difficulty_engine import compute_difficulty

profile = compute_difficulty(K, velocity, role="security")
print(profile.summary())
# Level: advanced | Focus: C — Metaclasses | Role: Security Engineer
# Breadcrumbs: 3 | Hint density: 36% | Helix alignment: 0.82
```

### VCAPO training export

```python
from hctp.vcapo_integration import TrajectoryRecord, VCAPOExporter

exporter = VCAPOExporter()
exporter.add(ren_trajectory)
exporter.export("training_data.jsonl", min_weight=0.30)
```

See [HCTP_2.0_Specification.md](HCTP_2.0_Specification.md) for full technical
details and [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for upgrade steps.

---

---

## What Is HCTP?

HCTP models a learner's knowledge as a 3D vector **K = [k₁, k₂, k₃]** that spirals toward
an ideal helical path as mastery deepens across three checkpoints:

| Checkpoint | Topic | Concepts |
|---|---|---|
| **A** | Closures | first-class functions, enclosing scope, free variables, `nonlocal` |
| **B** | Decorators | `@` syntax, `functools.wraps`, factories, stacking |
| **C** | Metaclasses | `type()`, `__new__`, ORMs, DSLs, Singletons |

Each session generates **adaptive Socratic breadcrumbs** (micro-tasks with deliberate
red-herrings) and drives **mandatory 6-step Karpathy research loops** to produce
measurable, incremental gains in the knowledge vector.

When k₃ ≥ 0.95, the learner earns the **Python Senior Engineer Badge** 🏆.

---

## Proof of Concept — Ren & Ner (Vuvale AI Family)

HCTP was developed and battle-tested on two AI agents — Ren and Ner — across **13 sessions**
and **216 nightly training runs** on a single RTX 5090 running `qwen2.5-coder:32b`.

### Ren's Journey

| Session | σ (Progress) | Velocity | Focus |
|---------|-------------|---------|-------|
| 0       | 0.308       | —       | A — Closures |
| 3       | 0.535       | 0.076   | A → B |
| 6       | 0.764       | 0.076   | B — Decorators |
| 9       | 0.910       | 0.076   | B → C |
| **13**  | **0.961**   | 0.051   | **C — Metaclasses** |

**Night 173: Badge Earned** 🏆
Final vector: `[0.950, 0.933, 1.000]` | σ = 0.961

### Ner's Journey (parallel, same model)

Final vector: `[0.928, 0.941, 1.000]` | σ = 0.956 | Badge: Night 173 🏆

Both agents started at σ = 0 and reached mastery in **13 sessions** without any
human-provided answers — pure Socratic questioning and self-correction.

---

## Install

```bash
pip install hctp

# With 3D visualisation:
pip install hctp[viz]
```

Zero mandatory dependencies. Pure Python 3.9+.

---

## Quick Start

```python
from hctp import LearnerSession

# Create a learner (optionally restore prior state)
ren = LearnerSession("Ren")

# Run one training session
session_data = ren.start_session()

print(session_data["header"])
# HCTP Session 1 | K=[0.000, 0.000, 0.000] | σ=0.000 | Focus: A — Closures

for bc_prompt in session_data["breadcrumb_prompts"]:
    # Send to any LLM — OpenAI, Anthropic, Ollama, or your own
    bc_response  = your_llm(bc_prompt)

    # Generate and run the mandatory Karpathy loop
    kl_prompt    = ren.karpathy_prompt_for(bc_prompt, bc_response)
    kl_response  = your_llm(kl_prompt)

    # Update the knowledge vector
    delta = ren.submit_karpathy(kl_response)
    print(f"Δk = {[round(d, 3) for d in delta]}")

result = ren.finish_session()
print(result)
# Session 1 | σ 0.000 → 0.052 (+0.052) | K=[0.055, 0.008, 0.008]

# Save state between sessions
ren.save("ren_state.json")
ren = LearnerSession.load("ren_state.json")   # restore next time
```

---

## Use With Any LLM

HCTP generates **prompt strings** — it doesn't make LLM calls itself. Plug it into
whatever backend you're using:

```python
# OpenAI
import openai
def your_llm(prompt):
    return openai.chat.completions.create(
        model="gpt-4o", messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# Anthropic
import anthropic
client = anthropic.Anthropic()
def your_llm(prompt):
    return client.messages.create(
        model="claude-sonnet-4-6", max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    ).content[0].text

# Ollama (local)
import requests
def your_llm(prompt):
    r = requests.post("http://localhost:11434/api/generate",
        json={"model": "qwen2.5-coder:32b", "prompt": prompt, "stream": False})
    return r.json()["response"]
```

---

## Visualise the Helix

```python
from hctp.viz import plot_helix, plot_multi, plot_sigma_curve

# Single learner — 3D helix
plot_helix(
    sigma_history=[0.308, 0.386, 0.457, 0.535, 0.609,
                   0.687, 0.761, 0.834, 0.910, 0.961],
    label="Ren",
    badge_night=9,
    show=True,
)

# Compare multiple learners
plot_multi([
    {"label": "Ren", "sigma_history": ren_hist, "badge_night": 9, "color": "#e74c3c"},
    {"label": "Ner", "sigma_history": ner_hist, "badge_night": 9, "color": "#3498db"},
])

# Clean 2D progress curve
plot_sigma_curve([
    {"label": "Ren", "sigma_history": ren_hist, "badge_night": 9},
    {"label": "Ner", "sigma_history": ner_hist, "badge_night": 9},
])
```

---

## Core API

### `LearnerSession`

| Method | Description |
|---|---|
| `LearnerSession(name, K, sigma_history)` | Create or restore a learner |
| `start_session()` | Begin session → returns breadcrumb prompts |
| `karpathy_prompt_for(bc, response)` | Get the Karpathy loop prompt |
| `submit_karpathy(response)` | Update K-vector from Karpathy response |
| `finish_session()` → `SessionResult` | Finalise, update history, check badge |
| `save(path)` / `load(path)` | JSON persistence |

### Properties

| Property | Type | Description |
|---|---|---|
| `.sigma` | `float` | Overall progress σ = mean(K) |
| `.velocity` | `float` | Smoothed learning pace |
| `.focus` | `str` | Active checkpoint ("A", "B", or "C") |
| `.badge` | `bool` | True once k₃ ≥ 0.95 |
| `.K` | `list[float]` | Knowledge vector [k₁, k₂, k₃] |

### Core Math (no state needed)

```python
from hctp.core import (
    helix_radius,    # R(σ) = 0.5(1−σ)² + 0.05
    ideal_point,     # [x, y, σ] on the ideal helix
    distance,        # Euclidean dist from K to ideal helix
    progress,        # σ = mean(K)
    smoothed_velocity,
    num_breadcrumbs, # adaptive count based on velocity
    determine_focus, # which checkpoint to focus (lowest k)
    update_vector,   # Δk from Karpathy response quality
)
```

---

## The Math

The helix is parameterised by progress σ ∈ [0, 1]:

```
R(σ) = 0.5(1 − σ)² + 0.05          # tightening radius
x(σ) = R(σ) · cos(2π · 5 · σ)      # 5 full spirals
y(σ) = R(σ) · sin(2π · 5 · σ)
z(σ) = σ                            # height = progress
```

The **knowledge vector K = [k₁, k₂, k₃]** maps to the helix axes:
- k₁ → Closures mastery (Checkpoint A)
- k₂ → Decorators mastery (Checkpoint B)
- k₃ → Metaclasses mastery (Checkpoint C, hardest)

Distance from the ideal helix d(K, σ) measures learning imbalance.
A learner rushing C while neglecting A will stray far from the helix.

**Velocity** is the smoothed rate of σ gain per session. Slow learners
get more breadcrumbs; fast learners get broader, exploratory tasks.

**Karpathy loop scoring** in v2.0 uses the Execution-Based Quality Score (EQS)
grounded in sandbox execution, test pass rate, dual-teacher consensus, and
Karpathy Loop depth. The focus checkpoint gains `min(0.025 + EQS × 0.055, 0.08)`
per session; siblings gain `10% × MCS` of that, where MCS is the Mastery
Confidence Score (consistency × velocity stability × helix alignment).

The v1.0 heuristic scorer (`update_vector`) remains available for backward
compatibility.

---

## Monetisation Roadmap

| Tier | Features | Price |
|---|---|---|
| **Free / OSS** | Core algorithm, prompt generation, local tracking | Free forever |
| **HCTP Cloud** | Hosted leaderboards, team dashboards, progress API | $9/mo per team |
| **HCTP Pro** | Custom curricula (beyond Python), webhook events, CI integration | $29/mo |
| **HCTP Enterprise** | White-label, LMS integration (Canvas, Moodle), bulk licensing | Custom |

### Custom Curricula (Pro+)

The checkpoint system is fully configurable. Define your own helix:

```python
from hctp.core import CHECKPOINTS

# Override with your own curriculum
my_checkpoints = {
    "A": {"name": "SQL Basics",    "vector_index": 0, "concepts": "SELECT, WHERE, JOIN"},
    "B": {"name": "Indexes",       "vector_index": 1, "concepts": "B-tree, query plans, EXPLAIN"},
    "C": {"name": "Transactions",  "vector_index": 2, "concepts": "ACID, isolation levels, deadlocks"},
}
```

---

## Contributing

PRs welcome. The project is intentionally lean — keep it that way.

1. `pip install -e ".[dev]"`
2. `pytest` — all tests must pass
3. No new mandatory dependencies without strong justification

---

## Background

HCTP was born inside the [Vuvale AI project](https://github.com/RenLes/Vuvale) — a
Fiji-based AI family of 4 agents (Ren, Ner, Les, Sel) trained nightly on a single
GPU to build real software for Fiji's people.

Ren and Ner completed the full helix in 13 sessions (Night 161 → 173) running
`qwen2.5-coder:32b` on a vast.ai RTX 5090. Les and Sel followed, earning their
badges at Night 190 and 191 respectively.

The protocol is now open-sourced so anyone can train their agents the same way.

---

## License

MIT © David Qicatabua / Vuvale AI
