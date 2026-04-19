# HCTP 2.0 — Technical Specification

> **Helix Calculus Training Protocol, Version 2.0**
> David Qicatabua / Vuvale AI · April 2026

---

## 1. Overview

HCTP 2.0 is a rigorous upgrade to the original Helix Calculus Training Protocol. The
core philosophy is unchanged: model a learner's knowledge as a 3D vector **K = [k_A,
k_B, k_C]** spiralling toward an ideal helical path as mastery deepens. What changes
in 2.0 is how progress is *measured* and how difficulty is *adapted*.

### What changed and why

| Area | v1.0 | v2.0 |
|---|---|---|
| Quality scoring | Text heuristics (marker scanning) | Execution-Based Quality Score (EQS) |
| K-vector base gain | 0.020 | 0.025 |
| K-vector max gain | 0.060 (soft) | 0.080 (hard cap) |
| Sibling spillover | 15 % (fixed) | 10 % × MCS (confidence-modulated) |
| Difficulty signal | Velocity only | Velocity + helix alignment |
| Role awareness | None | Security / DS / DevOps weights |
| Fine-tuning output | None | VCAPO trajectory export |

The fundamental problem with v1.0 quality scoring was that a learner could earn
maximum gain simply by writing a long response containing certain keywords. This
allowed low-quality but verbose outputs to inflate the knowledge vector.

v2.0 replaces this with EQS — a score grounded in **what the code actually does**,
not what words surround it.

---

## 2. The Helix Geometry (unchanged from v1.0)

The ideal helix is parameterised by progress σ ∈ [0, 1]:

```
R(σ) = 0.5 × (1 − σ)² + 0.05       # radius tightens as mastery converges
x(σ) = R(σ) × cos(2π × 5 × σ)      # 5 full spirals
y(σ) = R(σ) × sin(2π × 5 × σ)
z(σ) = σ                             # height equals progress
```

The **knowledge vector K = [k_A, k_B, k_C]** maps to 3D space. σ = mean(K).
Distance from the ideal helix measures imbalance — a learner rushing Checkpoint C
while neglecting A will drift far from the helix and score lower on alignment.

---

## 3. Execution-Based Quality Score (EQS)

### 3.1 Components

| Component | Symbol | Weight | Source |
|---|---|---|---|
| Sandbox execution | E | 0.25 | Did code run without errors? (bool) |
| Test pass rate | T | 0.30 | Fraction of automated tests passing [0, 1] |
| Dual-Teacher score | D | 0.30 | Harmonic mean of Grok + Claude scores |
| Karpathy Loop depth | L | 0.15 | Steps 1–6 completed [0, 6] |

**Weights sum to 1.0.**

### 3.2 Formula

```
EQS = 0.25 × E + 0.30 × T + 0.30 × D + 0.15 × (L / 6)
```

where the Dual-Teacher score uses the **harmonic mean** of Grok and Claude:

```
D = 2 × G × C / (G + C)    if G + C > 0, else 0
```

The harmonic mean penalises teacher disagreement more than the arithmetic mean.
If Grok gives 0.9 and Claude gives 0.1, the harmonic mean is 0.18 (not 0.50),
preventing a learner from gaming a high score by impressing one teacher while
producing garbage for the other.

### 3.3 Score range

| EQS | Interpretation |
|---|---|
| 0.00 – 0.30 | Poor: code fails, no tests pass, teachers disagree |
| 0.30 – 0.55 | Partial: code runs, some tests pass, one teacher convinced |
| 0.55 – 0.75 | Good: most tests pass, teachers broadly agree |
| 0.75 – 1.00 | Excellent: all pass, both teachers aligned, full K-loop |

### 3.4 Implementation

```python
from hctp.scoring import EQSComponents

eqs = EQSComponents(
    sandbox_pass=True,
    test_pass_rate=0.85,
    grok_score=0.80,
    claude_score=0.75,
    karpathy_depth=5,
).compute()

print(eqs.score)        # 0.7025
print(eqs.summary())    # full breakdown
```

---

## 4. Mastery Confidence Score (MCS)

### 4.1 Purpose

MCS answers: *"How confident are we that this learner has genuinely consolidated
knowledge at their current σ level?"*

A high MCS means: consistent EQS, stable velocity, on-helix growth.
A low MCS means: erratic quality, rushing/stalling, or imbalanced checkpoints.

### 4.2 Formula

```
MCS = 0.40 × consistency + 0.35 × velocity_stability + 0.25 × helix_alignment
```

**Consistency** (EQS variance over last 5 sessions):
```
consistency = max(0, 1 − variance(EQS_history[-5:]) / 0.25)
```

**Velocity stability** (how close to TARGET_VELOCITY = 0.15):
```
deviation        = |avg_velocity − 0.15|
velocity_stability = max(0, 1 − deviation / 0.15)
```

**Helix alignment** (how close K is to the ideal helix):
```
alignment = max(0, 1 − distance(K, σ) / 0.6)
```

### 4.3 How MCS is used

MCS multiplies the sibling spillover in `update_vector_v2`:

```
delta_sibling = delta_focus × SPILLOVER × MCS
```

A learner with MCS = 0.8 propagates 8 % of their focus gain to siblings.
A learner with MCS = 0.3 propagates only 3 %.

This ensures that knowledge does not spread laterally until the learner has
demonstrated stable mastery at their current level.

### 4.4 Implementation

```python
from hctp.scoring import mastery_confidence_score

mcs = mastery_confidence_score(
    eqs_history=[0.62, 0.65, 0.68, 0.70],
    sigma_history=[0.31, 0.46, 0.61, 0.76],
    K=[0.80, 0.74, 0.72],
)
# mcs ≈ 0.71
```

---

## 5. K-Vector Progression (v2.0)

### 5.1 Constants

| Parameter | v1.0 | v2.0 | Rationale |
|---|---|---|---|
| BASE_GAIN | 0.020 | 0.025 | Reward consistent effort more |
| MAX_GAIN | 0.060 (soft) | 0.080 (hard) | Allow excellent sessions to matter more |
| SPILLOVER | 0.150 (fixed) | 0.10 × MCS | Tie lateral spread to genuine mastery |

### 5.2 Update formula

```
delta_focus   = min(BASE_GAIN + EQS × 0.055, MAX_GAIN)
delta_sibling = delta_focus × 0.10 × MCS

K_new[i] = clamp(K[i] + delta[i], 0, 1)
```

EQS range [0, 1] gives focus gain range [0.025, 0.080].

### 5.3 Implementation

```python
from hctp.hctp_vector import update_vector_v2

K_new, delta = update_vector_v2(
    K=[0.72, 0.60, 0.55],
    eqs_score=0.74,
    focus_checkpoint="C",
    mcs=0.68,
)
```

---

## 6. Adaptive Difficulty Engine

### 6.1 Difficulty score

```
D = 0.5 × velocity_factor + 0.5 × helix_alignment

velocity_factor = min(velocity / 0.15, 1.5) / 1.5
helix_alignment = max(0, 1 − distance(K, σ) / 0.6)
```

| D range | Level | Description |
|---|---|---|
| [0.00, 0.25) | foundation | Many hints, more breadcrumbs |
| [0.25, 0.50) | standard | Moderate hints |
| [0.50, 0.75) | advanced | Few hints, broader tasks |
| [0.75, 1.00] | expert | No hints, exploratory tasks |

### 6.2 Role-specific adjustments

Role weights scale the breadcrumb count and inject domain context into prompts.

| Role | Closures | Decorators | Metaclasses | Extra emphasis |
|---|---|---|---|---|
| generic | 1.0× | 1.0× | 1.0× | — |
| security | 1.1× | 1.3× | **1.5×** | Injection attacks, sandboxing |
| data_scientist | 1.0× | 1.2× | 1.1× | Pipelines, memoization, schema |
| devops | 1.1× | **1.4×** | 1.0× | Retry, config injection, plugins |

A Security Engineer at Checkpoint C (Metaclasses) gets 1.5× the standard
breadcrumb count because metaclasses have high-value security implications.

### 6.3 Hint density

```
hint_density = max(0, 1 − D)
```

Learners at foundation level (D ≈ 0.15) get ~85 % of breadcrumbs with hints.
Expert learners (D ≈ 0.85) get ~15 % with hints — the rest are pure challenges.

### 6.4 Implementation

```python
from hctp.difficulty_engine import compute_difficulty

profile = compute_difficulty(
    K=[0.72, 0.60, 0.55],
    velocity=0.18,
    role="security",
)
print(profile.summary())
# Difficulty Profile:
#   Level:           advanced
#   Breadcrumbs:     3
#   Focus:           C — Metaclasses
#   Role:            Security Engineer
#   Helix alignment: 0.82
#   Velocity ratio:  1.20×
#   Hint density:    36%
#   Role emphasis:   injection attacks via closures, monkey-patching vulnerabilities
```

---

## 7. VCAPO Fine-Tuning Integration

### 7.1 Motivation

VCAPO (Value-Calibrated Adaptive Policy Optimization) uses learner trajectories
as supervised fine-tuning signal. Not all trajectories are equal — erratic,
off-helix paths should influence training less than stable, high-EQS paths.

### 7.2 Trajectory weight formula

```
w(τ) = 0.40 × EQS_mean(τ) + 0.35 × alignment_mean(τ) + 0.25 × kloop_completion(τ)
```

All three signals are in [0, 1], so w ∈ [0, 1].

### 7.3 Output format (JSONL)

Each trajectory is exported as a JSONL line:

```json
{
  "learner_id": "Ren",
  "role": "generic",
  "badge_earned": true,
  "weight": 0.643,
  "session_count": 13,
  "eqs_mean": 0.71,
  "sigma_final": 0.961,
  "sessions": [...]
}
```

The `weight` field is consumed by the VCAPO training loop to scale the
cross-entropy loss contribution of each trajectory.

### 7.4 Implementation

```python
from hctp.vcapo_integration import TrajectoryRecord, VCAPOExporter

traj = TrajectoryRecord(
    learner_id="Ren",
    sessions=session_dicts,
    eqs_scores=eqs_list,
    sigma_history=sigma_list,
    K_history=K_list,
    role="generic",
    badge_earned=True,
)

exporter = VCAPOExporter()
exporter.add(traj)
n = exporter.export("training_data.jsonl", min_weight=0.30)
print(exporter.weight_summary())
```

---

## 8. Backward Compatibility

All v1.0 public API is fully preserved:

- `LearnerSession`, `SessionResult` — unchanged
- `update_vector` — unchanged (heuristic scorer, still functional)
- `breadcrumb_prompt`, `karpathy_loop_prompt`, `session_header` — unchanged
- All `hctp.core` functions and constants — unchanged

v2.0 adds *new* symbols alongside old ones. No imports break. See
[MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for how to adopt v2.0 features
incrementally.

---

## 9. Design Invariants

These invariants must hold in any correct implementation:

1. **K values stay in [0, 1]** — clamped after every update.
2. **σ = mean(K)** — never set independently.
3. **EQS ∈ [0, 1]** — weights sum to 1.0 by construction.
4. **MCS ∈ [0, 1]** — three signals weighted to sum to 1.0.
5. **delta_focus ≤ MAX_GAIN** — hard cap enforced in `update_vector_v2`.
6. **weight ∈ [0, 1]** — VCAPO weights clamped by construction.
7. **All v1.0 tests must still pass** — backward compatibility is non-negotiable.
