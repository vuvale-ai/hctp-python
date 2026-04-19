# HCTP 1.0 → 2.0 Migration Guide

> This guide explains how to upgrade an existing HCTP 1.0 integration to use
> the new v2.0 features.  **All v1.0 code continues to work unchanged** — you
> can adopt v2.0 features incrementally, one module at a time.

---

## Is migration required?

No. HCTP 2.0 is fully backward compatible.

If you have existing code using `LearnerSession`, `update_vector`,
`breadcrumb_prompt`, or any `hctp.core` function, it will continue to work
exactly as before.

Migration is worthwhile when you want:
- More accurate quality scoring (EQS vs. text heuristics)
- Role-specific difficulty tuning
- VCAPO training data export
- Mastery Confidence Score for spillover modulation

---

## Step 1 — Upgrade the package

```bash
pip install --upgrade hctp
```

Verify:

```python
import hctp
print(hctp.__version__)  # "2.0.0"
```

---

## Step 2 — Replace `update_vector` with `update_vector_v2` (optional)

### Before (v1.0)

```python
from hctp import update_vector

K_new, delta = update_vector(
    K=[0.5, 0.4, 0.3],
    karpathy_response="...long response with error, fix, refactor...",
    focus_checkpoint="B",
)
```

The v1.0 function scans the response text for marker words (`"error"`, `"fix"`,
`"refactor"`, `"lesson"`, etc.) and applies heuristic bonuses. This is fragile —
a verbose but shallow response can score as high as a rigorous one.

### After (v2.0)

```python
from hctp.scoring import EQSComponents, mastery_confidence_score
from hctp.hctp_vector import update_vector_v2

# 1. Collect EQS inputs from your execution environment
eqs = EQSComponents(
    sandbox_pass=True,          # did the code run?
    test_pass_rate=0.80,        # fraction of pytest tests passing
    grok_score=0.75,            # Grok's holistic score [0, 1]
    claude_score=0.70,          # Claude's holistic score [0, 1]
    karpathy_depth=5,           # how many of 6 K-loop steps were completed
).compute()

# 2. Compute Mastery Confidence Score from session history
mcs = mastery_confidence_score(
    eqs_history=[0.60, 0.65, 0.72],   # last few EQS scores
    sigma_history=[0.31, 0.46, 0.61], # last few σ values
    K=[0.72, 0.60, 0.55],
)

# 3. Update the knowledge vector
K_new, delta = update_vector_v2(
    K=[0.72, 0.60, 0.55],
    eqs_score=eqs.score,
    focus_checkpoint="B",
    mcs=mcs,
)
```

---

## Step 3 — Replace velocity-only difficulty with the full engine (optional)

### Before (v1.0)

```python
from hctp.core import num_breadcrumbs, determine_focus

n = num_breadcrumbs(velocity)       # velocity only
focus = determine_focus(K)
```

### After (v2.0)

```python
from hctp.difficulty_engine import compute_difficulty

profile = compute_difficulty(
    K=K,
    velocity=velocity,
    role="security",   # or "data_scientist", "devops", "generic"
)

n     = profile.breadcrumb_count    # role-adjusted count
focus = profile.focus_checkpoint    # same as determine_focus(K)
level = profile.level               # "foundation" / "standard" / "advanced" / "expert"
hints = profile.hint_density        # fraction of breadcrumbs that include hints

print(profile.summary())
```

---

## Step 4 — Export training trajectories to VCAPO (optional)

This step is only relevant if you are training a model on HCTP data.

```python
from hctp.vcapo_integration import TrajectoryRecord, VCAPOExporter

# Build one TrajectoryRecord per learner
ren_traj = TrajectoryRecord(
    learner_id="Ren",
    sessions=[s.to_dict() for s in ren_session_results],
    eqs_scores=ren_eqs_list,             # one float per session
    sigma_history=ren.sigma_history,
    K_history=ren_K_history,             # one [k_A, k_B, k_C] per session
    role="generic",
    badge_earned=ren.badge,
)

exporter = VCAPOExporter()
exporter.add(ren_traj)
exporter.add(ner_traj)

# Export — skip trajectories with weight below 0.30
n = exporter.export("training_data.jsonl", min_weight=0.30)
print(f"Exported {n} trajectories")
print(exporter.weight_summary())
```

---

## Step 5 — Migrate existing saved state

Existing JSON state files produced by `LearnerSession.save()` are fully
compatible with v2.0.  No changes to the persistence format.

```python
# Load a v1.0 state file — works unchanged
ren = LearnerSession.load("ren_state.json")
```

If you want to attach EQS history to the saved state, add it as a custom field
in your application layer; the `LearnerSession` JSON schema is not versioned.

---

## Breaking changes

There are **no breaking changes** in HCTP 2.0.

All v1.0 symbols (`update_vector`, `LearnerSession`, `breadcrumb_prompt`, etc.)
are exported from `hctp` exactly as before.

The only change visible at the package level is `__version__` which changes
from `"0.1.2"` to `"2.0.0"`.

---

## Quick reference: v1.0 vs v2.0 imports

| Task | v1.0 | v2.0 |
|---|---|---|
| Update K-vector | `from hctp import update_vector` | `from hctp.hctp_vector import update_vector_v2` |
| Quality score | _(heuristic, internal)_ | `from hctp.scoring import EQSComponents` |
| Mastery confidence | _(not available)_ | `from hctp.scoring import mastery_confidence_score` |
| Difficulty profile | `num_breadcrumbs` + `determine_focus` | `from hctp.difficulty_engine import compute_difficulty` |
| Training export | _(not available)_ | `from hctp.vcapo_integration import VCAPOExporter` |

---

## Frequently asked questions

**Q: Do I have to stop using `update_vector`?**
No. The v1.0 heuristic function is still exported and works identically.
Use `update_vector_v2` when you have an execution environment that can provide
EQS inputs.

**Q: What if I don't have a sandbox or test runner?**
Set `sandbox_pass=False` and `test_pass_rate=0.0` in `EQSComponents`. The
score will be lower but still meaningful via the teacher and K-loop signals.

**Q: What if I only have one teacher (e.g. Claude only, no Grok)?**
Set both `grok_score` and `claude_score` to the same value. The harmonic mean
equals the arithmetic mean when both values are identical.

**Q: What if I don't know the Karpathy depth?**
Set `karpathy_depth=0`. This means 15 % of the EQS weight scores 0, which
lowers the score appropriately — it's an honest signal.

**Q: Can I use compute_difficulty without a role?**
Yes — `role="generic"` is the default. Role adjustments are purely additive.
