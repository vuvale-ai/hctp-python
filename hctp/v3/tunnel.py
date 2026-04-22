"""
hctp.v3.tunnel
~~~~~~~~~~~~~~
Data model for the Helical Tunnel Mastery Protocol.

HCTP 3.0 replaces the flat 3-checkpoint line from v2 with a rich,
hierarchical tunnel manifold:

    Macro-Tunnel (5–7 stages, long axis)
      └── Sub-Tunnel (3–5 per macro, skill tree branch)
            └── local K-vector (re-uses v2 checkpoint math)

Each macro-tunnel is a segment along the main helical axis σ ∈ [0, 1].
Inside each macro-tunnel the learner navigates its sub-tunnels. The
helical *centerline* is preserved from v1/v2 — what changes is that the
tunnel has **width** (productive deviation allowed) and **depth** (how
far into the sub-tunnel's expertise frontier the learner has gone).

Phase 1 ships with a pragmatic default of 4 Macro-Tunnels for advanced
Python. The structure is entirely data-driven, so you can swap in your
own curriculum without touching the library.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

__all__ = [
    "Hypothesis",
    "SubTunnel",
    "MacroTunnel",
    "TunnelState",
    "DEFAULT_TUNNEL_SYSTEM",
    "DEFAULT_WIDTH",
    "DRIFT_THRESHOLD",
]


# ── Default tunnel geometry knobs ──────────────────────────────────────────────

DEFAULT_WIDTH   = 0.25   # productive deviation radius around the centerline
DRIFT_THRESHOLD = 0.35   # drift_score above this triggers re-centering


# ── Dataclasses ────────────────────────────────────────────────────────────────

@dataclass
class Hypothesis:
    """A learner-generated prediction about the next powerful concept.

    Produced by the (Phase 2) Anticipatory Hypothesis Engine. Phase 1
    stores them as inert records so upstream code can already build
    the data path. When the curriculum reaches the predicted area the
    engine marks ``tested=True`` and records ``accuracy``.
    """

    concept: str
    rationale: str
    confidence: float                 # learner's subjective confidence in [0, 1]
    macro_tunnel: str
    sub_tunnel: str | None = None
    tested: bool = False
    accuracy: float | None = None     # teacher-judged accuracy once tested

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SubTunnel:
    """One branch of a macro-tunnel's skill tree.

    Args:
        key:       Short identifier unique inside its macro-tunnel.
        name:      Human-readable name.
        concepts:  Free-text summary of the concepts covered.
    """

    key: str
    name: str
    concepts: str


@dataclass(frozen=True)
class MacroTunnel:
    """A long segment of the main helical axis.

    Each macro-tunnel owns an ordered list of ``SubTunnel``s.

    Args:
        key:          Short identifier unique across the tunnel system.
        name:         Human-readable name (e.g. "Core Abstractions").
        description:  One-line motivation.
        sub_tunnels:  Ordered sub-tunnels; learners traverse these top-to-bottom.
    """

    key: str
    name: str
    description: str
    sub_tunnels: tuple[SubTunnel, ...]

    def sub_tunnel(self, key: str) -> SubTunnel:
        for st in self.sub_tunnels:
            if st.key == key:
                return st
        raise KeyError(f"sub_tunnel {key!r} not in macro-tunnel {self.key!r}")


@dataclass
class TunnelState:
    """Full tunnel-aware state of a single learner.

    Unlike v2's flat K-vector, the v3 state is multi-layered: the learner
    is positioned inside a specific sub-tunnel of a macro-tunnel, with
    a local K-vector capturing fine-grained mastery at that level.

    Args:
        macro_tunnel:       Current macro-tunnel key.
        sub_tunnel:         Current sub-tunnel key.
        position:           Local progress σ ∈ [0, 1] within the current sub-tunnel.
        depth:              Depth score ∈ [0, 1] — expertise within the sub-tunnel.
        width:              Permitted productive deviation radius.
        drift_score:        How far the learner is from the productive volume.
        compilation_level:  Count of sub-tunnels "compiled" into reusable primitives.
        K_local:            Fine-grained 3-dim mastery vector (reuses v2 math).
        hypothesis_history: Anticipatory predictions (Phase 2 populates this).
    """

    macro_tunnel: str
    sub_tunnel: str
    position: float = 0.0
    depth: float = 0.0
    width: float = DEFAULT_WIDTH
    drift_score: float = 0.0
    compilation_level: int = 0
    K_local: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    hypothesis_history: list[Hypothesis] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["hypothesis_history"] = [h.to_dict() for h in self.hypothesis_history]
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TunnelState":
        data = dict(data)
        data["hypothesis_history"] = [
            Hypothesis(**h) for h in data.get("hypothesis_history", [])
        ]
        return cls(**data)


# ── Default Phase-1 Python curriculum (4 Macro-Tunnels) ────────────────────────

DEFAULT_TUNNEL_SYSTEM: tuple[MacroTunnel, ...] = (
    MacroTunnel(
        key="M1",
        name="Foundations",
        description="Variables, control flow, functions, OOP basics, the stdlib.",
        sub_tunnels=(
            SubTunnel("F1", "Primitives & collections",
                      "numbers, strings, lists, dicts, sets, tuples, slicing"),
            SubTunnel("F2", "Functions & scope",
                      "def, return, arguments, *args/**kwargs, LEGB, closures intro"),
            SubTunnel("F3", "OOP basics",
                      "class, __init__, inheritance, dunder methods, properties"),
            SubTunnel("F4", "Standard library tour",
                      "itertools, functools, collections, pathlib, datetime"),
        ),
    ),
    MacroTunnel(
        key="M2",
        name="Core Abstractions",
        description="Closures, decorators, metaclasses, iterators, generators, "
                    "context managers — the spine of expressive Python.",
        sub_tunnels=(
            SubTunnel("A1", "Closures & first-class functions",
                      "enclosing scope, free variables, nonlocal, late binding"),
            SubTunnel("A2", "Decorators",
                      "@ syntax, functools.wraps, factories, stacking, class decorators"),
            SubTunnel("A3", "Iterators & generators",
                      "__iter__/__next__, yield, generator expressions, send/throw"),
            SubTunnel("A4", "Context managers",
                      "__enter__/__exit__, contextlib, nested/reusable managers"),
            SubTunnel("A5", "Metaclasses",
                      "type(), __new__, __init_subclass__, ORMs, DSLs, singletons"),
        ),
    ),
    MacroTunnel(
        key="M3",
        name="Systems & Composition",
        description="Async, concurrency, typing, packaging, testing — "
                    "engineering Python at the system scale.",
        sub_tunnels=(
            SubTunnel("S1", "Async & await",
                      "event loop, coroutines, asyncio, cancellation, back-pressure"),
            SubTunnel("S2", "Concurrency",
                      "threads, processes, GIL, concurrent.futures, synchronization"),
            SubTunnel("S3", "Static typing",
                      "typing module, generics, Protocols, TypedDict, mypy"),
            SubTunnel("S4", "Packaging & distribution",
                      "pyproject.toml, entry points, wheels, pypi, lockfiles"),
            SubTunnel("S5", "Testing & design-by-contract",
                      "pytest, fixtures, property-based, hypothesis, coverage"),
        ),
    ),
    MacroTunnel(
        key="M4",
        name="Meta-Engineering",
        description="Performance, profiling, C extensions, dynamic dispatch, DSLs.",
        sub_tunnels=(
            SubTunnel("E1", "Profiling & performance",
                      "cProfile, py-spy, memory_profiler, timeit, micro-benchmarks"),
            SubTunnel("E2", "C interop & extensions",
                      "ctypes, cffi, Cython, PyO3, the CPython ABI"),
            SubTunnel("E3", "Memory model & GC",
                      "refcounting, cycles, __slots__, weakref, generations"),
            SubTunnel("E4", "DSLs & code generation",
                      "ast, bytecode, descriptors, metaclass-driven DSLs"),
        ),
    ),
)
