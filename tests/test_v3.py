"""
Tests for HCTP 3.0 Phase-1 modules: tunnel, manifold, looking_glass,
breadcrumbs, session.
"""

import json

import pytest

from hctp.v3 import (
    DEFAULT_TUNNEL_SYSTEM,
    DRIFT_THRESHOLD,
    LookingGlassEvaluator,
    TunnelLearnerSession,
    TunnelManifold,
    TunnelState,
    Hypothesis,
    compute_drift,
    exploration_breadcrumb_prompt,
    re_centering_breadcrumb_prompt,
)


# ── Tunnel data model ──────────────────────────────────────────────────────────

class TestTunnelState:
    def test_default_state_fields(self):
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1")
        assert s.position == 0.0
        assert s.K_local == [0.0, 0.0, 0.0]
        assert s.compilation_level == 0
        assert s.hypothesis_history == []

    def test_roundtrip_dict(self):
        s = TunnelState(macro_tunnel="M2", sub_tunnel="A1",
                        K_local=[0.2, 0.3, 0.4])
        s.hypothesis_history.append(
            Hypothesis(concept="descriptors", rationale="r",
                       confidence=0.7, macro_tunnel="M2", sub_tunnel="A1")
        )
        r = TunnelState.from_dict(s.to_dict())
        assert r.macro_tunnel == "M2"
        assert r.K_local == [0.2, 0.3, 0.4]
        assert len(r.hypothesis_history) == 1
        assert r.hypothesis_history[0].concept == "descriptors"


# ── compute_drift (Euclidean, balanced-point version) ─────────────────────────

class TestComputeDrift:
    def test_empty_is_zero(self):
        assert compute_drift([]) == 0.0

    def test_all_zero_is_zero(self):
        # sigma = 0 → max_dev = 0 → drift guarded to 0
        assert compute_drift([0.0, 0.0, 0.0]) == 0.0

    def test_perfectly_balanced_is_zero(self):
        assert compute_drift([0.5, 0.5, 0.5]) == 0.0
        assert compute_drift([0.85, 0.85, 0.85]) == 0.0

    def test_max_imbalance_saturates(self):
        # One axis absorbs all progress → drift clips to 1.0
        assert compute_drift([1.0, 0.0, 0.0]) == 1.0

    def test_mild_lean_is_graduated(self):
        # K=[0.9,0.85,0.8] is a gentle staircase; Euclidean drift ≈ 0.235
        # at width=0.25 — far from the tunnel wall, but no longer pinned to 0
        # the way the old max−min metric handled any spread identically.
        assert compute_drift([0.9, 0.85, 0.8]) == pytest.approx(0.2353, abs=1e-3)

    def test_width_scales_drift_inversely(self):
        K = [0.7, 0.5, 0.3]
        tight = compute_drift(K, width=0.10)
        loose = compute_drift(K, width=0.50)
        assert tight >= loose
        assert loose <= 1.0

    def test_drift_is_clipped(self):
        assert 0.0 <= compute_drift([1.0, 0.0, 0.0]) <= 1.0
        assert 0.0 <= compute_drift([0.9, 0.85, 0.8], width=0.01) <= 1.0


# ── Manifold ───────────────────────────────────────────────────────────────────

class TestManifold:
    def test_default_has_four_macros(self):
        assert len(DEFAULT_TUNNEL_SYSTEM) == 4

    def test_macro_count(self):
        m = TunnelManifold()
        assert m.macro_count() == 4

    def test_unknown_macro_raises(self):
        m = TunnelManifold()
        with pytest.raises(KeyError):
            m.macro_tunnel("ZZ")

    def test_sub_indices(self):
        m = TunnelManifold()
        macro_idx, sub_idx = m.sub_tunnel_indices("M2", "A1")
        assert macro_idx == 1
        assert sub_idx == 0

    def test_global_sigma_monotonic(self):
        m = TunnelManifold()
        s_first = TunnelState(macro_tunnel="M1", sub_tunnel="F1")
        s_last = TunnelState(macro_tunnel="M4", sub_tunnel="E4",
                             K_local=[1.0, 1.0, 1.0], position=1.0)
        assert m.global_sigma(s_last) > m.global_sigma(s_first)

    def test_on_centerline_low_drift(self):
        m = TunnelManifold()
        # Perfectly balanced K → on the helix axis at σ=mean(K) → small drift
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[0.5, 0.5, 0.5])
        assert m.drift_score(s) < 0.5

    def test_drift_high_when_imbalanced(self):
        m = TunnelManifold()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[1.0, 0.0, 0.0])
        assert m.drift_score(s) > 0.5

    def test_metrics_inside_flag(self):
        m = TunnelManifold()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[0.5, 0.5, 0.5])
        metrics = m.measure(s)
        assert metrics.inside_tunnel is True
        assert 0.0 <= metrics.alignment <= 1.0

    def test_next_sub_tunnel_within_macro(self):
        m = TunnelManifold()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1")
        assert m.next_sub_tunnel(s) == ("M1", "F2")

    def test_next_sub_tunnel_crosses_macro(self):
        m = TunnelManifold()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F4")
        assert m.next_sub_tunnel(s) == ("M2", "A1")

    def test_next_sub_tunnel_end(self):
        m = TunnelManifold()
        s = TunnelState(macro_tunnel="M4", sub_tunnel="E4")
        assert m.next_sub_tunnel(s) is None


# ── Looking Glass ──────────────────────────────────────────────────────────────

class TestLookingGlass:
    def test_recommend_re_center_on_drift(self):
        ev = LookingGlassEvaluator()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[1.0, 0.0, 0.0])  # high drift
        rep = ev.evaluate(s)
        assert rep.recommended_action == "re_center"
        assert rep.metrics.inside_tunnel is False

    def test_recommend_advance_on_mastery(self):
        ev = LookingGlassEvaluator()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[0.85, 0.85, 0.85])
        rep = ev.evaluate(s)
        assert rep.recommended_action in ("advance", "compile")
        assert rep.next_focus is not None

    def test_recommend_continue_default(self):
        ev = LookingGlassEvaluator()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[0.4, 0.4, 0.4])
        rep = ev.evaluate(s)
        assert rep.recommended_action == "continue"

    def test_stall_detection_triggers_revisit(self):
        ev = LookingGlassEvaluator()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                        K_local=[0.4, 0.4, 0.4])
        rep = ev.evaluate(s, eqs_history=[0.20, 0.10, 0.30],
                         sigma_history=[0.0, 0.05, 0.06])
        assert rep.recommended_action in ("revisit", "continue", "re_center")
        assert any("stall" in n.lower() for n in rep.notes)

    def test_summary_is_string(self):
        ev = LookingGlassEvaluator()
        s = TunnelState(macro_tunnel="M1", sub_tunnel="F1")
        assert isinstance(ev.evaluate(s).summary(), str)


# ── Breadcrumb prompts ─────────────────────────────────────────────────────────

class TestBreadcrumbs:
    def _fixtures(self):
        m = TunnelManifold()
        s = TunnelState(macro_tunnel="M2", sub_tunnel="A1",
                        K_local=[0.3, 0.2, 0.1])
        metrics = m.measure(s)
        macro = m.macro_tunnel("M2")
        sub = macro.sub_tunnel("A1")
        return m, s, macro, sub, metrics

    def test_exploration_prompt_mentions_sub_tunnel(self):
        _, s, macro, sub, metrics = self._fixtures()
        prompt = exploration_breadcrumb_prompt("Ren", s, macro, sub, metrics,
                                               1, velocity=0.10)
        assert sub.name in prompt
        assert "Exploration" in prompt
        assert "Socratic" in prompt

    def test_recenter_prompt_forbids_tangents(self):
        _, s, macro, sub, metrics = self._fixtures()
        prompt = re_centering_breadcrumb_prompt("Ren", s, macro, sub, metrics, 1)
        assert "re-centering" in prompt.lower()
        assert sub.name in prompt
        assert "forbid" in prompt.lower() or "forbids" in prompt.lower() \
               or "invariant" in prompt.lower()


# ── TunnelLearnerSession ───────────────────────────────────────────────────────

class TestTunnelLearnerSession:
    def test_start_session_returns_expected_keys(self):
        sess = TunnelLearnerSession("Ren")
        data = sess.start_session()
        for key in ("header", "breadcrumb_prompts", "n_breadcrumbs",
                    "mode", "macro_tunnel", "sub_tunnel"):
            assert key in data
        assert data["mode"] == "exploration"

    def test_cannot_double_start(self):
        sess = TunnelLearnerSession("Ren")
        sess.start_session()
        with pytest.raises(RuntimeError):
            sess.start_session()

    def test_submit_requires_active_session(self):
        sess = TunnelLearnerSession("Ren")
        with pytest.raises(RuntimeError):
            sess.submit_response(0.5)

    def test_finish_without_start_raises(self):
        sess = TunnelLearnerSession("Ren")
        with pytest.raises(RuntimeError):
            sess.finish_session()

    def test_full_exploration_cycle(self):
        sess = TunnelLearnerSession("Ren")
        data = sess.start_session()
        for _ in data["breadcrumb_prompts"]:
            sess.submit_response(0.6)
        result = sess.finish_session()
        assert result.breadcrumbs_completed == data["n_breadcrumbs"]
        assert result.mode == "exploration"
        assert 0.0 <= result.eqs_score <= 1.0
        assert 0.0 <= result.mcs <= 1.0

    def test_re_centering_triggered_by_drift(self):
        # Pre-seed an off-helix K to force drift.
        state = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                            K_local=[1.0, 0.0, 0.0])
        sess = TunnelLearnerSession("Ner", state=state)
        data = sess.start_session()
        assert data["mode"] == "re_centering"
        assert data["n_breadcrumbs"] <= 3  # re-centering caps at 3

    def test_auto_advance_on_mastery(self):
        state = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                            K_local=[0.85, 0.85, 0.85])
        sess = TunnelLearnerSession("Ren", state=state, auto_advance=True)
        sess.start_session()
        sess.submit_response(0.9)
        result = sess.finish_session()
        assert result.advanced is True
        # After advance, should have jumped to the next sub-tunnel.
        assert sess.state.sub_tunnel != "F1"

    def test_no_advance_when_disabled(self):
        state = TunnelState(macro_tunnel="M1", sub_tunnel="F1",
                            K_local=[0.85, 0.85, 0.85])
        sess = TunnelLearnerSession("Ren", state=state, auto_advance=False)
        sess.start_session()
        sess.submit_response(0.9)
        result = sess.finish_session()
        assert result.advanced is False
        assert sess.state.sub_tunnel == "F1"

    def test_serialisation_roundtrip(self, tmp_path):
        sess = TunnelLearnerSession("Ren")
        sess.start_session()
        sess.submit_response(0.5)
        sess.finish_session()
        p = tmp_path / "state.json"
        sess.save(p)
        restored = TunnelLearnerSession.load(p)
        assert restored.name == "Ren"
        assert restored.state.macro_tunnel == sess.state.macro_tunnel
        assert restored.eqs_history == sess.eqs_history

    def test_to_v2_session_bridge(self):
        sess = TunnelLearnerSession("Ren")
        sess.start_session()
        sess.submit_response(0.5)
        sess.finish_session()
        v2 = sess.to_v2_session()
        assert v2.name == "Ren"
        assert len(v2.K) == 3


# ── Backward-compatibility smoke test ──────────────────────────────────────────

class TestBackwardCompat:
    def test_v2_api_still_works(self):
        from hctp import LearnerSession
        ren = LearnerSession("Ren")
        assert ren.sigma == 0.0

    def test_v3_entry_point(self):
        from hctp.v3 import TunnelLearnerSession as T
        assert T.__name__ == "TunnelLearnerSession"
