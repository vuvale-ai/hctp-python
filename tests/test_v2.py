"""
Tests for HCTP 2.0 modules: scoring, hctp_vector, difficulty_engine, vcapo_integration.
"""
import pytest
from hctp.scoring import (
    EQSComponents,
    EQSResult,
    mastery_confidence_score,
    EXEC_WEIGHT,
    TEST_WEIGHT,
    TEACHER_WEIGHT,
    DEPTH_WEIGHT,
    KARPATHY_MAX_DEPTH,
)
from hctp.hctp_vector import update_vector_v2, BASE_GAIN, MAX_GAIN, SPILLOVER
from hctp.difficulty_engine import compute_difficulty, ROLES, DIFFICULTY_LEVELS
from hctp.vcapo_integration import TrajectoryRecord, VCAPOExporter


# ── EQSComponents ──────────────────────────────────────────────────────────────

class TestEQSComponents:
    def test_weights_sum_to_one(self):
        assert abs(EXEC_WEIGHT + TEST_WEIGHT + TEACHER_WEIGHT + DEPTH_WEIGHT - 1.0) < 1e-9

    def test_perfect_score(self):
        eqs = EQSComponents(
            sandbox_pass=True, test_pass_rate=1.0,
            grok_score=1.0, claude_score=1.0,
            karpathy_depth=KARPATHY_MAX_DEPTH,
        ).compute()
        assert eqs.score == pytest.approx(1.0)

    def test_zero_score(self):
        eqs = EQSComponents(
            sandbox_pass=False, test_pass_rate=0.0,
            grok_score=0.0, claude_score=0.0,
            karpathy_depth=0,
        ).compute()
        assert eqs.score == pytest.approx(0.0)

    def test_exec_only(self):
        eqs = EQSComponents(sandbox_pass=True).compute()
        assert eqs.score == pytest.approx(EXEC_WEIGHT)

    def test_harmonic_mean_penalises_disagreement(self):
        # arithmetic mean of 0.9 and 0.1 = 0.50; harmonic = 0.18
        c = EQSComponents(grok_score=0.9, claude_score=0.1)
        assert c.dual_teacher_score == pytest.approx(0.18, abs=1e-3)

    def test_harmonic_mean_both_zero(self):
        c = EQSComponents(grok_score=0.0, claude_score=0.0)
        assert c.dual_teacher_score == 0.0

    def test_partial_depth(self):
        eqs = EQSComponents(karpathy_depth=3).compute()
        assert eqs.depth_component == pytest.approx(0.5)

    def test_validation_test_pass_rate(self):
        with pytest.raises(ValueError):
            EQSComponents(test_pass_rate=1.5)

    def test_validation_grok_score(self):
        with pytest.raises(ValueError):
            EQSComponents(grok_score=-0.1)

    def test_validation_karpathy_depth(self):
        with pytest.raises(ValueError):
            EQSComponents(karpathy_depth=7)

    def test_summary_contains_score(self):
        eqs = EQSComponents(sandbox_pass=True, test_pass_rate=0.5).compute()
        assert "EQS Score" in eqs.summary()


# ── mastery_confidence_score ────────────────────────────────────────────────

class TestMasteryConfidenceScore:
    def test_returns_float_in_range(self):
        mcs = mastery_confidence_score(
            eqs_history=[0.6, 0.65, 0.7],
            sigma_history=[0.3, 0.45, 0.6],
            K=[0.6, 0.55, 0.5],
        )
        assert 0.0 <= mcs <= 1.0

    def test_insufficient_history_neutral(self):
        # With single-point history, consistency and velocity both return 0.5
        mcs = mastery_confidence_score(
            eqs_history=[0.7],
            sigma_history=[0.5],
            K=[0.5, 0.5, 0.5],
        )
        assert 0.0 <= mcs <= 1.0

    def test_consistent_high_eqs_raises_mcs(self):
        mcs_consistent = mastery_confidence_score(
            eqs_history=[0.8, 0.81, 0.79, 0.80],
            sigma_history=[0.2, 0.35, 0.50, 0.65],
            K=[0.65, 0.60, 0.55],
        )
        mcs_erratic = mastery_confidence_score(
            eqs_history=[0.2, 0.9, 0.1, 0.9],
            sigma_history=[0.2, 0.35, 0.50, 0.65],
            K=[0.65, 0.60, 0.55],
        )
        assert mcs_consistent > mcs_erratic


# ── update_vector_v2 ────────────────────────────────────────────────────────

class TestUpdateVectorV2:
    def test_focus_gains_most(self):
        K = [0.5, 0.5, 0.5]
        K_new, delta = update_vector_v2(K, eqs_score=0.5, focus_checkpoint="A")
        assert delta[0] > delta[1]
        assert delta[0] > delta[2]

    def test_base_gain_minimum(self):
        _, delta = update_vector_v2([0.5, 0.5, 0.5], eqs_score=0.0, focus_checkpoint="A")
        assert delta[0] == pytest.approx(BASE_GAIN)

    def test_max_gain_cap(self):
        _, delta = update_vector_v2([0.0, 0.0, 0.0], eqs_score=1.0, focus_checkpoint="A")
        assert delta[0] <= MAX_GAIN

    def test_spillover_modulated_by_mcs(self):
        _, delta_high_mcs = update_vector_v2(
            [0.5, 0.5, 0.5], eqs_score=0.5, focus_checkpoint="A", mcs=1.0
        )
        _, delta_low_mcs = update_vector_v2(
            [0.5, 0.5, 0.5], eqs_score=0.5, focus_checkpoint="A", mcs=0.0
        )
        assert delta_high_mcs[1] > delta_low_mcs[1]
        assert delta_low_mcs[1] == pytest.approx(0.0)

    def test_spillover_rate(self):
        _, delta = update_vector_v2(
            [0.5, 0.5, 0.5], eqs_score=0.0, focus_checkpoint="A", mcs=1.0
        )
        expected_sibling = BASE_GAIN * SPILLOVER * 1.0
        assert delta[1] == pytest.approx(expected_sibling)

    def test_K_clamped_to_one(self):
        K_new, _ = update_vector_v2([0.99, 0.99, 0.99], eqs_score=1.0, focus_checkpoint="A")
        assert all(k <= 1.0 for k in K_new)

    def test_K_clamped_to_zero(self):
        K_new, _ = update_vector_v2([0.0, 0.0, 0.0], eqs_score=0.0, focus_checkpoint="A")
        assert all(k >= 0.0 for k in K_new)

    def test_invalid_focus(self):
        with pytest.raises(ValueError):
            update_vector_v2([0.5, 0.5, 0.5], 0.5, "Z")

    def test_invalid_eqs_score(self):
        with pytest.raises(ValueError):
            update_vector_v2([0.5, 0.5, 0.5], 1.5, "A")

    def test_invalid_mcs(self):
        with pytest.raises(ValueError):
            update_vector_v2([0.5, 0.5, 0.5], 0.5, "A", mcs=-0.1)

    def test_all_checkpoints(self):
        for cp in ["A", "B", "C"]:
            K_new, delta = update_vector_v2([0.5, 0.5, 0.5], 0.5, cp)
            focus_idx = ["A", "B", "C"].index(cp)
            assert delta[focus_idx] == max(delta)


# ── compute_difficulty ──────────────────────────────────────────────────────

class TestComputeDifficulty:
    def test_returns_difficulty_profile(self):
        from hctp.difficulty_engine import DifficultyProfile
        profile = compute_difficulty([0.5, 0.4, 0.3], velocity=0.15)
        assert isinstance(profile, DifficultyProfile)

    def test_level_is_valid(self):
        profile = compute_difficulty([0.5, 0.4, 0.3], velocity=0.15)
        assert profile.level in DIFFICULTY_LEVELS

    def test_fast_on_helix_is_expert(self):
        # Perfect helix tracking at high velocity → expert
        profile = compute_difficulty(
            [1/3, 1/3, 1/3],   # perfectly on-helix (σ=mean, K=uniform)
            velocity=0.30,      # 2× target
        )
        assert profile.level in ("advanced", "expert")

    def test_slow_off_helix_is_foundation(self):
        profile = compute_difficulty(
            [1.0, 0.0, 0.0],   # extreme off-helix
            velocity=0.01,
        )
        assert profile.level in ("foundation", "standard")

    def test_breadcrumb_count_positive(self):
        profile = compute_difficulty([0.5, 0.4, 0.3], velocity=0.10)
        assert profile.breadcrumb_count >= 1

    def test_security_role_boosts_metaclass_breadcrumbs(self):
        # Checkpoint C (Metaclasses) has 1.5× weight for security role
        p_generic  = compute_difficulty([0.9, 0.8, 0.1], velocity=0.10, role="generic")
        p_security = compute_difficulty([0.9, 0.8, 0.1], velocity=0.10, role="security")
        assert p_security.breadcrumb_count >= p_generic.breadcrumb_count

    def test_hint_density_range(self):
        profile = compute_difficulty([0.5, 0.4, 0.3], velocity=0.15)
        assert 0.0 <= profile.hint_density <= 1.0

    def test_invalid_role(self):
        with pytest.raises(ValueError):
            compute_difficulty([0.5, 0.4, 0.3], velocity=0.15, role="wizard")

    def test_all_roles_valid(self):
        for role in ROLES:
            profile = compute_difficulty([0.5, 0.4, 0.3], velocity=0.15, role=role)
            assert profile.role == role

    def test_summary_is_string(self):
        profile = compute_difficulty([0.5, 0.4, 0.3], velocity=0.15)
        s = profile.summary()
        assert isinstance(s, str)
        assert "Difficulty Profile" in s


# ── VCAPOExporter ───────────────────────────────────────────────────────────

class TestVCAPOExporter:
    def _make_traj(self, learner_id="T", badge=False) -> TrajectoryRecord:
        return TrajectoryRecord(
            learner_id=learner_id,
            sessions=[{"karpathy_depth": 5}],
            eqs_scores=[0.70],
            sigma_history=[0.0, 0.5],
            K_history=[[0.5, 0.5, 0.5]],
            role="generic",
            badge_earned=badge,
        )

    def test_add_computes_weight(self):
        exporter = VCAPOExporter()
        traj = self._make_traj()
        exporter.add(traj)
        assert traj.weight > 0.0

    def test_weight_in_range(self):
        traj = self._make_traj()
        traj.compute_weight()
        assert 0.0 <= traj.weight <= 1.0

    def test_export_writes_jsonl(self, tmp_path):
        exporter = VCAPOExporter()
        exporter.add(self._make_traj("A", badge=True))
        exporter.add(self._make_traj("B", badge=False))
        out = tmp_path / "out.jsonl"
        n = exporter.export(out)
        assert n == 2
        lines = out.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_export_min_weight_filter(self, tmp_path):
        exporter = VCAPOExporter()
        exporter.add(self._make_traj("A"))
        out = tmp_path / "out.jsonl"
        # Require weight > 1.0 — nothing should be written
        n = exporter.export(out, min_weight=2.0)
        assert n == 0

    def test_weight_summary_empty(self):
        exporter = VCAPOExporter()
        assert "No trajectories" in exporter.weight_summary()

    def test_weight_summary_with_data(self):
        exporter = VCAPOExporter()
        exporter.add(self._make_traj("A", badge=True))
        s = exporter.weight_summary()
        assert "Badge earners: 1" in s

    def test_len(self):
        exporter = VCAPOExporter()
        exporter.add(self._make_traj())
        exporter.add(self._make_traj())
        assert len(exporter) == 2

    def test_training_example_has_required_keys(self):
        traj = self._make_traj("X")
        traj.compute_weight()
        ex = traj.to_training_example()
        for key in ("learner_id", "role", "badge_earned", "weight",
                    "session_count", "eqs_mean", "sigma_final", "sessions"):
            assert key in ex
