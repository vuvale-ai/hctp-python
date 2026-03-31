"""Tests for hctp.tracker — LearnerSession lifecycle."""
import json
import pytest
from pathlib import Path
from hctp.tracker import LearnerSession, SessionResult


class TestLearnerSession:
    def test_initial_state(self):
        s = LearnerSession("Ren")
        assert s.K == [0.0, 0.0, 0.0]
        assert s.sigma == 0.0
        assert not s.badge
        assert s.sessions_completed == 0

    def test_start_returns_prompts(self):
        s = LearnerSession("Ren")
        data = s.start_session()
        assert "breadcrumb_prompts" in data
        assert len(data["breadcrumb_prompts"]) >= 1
        assert "header" in data
        assert "focus" in data

    def test_cannot_start_twice(self):
        s = LearnerSession("Ren")
        s.start_session()
        with pytest.raises(RuntimeError):
            s.start_session()

    def test_finish_without_start_raises(self):
        s = LearnerSession("Ren")
        with pytest.raises(RuntimeError):
            s.finish_session()

    def test_full_session_cycle(self):
        s = LearnerSession("Ren")
        s.start_session()
        delta = s.submit_karpathy("error fix refactor lesson self-propose ```python```")
        result = s.finish_session()

        assert isinstance(result, SessionResult)
        assert result.session_number == 1
        assert result.breadcrumbs_completed >= 1
        assert s.sessions_completed == 1
        assert s.sigma > 0.0

    def test_badge_earned_when_k3_threshold(self):
        # k3 is clearly the weakest — focus will be C, gaining directly toward 0.95
        s = LearnerSession("Ren", K=[0.99, 0.99, 0.909])
        s.start_session()
        # Rich response triggers all quality markers: ~0.06 gain on k3 → 0.909+0.06 = 0.969 ≥ 0.95
        s.submit_karpathy("error fix refactor lesson self-propose ```python``` connection next question")
        result = s.finish_session()
        assert result.badge_earned or s.badge

    def test_badge_earned_exactly_at_threshold(self):
        # K[2] already at threshold — badge on init
        s = LearnerSession("Ren", K=[0.96, 0.94, 0.95])
        assert s.badge

    def test_serialisation_round_trip(self, tmp_path):
        s = LearnerSession("Ren", K=[0.5, 0.6, 0.7],
                           sigma_history=[0.1, 0.2, 0.3])
        s.sessions_completed = 5
        path = tmp_path / "ren.json"
        s.save(path)

        loaded = LearnerSession.load(path)
        assert loaded.name == "Ren"
        assert loaded.K == [0.5, 0.6, 0.7]
        assert loaded.sessions_completed == 5

    def test_to_dict_from_dict(self):
        s = LearnerSession("Ner", K=[0.3, 0.4, 0.5])
        d = s.to_dict()
        s2 = LearnerSession.from_dict(d)
        assert s2.name == "Ner"
        assert s2.K == [0.3, 0.4, 0.5]

    def test_repr(self):
        s = LearnerSession("Sel", K=[0.1, 0.2, 0.3])
        r = repr(s)
        assert "Sel" in r
        assert "sigma" in r

    def test_velocity_with_history(self):
        s = LearnerSession("Les", sigma_history=[0.1, 0.2, 0.3, 0.4])
        assert s.velocity > 0


class TestSessionResult:
    def test_sigma_gain(self):
        r = SessionResult(
            session_number=1, K_before=[0.0]*3, K_after=[0.3]*3,
            sigma_before=0.2, sigma_after=0.35,
            velocity=0.15, focus_checkpoint="A",
            breadcrumbs_completed=3,
        )
        assert r.sigma_gain == pytest.approx(0.15)

    def test_str_includes_session_number(self):
        r = SessionResult(
            session_number=7, K_before=[0.0]*3, K_after=[0.6]*3,
            sigma_before=0.5, sigma_after=0.6,
            velocity=0.1, focus_checkpoint="B",
            breadcrumbs_completed=2,
        )
        assert "7" in str(r)

    def test_badge_str(self):
        r = SessionResult(
            session_number=13, K_before=[0.0]*3, K_after=[1.0]*3,
            sigma_before=0.9, sigma_after=0.97,
            velocity=0.07, focus_checkpoint="C",
            breadcrumbs_completed=3, badge_earned=True,
        )
        assert "BADGE" in str(r)
