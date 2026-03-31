"""Tests for hctp.core — pure math, no LLM required."""
import math
import pytest
from hctp.core import (
    helix_radius, ideal_point, distance, progress,
    smoothed_velocity, num_breadcrumbs, determine_focus, update_vector,
    NUM_SPIRALS, MASTERY_THRESHOLD,
)


class TestHelixRadius:
    def test_zero_sigma(self):
        assert helix_radius(0.0) == pytest.approx(0.55)

    def test_full_sigma(self):
        assert helix_radius(1.0) == pytest.approx(0.05)

    def test_half_sigma(self):
        assert helix_radius(0.5) == pytest.approx(0.175)

    def test_always_positive(self):
        for s in [0.0, 0.25, 0.5, 0.75, 1.0]:
            assert helix_radius(s) > 0


class TestIdealPoint:
    def test_returns_three_components(self):
        pt = ideal_point(0.5)
        assert len(pt) == 3

    def test_z_equals_sigma(self):
        for sigma in [0.0, 0.3, 0.7, 1.0]:
            assert ideal_point(sigma)[2] == pytest.approx(sigma)

    def test_radius_at_zero(self):
        pt = ideal_point(0.0)
        r = math.sqrt(pt[0] ** 2 + pt[1] ** 2)
        assert r == pytest.approx(helix_radius(0.0))


class TestDistance:
    def test_on_helix_is_near_zero(self):
        sigma = 0.5
        pt = ideal_point(sigma)
        d = distance(pt, sigma)
        assert d == pytest.approx(0.0, abs=1e-10)

    def test_off_helix_positive(self):
        K = [0.1, 0.2, 0.3]
        sigma = progress(K)
        assert distance(K, sigma) > 0


class TestProgress:
    def test_zero_vector(self):
        assert progress([0.0, 0.0, 0.0]) == 0.0

    def test_full_vector(self):
        assert progress([1.0, 1.0, 1.0]) == pytest.approx(1.0)

    def test_mean(self):
        assert progress([0.3, 0.6, 0.9]) == pytest.approx(0.6)


class TestSmoothedVelocity:
    def test_single_value_returns_zero(self):
        assert smoothed_velocity([0.5]) == 0.0

    def test_constant_progress_returns_zero(self):
        assert smoothed_velocity([0.5, 0.5, 0.5]) == pytest.approx(0.0)

    def test_positive_growth(self):
        assert smoothed_velocity([0.1, 0.2, 0.3]) > 0


class TestNumBreadcrumbs:
    def test_fast_learner_gets_minimum(self):
        assert num_breadcrumbs(0.20) == 2  # at or above target

    def test_slow_learner_gets_more(self):
        assert num_breadcrumbs(0.0) > 2

    def test_always_at_least_one(self):
        assert num_breadcrumbs(-1.0) >= 1


class TestDetermineFocus:
    def test_focus_on_lowest(self):
        assert determine_focus([0.1, 0.5, 0.9]) == "A"
        assert determine_focus([0.9, 0.1, 0.5]) == "B"
        assert determine_focus([0.5, 0.9, 0.1]) == "C"

    def test_equal_vector_returns_A(self):
        assert determine_focus([0.5, 0.5, 0.5]) == "A"


class TestUpdateVector:
    def test_focus_component_grows_most(self):
        K = [0.5, 0.5, 0.5]
        new_K, delta = update_vector(K, "error fix refactor lesson", "A")
        assert delta[0] > delta[1]
        assert delta[0] > delta[2]

    def test_spillover_to_siblings(self):
        K = [0.5, 0.5, 0.5]
        _, delta = update_vector(K, "error fix refactor lesson ```code```", "C")
        assert delta[0] > 0
        assert delta[1] > 0
        assert delta[2] > delta[0]

    def test_values_clamped_to_one(self):
        K = [1.0, 1.0, 1.0]
        new_K, _ = update_vector(K, "error fix lesson", "A")
        assert all(v <= 1.0 for v in new_K)

    def test_values_not_negative(self):
        K = [0.0, 0.0, 0.0]
        new_K, _ = update_vector(K, "", "A")
        assert all(v >= 0.0 for v in new_K)
