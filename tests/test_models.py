"""
Unit tests for core MCDM models.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from core.models import (
    ahp_weights,
    default_ahp_weights,
    merec,
    mabac,
    rank_frame_from_matrix,
    topsis,
)


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def simple_matrix() -> np.ndarray:
    """3 alternatives × 3 criteria, no NaN, no zeros."""
    # Criteria: [NPxG, SoT, xA] — all benefits
    return np.array([
        [0.50, 1.2, 0.15],  # Player A
        [0.70, 1.5, 0.20],  # Player B (dominates A)
        [0.30, 0.8, 0.10],  # Player C (dominated)
    ], dtype=float)


@pytest.fixture
def matrix_with_cost() -> np.ndarray:
    """4 criteria: 3 benefits + 1 cost (market value)."""
    return np.array([
        [0.50, 1.2, 0.15, 20_000_000],   # High value, high cost
        [0.70, 1.5, 0.20, 50_000_000],   # Best performance, highest cost
        [0.30, 0.8, 0.10,  5_000_000],   # Low performance, low cost
        [0.60, 1.3, 0.18, 10_000_000],   # Balanced
    ], dtype=float)


@pytest.fixture
def benefits_3() -> np.ndarray:
    return np.array([True, True, True], dtype=bool)


@pytest.fixture
def benefits_4() -> np.ndarray:
    return np.array([True, True, True, False], dtype=bool)


@pytest.fixture
def weights_3() -> np.ndarray:
    return np.array([0.5, 0.3, 0.2], dtype=float)


@pytest.fixture
def weights_4() -> np.ndarray:
    # Similar to AHP default: NPxG ~0.45, SoT ~0.14, xA ~0.14, Cost ~0.26
    return np.array([0.45, 0.14, 0.14, 0.27], dtype=float)


# ============================================================
# AHP TESTS
# ============================================================

class TestAHP:
    def test_default_weights_sum_to_one(self):
        w = default_ahp_weights()
        assert np.isclose(w.sum(), 1.0)

    def test_default_weights_shape(self):
        w = default_ahp_weights()
        assert w.shape == (4,)

    def test_ahp_consistency_check_passes(self):
        matrix = np.array([
            [1, 3, 3],
            [1/3, 1, 1],
            [1/3, 1, 1],
        ], dtype=float)
        w = ahp_weights(matrix)
        assert np.isclose(w.sum(), 1.0)

    def test_ahp_consistency_check_fails(self):
        # Highly inconsistent matrix
        matrix = np.array([
            [1, 9, 9],
            [1/9, 1, 9],
            [1/9, 1/9, 1],
        ], dtype=float)
        with pytest.raises(ValueError, match="consistency ratio"):
            ahp_weights(matrix)

    def test_ahp_non_square_raises(self):
        matrix = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        with pytest.raises(ValueError, match="square"):
            ahp_weights(matrix)


# ============================================================
# TOPSIS TESTS
# ============================================================

class TestTOPSIS:
    def test_range_zero_one(self, simple_matrix, benefits_3, weights_3):
        scores = topsis(simple_matrix, weights_3, benefits_3)
        assert scores.shape == (3,)
        assert np.all(scores >= 0.0), "Scores below 0"
        assert np.all(scores <= 1.0), "Scores above 1"

    def test_monotonicity_benefit(self, simple_matrix, benefits_3, weights_3):
        """Increasing a benefit criterion should not decrease score."""
        X = simple_matrix.copy()
        base_scores = topsis(X, weights_3, benefits_3)

        # Boost player 0 on criterion 0 (benefit)
        X[0, 0] *= 1.5
        new_scores = topsis(X, weights_3, benefits_3)

        assert new_scores[0] >= base_scores[0] - 1e-10, \
            "Score decreased after increasing benefit criterion"

    def test_no_nan_inf_clean_input(self, simple_matrix, benefits_3, weights_3):
        scores = topsis(simple_matrix, weights_3, benefits_3)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_handles_zeros_in_matrix(self, benefits_3, weights_3):
        X = np.array([
            [0.0, 1.0, 0.1],
            [0.5, 1.2, 0.15],
            [0.3, 0.8, 0.1],
        ], dtype=float)
        scores = topsis(X, weights_3, benefits_3)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_handles_extreme_values(self, benefits_3, weights_3):
        X = np.array([
            [1e-6, 1e-6, 1e-6],
            [1e6, 1e6, 1e6],
            [1.0, 1.0, 1.0],
        ], dtype=float)
        scores = topsis(X, weights_3, benefits_3)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))
        assert np.all(scores >= 0.0) and np.all(scores <= 1.0)

    def test_vector_vs_minmax_normalization(self, simple_matrix, benefits_3, weights_3):
        s_vec = topsis(simple_matrix, weights_3, benefits_3, normalize="vector")
        s_mm = topsis(simple_matrix, weights_3, benefits_3, normalize="minmax")
        # Both should produce valid scores in [0,1]
        assert np.all(s_vec >= 0) and np.all(s_vec <= 1)
        assert np.all(s_mm >= 0) and np.all(s_mm <= 1)


# ============================================================
# MABAC TESTS
# ============================================================

class TestMABAC:
    def test_no_nan_inf_clean_input(self, matrix_with_cost, benefits_4, weights_4):
        scores = mabac(matrix_with_cost, weights_4, benefits_4)
        assert scores.shape == (4,)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_monotonicity_benefit(self, matrix_with_cost, benefits_4, weights_4):
        X = matrix_with_cost.copy()
        base = mabac(X, weights_4, benefits_4)

        # Boost player 0 on benefit criterion
        X[0, 0] *= 1.5
        new = mabac(X, weights_4, benefits_4)
        assert new[0] >= base[0] - 1e-10

    def test_monotonicity_cost(self, matrix_with_cost, benefits_4, weights_4):
        X = matrix_with_cost.copy()
        base = mabac(X, weights_4, benefits_4)

        # REDUCE cost for player 0 (cost criterion = index 3)
        X[0, 3] *= 0.5
        new = mabac(X, weights_4, benefits_4)
        assert new[0] >= base[0] - 1e-10

    def test_handles_identical_rows(self, benefits_3, weights_3):
        X = np.array([
            [0.5, 1.0, 0.15],
            [0.5, 1.0, 0.15],
            [0.5, 1.0, 0.15],
        ], dtype=float)
        scores = mabac(X, weights_3, benefits_3)
        # All should get same score
        assert np.allclose(scores, scores[0])

    def test_handles_extreme_values(self, benefits_3, weights_3):
        X = np.array([
            [1e-9, 1e-9, 1e-9],
            [1e9, 1e9, 1e9],
            [1.0, 1.0, 1.0],
        ], dtype=float)
        scores = mabac(X, weights_3, benefits_3)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))


# ============================================================
# MEREC TESTS
# ============================================================

class TestMEREC:
    def test_returns_scores_and_weights(self, matrix_with_cost, benefits_4):
        scores, weights = merec(matrix_with_cost, benefits_4)
        assert scores.shape == (4,)
        assert weights.shape == (4,)
        assert np.isclose(weights.sum(), 1.0)

    def test_scores_in_zero_one(self, matrix_with_cost, benefits_4):
        scores, _ = merec(matrix_with_cost, benefits_4)
        assert np.all(scores >= 0.0), "MEREC scores below 0"
        assert np.all(scores <= 1.0), "MEREC scores above 1"

    def test_no_nan_inf(self, matrix_with_cost, benefits_4):
        scores, weights = merec(matrix_with_cost, benefits_4)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))
        assert not np.any(np.isnan(weights))
        assert not np.any(np.isinf(weights))

    def test_strictly_positive_input_required(self, benefits_3):
        """MEREC requires X > 0; zeros should be handled by eps."""
        X = np.array([
            [0.0, 1.0, 0.1],
            [0.5, 1.2, 0.15],
            [0.3, 0.8, 0.1],
        ], dtype=float)
        scores, weights = merec(X, benefits_3, eps=1e-12)
        assert not np.any(np.isnan(scores))
        assert not np.any(np.isinf(scores))

    def test_monotonicity_benefit(self, matrix_with_cost, benefits_4):
        X = matrix_with_cost.copy()
        base, _ = merec(X, benefits_4)

        X[0, 0] *= 1.5  # increase benefit
        new, _ = merec(X, benefits_4)
        assert new[0] >= base[0] - 1e-10

    def test_monotonicity_cost(self, matrix_with_cost, benefits_4):
        X = matrix_with_cost.copy()
        base, _ = merec(X, benefits_4)

        X[0, 3] *= 0.5  # decrease cost
        new, _ = merec(X, benefits_4)
        assert new[0] >= base[0] - 1e-10

    def test_weights_differ_from_uniform(self, matrix_with_cost, benefits_4):
        """With diverse data, MEREC weights should not be uniform."""
        _, weights = merec(matrix_with_cost, benefits_4)
        uniform = np.full(4, 0.25)
        # At least one weight should differ significantly from uniform
        assert not np.allclose(weights, uniform, atol=0.05)


# ============================================================
# INTEGRATION / RANK FRAME TESTS
# ============================================================

class TestRankFrame:
    def test_rank_frame_output_structure(self, matrix_with_cost, benefits_4, weights_4):
        df, w_merec = rank_frame_from_matrix(matrix_with_cost, weights_4, benefits_4)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["TOPSIS", "MABAC", "MEREC", "Rank_TOPSIS", "Rank_MABAC", "Rank_MEREC"]
        assert len(df) == 4
        assert np.isclose(w_merec.sum(), 1.0)

    def test_ranks_are_permutation(self, matrix_with_cost, benefits_4, weights_4):
        df, _ = rank_frame_from_matrix(matrix_with_cost, weights_4, benefits_4)
        for method in ["TOPSIS", "MABAC", "MEREC"]:
            ranks = df[f"Rank_{method}"].values
            assert set(ranks) == {1, 2, 3, 4}

    def test_no_nan_in_final_frame(self, matrix_with_cost, benefits_4, weights_4):
        df, _ = rank_frame_from_matrix(matrix_with_cost, weights_4, benefits_4)
        assert not df.isna().any().any()


# ============================================================
# EDGE CASES / ROBUSTNESS
# ============================================================

class TestEdgeCases:
    def test_single_alternative(self, benefits_3, weights_3):
        X = np.array([[0.5, 1.2, 0.15]], dtype=float)
        s1 = topsis(X, weights_3, benefits_3)
        s2 = mabac(X, weights_3, benefits_3)
        s3, _ = merec(X, benefits_3)
        assert s1[0] == 0.5  # TOPSIS single item = 0.5
        assert not np.any(np.isnan([s1, s2, s3]))

    def test_two_alternatives(self, benefits_3, weights_3):
        X = np.array([
            [0.5, 1.0, 0.1],
            [0.6, 1.2, 0.15],
        ], dtype=float)
        for fn in [topsis, mabac]:
            scores = fn(X, weights_3, benefits_3)
            assert not np.any(np.isnan(scores))

    def test_all_identical_criteria_values(self, benefits_3, weights_3):
        X = np.array([
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ], dtype=float)
        for fn in [topsis, mabac]:
            scores = fn(X, weights_3, benefits_3)
            # All should get equal score
            assert np.allclose(scores, scores[0])


# ============================================================
# PROPERTY-BASED TESTS (using hypothesis if available)
# ============================================================

try:
    from hypothesis import given, strategies as st, settings

    class TestPropertyBased:
        @settings(max_examples=50, deadline=None)
        @given(
            m=st.integers(2, 10),
            n=st.integers(2, 5),
        )
        def test_topsis_always_returns_valid_scores(self, m, n):
            """TOPSIS should never produce NaN/Inf for any valid random input."""
            X = np.random.rand(m, n) * 100 + 0.1  # strictly positive
            weights = np.random.dirichlet(np.ones(n))
            benefits = np.random.choice([True, False], size=n)
            # Ensure at least one benefit and one cost for variety
            if not benefits.any():
                benefits[0] = True
            if benefits.all():
                benefits[0] = False

            scores = topsis(X, weights, benefits)
            assert scores.shape == (m,)
            assert not np.any(np.isnan(scores))
            assert not np.any(np.isinf(scores))
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

        @settings(max_examples=50, deadline=None)
        @given(
            m=st.integers(2, 10),
            n=st.integers(2, 5),
        )
        def test_mabac_always_returns_finite_scores(self, m, n):
            X = np.random.rand(m, n) * 100 + 0.1
            weights = np.random.dirichlet(np.ones(n))
            benefits = np.random.choice([True, False], size=n)
            if not benefits.any():
                benefits[0] = True
            if benefits.all():
                benefits[0] = False

            scores = mabac(X, weights, benefits)
            assert scores.shape == (m,)
            assert not np.any(np.isnan(scores))
            assert not np.any(np.isinf(scores))

        @settings(max_examples=50, deadline=None)
        @given(
            m=st.integers(3, 10),
            n=st.integers(2, 5),
        )
        def test_merec_always_returns_valid(self, m, n):
            X = np.random.rand(m, n) * 100 + 0.1
            benefits = np.random.choice([True, False], size=n)
            if not benefits.any():
                benefits[0] = True
            if benefits.all():
                benefits[0] = False

            scores, weights = merec(X, benefits)
            assert scores.shape == (m,)
            assert weights.shape == (n,)
            assert not np.any(np.isnan(scores))
            assert not np.any(np.isinf(scores))
            assert np.isclose(weights.sum(), 1.0)
            assert np.all(scores >= 0.0)
            assert np.all(scores <= 1.0)

except ImportError:
    # Hypothesis not installed — skip property tests
    pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])