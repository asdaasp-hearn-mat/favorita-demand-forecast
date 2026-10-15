"""Unit tests for the Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE).

Validates:
1. Zero error on perfect predictions.
2. Exact analytical matching for hand-computed synthetic cases.
3. Asymmetric penalty behavior (perishables w=1.25 penalized higher than non-perishables w=1.00).
4. Boundary cases: zero sales, negative sales clipping, null/empty/shape mismatches.
5. Equivalence between NumPy and Polars implementations.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from src.models.metrics import (
    DEFAULT_NON_PERISHABLE_WEIGHT,
    DEFAULT_PERISHABLE_WEIGHT,
    nwrmsle_numpy,
    nwrmsle_polars,
)


class TestNWRMSLENumPy:
    """Test suite for the NumPy NWRMSLE implementation."""

    def test_zero_error_on_exact_match(self) -> None:
        """When predictions match ground truth exactly, error must be zero."""
        y_true = np.array([0.0, 10.0, 45.2, 120.0, 500.0])
        y_pred = np.array([0.0, 10.0, 45.2, 120.0, 500.0])
        weights = np.array([1.0, 1.25, 1.0, 1.25, 1.0])

        score = nwrmsle_numpy(y_true, y_pred, weights=weights)
        assert pytest.approx(score, abs=1e-12) == 0.0

    def test_hand_calculated_synthetic_case(self) -> None:
        """Verify metric output against a manually computed closed-form value.

        Item 1 (non-perishable, w=1.0): y_true=0, y_pred=1 -> diff = ln(2) - ln(1) = ln(2)
        Item 2 (perishable, w=1.25):     y_true=3, y_pred=7 -> diff = ln(8) - ln(4) = ln(2)
        Both have squared diff = (ln(2))^2.
        Weighted average = (1.0 * ln(2)^2 + 1.25 * ln(2)^2) / (1.0 + 1.25) = ln(2)^2.
        Root = ln(2) = 0.69314718056...
        """
        y_true = [0.0, 3.0]
        y_pred = [1.0, 7.0]
        weights = [1.0, 1.25]

        expected_error = math.log(2)
        calculated_error = nwrmsle_numpy(y_true, y_pred, weights=weights)
        assert pytest.approx(calculated_error, rel=1e-7) == expected_error

    def test_asymmetric_perishable_weight_impact(self) -> None:
        """An error on a perishable item must yield higher overall NWRMSLE

        than the identical error on a non-perishable item, ceteris paribus.
        """
        # Baseline: two items, one perishable (idx 0), one non-perishable (idx 1)
        weights = np.array(
            [DEFAULT_PERISHABLE_WEIGHT, DEFAULT_NON_PERISHABLE_WEIGHT]
        )  # [1.25, 1.0]

        # Case A: Error occurs solely on the perishable item
        y_true_a = np.array([10.0, 20.0])
        y_pred_a = np.array([20.0, 20.0])
        err_on_perishable = nwrmsle_numpy(y_true_a, y_pred_a, weights=weights)

        # Case B: Identical magnitude logarithmic error occurs on the non-perishable item
        # ln(21) - ln(11) ≈ 0.6466
        # Let's use identical ratio (y+1) factor = 2:
        # y_true=10 -> y_pred=21 (ratio 2)
        y_true_b = np.array([20.0, 10.0])
        y_pred_b = np.array([20.0, 21.0])
        err_on_non_perishable = nwrmsle_numpy(y_true_b, y_pred_b, weights=weights)

        # Perishable weighted error must be strictly greater
        assert err_on_perishable > err_on_non_perishable

    def test_clipping_negative_predictions(self) -> None:
        """Negative predictions must be clipped to zero by default without raising errors."""
        y_true = np.array([0.0, 10.0])
        y_pred = np.array([-5.0, 10.0])  # Negative value should be clipped to 0.0

        # With y_pred[0] clipped to 0.0, both items match exactly (0.0 vs 0.0, 10.0 vs 10.0)
        score = nwrmsle_numpy(y_true, y_pred, clip_negative=True)
        assert pytest.approx(score, abs=1e-9) == 0.0

    def test_negative_values_raise_when_unclipped(self) -> None:
        """With clip_negative=False, negative inputs must raise ValueError."""
        with pytest.raises(ValueError, match="Negative values encountered"):
            nwrmsle_numpy([0.0, 5.0], [-1.0, 5.0], clip_negative=False)

    def test_shape_and_length_mismatches(self) -> None:
        """Incompatible dimensions must raise ValueError."""
        with pytest.raises(ValueError, match="Shape mismatch"):
            nwrmsle_numpy([1.0, 2.0], [1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="Shape mismatch"):
            nwrmsle_numpy([1.0, 2.0], [1.0, 2.0], weights=[1.0])

    def test_empty_arrays_raise_error(self) -> None:
        """Empty input arrays must raise ValueError."""
        with pytest.raises(ValueError, match="Cannot calculate NWRMSLE on empty arrays"):
            nwrmsle_numpy([], [])

    def test_invalid_weights_raise_error(self) -> None:
        """Negative weights or all-zero weights must raise ValueError."""
        with pytest.raises(ValueError, match="Weights cannot be negative"):
            nwrmsle_numpy([1.0, 2.0], [1.0, 2.0], weights=[-1.0, 1.0])

        with pytest.raises(ValueError, match="Sum of weights must be strictly positive"):
            nwrmsle_numpy([1.0, 2.0], [1.0, 2.0], weights=[0.0, 0.0])


class TestNWRMSLEPolars:
    """Test suite for the Polars native NWRMSLE implementation."""

    def test_polars_numpy_equivalence(self) -> None:
        """Verify numerical equivalence between NumPy and Polars implementations."""
        rng = np.random.default_rng(seed=42)
        n_samples = 1_000

        y_true = rng.uniform(0.0, 200.0, size=n_samples)
        # Add heteroscedastic noise
        y_pred = y_true + rng.normal(0.0, 15.0, size=n_samples)
        y_pred = np.clip(y_pred, 0.0, None)
        perishable = rng.choice([0, 1], size=n_samples, p=[0.7, 0.3])

        weights = np.where(perishable == 1, 1.25, 1.0)

        numpy_score = nwrmsle_numpy(y_true, y_pred, weights=weights)

        df = pl.DataFrame(
            {
                "unit_sales": y_true,
                "prediction": y_pred,
                "perishable": perishable,
            }
        )

        polars_score = nwrmsle_polars(
            df,
            y_true_col="unit_sales",
            y_pred_col="prediction",
            weight_col=None,
            perishable_col="perishable",
        )

        assert pytest.approx(polars_score, rel=1e-8) == numpy_score

    def test_polars_lazyframe_support(self) -> None:
        """Verify seamless execution on Polars LazyFrame."""
        lf = pl.LazyFrame(
            {
                "unit_sales": [0.0, 3.0],
                "prediction": [1.0, 7.0],
                "weight": [1.0, 1.25],
            }
        )

        score = nwrmsle_polars(lf, weight_col="weight")
        assert pytest.approx(score, rel=1e-7) == math.log(2)
