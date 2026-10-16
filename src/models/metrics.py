"""Mathematical definition and implementations of the Normalized Weighted Root Mean Squared Logarithmic Error (NWRMSLE).

This module implements the official evaluation metric for the Corporación Favorita demand
forecasting competition (Store Sales - Time Series Forecasting).

Mathematical formulation:
-------------------------
Given:
    - y_i: Actual target sales value (y_i >= 0)
    - \\hat{y}_i: Predicted target sales value (\\hat{y}_i >= 0)
    - w_i: Weight associated with sample i (e.g., 1.25 for perishable, 1.00 for non-perishable)
    - n: Total number of observations

The metric is defined as:
    NWRMSLE = \\sqrt{ \\frac{\\sum_{i=1}^n w_i \\left( \\ln(\\hat{y}_i + 1) - \\ln(y_i + 1) \\right)^2}{\\sum_{i=1}^n w_i} }
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import numpy.typing as npt

DEFAULT_PERISHABLE_WEIGHT: float = 1.25
DEFAULT_NON_PERISHABLE_WEIGHT: float = 1.00


def nwrmsle_numpy(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    clip_negative: bool = True,
) -> float:
    """Compute the Normalized Weighted Root Mean Squared Logarithmic Error using NumPy.

    Parameters
    ----------
    y_true : ArrayLike
        True daily unit sales.
    y_pred : ArrayLike
        Predicted daily unit sales.
    weights : ArrayLike | None, default=None
        Sample weights. Perishable products typically have weight 1.25, non-perishable 1.0.
        If None, all observations are assigned a weight of 1.0.
    clip_negative : bool, default=True
        Whether to clip negative predicted and actual values to 0.0 before evaluating log1p.
        Demand cannot be negative; negative values would yield invalid logarithms.

    Returns
    -------
    float
        The scalar NWRMSLE error.

    Raises
    ------
    ValueError
        If inputs have mismatched lengths, invalid weights, or empty arrays.
    """
    y_true_arr = np.asarray(y_true, dtype=np.float64)
    y_pred_arr = np.asarray(y_pred, dtype=np.float64)

    if y_true_arr.size == 0 or y_pred_arr.size == 0:
        raise ValueError("Cannot calculate NWRMSLE on empty arrays.")

    if y_true_arr.shape != y_pred_arr.shape:
        raise ValueError(
            f"Shape mismatch: y_true has shape {y_true_arr.shape} while y_pred has shape {y_pred_arr.shape}."
        )

    if weights is None:
        weights_arr = np.ones_like(y_true_arr, dtype=np.float64)
    else:
        weights_arr = np.asarray(weights, dtype=np.float64)
        if weights_arr.shape != y_true_arr.shape:
            raise ValueError(
                f"Shape mismatch: weights has shape {weights_arr.shape} while y_true has shape {y_true_arr.shape}."
            )
        if np.any(weights_arr < 0):
            raise ValueError("Weights cannot be negative.")

    total_weight = float(np.sum(weights_arr))
    if total_weight <= 0.0:
        raise ValueError("Sum of weights must be strictly positive.")

    if clip_negative:
        y_true_arr = np.clip(y_true_arr, a_min=0.0, a_max=None)
        y_pred_arr = np.clip(y_pred_arr, a_min=0.0, a_max=None)
    else:
        if np.any(y_true_arr < 0.0) or np.any(y_pred_arr < 0.0):
            raise ValueError(
                "Negative values encountered in y_true or y_pred with clip_negative=False. "
                "Log1p is undefined for x < -1 and non-physical for sales."
            )

    log_true = np.log1p(y_true_arr)
    log_pred = np.log1p(y_pred_arr)
    squared_log_errors = (log_pred - log_true) ** 2

    weighted_mean_squared_log_error = np.sum(weights_arr * squared_log_errors) / total_weight
    return float(np.sqrt(weighted_mean_squared_log_error))


def nwrmsle_polars(
    df: pl.DataFrame | pl.LazyFrame,
    y_true_col: str = "unit_sales",
    y_pred_col: str = "prediction",
    weight_col: str | None = "weight",
    perishable_col: str | None = None,
    perishable_weight: float = DEFAULT_PERISHABLE_WEIGHT,
    non_perishable_weight: float = DEFAULT_NON_PERISHABLE_WEIGHT,
) -> float:
    """Compute NWRMSLE directly within Polars engine using vectorized columnar expressions.

    Can operate natively on `pl.DataFrame` or `pl.LazyFrame` with multithreaded SIMD execution.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        Input dataframe containing actuals, predictions, and weights/perishable flags.
    y_true_col : str, default='unit_sales'
        Column name of the ground truth sales.
    y_pred_col : str, default='prediction'
        Column name of the predicted sales.
    weight_col : str | None, default='weight'
        Existing weight column. If None or not present, uses `perishable_col` to derive weights.
    perishable_col : str | None, default=None
        Optional boolean/integer column indicating perishable items (1 or True).
        Used if `weight_col` is None or not found.
    perishable_weight : float, default=1.25
        Weight assigned to perishable items when derived from `perishable_col`.
    non_perishable_weight : float, default=1.00
        Weight assigned to non-perishable items when derived from `perishable_col`.

    Returns
    -------
    float
        The scalar NWRMSLE error.
    """
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Construct the weight expression
    schema_cols = lf.collect_schema().names()
    if weight_col is not None and weight_col in schema_cols:
        w_expr = pl.col(weight_col).cast(pl.Float64)
    elif perishable_col is not None and perishable_col in schema_cols:
        w_expr = (
            pl.when(pl.col(perishable_col).cast(pl.Boolean))
            .then(pl.lit(perishable_weight))
            .otherwise(pl.lit(non_perishable_weight))
            .cast(pl.Float64)
        )
    else:
        w_expr = pl.lit(1.0, dtype=pl.Float64)

    # Vectorized clipping and log1p computation
    clipped_true = (
        pl.when(pl.col(y_true_col) < 0.0)
        .then(pl.lit(0.0))
        .otherwise(pl.col(y_true_col))
        .cast(pl.Float64)
    )
    clipped_pred = (
        pl.when(pl.col(y_pred_col) < 0.0)
        .then(pl.lit(0.0))
        .otherwise(pl.col(y_pred_col))
        .cast(pl.Float64)
    )

    diff_log = clipped_pred.log1p() - clipped_true.log1p()
    sq_err = diff_log.pow(2)

    aggregated = lf.select(
        weighted_sq_sum=(w_expr * sq_err).sum(),
        sum_weights=w_expr.sum(),
    ).collect()

    weighted_sum = aggregated["weighted_sq_sum"][0]
    total_weights = aggregated["sum_weights"][0]

    if total_weights is None or total_weights <= 0.0:
        raise ValueError("Total sum of weights in Polars evaluation is non-positive or null.")

    if weighted_sum is None:
        raise ValueError("Calculated weighted sum is null.")

    return float(np.sqrt(weighted_sum / total_weights))
