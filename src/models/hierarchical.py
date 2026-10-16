"""Hierarchical forecast reconciliation engine for retail supply chains.

In retail hierarchies, independent forecasts at multiple aggregation levels
(e.g., store × family, city × family, national total) are fundamentally incoherent:
the sum of granular predictions does not match top-level predictions.

This module implements two foundational reconciliation algorithms:
1. **Bottom-Up Reconciliation:**
   Aggregates the bottom-level forecasts upwards through the summing matrix S:
       y_tilde = S * y_hat_bottom
   Guarantees 100% mathematical coherence, preserves low-level granularity, and
   avoids matrix inversion instabilities.

2. **Optimal Combination (MinT / WLS Reconciliation):**
   Reconciles forecasts across all levels by minimizing the trace of the
   reconciled forecast error covariance:
       y_tilde = S * (S^T * W^-1 * S)^-1 * S^T * W^-1 * y_hat
   where W is the error covariance matrix (e.g. diagonal variance scaling or OLS identity).
   Includes projection onto the non-negative orthant (sales >= 0).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Literal

import numpy as np
import polars as pl

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


class HierarchicalReconciler:
    """Manages hierarchical summing matrices and forecast reconciliation."""

    def __init__(
        self,
        method: Literal["bottom_up", "ols", "wls"] = "bottom_up",
    ) -> None:
        self.method = method

    @staticmethod
    def bottom_up_aggregate(
        df_bottom: pl.DataFrame,
        prediction_col: str = "prediction",
        hierarchy_group_cols: list[str] | None = None,
        date_col: str = "date",
    ) -> pl.DataFrame:
        """Aggregate bottom-level predictions upwards to higher hierarchical levels.

        Parameters
        ----------
        df_bottom : pl.DataFrame
            Granular predictions at (date, store_nbr, family).
        prediction_col : str, default='prediction'
            Column containing unit sales predictions.
        hierarchy_group_cols : list[str] | None
            Columns to aggregate by (e.g., ['date', 'family'] for national-by-family).
        date_col : str, default='date'
            Date column.
        """
        group = hierarchy_group_cols or [date_col, "family"]
        return df_bottom.group_by(group).agg(
            pl.col(prediction_col).sum().alias(f"{prediction_col}_aggregated")
        )

    @staticmethod
    def reconcile_mint_ols(
        S: npt.NDArray[np.float64],
        y_hat: npt.NDArray[np.float64],
        clip_non_negative: bool = True,
    ) -> npt.NDArray[np.float64]:
        """Perform MinT Ordinary Least Squares reconciliation: P = (S^T S)^-1 S^T.

        Parameters
        ----------
        S : NDArray[float64]
            Summing matrix of dimension (n_total, n_bottom).
        y_hat : NDArray[float64]
            Base forecasts vector of length n_total (or matrix n_total x n_time).
        clip_non_negative : bool, default=True
            Enforce demand non-negativity on reconciled bottom forecasts.

        Returns
        -------
        NDArray[float64]
            Coherent reconciled forecasts of dimension (n_total, ...).
        """
        # S_pinv = (S^T S)^-1 S^T
        StS = S.T @ S
        S_pinv = np.linalg.pinv(StS) @ S.T

        # Bottom-level reconciled: y_tilde_bottom = S_pinv @ y_hat
        y_tilde_bottom = S_pinv @ y_hat
        if clip_non_negative:
            y_tilde_bottom = np.clip(y_tilde_bottom, a_min=0.0, a_max=None)

        # Full reconciled hierarchy: y_tilde = S @ y_tilde_bottom
        y_tilde_total = S @ y_tilde_bottom
        return y_tilde_total

    @staticmethod
    def build_summing_matrix_store_family(
        unique_stores: list[int],
        unique_families: list[str],
    ) -> tuple[npt.NDArray[np.float64], list[str], list[str]]:
        """Construct the structural Summing Matrix S for Favorita store-family hierarchy.

        Hierarchy levels:
        1. Top Level (1): Total national sales.
        2. Store Level (n_stores): Total sales per store across all families.
        3. Family Level (n_families): Total sales per family across all stores.
        4. Bottom Level (n_stores * n_families): Individual (store, family) series.

        Returns
        -------
        tuple[NDArray[float64], list[str], list[str]]
            Summing matrix S, all series labels, bottom series labels.
        """
        n_s = len(unique_stores)
        n_f = len(unique_families)
        n_bottom = n_s * n_f

        bottom_labels = [f"s{s}_f{f}" for s in unique_stores for f in unique_families]

        # 1. Total (1 row of all ones)
        row_total = np.ones((1, n_bottom), dtype=np.float64)

        # 2. Store totals (n_stores rows)
        rows_stores = np.zeros((n_s, n_bottom), dtype=np.float64)
        for i, _ in enumerate(unique_stores):
            rows_stores[i, i * n_f : (i + 1) * n_f] = 1.0

        # 3. Family totals (n_families rows)
        rows_families = np.zeros((n_f, n_bottom), dtype=np.float64)
        for j, _ in enumerate(unique_families):
            for i in range(n_s):
                rows_families[j, i * n_f + j] = 1.0

        # 4. Bottom level (Identity matrix of size n_bottom)
        rows_bottom = np.eye(n_bottom, dtype=np.float64)

        # Stack into complete Summing Matrix S
        S = np.vstack([row_total, rows_stores, rows_families, rows_bottom])

        all_labels = (
            ["TOTAL"]
            + [f"store_{s}" for s in unique_stores]
            + [f"family_{f}" for f in unique_families]
            + bottom_labels
        )

        return S, all_labels, bottom_labels
