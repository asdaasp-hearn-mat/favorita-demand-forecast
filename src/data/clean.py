"""Data sanitization, missing-grid completion, and metadata enrichment pipeline.

Provides functional, vectorized transformations over Polars LazyFrames to:
1. Enforce physical non-negativity of demand (clip negative returns to 0.0).
2. Complete Cartesian grids (date × store × family) to guarantee continuous lag intervals.
3. Impute missing exogenous signals (oil price weekends/holidays forward-fill).
4. Provide numerically stable log1p / expm1 transformations for variance stabilization.
5. Enrich series with store hierarchy metadata (city, state, store type, cluster).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, overload

import polars as pl

if TYPE_CHECKING:
    import datetime

logger = logging.getLogger(__name__)


def sanitize_sales_target(
    lf: pl.LazyFrame,
    target_col: str = "unit_sales",
    clip_lower: float = 0.0,
) -> pl.LazyFrame:
    """Ensure sales targets are non-null and physically valid (non-negative).

    Parameters
    ----------
    lf : pl.LazyFrame
        Input LazyFrame containing sales data.
    target_col : str, default='unit_sales'
        Target sales column name.
    clip_lower : float, default=0.0
        Lower bound for sales. Negative sales (e.g., returns) are clipped to this value.
    """
    return lf.with_columns(
        pl.when(pl.col(target_col) < clip_lower)
        .then(pl.lit(clip_lower, dtype=pl.Float32))
        .otherwise(pl.col(target_col).fill_null(clip_lower))
        .cast(pl.Float32)
        .alias(target_col)
    )


def apply_log1p_target(
    lf: pl.LazyFrame,
    target_col: str = "unit_sales",
    output_col: str = "target_log1p",
) -> pl.LazyFrame:
    """Apply log1p transformation to stabilize variance and align with NWRMSLE optimization.

    Minimizing RMSE on ln(y + 1) directly corresponds to minimizing RMSLE in the original scale.
    """
    return lf.with_columns(pl.col(target_col).log1p().cast(pl.Float32).alias(output_col))


@overload
def invert_log1p_predictions(predictions: pl.Expr, clip_zero: bool = ...) -> pl.Expr: ...


@overload
def invert_log1p_predictions(predictions: pl.Series, clip_zero: bool = ...) -> pl.Series: ...


def invert_log1p_predictions(
    predictions: pl.Expr | pl.Series,
    clip_zero: bool = True,
) -> pl.Expr | pl.Series:
    """Invert log1p predictions back to the natural unit sales scale: y = exp(y_hat) - 1.

    Parameters
    ----------
    predictions : pl.Expr | pl.Series
        Predicted values in log1p space.
    clip_zero : bool, default=True
        Whether to enforce non-negative predictions.
    """
    if isinstance(predictions, pl.Expr):
        inv = predictions.exp() - 1.0
        return pl.when(inv < 0.0).then(pl.lit(0.0)).otherwise(inv) if clip_zero else inv
    else:
        inv_s = predictions.exp() - 1.0
        if clip_zero:
            inv_s = inv_s.clip(lower_bound=0.0)
        return inv_s


def build_complete_time_grid(
    start_date: datetime.date,
    end_date: datetime.date,
    stores_lf: pl.LazyFrame,
    families: list[str],
) -> pl.LazyFrame:
    """Generate an unbroken Cartesian product grid (date × store_nbr × family).

    In retail forecasting, days where a store sold zero units are frequently omitted
    from raw reporting rather than explicitly logged as 0. Creating a dense date grid
    is essential so that lagging operations (shift(k)) represent exact calendar days.

    Parameters
    ----------
    start_date : datetime.date
        Start date of the required grid.
    end_date : datetime.date
        End date of the required grid.
    stores_lf : pl.LazyFrame
        LazyFrame containing unique store numbers.
    families : list[str]
        List of all unique product families.
    """
    dates_df = pl.date_range(start_date, end_date, interval="1d", eager=True).to_frame("date")
    unique_stores = stores_lf.select("store_nbr").unique().collect()
    unique_families = pl.DataFrame({"family": families})

    # Cartesian join: dates x stores x families
    grid = dates_df.join(unique_stores, how="cross").join(unique_families, how="cross").lazy()
    return grid


def enrich_with_store_metadata(
    sales_lf: pl.LazyFrame,
    stores_lf: pl.LazyFrame,
    store_key: str = "store_nbr",
) -> pl.LazyFrame:
    """Enrich sales LazyFrame with store location, type, and cluster metadata.

    Parameters
    ----------
    sales_lf : pl.LazyFrame
        Sales time series frame.
    stores_lf : pl.LazyFrame
        Store attributes (city, state, type, cluster).
    store_key : str, default='store_nbr'
        Primary key column.
    """
    stores_clean = stores_lf.select(
        pl.col(store_key).cast(pl.Int16),
        pl.col("city").cast(pl.Categorical).alias("store_city"),
        pl.col("state").cast(pl.Categorical).alias("store_state"),
        pl.col("type").cast(pl.Categorical).alias("store_type"),
        pl.col("cluster").cast(pl.Int8).alias("store_cluster"),
    )
    return sales_lf.join(stores_clean, on=store_key, how="left")


def impute_missing_signals(
    lf: pl.LazyFrame,
    promo_col: str = "onpromotion",
    fill_promo_zero: bool = True,
) -> pl.LazyFrame:
    """Impute null exogenous columns (e.g. onpromotion nulls filled with 0)."""
    exprs: list[pl.Expr] = []
    if promo_col in lf.collect_schema().names() and fill_promo_zero:
        exprs.append(pl.col(promo_col).fill_null(0).cast(pl.Int32).alias(promo_col))

    if exprs:
        return lf.with_columns(exprs)
    return lf
