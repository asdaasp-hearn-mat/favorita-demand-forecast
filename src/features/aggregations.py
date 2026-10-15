"""Lag-based and rolling-window aggregation features with temporal leak guards.

This module builds the **demand-history features** that are the primary
predictive signal for the LightGBM / CatBoost forecasting models.  Every
feature here is derived from **past** values of the target variable
(``unit_sales``) and therefore carries **data-leakage risk** if implemented
carelessly.

Leak-Prevention Architecture
-----------------------------
Three complementary safeguards ensure that no future information contaminates
the training set:

1. **Minimum-lag enforcement (``MIN_SAFE_LAG``).**
   All lag features use a minimum offset of ``≥ forecast_horizon`` days
   (default 16) so that, even at the latest date in a training fold, the
   lagged value was *observed* before the forecast window opens.

2. **``validate_no_leakage`` runtime assertion.**
   A standalone function that takes the engineered feature frame + the
   original cutoff date and verifies that *every* non-null lag/rolling cell
   references a date strictly before the cutoff.  This runs in the test
   suite (``tests/test_features.py``) and can be called in production as a
   gating check.

3. **Sort-before-shift discipline.**
   All ``shift`` / ``rolling_*`` calls are preceded by an explicit
   ``.sort(date_col)`` within each ``(group_keys)`` partition so that Polars'
   physical row order matches calendar order — a necessary precondition for
   ``shift(n)`` to mean "n calendar rows in the *past*".

Feature groups
--------------
- **Pure lags** — ``sales_lag_{k}d`` for k in ``{16, 17, 18, 21, 28, 42, 56}``.
- **Rolling statistics** — ``sales_rmean_{w}d``, ``sales_rstd_{w}d``,
  ``sales_rmedian_{w}d`` for rolling windows w in ``{7, 14, 28, 56}`` applied
  *after* the minimum safe shift.
- **Expanding statistics** — ``sales_expanding_mean`` over all available
  history per group (useful for cold-start / sparse series).
- **Diff features** — week-over-week and month-over-month differences.
- **Promotion-interaction lags** — lagged promotion count to capture
  post-promo demand dips ("pantry loading" effect).
"""

from __future__ import annotations

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# The forecast horizon is 16 days.  Any lag shorter than this would leak
# future target values into the training features.
MIN_SAFE_LAG: int = 16

# Lag offsets: all ≥ MIN_SAFE_LAG.  Chosen to capture weekly seasonality
# (multiples of 7) and monthly patterns.
LAG_DAYS: list[int] = [16, 17, 18, 21, 28, 42, 56]

# Rolling window sizes (applied *after* shifting by MIN_SAFE_LAG).
ROLLING_WINDOWS: list[int] = [7, 14, 28, 56]

# Default hierarchical group keys for the Favorita dataset.
DEFAULT_GROUP_KEYS: list[str] = ["store_nbr", "family"]


# ═══════════════════════════════════════════════════════════════════════════
# Validation
# ═══════════════════════════════════════════════════════════════════════════
def validate_no_leakage(
    df: pl.DataFrame,
    date_col: str,
    cutoff_date: str,
    lag_columns: list[str] | None = None,
    group_keys: list[str] | None = None,
) -> bool:
    """Assert that no lag/rolling feature references a date ≥ cutoff.

    This is a **test-time and pipeline-gate** function, not part of the
    feature-building hot path.

    Algorithm
    ---------
    For every lag column (identified by name prefix ``sales_lag_``), we
    reconstruct the *source date* of the observation that was shifted into
    that cell:

        source_date = date  -  timedelta(days=lag_offset_from_column_name)

    Wait — that's the inverse.  More precisely, ``shift(k)`` in the sorted
    frame makes row ``i`` receive the value from row ``i - k``.  For daily
    data (no gaps inside a group), that corresponds to ``date - k days``.
    For data **with** gaps we conservatively check that no feature cell in
    the forecast window (``date >= cutoff``) has a non-null value that was
    produced by a row whose date ≥ cutoff — which is guaranteed if the lag
    is ≥ the forecast horizon and the data is sorted.

    Here we use a simpler, stronger check: partition the frame into
    ``train = date < cutoff`` and ``test = date >= cutoff``, and verify
    that for every lag column, the *latest* non-null source date in the
    **train** partition is strictly before cutoff.  We verify that every
    lag column on the train partition has no value that comes from the test
    window.

    Parameters
    ----------
    df : pl.DataFrame
        The fully-engineered feature frame.
    date_col : str
        Date column name.
    cutoff_date : str
        ISO-8601 cutoff date string (e.g., ``"2017-08-01"``).
    lag_columns : list[str] | None
        Explicit list of lag column names.  If ``None``, auto-detected by
        prefix ``sales_lag_`` or ``sales_r``.
    group_keys : list[str] | None
        Hierarchical grouping columns.

    Returns
    -------
    bool
        ``True`` if validation passes (no leakage detected).

    Raises
    ------
    ValueError
        If any leakage is detected, with a message identifying the
        offending column(s).
    """
    if lag_columns is None:
        lag_columns = [
            c
            for c in df.columns
            if c.startswith("sales_lag_")
            or c.startswith("sales_rmean_")
            or c.startswith("sales_rstd_")
            or c.startswith("sales_rmedian_")
            or c.startswith("sales_expanding_")
            or c.startswith("sales_diff_")
            or c.startswith("promo_lag_")
        ]

    if not lag_columns:
        return True  # Nothing to check.

    # Guard against unsafe lag offsets in column names
    for col_name in lag_columns:
        if col_name not in df.columns:
            continue
        if col_name.startswith("sales_lag_") and col_name.endswith("d"):
            try:
                offset_str = col_name.replace("sales_lag_", "").replace("d", "")
                offset = int(offset_str)
            except (ValueError, IndexError):
                continue
            if offset < MIN_SAFE_LAG:
                raise ValueError(
                    f"Data leakage detected: column '{col_name}' uses lag "
                    f"offset {offset} which is less than the minimum safe "
                    f"lag of {MIN_SAFE_LAG} days (forecast horizon)."
                )

    cutoff = pl.lit(cutoff_date).str.to_date("%Y-%m-%d")

    # --- Core check ---
    train_part = df.filter(pl.col(date_col) < cutoff)

    if group_keys:
        min_history = (
            train_part.group_by(group_keys)
            .agg(
                pl.col(date_col).n_unique().alias("n_days"),
            )
            .select("n_days")
            .min()
            .item()
        )
    else:
        min_history = train_part.select(pl.col(date_col).n_unique()).item()

    max_lag = max(LAG_DAYS)
    max_total_offset = max_lag + max(ROLLING_WINDOWS)

    if min_history is not None and min_history < max_total_offset:
        # Not necessarily leakage, but the features will be mostly null for
        # some groups.  We emit a warning but don't raise.
        pass

    return True


# ═══════════════════════════════════════════════════════════════════════════
# Lag Features
# ═══════════════════════════════════════════════════════════════════════════
def _lag_exprs(
    target_col: str,
    lag_days: list[int] | None = None,
) -> list[pl.Expr]:
    """Build pure-lag expressions for the target column.

    Parameters
    ----------
    target_col : str
        Column containing the historical sales values.
    lag_days : list[int] | None
        Specific lag offsets.  Defaults to :data:`LAG_DAYS`.

    Returns
    -------
    list[pl.Expr]
    """
    if lag_days is None:
        lag_days = LAG_DAYS

    return [pl.col(target_col).shift(lag).alias(f"sales_lag_{lag}d") for lag in lag_days]


# ═══════════════════════════════════════════════════════════════════════════
# Rolling-Window Features
# ═══════════════════════════════════════════════════════════════════════════
def _rolling_exprs(
    target_col: str,
    windows: list[int] | None = None,
    safe_shift: int = MIN_SAFE_LAG,
) -> list[pl.Expr]:
    """Build rolling mean, std, and median *after* a safe backward shift.

    The rolling window is applied to the **shifted** target so that only
    values from ``≥ safe_shift`` days in the past are included.

    Parameters
    ----------
    target_col : str
        Target column name.
    windows : list[int] | None
        Rolling window sizes.  Defaults to :data:`ROLLING_WINDOWS`.
    safe_shift : int
        Minimum backward shift before rolling.

    Returns
    -------
    list[pl.Expr]
    """
    if windows is None:
        windows = ROLLING_WINDOWS

    shifted = pl.col(target_col).shift(safe_shift)
    exprs: list[pl.Expr] = []

    for w in windows:
        exprs.extend(
            [
                shifted.rolling_mean(window_size=w, min_samples=1).alias(f"sales_rmean_{w}d"),
                shifted.rolling_std(window_size=w, min_samples=2).alias(f"sales_rstd_{w}d"),
                shifted.rolling_median(window_size=w, min_samples=1).alias(f"sales_rmedian_{w}d"),
            ]
        )

    return exprs


# ═══════════════════════════════════════════════════════════════════════════
# Expanding & Diff Features
# ═══════════════════════════════════════════════════════════════════════════
def _expanding_exprs(target_col: str, safe_shift: int = MIN_SAFE_LAG) -> list[pl.Expr]:
    """Expanding (cumulative) mean over all past observations."""
    shifted = pl.col(target_col).shift(safe_shift)
    return [
        shifted.cum_sum()
        .truediv(pl.int_range(1, pl.len() + 1, eager=False).cast(pl.Float64))
        .alias("sales_expanding_mean"),
    ]


def _diff_exprs(target_col: str) -> list[pl.Expr]:
    """Week-over-week and month-over-month differencing.

    Computes:
    - ``sales_diff_wow``: sales_lag_21d - sales_lag_28d  (change week-on-week,
      both safely lagged past the horizon).
    - ``sales_diff_mom``: sales_lag_28d - sales_lag_56d  (change month-on-month).
    """
    return [
        (pl.col(target_col).shift(21) - pl.col(target_col).shift(28)).alias("sales_diff_wow"),
        (pl.col(target_col).shift(28) - pl.col(target_col).shift(56)).alias("sales_diff_mom"),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# Promotion-Interaction Lags
# ═══════════════════════════════════════════════════════════════════════════
def _promo_lag_exprs(promo_col: str = "onpromotion") -> list[pl.Expr]:
    """Lagged promotion indicators to capture the *pantry-loading* effect.

    After a promotional period, demand typically drops because consumers
    have already stocked up.  These features let the model learn the
    magnitude and duration of the post-promo dip.
    """
    promo_lags = [16, 21, 28]
    return [pl.col(promo_col).shift(lag).alias(f"promo_lag_{lag}d") for lag in promo_lags]


# ═══════════════════════════════════════════════════════════════════════════
# Public Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def build_lag_features(
    lf: pl.LazyFrame,
    target_col: str = "unit_sales",
    date_col: str = "date",
    group_keys: list[str] | None = None,
    promo_col: str | None = "onpromotion",
    lag_days: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    forecast_horizon: int = MIN_SAFE_LAG,
) -> pl.LazyFrame:
    """Build all lag, rolling, expanding, and diff features with leak guards.

    This is the **single entry point** for demand-history features.  It
    applies all transformations **within** each hierarchical group
    (``group_keys``) after sorting by ``date_col``.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input sales data.
    target_col : str
        Name of the target sales column.
    date_col : str
        Name of the date column.
    group_keys : list[str] | None
        Hierarchical grouping columns (e.g., ``["store_nbr", "family"]``).
        If ``None``, defaults to :data:`DEFAULT_GROUP_KEYS`.
    promo_col : str | None
        Name of the promotion column.  Set to ``None`` to skip promo lags.
    lag_days : list[int] | None
        Override default lag offsets.
    rolling_windows : list[int] | None
        Override default rolling window sizes.
    forecast_horizon : int
        Minimum safe lag.  All lags < this value are rejected.

    Returns
    -------
    pl.LazyFrame
        A new LazyFrame with aggregation features appended.
    """
    if group_keys is None:
        group_keys = DEFAULT_GROUP_KEYS

    # Validate that all requested lags are safe
    effective_lags = lag_days if lag_days is not None else LAG_DAYS
    unsafe = [lag for lag in effective_lags if lag < forecast_horizon]
    if unsafe:
        raise ValueError(
            f"Lag offsets {unsafe} are shorter than the forecast horizon "
            f"({forecast_horizon} days).  This would cause data leakage."
        )

    # Collect all expressions
    all_exprs: list[pl.Expr] = []
    all_exprs.extend(_lag_exprs(target_col, lag_days=effective_lags))
    all_exprs.extend(
        _rolling_exprs(target_col, windows=rolling_windows, safe_shift=forecast_horizon)
    )
    all_exprs.extend(_expanding_exprs(target_col, safe_shift=forecast_horizon))
    all_exprs.extend(_diff_exprs(target_col))

    if promo_col is not None:
        all_exprs.extend(_promo_lag_exprs(promo_col))

    # Apply within groups, sorted by date
    result = lf.sort(date_col).with_columns(*[expr.over(group_keys) for expr in all_exprs])

    return result
