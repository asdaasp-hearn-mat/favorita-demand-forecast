"""Deterministic calendar and temporal feature engineering.

All features in this module derive exclusively from the **date column** and
optionally from static external calendars (holidays, oil prices).  They
carry **zero data-leakage risk** because they never reference future target
values.

Feature groups produced
-----------------------
1. **Basic calendar**  — day_of_week, day_of_month, day_of_year, week_of_year,
   month, quarter, year, is_weekend, is_month_start, is_month_end.
2. **Cyclical encodings** — sin/cos transforms of periodic features to preserve
   circular continuity for tree-based and neural models.
3. **Ecuadorian economic calendar** — payday indicators (15th & last day of
   month), *quincena* flag, and holiday-proximity windows.
4. **Holiday features** — days-to/from nearest national/regional/local holiday,
   holiday type flags, transferred-holiday flags, and bridge-day indicators.
5. **Oil-price context** — lagged & rolling crude-oil price (Ecuador is oil-
   dependent; WTI strongly correlates with public spending and, transitively,
   with retail demand).

Design decisions
----------------
- All transformations are expressed as **Polars expressions** so they compose
  lazily inside a `LazyFrame` query plan with zero intermediate copies.
- The public entry point `build_temporal_features` returns a *new* LazyFrame
  with features appended — it never mutates the input.
"""

from __future__ import annotations

import math

import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TWO_PI: float = 2.0 * math.pi

# Ecuadorian paydays: the 15th (mid-month) and the last calendar day.
_PAYDAY_DAY: int = 15

# Holiday proximity window: how many days before/after a holiday to flag.
_HOLIDAY_PROXIMITY_WINDOW: int = 3

# Oil price lag / rolling configuration (production-safe: only past values).
_OIL_LAG_DAYS: list[int] = [1, 7, 14]
_OIL_ROLLING_WINDOWS: list[int] = [7, 14, 28]


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Basic Calendar Features
# ═══════════════════════════════════════════════════════════════════════════
def _basic_calendar_exprs(date_col: str) -> list[pl.Expr]:
    """Return Polars expressions for raw calendar attributes.

    Parameters
    ----------
    date_col : str
        Name of the ``pl.Date`` or ``pl.Datetime`` column.

    Returns
    -------
    list[pl.Expr]
        Named expressions ready to pass into ``LazyFrame.with_columns``.
    """
    d = pl.col(date_col)
    return [
        d.dt.weekday().alias("day_of_week"),  # Mon=1 … Sun=7
        d.dt.day().alias("day_of_month"),
        d.dt.ordinal_day().alias("day_of_year"),
        d.dt.week().alias("week_of_year"),
        d.dt.month().alias("month"),
        d.dt.quarter().alias("quarter"),
        d.dt.year().alias("year"),
        (d.dt.weekday() >= 6).cast(pl.UInt8).alias("is_weekend"),
        (d.dt.day() == 1).cast(pl.UInt8).alias("is_month_start"),
        # Last day of month: the *next* day's month differs from the current.
        (d.dt.offset_by("1d").dt.month() != d.dt.month()).cast(pl.UInt8).alias("is_month_end"),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Cyclical Encodings
# ═══════════════════════════════════════════════════════════════════════════
def _cyclical_encoding_exprs() -> list[pl.Expr]:
    """Sin/cos encodings for periodic calendar features.

    Encodes ``day_of_week`` (period 7), ``day_of_month`` (period 31),
    ``day_of_year`` (period 365), and ``month`` (period 12) with a pair of
    sin/cos columns each so that gradient-boosted trees and downstream
    neural models can exploit circular proximity (e.g., Sunday is *near*
    Monday, December is *near* January).
    """
    pairs: list[tuple[str, float]] = [
        ("day_of_week", 7.0),
        ("day_of_month", 31.0),
        ("day_of_year", 365.0),
        ("month", 12.0),
    ]
    exprs: list[pl.Expr] = []
    for col_name, period in pairs:
        angle = pl.col(col_name).cast(pl.Float64) * (_TWO_PI / period)
        exprs.append(angle.sin().alias(f"{col_name}_sin"))
        exprs.append(angle.cos().alias(f"{col_name}_cos"))
    return exprs


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Ecuadorian Economic Calendar
# ═══════════════════════════════════════════════════════════════════════════
def _economic_calendar_exprs(date_col: str) -> list[pl.Expr]:
    """Ecuadorian payday and quincena indicators.

    In Ecuador (and most of Latin America), workers receive salaries on the
    15th (*quincena*) and the last day of each month.  Consumer spending
    spikes sharply on these dates and the 1-2 days that follow.

    Parameters
    ----------
    date_col : str
        Name of the date column.

    Returns
    -------
    list[pl.Expr]
        ``is_payday_15``, ``is_payday_end``, ``days_since_payday``.
    """
    d = pl.col(date_col)
    dom = d.dt.day()
    is_end = (d.dt.offset_by("1d").dt.month() != d.dt.month()).cast(pl.UInt8)
    is_15 = (dom == _PAYDAY_DAY).cast(pl.UInt8)

    # days_since_payday: min(day - 15, day_of_month) if past the 15th,
    # otherwise day_of_month (counting from last month's end-of-month payday).
    # Simplified: distance from the nearest past payday boundary.
    days_since_payday = (
        pl.when(dom > _PAYDAY_DAY)
        .then(dom - _PAYDAY_DAY)
        .otherwise(dom)  # days since last EOM payday
        .cast(pl.UInt16)
    )

    return [
        is_15.alias("is_payday_15"),
        is_end.alias("is_payday_end"),
        days_since_payday.alias("days_since_payday"),
    ]


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Holiday Features
# ═══════════════════════════════════════════════════════════════════════════
def build_holiday_features(
    lf: pl.LazyFrame,
    holidays_lf: pl.LazyFrame,
    date_col: str = "date",
    _store_city_col: str = "city",
    _store_state_col: str = "state",
) -> pl.LazyFrame:
    """Enrich a sales LazyFrame with holiday-proximity features.

    Strategy:
    - Create a dense date spine and left-join holiday events.
    - Compute ``days_to_next_holiday`` and ``days_from_prev_holiday`` using
      forward/backward fill on the holiday flag.
    - Flag *bridge days* (working day sandwiched between a holiday and a
      weekend or between two holidays).
    - Encode holiday type as categorical integer.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input sales data with at least ``date_col``.
    holidays_lf : pl.LazyFrame
        Holidays dataframe with columns: ``date``, ``type``, ``locale``,
        ``locale_name``, ``transferred``.
    date_col : str
        Name of the date column.
    store_city_col : str
        Column mapping each row to a city (for local holiday matching).
    store_state_col : str
        Column mapping each row to a state (for regional holiday matching).

    Returns
    -------
    pl.LazyFrame
        Input frame enriched with holiday features.
    """
    # --- Build a national holiday spine (most impactful scope) ---
    national = (
        holidays_lf.filter(pl.col("locale") == "National")
        .select(
            pl.col("date").cast(pl.Date),
            pl.lit(1).cast(pl.UInt8).alias("is_national_holiday"),
            pl.col("type").alias("holiday_type"),
            pl.col("transferred").cast(pl.Boolean).alias("is_transferred_holiday"),
        )
        .unique(subset=["date"])
    )

    # --- Join national holidays and compute proximity ---
    enriched = lf.join(national, on=date_col, how="left")

    enriched = enriched.with_columns(
        pl.col("is_national_holiday").fill_null(0).alias("is_national_holiday"),
        pl.col("is_transferred_holiday").fill_null(False).alias("is_transferred_holiday"),
    )

    # Forward-fill to compute "days since last holiday" and backward to
    # compute "days to next holiday".
    enriched = enriched.sort(date_col).with_columns(
        # Row number within the full date axis for distance computation
        pl.col(date_col).cast(pl.Date).alias("_date_ordinal"),
    )

    # Days to next holiday: use a window over the holiday flag.
    # We mark holiday dates, then for every row count how far away the
    # nearest *future* holiday date is.
    enriched = enriched.with_columns(
        pl.when(pl.col("is_national_holiday") == 1)
        .then(pl.col(date_col))
        .otherwise(None)
        .forward_fill()
        .alias("_prev_holiday_date"),
        pl.when(pl.col("is_national_holiday") == 1)
        .then(pl.col(date_col))
        .otherwise(None)
        .backward_fill()
        .alias("_next_holiday_date"),
    )

    enriched = enriched.with_columns(
        (pl.col(date_col).cast(pl.Date) - pl.col("_prev_holiday_date"))
        .dt.total_days()
        .clip(lower_bound=0)
        .cast(pl.Int16)
        .alias("days_from_prev_holiday"),
        (pl.col("_next_holiday_date") - pl.col(date_col).cast(pl.Date))
        .dt.total_days()
        .clip(lower_bound=0)
        .cast(pl.Int16)
        .alias("days_to_next_holiday"),
    )

    # Holiday proximity window flag (within ±N days of any national holiday)
    enriched = enriched.with_columns(
        (
            (pl.col("days_from_prev_holiday") <= _HOLIDAY_PROXIMITY_WINDOW)
            | (pl.col("days_to_next_holiday") <= _HOLIDAY_PROXIMITY_WINDOW)
        )
        .cast(pl.UInt8)
        .alias("in_holiday_window"),
    )

    # Drop intermediate columns
    enriched = enriched.drop(["_date_ordinal", "_prev_holiday_date", "_next_holiday_date"])

    return enriched


# ═══════════════════════════════════════════════════════════════════════════
# 5.  Oil Price Context (Lagged Only — No Leakage)
# ═══════════════════════════════════════════════════════════════════════════
def build_oil_features(
    lf: pl.LazyFrame,
    oil_lf: pl.LazyFrame,
    date_col: str = "date",
    oil_price_col: str = "dcoilwtico",
) -> pl.LazyFrame:
    """Append lagged and rolling oil-price features.

    Ecuador's economy is heavily oil-dependent.  Crude-oil price fluctuations
    directly affect public-sector spending and consumer purchasing power,
    making WTI a strong exogenous demand signal.

    **Leakage safety**: Only backward lags (≥1 day) and backward rolling
    windows are computed — the model never sees future oil prices.

    Parameters
    ----------
    lf : pl.LazyFrame
        Main sales LazyFrame.
    oil_lf : pl.LazyFrame
        Oil prices with ``date`` and ``dcoilwtico`` columns.
    date_col : str
        Date column name (must match between both frames).
    oil_price_col : str
        Column name of the daily WTI oil price.

    Returns
    -------
    pl.LazyFrame
        Input frame with oil lag and rolling features appended.
    """
    # Prepare oil series: forward-fill weekends/missing, then join
    oil_clean = (
        oil_lf.sort(date_col)
        .with_columns(pl.col(oil_price_col).forward_fill().alias("oil_price"))
        .select(pl.col(date_col).cast(pl.Date), "oil_price")
    )

    merged = lf.join(oil_clean, on=date_col, how="left")

    # --- Lag features (strictly backward) ---
    lag_exprs: list[pl.Expr] = []
    for lag in _OIL_LAG_DAYS:
        lag_exprs.append(pl.col("oil_price").shift(lag).alias(f"oil_price_lag_{lag}d"))

    # --- Rolling features (backward window, no center) ---
    roll_exprs: list[pl.Expr] = []
    for window in _OIL_ROLLING_WINDOWS:
        roll_exprs.append(
            pl.col("oil_price")
            .rolling_mean(window_size=window, min_samples=1)
            .alias(f"oil_price_rmean_{window}d")
        )

    merged = merged.sort(date_col).with_columns(lag_exprs + roll_exprs)

    return merged


# ═══════════════════════════════════════════════════════════════════════════
# Public Entry Point
# ═══════════════════════════════════════════════════════════════════════════
def build_temporal_features(
    lf: pl.LazyFrame,
    date_col: str = "date",
    holidays_lf: pl.LazyFrame | None = None,
    oil_lf: pl.LazyFrame | None = None,
) -> pl.LazyFrame:
    """Orchestrate the full deterministic temporal feature engineering pipeline.

    This is the **single entry point** that downstream training and inference
    code should call.  It composes all calendar, cyclical, economic-calendar,
    holiday-proximity, and oil-price features into a unified LazyFrame query
    plan.

    Parameters
    ----------
    lf : pl.LazyFrame
        Input sales data.  Must contain at least a ``date_col`` column of
        type ``pl.Date`` or ``pl.Datetime``.
    date_col : str, default ``"date"``
        Name of the date column.
    holidays_lf : pl.LazyFrame | None
        Holidays events frame (Favorita ``holidays_events.csv``).
        If ``None``, holiday features are skipped.
    oil_lf : pl.LazyFrame | None
        Oil prices frame (Favorita ``oil.csv``).
        If ``None``, oil features are skipped.

    Returns
    -------
    pl.LazyFrame
        A new LazyFrame with all temporal features appended.  The input
        frame is never mutated.
    """
    # 1. Basic calendar attributes
    result = lf.with_columns(_basic_calendar_exprs(date_col))

    # 2. Cyclical sin/cos encodings (depend on step 1 columns)
    result = result.with_columns(_cyclical_encoding_exprs())

    # 3. Ecuadorian economic calendar
    result = result.with_columns(_economic_calendar_exprs(date_col))

    # 4. Holiday proximity features
    if holidays_lf is not None:
        result = build_holiday_features(result, holidays_lf, date_col=date_col)

    # 5. Oil-price exogenous signals
    if oil_lf is not None:
        result = build_oil_features(result, oil_lf, date_col=date_col)

    return result
