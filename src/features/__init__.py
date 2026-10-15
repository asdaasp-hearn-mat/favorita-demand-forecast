"""Feature engineering subpackage for the Favorita demand forecasting pipeline.

This package provides two core modules:

- **temporal**: Calendar-based and deterministic temporal features (day-of-week,
  month, holiday proximity, payday indicators, etc.) that are safe to compute
  at any point because they depend only on the date itself.
- **aggregations**: Lag-based and rolling-window aggregations over historical
  sales data.  These features carry **data-leakage risk** and include built-in
  temporal guards that prevent any future information from leaking into the
  training set.
"""
