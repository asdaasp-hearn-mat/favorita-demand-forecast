"""Integration and unit tests for the data ingestion and cleaning pipeline.

Validates:
1. Sales sanitization (negative clipping, null imputation).
2. Log1p transformation and its inverse.
3. Cartesian time grid generation (zero-hole completion).
4. Store metadata joining.
5. DataLoader scanning and perishable weighting rules.
"""

from __future__ import annotations

import datetime
from pathlib import Path

import polars as pl
import pytest

from src.data.clean import (
    apply_log1p_target,
    build_complete_time_grid,
    enrich_with_store_metadata,
    impute_missing_signals,
    invert_log1p_predictions,
    sanitize_sales_target,
)
from src.data.loader import DataLoader


class TestDataCleaningPipeline:
    """Test suite for data cleaning, type coercion, and grid generation."""

    def test_sanitize_sales_negative_and_nulls(self) -> None:
        """Negative sales (returns) and nulls must be clipped to 0.0."""
        lf = pl.LazyFrame(
            {
                "unit_sales": [-15.5, 0.0, 42.0, None, -0.01],
            }
        )
        cleaned = sanitize_sales_target(lf, target_col="unit_sales").collect()
        expected = [0.0, 0.0, 42.0, 0.0, 0.0]
        assert cleaned["unit_sales"].to_list() == pytest.approx(expected)

    def test_log1p_and_inverse_exact_reconstruction(self) -> None:
        """Invertibility: expm1(log1p(y)) must equal y within floating-point tolerance."""
        original_vals = [0.0, 1.0, 15.5, 120.0, 9500.25]
        lf = pl.LazyFrame({"unit_sales": original_vals})

        transformed = apply_log1p_target(lf).collect()
        log_vals = transformed["target_log1p"]

        # Invert via Expr
        inverted_expr = (
            transformed.lazy()
            .with_columns(invert_log1p_predictions(pl.col("target_log1p")).alias("reconstructed"))
            .collect()
        )
        assert inverted_expr["reconstructed"].to_list() == pytest.approx(original_vals, rel=1e-5)

        # Invert via Series
        inverted_series = invert_log1p_predictions(log_vals)
        assert inverted_series.to_list() == pytest.approx(original_vals, rel=1e-5)

    def test_build_complete_time_grid_dimensions(self) -> None:
        """Cartesian grid size must equal n_dates * n_stores * n_families exactly."""
        start_date = datetime.date(2017, 1, 1)
        end_date = datetime.date(2017, 1, 5)  # 5 days
        stores_lf = pl.LazyFrame({"store_nbr": [1, 2, 3]})  # 3 stores
        families = ["BEVERAGES", "MEATS", "PRODUCE", "CLEANING"]  # 4 families

        grid = build_complete_time_grid(start_date, end_date, stores_lf, families).collect()

        expected_rows = 5 * 3 * 4  # 60 rows
        assert grid.height == expected_rows
        assert set(grid.columns) == {"date", "store_nbr", "family"}

    def test_enrich_with_store_metadata(self) -> None:
        """Store metadata attributes (city, state, type, cluster) must be joined properly."""
        sales_lf = pl.LazyFrame(
            {
                "store_nbr": [1, 2, 99],
                "sales": [10.0, 20.0, 30.0],
            }
        )
        stores_lf = pl.LazyFrame(
            {
                "store_nbr": [1, 2],
                "city": ["Quito", "Guayaquil"],
                "state": ["Pichincha", "Guayas"],
                "type": ["A", "D"],
                "cluster": [1, 13],
            }
        )

        enriched = enrich_with_store_metadata(sales_lf, stores_lf).collect()

        assert "store_city" in enriched.columns
        assert "store_state" in enriched.columns
        assert "store_type" in enriched.columns
        assert "store_cluster" in enriched.columns

        row_store_1 = enriched.filter(pl.col("store_nbr") == 1).row(0, named=True)
        assert row_store_1["store_city"] == "Quito"
        assert row_store_1["store_cluster"] == 1

        # Store 99 is not in stores table -> should be null
        row_store_99 = enriched.filter(pl.col("store_nbr") == 99).row(0, named=True)
        assert row_store_99["store_city"] is None

    def test_impute_missing_signals(self) -> None:
        """Null onpromotion values should be filled with 0."""
        lf = pl.LazyFrame({"onpromotion": [1, None, 5, None]})
        imputed = impute_missing_signals(lf).collect()
        assert imputed["onpromotion"].to_list() == [1, 0, 5, 0]


class TestDataLoaderSynthetic:
    """Test suite for DataLoader lazy scanning and perishable weight tagging."""

    def test_scan_train_with_perishable_weights(self, tmp_path: Path) -> None:
        """Perishable families must be assigned weight=1.25 and non-perishables weight=1.00."""
        raw_dir = tmp_path / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        train_csv = raw_dir / "train.csv"

        df_train = pl.DataFrame(
            {
                "id": [1, 2, 3],
                "date": [
                    datetime.date(2017, 1, 1),
                    datetime.date(2017, 1, 1),
                    datetime.date(2017, 1, 1),
                ],
                "store_nbr": [1, 1, 1],
                "family": ["PRODUCE", "GROCERY I", "MEATS"],
                "sales": [100.0, 50.0, 80.0],
                "onpromotion": [0, 1, 0],
            }
        )
        df_train.write_csv(train_csv)

        loader = DataLoader(raw_dir=raw_dir, processed_dir=raw_dir / "processed")
        scanned = loader.scan_train(use_parquet=False).collect()

        assert "unit_sales" in scanned.columns
        assert "weight" in scanned.columns
        assert "perishable" in scanned.columns

        produce_row = scanned.filter(pl.col("family") == "PRODUCE").row(0, named=True)
        grocery_row = scanned.filter(pl.col("family") == "GROCERY I").row(0, named=True)
        meats_row = scanned.filter(pl.col("family") == "MEATS").row(0, named=True)

        assert produce_row["weight"] == 1.25
        assert produce_row["perishable"] == 1
        assert meats_row["weight"] == 1.25
        assert meats_row["perishable"] == 1

        assert grocery_row["weight"] == 1.00
        assert grocery_row["perishable"] == 0
