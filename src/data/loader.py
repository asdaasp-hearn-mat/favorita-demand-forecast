"""High-performance data ingestion engine for Corporación Favorita datasets.

Leverages Polars LazyFrames and DuckDB in-process OLAP to stream and scan
multi-gigabyte raw CSV files with optimal column schema downcasting, zero
redundant in-memory copies, and high-throughput Parquet staging.

Architecture principles:
------------------------
1. Streaming / Lazy execution: Never load entire uncompressed CSVs into RAM eagerly.
2. Type downcasting: Downcast integers (e.g., store_nbr -> Int16, cluster -> Int8)
   and floats (sales -> Float32) to reduce memory footprint by ~65%.
3. DuckDB integration: Expose zero-copy Arrow/DuckDB integration for fast SQL-based
   audits, filtering, or aggregations.
4. Parquet caching: Provide automated staging to Snappy/ZSTD-compressed Parquet.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
import polars as pl

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)

# Official list of perishable product families in Corporación Favorita
# Items belonging to these categories receive w_i = 1.25 in NWRMSLE.
DEFAULT_PERISHABLE_FAMILIES: frozenset[str] = frozenset(
    {
        "BREAD/BAKERY",
        "DAIRY",
        "DELI",
        "EGGS",
        "MEATS",
        "POULTRY",
        "PREPARED FOODS",
        "PRODUCE",
        "SEAFOOD",
    }
)


@dataclass(frozen=True)
class RawDataSchemas:
    """Explicit schema mappings for Favorita CSV datasets to prevent type inference overhead."""

    train_schema: dict[str, pl.DataType] = None  # type: ignore[assignment]
    test_schema: dict[str, pl.DataType] = None  # type: ignore[assignment]
    stores_schema: dict[str, pl.DataType] = None  # type: ignore[assignment]
    oil_schema: dict[str, pl.DataType] = None  # type: ignore[assignment]
    holidays_schema: dict[str, pl.DataType] = None  # type: ignore[assignment]
    transactions_schema: dict[str, pl.DataType] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "train_schema",
            {
                "id": pl.Int64,
                "date": pl.Date,
                "store_nbr": pl.Int16,
                "family": pl.String,
                "sales": pl.Float32,
                "onpromotion": pl.Int32,
            },
        )
        object.__setattr__(
            self,
            "test_schema",
            {
                "id": pl.Int64,
                "date": pl.Date,
                "store_nbr": pl.Int16,
                "family": pl.String,
                "onpromotion": pl.Int32,
            },
        )
        object.__setattr__(
            self,
            "stores_schema",
            {
                "store_nbr": pl.Int16,
                "city": pl.String,
                "state": pl.String,
                "type": pl.String,
                "cluster": pl.Int8,
            },
        )
        object.__setattr__(
            self,
            "oil_schema",
            {
                "date": pl.Date,
                "dcoilwtico": pl.Float32,
            },
        )
        object.__setattr__(
            self,
            "holidays_schema",
            {
                "date": pl.Date,
                "type": pl.String,
                "locale": pl.String,
                "locale_name": pl.String,
                "description": pl.String,
                "transferred": pl.Boolean,
            },
        )
        object.__setattr__(
            self,
            "transactions_schema",
            {
                "date": pl.Date,
                "store_nbr": pl.Int16,
                "transactions": pl.Int32,
            },
        )


class DataLoader:
    """Manages high-throughput ingestion, lazy scanning, and Parquet caching."""

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
        perishable_families: Sequence[str] | frozenset[str] = DEFAULT_PERISHABLE_FAMILIES,
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.perishable_families = frozenset(perishable_families)
        self.schemas = RawDataSchemas()

    def scan_train(self, use_parquet: bool = True) -> pl.LazyFrame:
        """Scan training dataset as a LazyFrame.

        Parameters
        ----------
        use_parquet : bool
            If True and the cached parquet file exists in `processed_dir`, scans
            the parquet file instead of raw CSV.
        """
        parquet_path = self.processed_dir / "train.parquet"
        if use_parquet and parquet_path.exists():
            return pl.scan_parquet(parquet_path)

        csv_path = self.raw_dir / "train.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Training data not found at {csv_path} or {parquet_path}")

        # Scan with schema overrides and normalize column name 'sales' -> 'unit_sales'
        lf = pl.scan_csv(csv_path, schema_overrides=self.schemas.train_schema)
        if (
            "sales" in lf.collect_schema().names()
            and "unit_sales" not in lf.collect_schema().names()
        ):
            lf = lf.rename({"sales": "unit_sales"})

        # Append boolean perishable indicator and associated NWRMSLE sample weight
        lf = lf.with_columns(
            pl.col("family")
            .is_in(list(self.perishable_families))
            .cast(pl.UInt8)
            .alias("perishable"),
            pl.when(pl.col("family").is_in(list(self.perishable_families)))
            .then(pl.lit(1.25, dtype=pl.Float64))
            .otherwise(pl.lit(1.00, dtype=pl.Float64))
            .alias("weight"),
        )
        return lf

    def scan_test(self, use_parquet: bool = True) -> pl.LazyFrame:
        """Scan test dataset as a LazyFrame."""
        parquet_path = self.processed_dir / "test.parquet"
        if use_parquet and parquet_path.exists():
            return pl.scan_parquet(parquet_path)

        csv_path = self.raw_dir / "test.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Test data not found at {csv_path} or {parquet_path}")

        lf = pl.scan_csv(csv_path, schema_overrides=self.schemas.test_schema)
        return lf.with_columns(
            pl.col("family")
            .is_in(list(self.perishable_families))
            .cast(pl.UInt8)
            .alias("perishable"),
            pl.when(pl.col("family").is_in(list(self.perishable_families)))
            .then(pl.lit(1.25, dtype=pl.Float64))
            .otherwise(pl.lit(1.00, dtype=pl.Float64))
            .alias("weight"),
        )

    def scan_stores(self) -> pl.LazyFrame:
        """Scan store metadata as a LazyFrame."""
        csv_path = self.raw_dir / "stores.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Stores file not found at {csv_path}")
        return pl.scan_csv(csv_path, schema_overrides=self.schemas.stores_schema)

    def scan_oil(self) -> pl.LazyFrame:
        """Scan oil prices as a LazyFrame with forward-filled prices."""
        csv_path = self.raw_dir / "oil.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Oil file not found at {csv_path}")
        return (
            pl.scan_csv(csv_path, schema_overrides=self.schemas.oil_schema)
            .sort("date")
            .with_columns(pl.col("dcoilwtico").forward_fill())
        )

    def scan_holidays(self) -> pl.LazyFrame:
        """Scan holidays dataset as a LazyFrame."""
        csv_path = self.raw_dir / "holidays_events.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"Holidays file not found at {csv_path}")
        return pl.scan_csv(csv_path, schema_overrides=self.schemas.holidays_schema)

    def stage_to_parquet(self, compression: str = "snappy") -> dict[str, Path]:
        """Convert all raw CSV files into optimized, compressed Parquet files.

        Parameters
        ----------
        compression : str, default='snappy'
            Compression codec: 'snappy', 'zstd', or 'uncompressed'.

        Returns
        -------
        dict[str, Path]
            Mapping of dataset names to staged Parquet file paths.
        """
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        staged: dict[str, Path] = {}

        tasks = [
            ("train", self.scan_train(use_parquet=False)),
            ("test", self.scan_test(use_parquet=False)),
            ("stores", self.scan_stores()),
            ("oil", self.scan_oil()),
            ("holidays", self.scan_holidays()),
        ]

        for name, lazy_frame in tasks:
            out_path = self.processed_dir / f"{name}.parquet"
            logger.info("Staging %s to %s...", name, out_path)
            # Execute in streaming mode where supported to minimize peak RAM
            lazy_frame.sink_parquet(out_path, compression=compression)  # type: ignore[call-overload]
            staged[name] = out_path

        return staged

    def duckdb_query(self, query: str) -> duckdb.DuckDBPyRelation:
        """Execute in-process SQL queries using DuckDB over registered datasets.

        Automatically exposes `train`, `stores`, `oil`, and `holidays` as virtual tables.

        Parameters
        ----------
        query : str
            Standard SQL query to evaluate.
        """
        con = duckdb.connect()
        # Register views lazily via Arrow
        try:
            train_arrow = self.scan_train().collect().to_arrow()
            con.register("train", train_arrow)
        except Exception as e:
            logger.debug("Could not register train table: %s", e)

        try:
            stores_arrow = self.scan_stores().collect().to_arrow()
            con.register("stores", stores_arrow)
        except Exception as e:
            logger.debug("Could not register stores table: %s", e)

        try:
            oil_arrow = self.scan_oil().collect().to_arrow()
            con.register("oil", oil_arrow)
        except Exception as e:
            logger.debug("Could not register oil table: %s", e)

        try:
            holidays_arrow = self.scan_holidays().collect().to_arrow()
            con.register("holidays", holidays_arrow)
        except Exception as e:
            logger.debug("Could not register holidays table: %s", e)

        return con.sql(query)
