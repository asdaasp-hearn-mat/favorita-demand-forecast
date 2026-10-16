"""Training orchestrator with expanding-window time-series cross-validation.

Features:
---------
1. Temporal Expanding-Window CV: Strictly preserves causality; never shuffles data.
2. End-to-end integration: Connects Polars feature pipelines, NWRMSLE weighting, and LightGBM.
3. Artifact management: Persists models, metrics logs, and feature importances.
4. CLI runnable: `python -m src.models.train --config configs/model_config.yaml`
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from src.data.loader import DEFAULT_PERISHABLE_FAMILIES, DataLoader
from src.features.aggregations import build_lag_features
from src.features.temporal import build_temporal_features
from src.models.lgbm_forecaster import LGBMForecaster
from src.models.metrics import nwrmsle_numpy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def generate_time_series_folds(
    dates: list[datetime.date],
    n_folds: int = 4,
    horizon_days: int = 16,
) -> list[tuple[datetime.date, datetime.date, datetime.date]]:
    """Generate expanding-window training and fixed-horizon validation folds.

    Parameters
    ----------
    dates : list[date]
        Sorted list of unique dates available in the dataset.
    n_folds : int, default=4
        Number of cross-validation folds.
    horizon_days : int, default=16
        Length of the validation window in days.

    Returns
    -------
    list[tuple[date, date, date]]
        List of (train_start, val_start, val_end) date cutoffs.
    """
    sorted_dates = sorted(dates)
    max_date = sorted_dates[-1]

    folds: list[tuple[datetime.date, datetime.date, datetime.date]] = []
    for k in range(n_folds, 0, -1):
        # Val end is shifted backward by (k - 1) * horizon_days from max_date
        val_end = max_date - datetime.timedelta(days=(k - 1) * horizon_days)
        val_start = val_end - datetime.timedelta(days=horizon_days - 1)
        train_start = sorted_dates[0]
        folds.append((train_start, val_start, val_end))

    return folds


def prepare_feature_matrices(
    df: pl.DataFrame,
    feature_cols: list[str],
    target_col: str = "unit_sales",
    weight_col: str = "weight",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract NumPy feature matrix X, log1p target y, and sample weights from Polars DataFrame."""
    X = df.select(feature_cols).to_numpy()
    y_raw = df[target_col].to_numpy()
    y_log1p = np.log1p(np.clip(y_raw, 0.0, None))
    weights = df[weight_col].to_numpy() if weight_col in df.columns else np.ones_like(y_raw)
    return X, y_log1p, weights


def generate_synthetic_benchmark() -> pl.DataFrame:
    """Generate realistic benchmark daily retail time-series for initial model training."""
    rng = np.random.default_rng(seed=42)
    start = datetime.date(2017, 1, 1)
    n_days = 220
    dates = [start + datetime.timedelta(days=i) for i in range(n_days)]

    stores = [1, 2, 3]
    families = ["GROCERY I", "BEVERAGES", "PRODUCE", "MEATS"]

    records = []
    for d in dates:
        day_of_week = d.weekday()
        weekend_mult = 1.3 if day_of_week >= 5 else 1.0
        for s in stores:
            store_mult = 1.0 + (s * 0.1)
            for f in families:
                is_perish = f in DEFAULT_PERISHABLE_FAMILIES
                base_demand = 90.0 if is_perish else 60.0
                on_promo = int(rng.binomial(1, 0.15))
                promo_mult = 1.4 if on_promo else 1.0
                noise = rng.normal(0, 5.0)
                sales = max(base_demand * store_mult * weekend_mult * promo_mult + noise, 0.0)
                weight = 1.25 if is_perish else 1.00

                records.append(
                    {
                        "date": d,
                        "store_nbr": s,
                        "family": f,
                        "unit_sales": round(float(sales), 2),
                        "onpromotion": on_promo,
                        "weight": weight,
                    }
                )

    return pl.DataFrame(records)


def train_pipeline(config_path: str | Path = "configs/model_config.yaml") -> dict[str, Any]:
    """Execute the end-to-end model training workflow."""
    cfg_path = Path(config_path)
    with open(cfg_path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Loaded training configuration from %s", cfg_path)
    exp_cfg = config.get("experiment", {})
    horizon = exp_cfg.get("forecast_horizon", 16)
    n_folds = exp_cfg.get("n_folds", 4)
    lgb_params = config.get("lightgbm", {})

    output_dir = Path("models")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for raw or sample data
    raw_path = Path("data/raw/train.csv")
    sample_path = Path("data/sample/train_sample.parquet")

    if raw_path.exists():
        logger.info("Loading Kaggle training data from %s...", raw_path)
        loader = DataLoader()
        df = loader.scan_train(use_parquet=False).collect()
    elif sample_path.exists():
        logger.info("Loading sample dataset from %s...", sample_path)
        df = pl.read_parquet(sample_path)
    else:
        logger.info("No raw data found in data/raw/. Generating representative benchmark sample...")
        df = generate_synthetic_benchmark()
        sample_path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(sample_path)
        logger.info("Saved benchmark sample dataset to %s (%d rows)", sample_path, len(df))

    # Feature engineering
    logger.info("Engineering temporal and lag features...")
    df_feat = build_temporal_features(df.lazy()).collect()
    df_feat = build_lag_features(df_feat.lazy(), target_col="unit_sales").collect()

    # Identify features
    ignore_cols = {"date", "unit_sales", "weight", "id"}
    feature_cols = [
        c
        for c in df_feat.columns
        if c not in ignore_cols
        and df_feat[c].dtype.is_numeric()
        and not df_feat[c].dtype.is_temporal()
    ]
    logger.info("Prepared %d feature columns.", len(feature_cols))

    # Drop nulls from lag shift
    df_clean = df_feat.drop_nulls(subset=feature_cols)

    # Generate time series folds
    unique_dates = df_clean["date"].unique().sort().to_list()
    folds = generate_time_series_folds(unique_dates, n_folds=n_folds, horizon_days=horizon)

    fold_scores: list[float] = []
    last_forecaster: LGBMForecaster | None = None

    for i, (train_start, val_start, val_end) in enumerate(folds, 1):
        df_train = df_clean.filter((pl.col("date") >= train_start) & (pl.col("date") < val_start))
        df_val = df_clean.filter((pl.col("date") >= val_start) & (pl.col("date") <= val_end))

        if len(df_train) == 0 or len(df_val) == 0:
            continue

        X_train, y_train, w_train = prepare_feature_matrices(df_train, feature_cols)
        X_val, y_val, w_val = prepare_feature_matrices(df_val, feature_cols)

        forecaster = LGBMForecaster(
            params={
                "objective": "regression",
                "learning_rate": lgb_params.get("learning_rate", 0.05),
                "num_leaves": lgb_params.get("num_leaves", 31),
                "min_child_samples": 5,
                "verbose": -1,
                "n_jobs": 1,
            }
        )

        forecaster.fit(
            X_train=X_train,
            y_train_log1p=y_train,
            weights_train=w_train,
            X_val=X_val,
            y_val_log1p=y_val,
            weights_val=w_val,
            feature_names=feature_cols,
            num_boost_round=100,
            early_stopping_rounds=20,
        )

        preds = forecaster.predict(X_val)
        y_val_natural = np.expm1(y_val)
        score = float(nwrmsle_numpy(y_val_natural, preds, w_val))
        fold_scores.append(score)
        logger.info(
            "Fold %d/%d (%s to %s) - NWRMSLE: %.4f", i, len(folds), val_start, val_end, score
        )
        last_forecaster = forecaster

    mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
    logger.info("Cross-Validation Complete! Mean NWRMSLE: %.4f", mean_score)

    # Save artifacts
    if last_forecaster is not None:
        model_path = output_dir / "lgbm_forecaster.txt"
        last_forecaster.save(model_path)
        logger.info("Saved trained LightGBM model to %s", model_path)

        importances = last_forecaster.get_feature_importance()
        imp_path = output_dir / "feature_importance.json"
        with open(imp_path, "w", encoding="utf-8") as f:
            json.dump(importances, f, indent=2)
        logger.info("Saved feature importance rankings to %s", imp_path)

        feat_cols_path = output_dir / "feature_cols.json"
        with open(feat_cols_path, "w", encoding="utf-8") as f:
            json.dump(feature_cols, f, indent=2)
        logger.info("Saved feature column list to %s", feat_cols_path)

    metrics_path = output_dir / "training_metrics.json"
    results = {
        "mean_nwrmsle": round(mean_score, 4),
        "fold_nwrmsle": [round(s, 4) for s in fold_scores],
        "n_folds": len(fold_scores),
        "horizon_days": horizon,
        "n_features": len(feature_cols),
        "model_type": "LightGBM",
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Saved training metrics summary to %s", metrics_path)

    print("\n" + "=" * 60)
    print("           TRAINING & CROSS-VALIDATION RESULTS")
    print("=" * 60)
    print(f"  Folds Evaluated:    {len(fold_scores)}")
    print(f"  Forecast Horizon:   {horizon} days")
    print(f"  Features Used:      {len(feature_cols)}")
    print(f"  Mean NWRMSLE:       {mean_score:.4f}")
    for k, sc in enumerate(fold_scores, 1):
        print(f"    - Fold {k}:          {sc:.4f}")
    print(f"  Model Artifact:     {output_dir / 'lgbm_forecaster.txt'}")
    print(f"  Metrics Artifact:   {metrics_path}")
    print("=" * 60 + "\n")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train demand forecasting models.")
    parser.add_argument(
        "--config", type=str, default="configs/model_config.yaml", help="Path to YAML config."
    )
    args = parser.parse_args()
    train_pipeline(args.config)
