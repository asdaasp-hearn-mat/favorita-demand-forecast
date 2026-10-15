"""Unit tests for LightGBM forecaster wrapper and Hierarchical reconciliation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.models.hierarchical import HierarchicalReconciler
from src.models.lgbm_forecaster import LGBMForecaster


class TestHierarchicalReconciler:
    """Test suite for hierarchical summing matrices and reconciliation algorithms."""

    def test_summing_matrix_dimensions(self) -> None:
        """Matrix S dimensions must equal (1 + n_stores + n_families + n_bottom, n_bottom)."""
        stores = [1, 2]
        families = ["GROCERY", "MEATS", "PRODUCE"]
        n_bottom = len(stores) * len(families)  # 6
        n_total = 1 + len(stores) + len(families) + n_bottom  # 1 + 2 + 3 + 6 = 12

        S, all_labels, bottom_labels = HierarchicalReconciler.build_summing_matrix_store_family(
            stores, families
        )

        assert S.shape == (n_total, n_bottom)
        assert len(all_labels) == n_total
        assert len(bottom_labels) == n_bottom

        # Top row (Total) must sum all bottom series (sum of row == n_bottom)
        assert np.sum(S[0, :]) == pytest.approx(n_bottom)

        # Store rows must sum across all families for that store (sum of row == n_families)
        assert np.sum(S[1, :]) == pytest.approx(len(families))
        assert np.sum(S[2, :]) == pytest.approx(len(families))

        # Bottom rows must form an identity matrix
        assert np.allclose(S[-n_bottom:, :], np.eye(n_bottom))

    def test_mint_ols_reconciliation_coherence(self) -> None:
        """Reconciled forecasts must be mathematically coherent: S @ y_tilde_bottom == y_tilde_total."""
        stores = [1, 2]
        families = ["DAIRY", "DELI"]
        S, _, _ = HierarchicalReconciler.build_summing_matrix_store_family(stores, families)

        # Generate incoherent base forecasts
        rng = np.random.default_rng(seed=42)
        y_hat_incoherent = np.asarray(rng.uniform(50.0, 150.0, size=S.shape[0]), dtype=np.float64)

        reconciled = HierarchicalReconciler.reconcile_mint_ols(
            S, y_hat_incoherent, clip_non_negative=True
        )

        n_bottom = len(stores) * len(families)
        y_tilde_bottom = reconciled[-n_bottom:]

        # Test coherence: top rows must equal matrix product S @ y_tilde_bottom
        expected_coherent = S @ y_tilde_bottom
        assert np.allclose(reconciled, expected_coherent, atol=1e-7)

    def test_bottom_up_aggregation(self) -> None:
        """Bottom-up aggregation must match grouped sum exactly."""
        df_bottom = pl.DataFrame(
            {
                "date": ["2017-08-01", "2017-08-01", "2017-08-01"],
                "store_nbr": [1, 2, 3],
                "family": ["MEATS", "MEATS", "MEATS"],
                "prediction": [10.0, 25.0, 40.0],
            }
        )
        agg = HierarchicalReconciler.bottom_up_aggregate(
            df_bottom, prediction_col="prediction", hierarchy_group_cols=["date", "family"]
        )
        assert agg["prediction_aggregated"][0] == 75.0


class TestLGBMForecaster:
    """Test suite for LGBMForecaster model training and inference."""

    def test_fit_and_predict_synthetic(self, tmp_path: Path) -> None:
        """Verify model training, non-negative predictions, and model persistence."""
        rng = np.random.default_rng(seed=42)
        n_samples = 200

        X = rng.normal(0, 1, size=(n_samples, 5))
        y = np.clip(10.0 + 3.0 * X[:, 0] + rng.normal(0, 1, size=n_samples), 0, None)
        y_log1p = np.log1p(y)
        weights = np.ones(n_samples)

        forecaster = LGBMForecaster(
            params={
                "objective": "regression",
                "learning_rate": 0.1,
                "num_leaves": 15,
                "min_child_samples": 5,
                "verbose": -1,
                "n_jobs": 1,
            }
        )

        forecaster.fit(
            X_train=X[:150],
            y_train_log1p=y_log1p[:150],
            weights_train=weights[:150],
            X_val=X[150:],
            y_val_log1p=y_log1p[150:],
            weights_val=weights[150:],
            feature_names=[f"feat_{i}" for i in range(5)],
            num_boost_round=30,
            early_stopping_rounds=10,
        )

        preds = forecaster.predict(X[150:], clip_zero=True)
        assert preds.shape == (50,)
        assert np.all(preds >= 0.0)

        # Feature importance check
        fi = forecaster.get_feature_importance()
        assert len(fi) == 5
        assert "feat_0" in fi

        # Model persistence check
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        model_file = models_dir / "lgbm_test.txt"
        forecaster.save(model_file)
        loaded = LGBMForecaster.load(model_file)
        loaded_preds = loaded.predict(X[150:])
        assert np.allclose(preds, loaded_preds)
