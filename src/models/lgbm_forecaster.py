"""LightGBM forecaster wrapper optimized for the NWRMSLE metric.

Mathematical formulation and loss alignment:
---------------------------------------------
The official NWRMSLE metric is:
    NWRMSLE = sqrt( sum( w_i * (ln(y_hat_i + 1) - ln(y_i + 1))^2 ) / sum(w_i) )

When the target is trained in log1p space:
    z_i = ln(y_i + 1)
and the model predicts:
    z_hat_i = ln(y_hat_i + 1)

The objective becomes minimizing the Weighted Mean Squared Error (WMSE):
    L(z_hat, z) = sum( w_i * (z_hat_i - z_i)^2 )

Thus, standard LightGBM regression (L2 loss) with `sample_weight=w_i` natively
optimizes the exact surrogate gradient for NWRMSLE without custom Hessian
instabilities, while early stopping is governed directly by the exact NWRMSLE metric.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import lightgbm as lgb
import numpy as np

from src.models.metrics import nwrmsle_numpy

if TYPE_CHECKING:
    import numpy.typing as npt

logger = logging.getLogger(__name__)


def lgb_nwrmsle_eval(
    preds: npt.NDArray[np.float64],
    dataset: lgb.Dataset,
) -> tuple[str, float, bool]:
    """Custom LightGBM evaluation metric computing exact NWRMSLE on raw validation arrays.

    Parameters
    ----------
    preds : NDArray[float64]
        Predictions in log1p space.
    dataset : lgb.Dataset
        Validation dataset containing ground truth targets and sample weights.

    Returns
    -------
    tuple[str, float, bool]
        ('nwrmsle', error_value, is_higher_better=False)
    """
    labels = dataset.get_label()
    weights = dataset.get_weight()

    # Reconstruct original units: y = exp(z) - 1
    y_true = np.expm1(np.clip(labels, a_min=0.0, a_max=None))
    y_pred = np.expm1(np.clip(preds, a_min=0.0, a_max=None))

    score = nwrmsle_numpy(y_true=y_true, y_pred=y_pred, weights=weights, clip_negative=True)
    return "nwrmsle", float(score), False


class LGBMForecaster:
    """Production wrapper for LightGBM time-series demand forecasting."""

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        categorical_features: list[str] | None = None,
    ) -> None:
        self.params: dict[str, Any] = params or {
            "objective": "regression",
            "metric": "None",  # Use custom eval metric exclusively
            "boosting_type": "gbdt",
            "num_leaves": 127,
            "learning_rate": 0.03,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "min_child_samples": 50,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "n_jobs": -1,
            "verbose": -1,
        }
        self.categorical_features: list[str] = categorical_features or []
        self.booster: lgb.Booster | None = None
        self.feature_names: list[str] = []
        self.best_iteration: int = 0
        self.best_score: float = float("inf")

    def fit(
        self,
        X_train: npt.NDArray[np.float64] | Any,
        y_train_log1p: npt.NDArray[np.float64],
        weights_train: npt.NDArray[np.float64],
        X_val: npt.NDArray[np.float64] | Any | None = None,
        y_val_log1p: npt.NDArray[np.float64] | None = None,
        weights_val: npt.NDArray[np.float64] | None = None,
        feature_names: list[str] | None = None,
        num_boost_round: int = 1500,
        early_stopping_rounds: int = 100,
    ) -> LGBMForecaster:
        """Train LightGBM with sample weights and custom NWRMSLE early stopping."""
        if feature_names:
            self.feature_names = feature_names

        train_data = lgb.Dataset(
            data=X_train,
            label=y_train_log1p,
            weight=weights_train,
            feature_name=self.feature_names or "auto",
            categorical_feature=self.categorical_features or "auto",
            free_raw_data=False,
        )

        valid_sets = [train_data]
        valid_names = ["train"]

        if X_val is not None and y_val_log1p is not None:
            val_data = lgb.Dataset(
                data=X_val,
                label=y_val_log1p,
                weight=weights_val if weights_val is not None else np.ones_like(y_val_log1p),
                reference=train_data,
                feature_name=self.feature_names or "auto",
                categorical_feature=self.categorical_features or "auto",
                free_raw_data=False,
            )
            valid_sets.append(val_data)
            valid_names.append("valid")

        callbacks = [
            lgb.early_stopping(stopping_rounds=early_stopping_rounds, verbose=False),
            lgb.log_evaluation(period=100),
        ]

        logger.info("Training LightGBM forecaster with %d max rounds...", num_boost_round)
        self.booster = lgb.train(
            params=self.params,
            train_set=train_data,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=lgb_nwrmsle_eval,
            callbacks=callbacks,  # type: ignore[arg-type]
        )

        self.best_iteration = self.booster.best_iteration
        if "valid" in self.booster.best_score and "nwrmsle" in self.booster.best_score["valid"]:
            self.best_score = float(self.booster.best_score["valid"]["nwrmsle"])
            logger.info(
                "Best iteration: %d with validation NWRMSLE: %.5f",
                self.best_iteration,
                self.best_score,
            )

        return self

    def predict_log1p(self, X: npt.NDArray[np.float64] | Any) -> npt.NDArray[np.float64]:
        """Predict target in log1p space."""
        if self.booster is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self.booster.predict(X, num_iteration=self.best_iteration)
        return np.asarray(preds, dtype=np.float64)

    def predict(
        self, X: npt.NDArray[np.float64] | Any, clip_zero: bool = True
    ) -> npt.NDArray[np.float64]:
        """Predict sales in natural units: y_hat = max(exp(z_hat) - 1, 0)."""
        preds_log1p = self.predict_log1p(X)
        y_hat = np.expm1(preds_log1p)
        if clip_zero:
            y_hat = np.clip(y_hat, a_min=0.0, a_max=None)
        return y_hat

    def get_feature_importance(self, importance_type: str = "gain") -> dict[str, float]:
        """Return feature importance sorted descending."""
        if self.booster is None:
            raise RuntimeError("Model is not fitted yet.")
        importances = self.booster.feature_importance(importance_type=importance_type)
        names = self.booster.feature_name()
        return dict(
            sorted(
                zip(names, [float(x) for x in importances], strict=False),
                key=lambda kv: kv[1],
                reverse=True,
            )
        )

    def save(self, model_path: str | Path) -> None:
        """Save model booster to disk."""
        if self.booster is None:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(model_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.booster.save_model(str(path))
        logger.info("Saved LightGBM model to %s", path)

    @classmethod
    def load(cls, model_path: str | Path) -> LGBMForecaster:
        """Load trained LightGBM booster from disk."""
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model file not found at {path}")
        instance = cls()
        instance.booster = lgb.Booster(model_file=str(path))
        instance.feature_names = instance.booster.feature_name()
        return instance
