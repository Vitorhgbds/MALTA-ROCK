"""Shared model interface: ``train(X, y)``, ``predict(X)`` and ``evaluate(X, y)``."""

from __future__ import annotations

import inspect
import logging
import os
import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, NamedTuple, Self

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler

from MaltaRock.evaluation.metrics import compute_metrics
from MaltaRock.exceptions import SchemaError

if TYPE_CHECKING:
    import optuna

logger = logging.getLogger(__name__)


class ModelDataset(NamedTuple):
    """Legacy container used by the reference implementation.

    Models still accept it: ``model.train(dataset)`` uses ``train`` /
    ``train_target`` and ``model.evaluate(dataset)`` uses ``test`` /
    ``test_target``.
    """

    train: Any
    train_target: Any
    test: Any
    test_target: Any


def _unpack(X: Any, y: Any, part: str) -> tuple[Any, Any]:
    if isinstance(X, ModelDataset):
        return (X.train, X.train_target) if part == "train" else (X.test, X.test_target)
    if y is None:
        raise TypeError("y is required unless X is a ModelDataset")
    return X, y


def _drop_missing_targets(X: Any, y: Any) -> tuple[Any, Any]:
    """Remove rows with a missing target, as every reference model did."""
    y_series = y if isinstance(y, pd.Series) else pd.Series(np.asarray(y, dtype=object))
    if len(X) != len(y_series):
        raise ValueError(f"X and y differ in length ({len(X)} != {len(y_series)})")
    mask = y_series.notna().to_numpy()
    if mask.all():
        return X, y_series
    X_kept = X[mask] if isinstance(X, (pd.DataFrame, np.ndarray)) else np.asarray(X)[mask]
    return X_kept, y_series[mask]


class BaseModel(ABC):
    """Common interface of every MaltaRock model."""

    model: Any = None

    @abstractmethod
    def train(self, X: Any, y: Any = None) -> Self:
        """Fit the model on training wells; returns ``self``."""

    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Predict class names for ``X``."""

    @abstractmethod
    def evaluate(self, X: Any, y: Any = None) -> dict[str, Any]:
        """Evaluate on held-out wells; see :func:`MaltaRock.evaluation.compute_metrics`."""

    # -- hyperparameters ----------------------------------------------------
    def get_params(self) -> dict[str, Any]:
        """Return the constructor arguments of this model."""
        signature = inspect.signature(type(self).__init__)
        return {
            name: getattr(self, name)
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind not in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD)
            and hasattr(self, name)
        }

    def set_params(self, **params: Any) -> Self:
        """Update constructor arguments (takes effect on the next ``train``)."""
        valid = self.get_params()
        unknown = set(params) - set(valid)
        if unknown:
            raise ValueError(f"Unknown parameter(s) for {type(self).__name__}: {sorted(unknown)}")
        for name, value in params.items():
            setattr(self, name, value)
        return self

    # -- persistence --------------------------------------------------------
    def save(self, file_path: str | os.PathLike[str]) -> None:
        """Pickle the fitted model (estimator, preprocessors and parameters)."""
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        with open(file_path, "wb") as handle:
            pickle.dump({"class": type(self).__name__, "state": self.__getstate__()}, handle)
        logger.info("Model saved to %s", file_path)

    def load(self, file_path: str | os.PathLike[str]) -> Self:
        """Restore a model saved with :meth:`save`. Only load trusted files."""
        with open(file_path, "rb") as handle:
            payload = pickle.load(handle)
        if payload.get("class") != type(self).__name__:
            raise TypeError(f"{file_path} holds a {payload.get('class')}, not a {type(self).__name__}")
        self.__setstate__(payload["state"])
        logger.info("Model loaded from %s", file_path)
        return self

    def __getstate__(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)

    # -- hyperparameter optimisation -----------------------------------------
    def optimize(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        search_space: dict[str, Any] | None = None,
        n_trials: int = 15,
        **kwargs: Any,
    ) -> optuna.Study:
        """Tune hyperparameters with group-aware CV on the training wells.

        The best parameters are applied with :meth:`set_params`; call
        :meth:`train` afterwards to fit the final model. See
        :func:`MaltaRock.optimization.optimize` for the keyword arguments.
        """
        from MaltaRock.optimization import optimize

        study = optimize(type(self), X, y, search_space, n_trials=n_trials,
                         base_params=self.get_params(), **kwargs)
        self.set_params(**study.best_trial.user_attrs["model_params"])
        return study


class TabularModel(BaseModel):
    """A classical estimator preceded by median imputation and standardisation.

    This is the preprocessing of every reference tabular model:
    ``SimpleImputer(strategy="median")`` → ``StandardScaler`` →
    estimator, with targets encoded by ``LabelEncoder``. All three
    preprocessors are fitted on the training data only.
    """

    def __init__(self) -> None:
        self.model = None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        self.feature_names_: list[str] | None = None

    @abstractmethod
    def _make_estimator(self) -> Any:
        """Build the unfitted estimator from the current parameters."""

    @staticmethod
    def _without_none(**params: Any) -> dict[str, Any]:
        """Drop ``None`` values so that the estimator's own defaults apply."""
        return {name: value for name, value in params.items() if value is not None}

    def _check_features(self, X: Any) -> Any:
        if self.feature_names_ is None or not isinstance(X, pd.DataFrame):
            return X
        missing = [name for name in self.feature_names_ if name not in X.columns]
        if missing:
            raise SchemaError(f"X is missing feature column(s) seen during training: {missing}")
        return X[self.feature_names_]

    def _transform(self, X: Any) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        return self.scaler.transform(self.imputer.transform(self._check_features(X)))

    def train(self, X: Any, y: Any = None) -> Self:
        """Fit imputer, scaler, label encoder and estimator on ``(X, y)``.

        Args:
            X: Training features (DataFrame recommended) or a legacy
                :class:`ModelDataset`.
            y: Training targets (class names).
        """
        X, y = _drop_missing_targets(*_unpack(X, y, "train"))
        logger.info("Training %s on %d rows.", type(self).__name__, len(X))
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        self.encoder = LabelEncoder()
        X_scaled = self.scaler.fit_transform(self.imputer.fit_transform(X))
        y_encoded = self.encoder.fit_transform(np.asarray(y, dtype=object))
        self.model = self._make_estimator()
        self.model.fit(X_scaled, y_encoded)
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict class names for every row of ``X``."""
        return self.encoder.inverse_transform(self.model_predict(self._transform(X)))

    def model_predict(self, X_scaled: np.ndarray) -> np.ndarray:
        return self.model.predict(X_scaled)

    def evaluate(self, X: Any, y: Any = None) -> dict[str, Any]:
        """Evaluate on ``(X, y)``; rows with a missing target are ignored.

        Returns:
            The metrics of :func:`~MaltaRock.evaluation.compute_metrics`
            (``accuracy``, ``f1_macro``, ``f1_weighted``, ``per_class``,
            ``confusion_matrix``, ...) plus model-specific extras.
        """
        X, y = _drop_missing_targets(*_unpack(X, y, "test"))
        predictions = self.predict(X)
        results = compute_metrics(np.asarray(y, dtype=object), predictions, labels=list(self.encoder.classes_))
        results.update(self._extra_results())
        logger.info(
            "%s — accuracy %.4f | F1 macro %.4f | F1 weighted %.4f",
            type(self).__name__, results["accuracy"], results["f1_macro"], results["f1_weighted"],
        )
        return results

    def _extra_results(self) -> dict[str, Any]:
        return {}

    def _input_feature_names(self) -> list[str] | None:
        """Feature names after imputation (all-NaN training columns are dropped)."""
        try:
            return [str(name) for name in self.imputer.get_feature_names_out()]
        except (AttributeError, ValueError):
            return None
