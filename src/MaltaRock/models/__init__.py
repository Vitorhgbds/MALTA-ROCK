"""Model implementations.

Every model exposes ``train(X, y)``, ``predict(X)`` and ``evaluate(X, y)``.
Model classes are imported lazily so that optional dependencies (``torch``
for LSTM/BILSTM, ``xgboost`` for XGBoost) are only needed when used.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

from MaltaRock.models.base import BaseModel, ModelDataset, TabularModel

if TYPE_CHECKING:
    from MaltaRock.models.bilstm import BILSTM
    from MaltaRock.models.forest import RandomForest
    from MaltaRock.models.knn import KNN
    from MaltaRock.models.logistic import LogisticRegression
    from MaltaRock.models.lstm import LSTM
    from MaltaRock.models.tree import DecisionTree
    from MaltaRock.models.xgboost import XGBoost

MODEL_MODULES: dict[str, str] = {
    "RandomForest": "MaltaRock.models.forest",
    "DecisionTree": "MaltaRock.models.tree",
    "KNN": "MaltaRock.models.knn",
    "LogisticRegression": "MaltaRock.models.logistic",
    "XGBoost": "MaltaRock.models.xgboost",
    "LSTM": "MaltaRock.models.lstm",
    "BILSTM": "MaltaRock.models.bilstm",
}

__all__ = ["BaseModel", "ModelDataset", "TabularModel", "get_model_class", *MODEL_MODULES]


def get_model_class(name: str) -> type[BaseModel]:
    """Return a model class by its name (the ``model_type`` used in the YAML configs)."""
    try:
        module = importlib.import_module(MODEL_MODULES[name])
    except KeyError:
        raise ValueError(f"Unknown model {name!r}; choose from {sorted(MODEL_MODULES)}") from None
    return getattr(module, name)


def __getattr__(name: str) -> Any:
    if name in MODEL_MODULES:
        return get_model_class(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
