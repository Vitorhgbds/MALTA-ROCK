"""MALTA-ROCK: reproducible lithology classification on MALTA-WLOGS.

Typical workflow::

    import MaltaRock

    MaltaRock.download_dataset("./data")
    features = MaltaRock.fetch_data("./data", total_wells=400, window=5,
                                    labels=MaltaRock.CORE15_LABELS)
    X_train, y_train, X_test, y_test = MaltaRock.split_train_test(
        features, test_size=0.2, random_state=42)
    model = MaltaRock.RandomForest(random_state=42)
    model.train(X_train, y_train)
    print(model.evaluate(X_test, y_test))

Attributes are imported lazily, so ``import MaltaRock`` stays light and the
optional ``torch``/``xgboost`` dependencies are only needed by the models
that use them.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"

_EXPORTS: dict[str, str] = {
    # data
    "download_dataset": "MaltaRock.data.download",
    "is_dataset_complete": "MaltaRock.data.download",
    "read_manifest": "MaltaRock.data.download",
    "fetch_data": "MaltaRock.data.features",
    "select_wells": "MaltaRock.data.features",
    "split_train_test": "MaltaRock.data.split",
    "split_wells": "MaltaRock.data.split",
    "get_groups": "MaltaRock.data.split",
    "save_well_assignment": "MaltaRock.data.split",
    "load_well_assignment": "MaltaRock.data.split",
    "TrainTestSplit": "MaltaRock.data.split",
    "build_sequences": "MaltaRock.data.sequences",
    "CORE15_LABELS": "MaltaRock.data.schema",
    "LOG_COLUMNS": "MaltaRock.data.schema",
    "REPO_ID": "MaltaRock.data.schema",
    # evaluation / optimisation
    "compute_metrics": "MaltaRock.evaluation.metrics",
    "save_results": "MaltaRock.evaluation.metrics",
    "optimize": "MaltaRock.optimization",
    # models
    "BaseModel": "MaltaRock.models.base",
    "ModelDataset": "MaltaRock.models.base",
    "get_model_class": "MaltaRock.models",
    "RandomForest": "MaltaRock.models.forest",
    "DecisionTree": "MaltaRock.models.tree",
    "KNN": "MaltaRock.models.knn",
    "LogisticRegression": "MaltaRock.models.logistic",
    "XGBoost": "MaltaRock.models.xgboost",
    "LSTM": "MaltaRock.models.lstm",
    "BILSTM": "MaltaRock.models.bilstm",
    # exceptions
    "MaltaRockError": "MaltaRock.exceptions",
    "DatasetNotFoundError": "MaltaRock.exceptions",
    "DatasetDownloadError": "MaltaRock.exceptions",
    "DatasetIntegrityError": "MaltaRock.exceptions",
    "InvalidPathError": "MaltaRock.exceptions",
    "SchemaError": "MaltaRock.exceptions",
    "SplitError": "MaltaRock.exceptions",
}

__all__ = sorted(_EXPORTS)

if TYPE_CHECKING:
    from MaltaRock.data.download import download_dataset, is_dataset_complete, read_manifest
    from MaltaRock.data.features import fetch_data, select_wells
    from MaltaRock.data.schema import CORE15_LABELS, LOG_COLUMNS, REPO_ID
    from MaltaRock.data.sequences import build_sequences
    from MaltaRock.data.split import (
        TrainTestSplit,
        get_groups,
        load_well_assignment,
        save_well_assignment,
        split_train_test,
        split_wells,
    )
    from MaltaRock.evaluation.metrics import compute_metrics, save_results
    from MaltaRock.exceptions import (
        DatasetDownloadError,
        DatasetIntegrityError,
        DatasetNotFoundError,
        InvalidPathError,
        MaltaRockError,
        SchemaError,
        SplitError,
    )
    from MaltaRock.models import get_model_class
    from MaltaRock.models.base import BaseModel, ModelDataset
    from MaltaRock.models.bilstm import BILSTM
    from MaltaRock.models.forest import RandomForest
    from MaltaRock.models.knn import KNN
    from MaltaRock.models.logistic import LogisticRegression
    from MaltaRock.models.lstm import LSTM
    from MaltaRock.models.tree import DecisionTree
    from MaltaRock.models.xgboost import XGBoost
    from MaltaRock.optimization import optimize


def __getattr__(name: str) -> Any:
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module 'MaltaRock' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted([*globals(), *_EXPORTS])
