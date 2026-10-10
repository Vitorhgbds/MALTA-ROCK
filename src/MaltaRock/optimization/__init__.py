"""Optuna hyperparameter optimisation with well-grouped cross-validation.

The reference ``run.py`` scored every Optuna trial on the held-out *test*
wells, which leaks test information into model selection. Here each trial
is scored by ``GroupKFold`` cross-validation over the *training* wells only;
the test wells are never seen until the final ``evaluate`` call.

Search spaces use the format of the reference YAML files
(``models/configs/hyperparam-tuning/*.yaml``, key ``params``):

* ``{"low": a, "high": b[, "step": s, "log": bool]}`` → ``suggest_int`` when
  ``low`` is an int, ``suggest_float`` when it is a float;
* a list → ``suggest_categorical``;
* any other value → fixed parameter.
"""

from __future__ import annotations

import logging
from importlib import resources
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from MaltaRock.data.split import get_groups

if TYPE_CHECKING:
    import optuna

    from MaltaRock.models.base import BaseModel

logger = logging.getLogger(__name__)

__all__ = ["load_search_space", "optimize", "suggest_params"]

_TUNING_FILES = {
    "RandomForest": "forest.yaml",
    "DecisionTree": "decision-tree.yaml",
    "KNN": "knn.yaml",
    "LogisticRegression": "logistic-regression.yaml",
    "XGBoost": "xgboost.yaml",
    "LSTM": "lstm.yaml",
    "BILSTM": "bilstm.yaml",
}


def load_search_space(model_name: str) -> dict[str, Any]:
    """Load the reference search space (``params``) for ``model_name``."""
    import yaml

    filename = _TUNING_FILES[model_name]
    config_file = resources.files("MaltaRock.models") / "configs" / "hyperparam-tuning" / filename
    return dict(yaml.safe_load(config_file.read_text(encoding="utf-8"))["params"])


def suggest_params(trial: optuna.Trial, search_space: dict[str, Any]) -> dict[str, Any]:
    """Sample one parameter set (same rules as the reference ``load_model_params``)."""
    params: dict[str, Any] = {}
    for name, spec in search_space.items():
        if isinstance(spec, dict):
            low = spec.get("low")
            if isinstance(low, int) and not isinstance(low, bool) and isinstance(spec.get("high"), int):
                params[name] = trial.suggest_int(name, **spec)
            else:
                params[name] = trial.suggest_float(name, **spec)
        elif isinstance(spec, list):
            params[name] = trial.suggest_categorical(name, spec)
        else:
            params[name] = spec
    return params


def optimize(
    model_class: type[BaseModel],
    X: pd.DataFrame,
    y: pd.Series,
    search_space: dict[str, Any] | None = None,
    n_trials: int = 15,
    metric: str = "f1_macro",
    n_splits: int = 3,
    random_state: int | None = 42,
    groups: Any = None,
    base_params: dict[str, Any] | None = None,
    storage: str | None = None,
    study_name: str | None = None,
) -> optuna.Study:
    """Run an Optuna study using only the given (training) wells.

    Args:
        model_class: A MaltaRock model class, e.g. ``MaltaRock.RandomForest``.
        X: Training features with the ``(well, depth)`` index from
            :func:`MaltaRock.split_train_test` (or pass ``groups``).
        y: Training targets.
        search_space: Parameter specification; defaults to the reference
            search space of ``model_class``.
        n_trials: Trial budget.
        metric: Key of the evaluation results to maximise.
        n_splits: Number of ``GroupKFold`` folds over wells.
        random_state: Seed for the fold assignment and the TPE sampler.
        groups: Well id per row, if ``X`` has no ``well`` index level.
        base_params: Fixed constructor arguments, overridden by sampled ones.
        storage: Optuna storage URL (e.g. ``"sqlite:///optuna.db"``).
        study_name: Study name (required to resume a persisted study).

    Returns:
        The finished :class:`optuna.Study`. ``study.best_params`` holds the
        best sampled parameters and ``study.best_trial.user_attrs["model_params"]``
        the complete constructor arguments of the best trial.
    """
    import optuna

    groups = np.asarray(groups) if groups is not None else get_groups(X)
    if len(np.unique(groups)) < n_splits:
        raise ValueError(f"Need at least {n_splits} training wells for {n_splits}-fold group CV.")
    if search_space is None:
        search_space = load_search_space(model_class.__name__)
    folds = list(GroupKFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(X, y, groups))

    def objective(trial: optuna.Trial) -> float:
        params = {**(base_params or {}), **suggest_params(trial, search_space)}
        scores = []
        for fold, (fit_idx, val_idx) in enumerate(folds):
            model = model_class(**params)
            model.train(X.iloc[fit_idx], y.iloc[fit_idx])
            results = model.evaluate(X.iloc[val_idx], y.iloc[val_idx])
            scores.append(results[metric])
            trial.report(float(np.mean(scores)), fold)
        score = float(np.mean(scores))
        trial.set_user_attr("fold_scores", scores)
        trial.set_user_attr("model_params", params)
        logger.info("Trial %d: %s=%.4f params=%s", trial.number, metric, score, params)
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=random_state),
        storage=storage,
        study_name=study_name,
        load_if_exists=storage is not None,
    )
    study.optimize(objective, n_trials=n_trials, gc_after_trial=True)
    logger.info("Best %s=%.4f with %s", metric, study.best_value, study.best_params)
    return study
