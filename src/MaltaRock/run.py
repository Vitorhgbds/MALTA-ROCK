"""Config-driven experiment runner (successor of the reference ``run.py``).

Reads the YAML files in ``MaltaRock/models/configs``: ``policy`` selects the
features and the well split, ``params`` holds fixed hyperparameters or an
Optuna search space. When ``params`` contains a search space, the study is
run with well-grouped cross-validation on the training wells, the final
model is retrained on all training wells with the best parameters and only
then evaluated on the test wells.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from MaltaRock.data.features import fetch_data
from MaltaRock.data.split import load_well_assignment, split_train_test
from MaltaRock.evaluation.metrics import save_results
from MaltaRock.models import get_model_class
from MaltaRock.optimization import optimize

logger = logging.getLogger(__name__)

_POLICY_FETCH_KEYS = ("total_wells", "apply_scaling", "apply_presence", "apply_rolling_stats", "labels")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment configuration."""
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _is_search_space(params: dict[str, Any]) -> bool:
    return any(isinstance(value, (dict, list)) for value in params.values())


def run_experiment(
    config: dict[str, Any],
    data_path: str | Path = "./data",
    revision: str | None = None,
    n_trials: int | None = None,
    cv_folds: int = 3,
    storage: str | None = None,
    output_dir: str | Path | None = None,
    split_file: str | Path | None = None,
) -> dict[str, Any]:
    """Run one experiment configuration and return the test-set results."""
    policy = dict(config.get("policy", {}))
    fetch_kwargs = {key: policy[key] for key in _POLICY_FETCH_KEYS if key in policy}
    if "rolling_window" in policy:
        # The reference DefaultFeatures expected ``window``; ``rolling_window``
        # was silently ignored there, so its rolling-10 runs used window=5.
        fetch_kwargs["window"] = policy["rolling_window"]
    elif "window" in policy:
        fetch_kwargs["window"] = policy["window"]

    features = fetch_data(data_path, revision=revision, **fetch_kwargs)
    split_kwargs: dict[str, Any] = (
        load_well_assignment(split_file)
        if split_file
        else {"test_size": policy.get("test_size", 0.2), "random_state": policy.get("random_state", 42)}
    )
    split = split_train_test(features, include_depth=policy.get("include_depth", False), **split_kwargs)
    X_train, y_train, X_test, y_test = split

    model_type = config["model_type"]
    model_class = get_model_class(model_type)
    params = dict(config.get("params", {}))
    study_name = f"{model_type.lower()}-policy-{policy.get('name', 'default')}"
    if _is_search_space(params):
        study = optimize(
            model_class, X_train, y_train, params,
            n_trials=n_trials or config.get("trials", 1),
            n_splits=cv_folds,
            random_state=policy.get("random_state", 42),
            storage=storage,
            study_name=study_name if storage else None,
        )
        params = study.best_trial.user_attrs["model_params"]

    model = model_class(**params)
    model.train(X_train, y_train)
    results = model.evaluate(X_test, y_test)
    results["experiment"] = {
        "model_type": model_type,
        "params": params,
        "policy": policy,
        "dataset": features.attrs.get("malta_rock"),
        "train_wells": split.train_wells,
        "test_wells": split.test_wells,
    }
    if output_dir is not None:
        out = Path(output_dir) / study_name
        out.mkdir(parents=True, exist_ok=True)
        save_results(results, out / "results.json")
        split.save_assignment(out / "well_split.json", source=str(split_file) if split_file else split_kwargs)
        logger.info("Results written to %s", out)
    else:
        logger.info("Results: %s", json.dumps({k: results[k] for k in ("accuracy", "f1_macro", "f1_weighted")}))
    return results
