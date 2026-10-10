from __future__ import annotations

import importlib.util

import numpy as np
import pandas as pd
import pytest
from conftest import make_bronze_frame

import MaltaRock
from MaltaRock.data.features import prepare_features
from MaltaRock.data.sequences import build_sequences
from MaltaRock.exceptions import SchemaError

METRIC_KEYS = {
    "accuracy", "f1_macro", "f1_weighted", "f1_macro_all_classes", "per_class",
    "classification_report", "confusion_matrix", "labels", "absent_classes", "unseen_classes",
}


@pytest.fixture
def split():
    features = prepare_features(make_bronze_frame(n_wells=12, rows_per_well=40), window=3)
    return MaltaRock.split_train_test(features, test_size=0.25, random_state=42)


def test_random_forest_train_predict_evaluate(split):
    X_train, y_train, X_test, y_test = split
    model = MaltaRock.RandomForest(random_state=42, n_estimators=20, n_jobs=1)
    assert model.train(X_train, y_train) is model
    predictions = model.predict(X_test)
    assert len(predictions) == len(X_test)
    assert set(predictions) <= set(y_train)
    results = model.evaluate(X_test, y_test)
    assert METRIC_KEYS <= set(results)
    assert results["accuracy"] > 0.6
    assert results["n_samples"] == len(y_test)
    assert len(results["feature_importances"]) == len(results["feature_names"])
    assert np.array(results["confusion_matrix"]).shape == (len(results["labels"]),) * 2
    for stats in results["per_class"].values():
        assert set(stats) == {"precision", "recall", "f1", "support"}


def test_random_forest_defaults_work(split):
    X_train, y_train, X_test, y_test = split
    results = MaltaRock.RandomForest(random_state=0, n_jobs=1).train(X_train, y_train).evaluate(X_test, y_test)
    assert 0 <= results["f1_macro"] <= 1


def test_absent_classes_are_reported(split):
    X_train, y_train, X_test, y_test = split
    model = MaltaRock.RandomForest(n_estimators=10, n_jobs=1).train(X_train, y_train)
    keep = (y_test != "SILTITO").to_numpy()
    results = model.evaluate(X_test[keep], y_test[keep])
    assert results["per_class"]["SILTITO"]["support"] == 0
    assert "SILTITO" in results["labels"]
    if "SILTITO" not in results["absent_classes"]:  # predicted but absent from y_true
        pytest.fail("absent class not reported")


def test_metrics_report_unseen_classes():
    results = MaltaRock.compute_metrics(["A", "B", "C"], ["A", "B", "B"], labels=["A", "B"])
    assert results["unseen_classes"] == ["C"]
    assert results["per_class"]["C"]["recall"] == 0.0


def test_missing_feature_column_at_predict(split):
    X_train, y_train, X_test, _ = split
    model = MaltaRock.RandomForest(n_estimators=5, n_jobs=1).train(X_train, y_train)
    with pytest.raises(SchemaError):
        model.predict(X_test.drop(columns=["GR"]))


def test_legacy_model_dataset_still_supported(split):
    X_train, y_train, X_test, y_test = split
    data = MaltaRock.ModelDataset(X_train, y_train, X_test, y_test)
    model = MaltaRock.RandomForest(n_estimators=10, n_jobs=1)
    model.train(data)
    assert model.evaluate(data)["n_samples"] == len(y_test)


def test_save_and_load_roundtrip(split, tmp_path):
    X_train, y_train, X_test, _ = split
    model = MaltaRock.RandomForest(n_estimators=10, n_jobs=1).train(X_train, y_train)
    model.save(tmp_path / "rf.pkl")
    restored = MaltaRock.RandomForest().load(tmp_path / "rf.pkl")
    np.testing.assert_array_equal(restored.predict(X_test), model.predict(X_test))
    assert restored.get_params()["n_estimators"] == 10


@pytest.mark.parametrize(
    "name, params",
    [
        ("DecisionTree", {"max_depth": 5}),
        ("KNN", {"n_neighbors": 5, "n_jobs": 1}),
        ("LogisticRegression", {"max_iter": 200, "n_jobs": 1}),
        pytest.param("XGBoost", {"n_estimators": 10, "learning_rate": 0.3, "n_jobs": 1},
                     marks=pytest.mark.skipif(importlib.util.find_spec("xgboost") is None,
                                              reason="xgboost not installed")),
    ],
)
def test_other_tabular_models(split, name, params):
    X_train, y_train, X_test, y_test = split
    model = getattr(MaltaRock, name)(**params).train(X_train, y_train)
    assert METRIC_KEYS <= set(model.evaluate(X_test, y_test))


def test_optuna_uses_only_training_wells(split, monkeypatch):
    pytest.importorskip("optuna")
    X_train, y_train, X_test, _ = split
    seen: set[str] = set()
    original_train = MaltaRock.RandomForest.train

    def spy(self, X, y=None):
        seen.update(MaltaRock.get_groups(X))
        return original_train(self, X, y)

    monkeypatch.setattr(MaltaRock.RandomForest, "train", spy)
    model = MaltaRock.RandomForest(n_jobs=1, random_state=0)
    space = {"n_estimators": [5, 10], "min_samples_leaf": {"low": 1, "high": 4}}
    study = model.optimize(X_train, y_train, search_space=space, n_trials=2, n_splits=3)
    assert len(study.trials) == 2
    assert seen <= set(split.train_wells)
    assert not seen & set(split.test_wells)
    assert model.n_estimators == study.best_params["n_estimators"]
    assert model.min_samples_leaf == study.best_params["min_samples_leaf"]


def test_reference_search_space_loads():
    pytest.importorskip("yaml")
    from MaltaRock.optimization import load_search_space

    space = load_search_space("RandomForest")
    assert space["n_estimators"] == [200, 300, 400, 500]


def test_sequences_never_cross_wells():
    groups = np.array(["A"] * 5 + ["B"] * 4)
    values = np.arange(9, dtype=float)[:, None]
    seqs, rows = build_sequences(values, groups, 3, "centered")
    for seq, row in zip(seqs, rows, strict=True):
        window_rows = seq[:, 0].astype(int)
        assert len(set(groups[window_rows])) == 1 and groups[row] == groups[window_rows[0]]
        assert window_rows[1] == row  # centred
    seqs, rows = build_sequences(values, groups, 3, "trailing")
    for seq, row in zip(seqs, rows, strict=True):
        window_rows = seq[:, 0].astype(int)
        assert len(set(groups[window_rows])) == 1 and groups[row] == groups[window_rows[0]]
        assert row == window_rows[-1] + 1
    with pytest.raises(ValueError):
        build_sequences(values, np.array(list("AABBAAAAA")), 3, "centered")


@pytest.mark.parametrize("name", ["LSTM", "BILSTM"])
def test_sequence_models_smoke(split, name):
    pytest.importorskip("torch")
    X_train, y_train, X_test, y_test = split
    model = getattr(MaltaRock, name)(sequence_length=5, hidden_size=8, num_layers=1, epochs=2,
                                     batch_size=64, num_workers=0, patience=1, device="cpu")
    model.train(X_train, y_train)
    predictions = model.predict(X_test)
    assert isinstance(predictions, pd.Series)
    assert set(MaltaRock.get_groups(predictions)) <= set(split.test_wells)
    results = model.evaluate(X_test, y_test)
    assert METRIC_KEYS <= set(results)
    assert results["n_samples"] + results["n_rows_without_window"] == len(y_test)


def test_config_runner_end_to_end(local_dataset, tmp_path):
    from MaltaRock.run import run_experiment

    config = {
        "model_type": "RandomForest",
        "params": {"n_estimators": [5, 10], "n_jobs": 1},
        "policy": {"name": "tiny", "total_wells": 8, "apply_scaling": True, "apply_presence": True,
                   "apply_rolling_stats": True, "rolling_window": 3, "labels": ["ARENITO", "FOLHELHO"],
                   "test_size": 0.25, "random_state": 42},
        "trials": 2,
    }
    results = run_experiment(config, data_path=local_dataset, cv_folds=2, output_dir=tmp_path)
    experiment = results["experiment"]
    assert experiment["dataset"]["window"] == 3
    assert not set(experiment["train_wells"]) & set(experiment["test_wells"])
    assert (tmp_path / "randomforest-policy-tiny" / "results.json").is_file()
    reloaded = MaltaRock.load_well_assignment(tmp_path / "randomforest-policy-tiny" / "well_split.json")
    assert set(reloaded["test_wells"]) == set(experiment["test_wells"])
