from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import make_bronze_frame
from sklearn.model_selection import train_test_split

import MaltaRock
from MaltaRock.data.features import prepare_features
from MaltaRock.exceptions import SchemaError, SplitError


@pytest.fixture
def features() -> pd.DataFrame:
    return prepare_features(make_bronze_frame(n_wells=10, rows_per_well=20), window=3)


def wells_of(X: pd.DataFrame) -> set[str]:
    return set(MaltaRock.get_groups(X))


def test_partitions_are_disjoint_and_cover_all_wells(features):
    X_train, y_train, X_test, y_test = MaltaRock.split_train_test(features, test_size=0.2, random_state=42)
    assert not wells_of(X_train) & wells_of(X_test)
    assert wells_of(X_train) | wells_of(X_test) == set(features.well)
    assert len(wells_of(X_test)) == 2
    assert len(X_train) + len(X_test) == len(features)


def test_rows_and_labels_stay_aligned(features):
    X_train, y_train, X_test, y_test = MaltaRock.split_train_test(features, random_state=1)
    for X, y in ((X_train, y_train), (X_test, y_test)):
        assert X.index.equals(y.index)
        lookup = features.set_index(["well", "depth"])
        np.testing.assert_array_equal(lookup.loc[X.index, "rock"].to_numpy(), y.to_numpy())
        np.testing.assert_allclose(lookup.loc[X.index, "GR"].to_numpy(), X["GR"].to_numpy(), equal_nan=True)
        # depth ordering within each well
        frame = X.index.to_frame(index=False)
        assert frame.groupby("well")["depth"].apply(lambda d: d.is_monotonic_increasing).all()


def test_metadata_columns_never_in_X(features):
    X_train, _, X_test, _ = MaltaRock.split_train_test(features)
    for X in (X_train, X_test):
        assert not {"well", "depth", "rock"} & set(X.columns)
    X_depth, *_ = MaltaRock.split_train_test(features, include_depth=True)
    assert "depth" in X_depth.columns and "well" not in X_depth.columns and "rock" not in X_depth.columns


def test_fixed_seed_is_deterministic_and_matches_reference(features):
    a = MaltaRock.split_train_test(features, test_size=0.3, random_state=7)
    b = MaltaRock.split_train_test(features.sample(frac=1.0, random_state=0), test_size=0.3, random_state=7)
    assert a.train_wells == b.train_wells and a.test_wells == b.test_wells
    ref_train, ref_test = train_test_split(sorted(features.well.unique()), test_size=0.3, random_state=7)
    assert set(a.train_wells) == set(ref_train) and set(a.test_wells) == set(ref_test)
    c = MaltaRock.split_train_test(features, test_size=0.3, random_state=8)
    assert set(c.test_wells) != set(a.test_wells)


def test_explicit_assignment(features):
    split = MaltaRock.split_train_test(features, train_wells=["W000", "W001"], test_wells=["W009"])
    assert set(split.train_wells) == {"W000", "W001"} and split.test_wells == ["W009"]
    only_test = MaltaRock.split_train_test(features, test_wells=["W003", "W004"])
    assert set(only_test.test_wells) == {"W003", "W004"} and len(only_test.train_wells) == 8
    only_train = MaltaRock.split_train_test(features, train_wells=["W003"])
    assert only_train.train_wells == ["W003"] and len(only_train.test_wells) == 9


@pytest.mark.parametrize(
    "kwargs",
    [
        {"train_wells": ["W000"], "test_wells": ["W000", "W001"]},  # overlap
        {"test_wells": ["NOPE"]},                                   # unknown
        {"train_wells": ["W000", "UNKNOWN"]},                       # unknown
        {"train_wells": list(f"W{i:03d}" for i in range(10))},      # empty test
    ],
)
def test_invalid_explicit_assignment(features, kwargs):
    with pytest.raises(SplitError):
        MaltaRock.split_train_test(features, **kwargs)


def test_invalid_test_size(features):
    with pytest.raises(SplitError):
        MaltaRock.split_train_test(features, test_size=1.5)


def test_requires_metadata_columns(features):
    with pytest.raises(SchemaError):
        MaltaRock.split_train_test(features.drop(columns=["well"]))


def test_save_and_reload_assignment(features, tmp_path):
    split = MaltaRock.split_train_test(features, random_state=3)
    split.save_assignment(tmp_path / "split.json", random_state=3)
    again = MaltaRock.split_train_test(features, **MaltaRock.load_well_assignment(tmp_path / "split.json"))
    assert set(again.train_wells) == set(split.train_wells)
    assert set(again.test_wells) == set(split.test_wells)
    pd.testing.assert_frame_equal(again.X_test, split.X_test)
