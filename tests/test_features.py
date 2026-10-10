from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from conftest import bronze_repo_files, make_bronze_frame, parquet_bytes

import MaltaRock
from MaltaRock.data import download as dl
from MaltaRock.data.features import add_rolling_statistics, prepare_features
from MaltaRock.data.schema import LOG_COLUMNS, VALID_RANGES
from MaltaRock.exceptions import DatasetNotFoundError, SchemaError


def reference_rolling(df: pd.DataFrame, window: int) -> pd.DataFrame:
    """The reference implementation (slide/models/featurePolicy.py)."""
    df = df.sort_values(["well", "depth"]).copy()
    for f in LOG_COLUMNS:
        df[f"{f}_rolling_mean"] = df.groupby("well")[f].transform(
            lambda x: x.rolling(window=window, min_periods=1).mean())
        df[f"{f}_rolling_std"] = df.groupby("well")[f].transform(
            lambda x: x.rolling(window=window, min_periods=1).std())
    return df


def test_fetch_data_schema_and_column_order(local_dataset):
    features = MaltaRock.fetch_data(local_dataset, window=3)
    expected = (
        ["well", "depth", *LOG_COLUMNS]
        + [f"{c}_present" for c in LOG_COLUMNS]
        + [f"{c}_{s}" for c in LOG_COLUMNS for s in ("rolling_mean", "rolling_std")]
        + ["rock"]
    )
    assert list(features.columns) == expected
    # shard 0 stores DT with the Arrow null type, as some published shards do
    raw = MaltaRock.fetch_data(local_dataset, apply_scaling=False, apply_rolling_stats=False)
    assert raw["DT"].dtype == np.float64
    assert raw["DT"].isna().any() and raw["DT"].notna().any()
    assert features.attrs["malta_rock"]["revision"] == "a" * 40
    assert features[["well", "depth"]].equals(
        features[["well", "depth"]].sort_values(["well", "depth"]).reset_index(drop=True)
    )


def test_fetch_data_is_offline_after_download(local_dataset, monkeypatch):
    monkeypatch.setattr(dl, "HfApi", lambda: (_ for _ in ()).throw(AssertionError("network")))
    assert len(MaltaRock.fetch_data(local_dataset)) > 0


def test_fetch_data_downloads_when_missing(tmp_path, fake_hub, bronze_frame):
    fake_hub.add_revision("a" * 40, bronze_repo_files(bronze_frame), alias="main")
    features = MaltaRock.fetch_data(tmp_path, apply_rolling_stats=False)
    assert len(features) == len(bronze_frame)
    assert fake_hub.info_calls == 1


def test_fetch_data_without_download_raises(tmp_path):
    with pytest.raises(DatasetNotFoundError):
        MaltaRock.fetch_data(tmp_path, download=False)


def test_missing_required_column_raises_schema_error(tmp_path, fake_hub, bronze_frame):
    files = bronze_repo_files(bronze_frame)
    files["Features/train-00001.parquet"] = parquet_bytes(bronze_frame.drop(columns=["rock"]).iloc[:5])
    fake_hub.add_revision("a" * 40, files, alias="main")
    with pytest.raises(SchemaError, match="rock"):
        MaltaRock.fetch_data(tmp_path)


def test_wrong_column_type_raises_schema_error(tmp_path, fake_hub, bronze_frame):
    files = bronze_repo_files(bronze_frame)
    bad = bronze_frame.iloc[:5].copy()
    bad["GR"] = "text"
    files["Features/train-00001.parquet"] = parquet_bytes(bad)
    fake_hub.add_revision("a" * 40, files, alias="main")
    with pytest.raises(SchemaError, match="GR"):
        MaltaRock.fetch_data(tmp_path)


def test_rolling_matches_reference_and_respects_depth_and_wells():
    raw = make_bronze_frame(n_wells=4, rows_per_well=12)
    raw.loc[3, "GR"] = np.nan
    shuffled = raw.sample(frac=1.0, random_state=0)  # unsorted input
    features = prepare_features(shuffled, apply_scaling=False, apply_presence=False, window=4)
    expected = reference_rolling(raw, 4).reset_index(drop=True)
    for column in [c for c in features.columns if "rolling" in c]:
        np.testing.assert_allclose(features[column], expected[column], equal_nan=True)
    # first row of every well only sees itself: mean == value, std is NaN
    first = features.groupby("well").head(1)
    np.testing.assert_allclose(first["SP_rolling_mean"], first["SP"])
    assert first["SP_rolling_std"].isna().all()


def test_rolling_never_crosses_well_boundaries():
    df = pd.DataFrame({
        "well": ["A"] * 3 + ["B"] * 3,
        "depth": [1.0, 2.0, 3.0] * 2,
        **{c: [1.0, 1.0, 1.0, 100.0, 100.0, 100.0] for c in LOG_COLUMNS},
        "rock": ["X"] * 6,
    })
    add_rolling_statistics(df, window=5)
    assert (df.loc[df.well == "B", "GR_rolling_mean"] == 100.0).all()
    assert (df.loc[df.well == "A", "GR_rolling_mean"] == 1.0).all()


def test_rolling_can_be_disabled_and_window_is_configurable(local_dataset):
    plain = MaltaRock.fetch_data(local_dataset, apply_rolling_stats=False, apply_presence=False)
    assert not any("rolling" in c or "present" in c for c in plain.columns)
    w2 = MaltaRock.fetch_data(local_dataset, window=2)
    w8 = MaltaRock.fetch_data(local_dataset, window=8)
    assert not np.allclose(w2["SP_rolling_mean"], w8["SP_rolling_mean"])
    with pytest.raises(ValueError):
        MaltaRock.fetch_data(local_dataset, window=0)


def test_valid_ranges_and_presence():
    raw = make_bronze_frame(n_wells=1, rows_per_well=4)
    raw.loc[0, "GR"] = 250.0      # above range
    raw.loc[1, "RHOB"] = 1.95     # inclusive lower bound
    raw.loc[2, "SP"] = np.nan
    out = prepare_features(raw, apply_rolling_stats=False)
    assert np.isnan(out.loc[0, "GR"]) and out.loc[0, "GR_present"] == 0
    assert out.loc[1, "RHOB"] == 1.95 and out.loc[1, "RHOB_present"] == 1
    assert out.loc[2, "SP_present"] == 0
    for column, (low, high) in VALID_RANGES.items():
        values = out[column].dropna()
        assert ((values >= low) & (values <= high)).all()
    unscaled = prepare_features(raw, apply_scaling=False, apply_rolling_stats=False)
    assert unscaled.loc[0, "GR"] == 250.0


def test_labels_filter_case_insensitive(local_dataset):
    features = MaltaRock.fetch_data(local_dataset, labels=["arenito", "Folhelho"])
    assert set(features["rock"]) == {"ARENITO", "FOLHELHO"}


def test_total_wells_selects_first_sorted_wells(tmp_path, fake_hub):
    frame = make_bronze_frame(n_wells=6)
    frame.loc[frame.well == "W000", "rock"] = "GRANITO"  # no requested label
    shuffled = frame.sample(frac=1.0, random_state=3)
    fake_hub.add_revision("a" * 40, bronze_repo_files(shuffled), alias="main")
    with_labels = MaltaRock.fetch_data(tmp_path, total_wells=3, labels=["ARENITO", "FOLHELHO", "SILTITO"])
    assert sorted(with_labels.well.unique()) == ["W001", "W002", "W003"]
    # without labels W000 still counts (it only has rows with a non-null class here)
    without = MaltaRock.fetch_data(tmp_path, total_wells=3)
    assert sorted(without.well.unique()) == ["W000", "W001", "W002"]


def test_prepare_features_requires_columns():
    with pytest.raises(SchemaError):
        prepare_features(pd.DataFrame({"well": ["A"], "depth": [1.0]}))
