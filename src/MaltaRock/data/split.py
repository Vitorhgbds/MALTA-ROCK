"""Well-wise train/test splitting.

Rows are never split individually: every well (and therefore every depth
sample of that well) belongs to exactly one partition.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from MaltaRock.data.schema import DEPTH, METADATA_COLUMNS, TARGET, WELL
from MaltaRock.exceptions import SchemaError, SplitError

logger = logging.getLogger(__name__)


class TrainTestSplit(NamedTuple):
    """Result of :func:`split_train_test`; unpacks as ``X_train, y_train, X_test, y_test``.

    ``X_*`` are DataFrames containing only predictive features. Their index
    is a ``(well, depth)`` MultiIndex shared with ``y_*`` so that group
    information stays available for group-aware validation and sequence
    models without leaking into the features.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series

    @property
    def train_wells(self) -> list[str]:
        return list(pd.unique(get_groups(self.X_train)))

    @property
    def test_wells(self) -> list[str]:
        return list(pd.unique(get_groups(self.X_test)))

    def save_assignment(self, path: str | os.PathLike[str], **metadata: Any) -> None:
        """Write the train/test well lists to a JSON file (see :func:`load_well_assignment`)."""
        save_well_assignment(path, self.train_wells, self.test_wells, **metadata)


def get_groups(X: pd.DataFrame | pd.Series) -> np.ndarray:
    """Return the well id of every row of an ``X``/``y`` produced by :func:`split_train_test`."""
    if isinstance(X.index, pd.MultiIndex) and WELL in X.index.names:
        return np.asarray(X.index.get_level_values(WELL))
    if X.index.name == WELL:
        return np.asarray(X.index)
    raise SchemaError(
        "Cannot determine well groups: expected an index level named 'well' "
        "(as produced by MaltaRock.split_train_test)."
    )


def split_wells(
    wells: Iterable[str],
    test_size: float | int = 0.2,
    random_state: int | None = 42,
) -> tuple[list[str], list[str]]:
    """Randomly assign unique wells to train and test.

    The wells are sorted first and then passed to
    :func:`sklearn.model_selection.train_test_split`, which reproduces the
    reference implementation (its well list was sorted by the SQL
    ``ORDER BY well``), so the assignment is identical for a given seed.
    """
    unique = sorted(set(wells))
    if len(unique) < 2:
        raise SplitError(f"At least two wells are required for a split, got {len(unique)}.")
    try:
        train, test = train_test_split(unique, test_size=test_size, random_state=random_state)
    except ValueError as exc:
        raise SplitError(str(exc)) from exc
    return list(train), list(test)


def _resolve_explicit(
    available: set[str],
    train_wells: Sequence[str] | None,
    test_wells: Sequence[str] | None,
) -> tuple[list[str], list[str]]:
    train = list(dict.fromkeys(train_wells)) if train_wells is not None else None
    test = list(dict.fromkeys(test_wells)) if test_wells is not None else None
    for name, wells in (("train_wells", train), ("test_wells", test)):
        if wells is None:
            continue
        unknown = sorted(set(wells) - available)
        if unknown:
            raise SplitError(f"{name} contains {len(unknown)} unknown well(s): {unknown[:10]}")
    if train is not None and test is not None:
        overlap = sorted(set(train) & set(test))
        if overlap:
            raise SplitError(f"{len(overlap)} well(s) appear in both partitions: {overlap[:10]}")
        excluded = available - set(train) - set(test)
        if excluded:
            logger.warning("%d well(s) are in neither partition and are dropped.", len(excluded))
    elif train is not None:
        test = sorted(available - set(train))
    else:
        assert test is not None
        train = sorted(available - set(test))
    if not train or not test:
        raise SplitError("Both partitions must contain at least one well.")
    return train, test


def _partition(
    features: pd.DataFrame,
    wells: Sequence[str],
    feature_columns: list[str],
) -> tuple[pd.DataFrame, pd.Series]:
    part = features[features[WELL].isin(wells)].sort_values([WELL, DEPTH], kind="mergesort")
    index = pd.MultiIndex.from_arrays([part[WELL].to_numpy(), part[DEPTH].to_numpy()], names=[WELL, DEPTH])
    X = part[feature_columns].set_axis(index, axis=0)
    y = part[TARGET].set_axis(index, axis=0)
    return X, y


def split_train_test(
    features: pd.DataFrame,
    test_size: float | int = 0.2,
    random_state: int | None = 42,
    *,
    train_wells: Sequence[str] | None = None,
    test_wells: Sequence[str] | None = None,
    include_depth: bool = False,
) -> TrainTestSplit:
    """Split a feature table by unique well.

    Either a random split (``test_size``/``random_state``) or an explicit
    assignment (``train_wells`` and/or ``test_wells``) is used. With only one
    explicit list, every other well goes to the other partition.

    Args:
        features: Output of :func:`MaltaRock.fetch_data` (needs ``well``,
            ``depth`` and ``rock`` columns).
        test_size: Fraction (float) or number (int) of *wells* in the test set.
        random_state: Seed for the random well assignment.
        train_wells: Explicit training wells.
        test_wells: Explicit test wells.
        include_depth: Also use ``depth`` as a feature column. The reference
            implementation did this; it is disabled by default because depth
            is sample metadata (see README, "Methodological notes").

    Returns:
        A :class:`TrainTestSplit` that unpacks as
        ``X_train, y_train, X_test, y_test``. Rows are sorted by well and
        depth. ``X`` never contains ``well`` or ``rock`` (and contains
        ``depth`` only with ``include_depth=True``); the well and depth of
        every row are in the ``(well, depth)`` index.

    Raises:
        SchemaError: ``features`` lacks the metadata columns.
        SplitError: The assignment is invalid (unknown or overlapping wells,
            empty partition, invalid ``test_size``).
    """
    missing = [column for column in METADATA_COLUMNS if column not in features.columns]
    if missing:
        raise SchemaError(f"features is missing column(s): {', '.join(missing)}")
    available = set(features[WELL].dropna().unique().tolist())

    if train_wells is None and test_wells is None:
        train, test = split_wells(available, test_size, random_state)
    else:
        train, test = _resolve_explicit(available, train_wells, test_wells)

    feature_columns = [column for column in features.columns if column not in METADATA_COLUMNS]
    if include_depth:
        feature_columns = [DEPTH, *feature_columns]
    if not feature_columns:
        raise SchemaError("features does not contain any predictive column.")

    X_train, y_train = _partition(features, train, feature_columns)
    X_test, y_test = _partition(features, test, feature_columns)
    logger.info(
        "Well-wise split: %d train wells (%d rows), %d test wells (%d rows).",
        len(train), len(X_train), len(test), len(X_test),
    )
    return TrainTestSplit(X_train, y_train, X_test, y_test)


def save_well_assignment(
    path: str | os.PathLike[str],
    train_wells: Sequence[str],
    test_wells: Sequence[str],
    **metadata: Any,
) -> None:
    """Save a train/test well assignment as JSON for later reuse."""
    payload = {"train_wells": sorted(train_wells), "test_wells": sorted(test_wells), "metadata": metadata}
    Path(path).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def load_well_assignment(path: str | os.PathLike[str]) -> dict[str, list[str]]:
    """Load an assignment saved by :func:`save_well_assignment`.

    Returns:
        ``{"train_wells": [...], "test_wells": [...]}``, ready to be passed as
        keyword arguments to :func:`split_train_test`.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    try:
        return {"train_wells": list(payload["train_wells"]), "test_wells": list(payload["test_wells"])}
    except (KeyError, TypeError) as exc:
        raise SplitError(f"{path} is not a valid well assignment file.") from exc
