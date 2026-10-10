"""Feature preparation reproducing the reference ``DefaultFeatures`` policy.

The reference implementation (``GeoAI-ANP-DataExtractor/slide/models``)
built the feature table with one SQL query followed by pandas
transformations. :func:`fetch_data` reproduces it step by step:

1. **Well subset** (``total_wells``): the first ``N`` distinct wells in
   ascending (binary) well-id order. The SQL ``SELECT DISTINCT well ...
   LIMIT N`` was executed through the ``(well, depth)`` index, so this order
   is what the paper's experiments used. With ``labels`` the candidate wells
   are those having at least one row of a requested class; without
   ``labels`` every well is a candidate (as in the SQL).
2. **Target-label filter**: rows whose ``rock`` matches one of ``labels``
   (ASCII case-insensitive, like SQLite ``lower()``); without ``labels``,
   rows with a non-null ``rock``. Rows with a null ``well`` are dropped
   (the SQL ``JOIN ... USING (well)`` dropped them).
3. **Outlier handling** (``apply_scaling``): values outside the physically
   plausible ranges in :data:`~MaltaRock.data.schema.VALID_RANGES` become
   NaN. Despite its name, this flag never rescales values.
4. Sorting by ``well`` then ``depth``.
5. **Presence indicators** (``apply_presence``): ``<LOG>_present`` is 1 when
   the (range-filtered) value is present, 0 otherwise.
6. **Rolling statistics** (``apply_rolling_stats``): ``<LOG>_rolling_mean``
   and ``<LOG>_rolling_std`` over a trailing window of ``window`` rows with
   ``min_periods=1``, computed independently for each well in depth order.

All these steps are deterministic and use only values from the same row or
the same well, so they can safely run before the train/test split. Missing
values are *not* imputed here: imputation and standardisation are learned
from the training wells only, inside each model.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Iterable

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc

from MaltaRock.data.loader import ensure_local_dataset, iter_bronze_tables
from MaltaRock.data.schema import (
    DEPTH,
    LOG_COLUMNS,
    REQUIRED_COLUMNS,
    TARGET,
    VALID_RANGES,
    WELL,
)
from MaltaRock.exceptions import SchemaError

logger = logging.getLogger(__name__)

PRESENCE_SUFFIX = "_present"
ROLLING_MEAN_SUFFIX = "_rolling_mean"
ROLLING_STD_SUFFIX = "_rolling_std"


def _normalise_labels(labels: Iterable[str] | None) -> list[str] | None:
    if labels is None:
        return None
    if isinstance(labels, str):
        raise TypeError("labels must be a sequence of class names, not a single string")
    normalised = sorted({pc.ascii_lower(pa.scalar(str(label))).as_py() for label in labels})
    if not normalised:
        raise ValueError("labels must contain at least one class name")
    return normalised


def _label_mask(table: pa.Table, lowered_labels: list[str] | None) -> pa.Array:
    rock = table.column(TARGET)
    if lowered_labels is None:
        return pc.is_valid(rock)
    return pc.fill_null(pc.is_in(pc.ascii_lower(rock), value_set=pa.array(lowered_labels)), False)


def select_wells(
    path: str | os.PathLike[str],
    total_wells: int,
    labels: Iterable[str] | None = None,
    revision: str | None = None,
    download: bool = True,
) -> list[str]:
    """Return the ``total_wells`` wells selected by the reference SQL query.

    See the module documentation for the exact ordering rule.
    """
    if total_wells < 1:
        raise ValueError("total_wells must be a positive integer")
    lowered = _normalise_labels(labels)
    manifest = ensure_local_dataset(path, revision, download)
    candidates: set[str] = set()
    has_null_well = False
    for table in iter_bronze_tables(path, manifest, columns=(WELL, TARGET)):
        if lowered is not None:
            table = table.filter(_label_mask(table, lowered))
        wells = table.column(WELL)
        has_null_well = has_null_well or wells.null_count > 0
        candidates.update(pc.unique(wells.drop_null()).to_pylist())
    # SQLite sorts NULL first in the index and LIMIT counts it, while the
    # subsequent JOIN drops it; Python's str ordering equals SQLite's BINARY
    # collation for UTF-8 text.
    limit = total_wells - 1 if has_null_well else total_wells
    return sorted(candidates)[:limit]


def apply_valid_ranges(df: pd.DataFrame) -> pd.DataFrame:
    """Replace values outside :data:`VALID_RANGES` by NaN (bounds inclusive)."""
    for column, (low, high) in VALID_RANGES.items():
        values = df[column]
        df[column] = values.where(values.between(low, high))
    return df


def add_presence_indicators(df: pd.DataFrame, columns: Iterable[str] = LOG_COLUMNS) -> list[str]:
    """Add ``<column>_present`` (int 0/1) columns in place; return their names."""
    new_columns = []
    for column in columns:
        name = f"{column}{PRESENCE_SUFFIX}"
        df[name] = df[column].notna().astype(int)
        new_columns.append(name)
    return new_columns


def add_rolling_statistics(
    df: pd.DataFrame,
    window: int = 5,
    columns: Iterable[str] = LOG_COLUMNS,
) -> list[str]:
    """Add per-well trailing rolling mean/std columns in place.

    ``df`` must be sorted by ``well`` and ``depth`` and have a unique index.
    Windows never cross a well boundary; the first rows of each well use the
    available observations (``min_periods=1``), so the standard deviation of
    a single observation is NaN, exactly as in the reference implementation.

    Returns:
        The names of the new columns, ``mean`` and ``std`` interleaved per log.
    """
    if not isinstance(window, (int, np.integer)) or isinstance(window, bool) or window < 1:
        raise ValueError(f"window must be a positive integer, got {window!r}")
    columns = list(columns)
    rolling = df.groupby(WELL, sort=False)[columns].rolling(window=int(window), min_periods=1)
    means = rolling.mean().reset_index(level=0, drop=True)
    stds = rolling.std().reset_index(level=0, drop=True)
    new_columns = []
    for column in columns:
        df[f"{column}{ROLLING_MEAN_SUFFIX}"] = means[column]
        df[f"{column}{ROLLING_STD_SUFFIX}"] = stds[column]
        new_columns.extend([f"{column}{ROLLING_MEAN_SUFFIX}", f"{column}{ROLLING_STD_SUFFIX}"])
    return new_columns


def prepare_features(
    raw: pd.DataFrame,
    apply_scaling: bool = True,
    apply_presence: bool = True,
    apply_rolling_stats: bool = True,
    window: int = 5,
) -> pd.DataFrame:
    """Apply steps 3–6 of the methodology to an already filtered raw table.

    Args:
        raw: Table with at least :data:`REQUIRED_COLUMNS`.

    Returns:
        A new DataFrame sorted by well and depth with a fresh ``RangeIndex``
        and columns ``well, depth, <logs>, [presence], [rolling], rock``.

    Raises:
        SchemaError: A required column is missing.
    """
    missing = [column for column in REQUIRED_COLUMNS if column not in raw.columns]
    if missing:
        raise SchemaError(f"Feature table is missing required column(s): {', '.join(missing)}")
    df = raw.loc[:, list(REQUIRED_COLUMNS)].copy()
    for column in (DEPTH, *LOG_COLUMNS):
        df[column] = pd.to_numeric(df[column], errors="raise").astype("float64")

    if apply_scaling:
        df = apply_valid_ranges(df)
    df = df.sort_values([WELL, DEPTH], kind="mergesort").reset_index(drop=True)

    extra: list[str] = []
    if apply_presence:
        extra += add_presence_indicators(df)
    if apply_rolling_stats:
        extra += add_rolling_statistics(df, window)
    return df[[WELL, DEPTH, *LOG_COLUMNS, *extra, TARGET]]


def fetch_data(
    path: str | os.PathLike[str] = "./data",
    total_wells: int | None = None,
    apply_scaling: bool = True,
    apply_presence: bool = True,
    apply_rolling_stats: bool = True,
    window: int = 5,
    labels: Iterable[str] | None = None,
    revision: str | None = None,
    download: bool = True,
) -> pd.DataFrame:
    """Load the local bronze features and engineer the model features.

    If ``path`` does not hold a complete download (see
    :func:`MaltaRock.download_dataset`) and ``download`` is true, the dataset
    is downloaded first; otherwise the cached files are read without network
    access.

    Args:
        path: Root dataset directory.
        total_wells: Keep only the first ``N`` wells (see module docs);
            ``None`` keeps every well.
        apply_scaling: Replace physically implausible log values by NaN.
        apply_presence: Add ``<LOG>_present`` indicator columns.
        apply_rolling_stats: Add per-well rolling mean/std columns.
        window: Rolling window size in rows.
        labels: Lithology classes to keep (case-insensitive). ``None`` keeps
            every row with a non-null class. The paper uses
            :data:`~MaltaRock.data.schema.CORE15_LABELS`.
        revision: Dataset revision to use/download (commit hash recommended).
        download: Download the dataset when it is not cached.

    Returns:
        A DataFrame with one row per depth sample, sorted by ``well`` and
        ``depth``, with columns ``well``, ``depth``, the ten log curves, the
        optional engineered columns and the target ``rock``. ``well`` and
        ``depth`` are metadata kept for the well-wise split; they are removed
        from ``X`` by :func:`MaltaRock.split_train_test`. Provenance is
        stored in ``DataFrame.attrs["malta_rock"]``.

    Raises:
        DatasetNotFoundError: Data is missing and ``download`` is false.
        DatasetDownloadError: Downloading failed.
        SchemaError: The local files do not have the expected schema.
    """
    lowered = _normalise_labels(labels)
    manifest = ensure_local_dataset(path, revision, download)

    selected: pa.Array | None = None
    if total_wells is not None:
        selected = pa.array(select_wells(path, total_wells, labels, revision, download=False))
        logger.info("Selected %d wells.", len(selected))

    def row_filter(table: pa.Table) -> pa.Array:
        mask = pc.and_(pc.is_valid(table.column(WELL)), _label_mask(table, lowered))
        if selected is not None:
            mask = pc.and_(mask, pc.is_in(table.column(WELL), value_set=selected))
        return mask

    tables = list(iter_bronze_tables(path, manifest, REQUIRED_COLUMNS, row_filter))
    raw = pa.concat_tables(tables).to_pandas() if tables else pd.DataFrame(columns=REQUIRED_COLUMNS)
    logger.info("Loaded %d rows from %d wells.", len(raw), raw[WELL].nunique())

    features = prepare_features(raw, apply_scaling, apply_presence, apply_rolling_stats, window)
    features.attrs["malta_rock"] = {
        "repo_id": manifest.repo_id,
        "config_name": manifest.config_name,
        "revision": manifest.revision,
        "total_wells": total_wells,
        "labels": list(labels) if labels is not None else None,
        "apply_scaling": apply_scaling,
        "apply_presence": apply_presence,
        "apply_rolling_stats": apply_rolling_stats,
        "window": window if apply_rolling_stats else None,
    }
    return features
