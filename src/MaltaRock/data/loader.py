"""Reading the locally cached bronze-feature Parquet files."""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from MaltaRock.data.download import (
    DatasetManifest,
    dataset_dir,
    download_dataset,
    is_dataset_complete,
    read_manifest,
)
from MaltaRock.data.schema import DEPTH, LOG_COLUMNS, REQUIRED_COLUMNS, TARGET, WELL
from MaltaRock.exceptions import DatasetNotFoundError, SchemaError

logger = logging.getLogger(__name__)

#: Canonical Arrow types. Some published shards store an all-null log curve
#: with the Arrow ``null`` type, so every file is cast to this schema.
ARROW_TYPES: dict[str, pa.DataType] = {
    WELL: pa.string(),
    DEPTH: pa.float64(),
    **{column: pa.float64() for column in LOG_COLUMNS},
    TARGET: pa.string(),
}


def ensure_local_dataset(
    path: str | os.PathLike[str],
    revision: str | None = None,
    download: bool = True,
) -> DatasetManifest:
    """Return the manifest of a complete local dataset, downloading if allowed.

    Raises:
        DatasetNotFoundError: The dataset is absent or incomplete and
            ``download`` is ``False``.
    """
    if not is_dataset_complete(path, revision):
        if not download:
            raise DatasetNotFoundError(
                f"No complete MALTA-WLOGS download{f' at revision {revision}' if revision else ''} "
                f"under {Path(path)}. Run MaltaRock.download_dataset({str(path)!r}) first."
            )
        download_dataset(path, revision=revision)
    manifest = read_manifest(path)
    if manifest is None:  # pragma: no cover - download_dataset always writes one
        raise DatasetNotFoundError(f"Manifest missing under {dataset_dir(path)}.")
    return manifest


def _cast_table(table: pa.Table, source: Path, columns: Sequence[str]) -> pa.Table:
    missing = [column for column in columns if column not in table.column_names]
    if missing:
        raise SchemaError(f"{source} is missing required column(s): {', '.join(missing)}")
    arrays = []
    for column in columns:
        target = ARROW_TYPES[column]
        array = table.column(column)
        if array.type == target:
            arrays.append(array)
            continue
        if pa.types.is_null(array.type) or (
            pa.types.is_floating(target)
            and (pa.types.is_integer(array.type) or pa.types.is_floating(array.type))
        ) or (pa.types.is_string(target) and pa.types.is_large_string(array.type)):
            arrays.append(array.cast(target))
            continue
        raise SchemaError(
            f"Column {column!r} in {source} has type {array.type}, expected {target}."
        )
    return pa.table(arrays, names=list(columns))


def iter_bronze_tables(
    path: str | os.PathLike[str],
    manifest: DatasetManifest,
    columns: Sequence[str] = REQUIRED_COLUMNS,
    row_filter: Callable[[pa.Table], pa.Array] | None = None,
) -> Iterator[pa.Table]:
    """Yield each published shard, in repository order, cast to the canonical schema.

    Shards are yielded in file-name order, which is the row order of the
    SQLite ``Features`` table exported by MALTA-WELD.

    Args:
        path: Root dataset directory (as given to :func:`download_dataset`).
        manifest: Manifest of the complete download.
        columns: Subset of :data:`REQUIRED_COLUMNS` to read.
        row_filter: Optional function returning a boolean mask per table.
    """
    unknown = set(columns) - set(ARROW_TYPES)
    if unknown:
        raise ValueError(f"Unknown column(s): {sorted(unknown)}")
    root = dataset_dir(path)
    for item in manifest.files:
        source = root / item.path
        try:
            schema_names = pq.read_schema(source).names
        except (OSError, pa.ArrowInvalid) as exc:
            raise SchemaError(f"Cannot read Parquet file {source}: {exc}") from exc
        missing = [column for column in columns if column not in schema_names]
        if missing:
            raise SchemaError(f"{source} is missing required column(s): {', '.join(missing)}")
        table = _cast_table(pq.read_table(source, columns=list(columns)), source, columns)
        if row_filter is not None:
            table = table.filter(row_filter(table))
        yield table


def ascii_lower(array: pa.Array | pa.ChunkedArray) -> pa.Array | pa.ChunkedArray:
    """Lower-case ASCII letters only, matching SQLite's ``lower()``."""
    return pc.ascii_lower(array)
