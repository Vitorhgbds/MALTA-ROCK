"""Offline fixtures: synthetic bronze-feature shards and a fake Hugging Face Hub."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from MaltaRock.data import download as download_module
from MaltaRock.data.schema import LOG_COLUMNS

CARD = {
    "configs": [
        {"config_name": "BronzeFeatures", "data_files": "Features/*.parquet", "default": True},
        {"config_name": "LogsFeatures", "data_files": "LogsFeatures/*.parquet"},
        {"config_name": "AGPSummary", "data_files": "AGPSummary/*.parquet"},
        {"config_name": "AGPLithology", "data_files": "AGPLithology/*.parquet"},
    ]
}
CLASSES = ["ARENITO", "FOLHELHO", "SILTITO"]


def make_bronze_frame(n_wells: int = 10, rows_per_well: int = 30, seed: int = 0) -> pd.DataFrame:
    """Synthetic rows with the published Features schema (incl. top/bottom)."""
    rng = np.random.default_rng(seed)
    frames = []
    for w in range(n_wells):
        depth = 1000.0 + 0.2 * np.arange(rows_per_well)
        rock = rng.choice(CLASSES, size=rows_per_well)
        frame = pd.DataFrame({"well": f"W{w:03d}", "depth": depth})
        for i, column in enumerate(LOG_COLUMNS):
            base = {"ARENITO": 0.0, "FOLHELHO": 1.0, "SILTITO": 2.0}
            frame[column] = np.array([base[r] for r in rock]) * (i + 1) + rng.normal(0, 0.1, rows_per_well)
        frame["GR"] = 50 + 40 * np.array([CLASSES.index(r) for r in rock]) + rng.normal(0, 5, rows_per_well)
        frame["top"] = depth
        frame["bottom"] = depth + 0.2
        frame["rock"] = rock
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def parquet_bytes(frame: pd.DataFrame, null_columns: tuple[str, ...] = ()) -> bytes:
    table = pa.Table.from_pandas(frame, preserve_index=False)
    for column in null_columns:  # mimic published shards with an all-null curve
        position = table.column_names.index(column)
        table = table.set_column(position, column, pa.nulls(len(table)))
    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()


class FakeHub:
    """In-memory replacement for ``HfApi`` and ``hf_hub_download``."""

    def __init__(self) -> None:
        self.revisions: dict[str, dict[str, bytes]] = {}
        self.info_calls = 0
        self.download_calls: list[str] = []
        self.fail_on: str | None = None
        self.corrupt: str | None = None

    def add_revision(self, sha: str, files: dict[str, bytes], alias: str | None = None) -> None:
        self.revisions[sha] = files
        if alias:
            self.revisions[alias] = files
            self._alias = {alias: sha}
        self._shas = getattr(self, "_shas", {})
        self._shas[sha] = sha
        if alias:
            self._shas[alias] = sha

    # HfApi --------------------------------------------------------------
    def dataset_info(self, repo_id: str, revision: str | None = None, files_metadata: bool = False):
        self.info_calls += 1
        key = revision or "main"
        if key not in self.revisions:
            response = SimpleNamespace(headers={}, status_code=404, request=None, url="", text="")
            raise download_module.hf_errors.RevisionNotFoundError(f"unknown revision {key}", response=response)
        files = self.revisions[key]
        siblings = [
            SimpleNamespace(
                rfilename=name,
                size=len(content),
                lfs=SimpleNamespace(sha256=hashlib.sha256(content).hexdigest(), size=len(content))
                if name.endswith(".parquet") else None,
            )
            for name, content in files.items()
        ]
        return SimpleNamespace(
            sha=self._shas[key], siblings=siblings, card_data=SimpleNamespace(to_dict=lambda: CARD)
        )

    # hf_hub_download ------------------------------------------------------
    def hf_hub_download(self, repo_id, filename, *, repo_type, revision, local_dir, force_download=False):
        self.download_calls.append(filename)
        if self.fail_on == filename:
            raise ConnectionError("simulated network failure")
        content = self.revisions[revision][filename]
        if self.corrupt == filename:
            content = content[:-1] + b"x"
        target = Path(local_dir) / filename
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        return str(target)


def bronze_repo_files(frame: pd.DataFrame, n_shards: int = 3) -> dict[str, bytes]:
    files: dict[str, bytes] = {
        "README.md": b"card",
        ".gitattributes": b"",
        "LogsFeatures/train-00000.parquet": b"not bronze",
        "AGPSummary/train-00000.parquet": b"not bronze",
    }
    for i, chunk in enumerate(np.array_split(np.arange(len(frame)), n_shards)):
        null_columns = ("DT",) if i == 0 else ()
        files[f"Features/train-{i:05d}.parquet"] = parquet_bytes(frame.iloc[chunk], null_columns)
    return files


@pytest.fixture
def fake_hub(monkeypatch: pytest.MonkeyPatch) -> FakeHub:
    hub = FakeHub()
    monkeypatch.setattr(download_module, "HfApi", lambda: hub)
    monkeypatch.setattr(download_module, "hf_hub_download", hub.hf_hub_download)
    return hub


@pytest.fixture
def bronze_frame() -> pd.DataFrame:
    return make_bronze_frame()


@pytest.fixture
def local_dataset(tmp_path: Path, fake_hub: FakeHub, bronze_frame: pd.DataFrame) -> Path:
    """A complete local dataset under ``tmp_path / 'data'`` (downloaded from the fake hub)."""
    fake_hub.add_revision("a" * 40, bronze_repo_files(bronze_frame), alias="main")
    root = tmp_path / "data"
    download_module.download_dataset(root)
    return root
