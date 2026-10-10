from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from conftest import bronze_repo_files, make_bronze_frame

from MaltaRock.data import download as dl
from MaltaRock.exceptions import DatasetDownloadError, DatasetIntegrityError, InvalidPathError

SHA_A = "a" * 40
SHA_B = "b" * 40


def test_file_selection_is_restricted_to_bronze_features():
    names = [
        "Features/train-00001.parquet", "Features/train-00000.parquet",
        "LogsFeatures/train-00000.parquet", "AGPSummary/train-00000.parquet",
        "AGPLithology/train-00000.parquet", "Features/nested/x.parquet",
        "Features/notes.txt", "README.md", ".gitattributes",
    ]
    siblings = [SimpleNamespace(rfilename=n, size=1, lfs=None) for n in names]
    selected = dl.select_bronze_files("Features/*.parquet", siblings)
    assert [f.path for f in selected] == ["Features/train-00000.parquet", "Features/train-00001.parquet"]


def test_pattern_comes_from_dataset_card():
    card = SimpleNamespace(to_dict=lambda: {"configs": [
        {"config_name": "LogsFeatures", "data_files": "LogsFeatures/*.parquet"},
        {"config_name": "BronzeFeatures", "data_files": [{"split": "train", "path": "Features/*.parquet"}]},
    ]})
    assert dl._bronze_pattern(card) == "Features/*.parquet"
    with pytest.raises(DatasetDownloadError):
        dl._bronze_pattern(SimpleNamespace(to_dict=lambda: {"configs": []}))


def test_download_writes_manifest_with_only_bronze_files(local_dataset, fake_hub):
    manifest = dl.read_manifest(local_dataset)
    assert manifest is not None
    assert manifest.revision == SHA_A
    assert manifest.data_files == "Features/*.parquet"
    assert all(f.path.startswith("Features/") for f in manifest.files)
    assert all(f.sha256 for f in manifest.files)
    assert sorted(fake_hub.download_calls) == [f.path for f in manifest.files]
    assert not (local_dataset / "MALTA-WLOGS" / "LogsFeatures").exists()


def test_complete_cache_skips_network(local_dataset, fake_hub, monkeypatch):
    def no_network(*args, **kwargs):
        raise AssertionError("network accessed")

    monkeypatch.setattr(dl, "HfApi", no_network)
    monkeypatch.setattr(dl, "hf_hub_download", no_network)
    assert dl.download_dataset(local_dataset) == local_dataset / "MALTA-WLOGS"
    assert dl.download_dataset(local_dataset, revision=SHA_A) == local_dataset / "MALTA-WLOGS"
    assert dl.download_dataset(local_dataset, revision="main") == local_dataset / "MALTA-WLOGS"
    assert dl.download_dataset(local_dataset, revision=SHA_A[:8]) == local_dataset / "MALTA-WLOGS"


def test_missing_file_triggers_refresh_of_that_file(local_dataset, fake_hub):
    manifest = dl.read_manifest(local_dataset)
    victim = manifest.files[1].path
    (local_dataset / "MALTA-WLOGS" / victim).unlink()
    assert not dl.is_dataset_complete(local_dataset)
    fake_hub.download_calls.clear()
    dl.download_dataset(local_dataset)
    assert fake_hub.download_calls == [victim]
    assert dl.is_dataset_complete(local_dataset, verify_checksums=True)


def test_truncated_file_is_not_complete(local_dataset):
    path = local_dataset / "MALTA-WLOGS" / dl.read_manifest(local_dataset).files[0].path
    path.write_bytes(path.read_bytes()[:-10])
    assert not dl.is_dataset_complete(local_dataset)


def test_same_size_corruption_detected_with_checksums(local_dataset):
    path = local_dataset / "MALTA-WLOGS" / dl.read_manifest(local_dataset).files[0].path
    data = bytearray(path.read_bytes())
    data[-1] ^= 0xFF
    path.write_bytes(bytes(data))
    assert dl.is_dataset_complete(local_dataset)
    assert not dl.is_dataset_complete(local_dataset, verify_checksums=True)


@pytest.mark.parametrize("content", ["{not json", json.dumps({"files": []}), json.dumps({"foo": 1})])
def test_invalid_manifest_triggers_refresh(local_dataset, fake_hub, content):
    (local_dataset / "MALTA-WLOGS" / "manifest.json").write_text(content)
    assert not dl.is_dataset_complete(local_dataset)
    calls = fake_hub.info_calls
    dl.download_dataset(local_dataset)
    assert fake_hub.info_calls == calls + 1
    assert dl.is_dataset_complete(local_dataset)


def test_changed_revision_triggers_refresh(local_dataset, fake_hub):
    fake_hub.add_revision(SHA_B, bronze_repo_files(make_bronze_frame(seed=1), n_shards=2))
    assert not dl.is_dataset_complete(local_dataset, revision=SHA_B)
    dl.download_dataset(local_dataset, revision=SHA_B)
    manifest = dl.read_manifest(local_dataset)
    assert manifest.revision == SHA_B and len(manifest.files) == 2
    # stale shard from the previous revision is removed
    assert not (local_dataset / "MALTA-WLOGS" / "Features" / "train-00002.parquet").exists()


def test_force_redownloads_everything(local_dataset, fake_hub):
    fake_hub.download_calls.clear()
    dl.download_dataset(local_dataset, force=True)
    assert len(fake_hub.download_calls) == len(dl.read_manifest(local_dataset).files)


def test_partial_download_is_not_complete(tmp_path, fake_hub, bronze_frame):
    fake_hub.add_revision(SHA_A, bronze_repo_files(bronze_frame), alias="main")
    fake_hub.fail_on = "Features/train-00001.parquet"
    with pytest.raises(DatasetDownloadError):
        dl.download_dataset(tmp_path)
    assert dl.read_manifest(tmp_path) is None
    assert not dl.is_dataset_complete(tmp_path)
    # resuming downloads only what is missing
    fake_hub.fail_on = None
    fake_hub.download_calls.clear()
    dl.download_dataset(tmp_path)
    assert fake_hub.download_calls == ["Features/train-00001.parquet", "Features/train-00002.parquet"]


def test_interrupted_refresh_invalidates_old_manifest(local_dataset, fake_hub):
    fake_hub.add_revision(SHA_B, bronze_repo_files(make_bronze_frame(seed=1)))
    fake_hub.fail_on = "Features/train-00000.parquet"
    with pytest.raises(DatasetDownloadError):
        dl.download_dataset(local_dataset, revision=SHA_B)
    assert not dl.is_dataset_complete(local_dataset)


def test_checksum_mismatch_raises_and_removes_file(tmp_path, fake_hub, bronze_frame):
    fake_hub.add_revision(SHA_A, bronze_repo_files(bronze_frame), alias="main")
    fake_hub.corrupt = "Features/train-00000.parquet"
    with pytest.raises(DatasetIntegrityError):
        dl.download_dataset(tmp_path)
    assert not (tmp_path / "MALTA-WLOGS" / "Features" / "train-00000.parquet").exists()
    assert dl.read_manifest(tmp_path) is None


def test_unknown_revision_raises_download_error(tmp_path, fake_hub):
    with pytest.raises(DatasetDownloadError):
        dl.download_dataset(tmp_path, revision="does-not-exist")


def test_path_that_is_a_file_is_rejected(tmp_path):
    target = tmp_path / "file.txt"
    target.write_text("x")
    with pytest.raises(InvalidPathError):
        dl.download_dataset(target)
