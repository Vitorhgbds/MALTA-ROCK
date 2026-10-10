"""Download and local caching of the MALTA-WLOGS bronze-feature table.

Only the files referenced by the ``BronzeFeatures`` configuration of the
public dataset card (``Features/*.parquet``) are downloaded. The other
configurations published in the same repository (``LogsFeatures``,
``AGPSummary`` and ``AGPLithology``) are never fetched.

Local layout::

    <path>/
    └── MALTA-WLOGS/
        ├── manifest.json            # written last, atomically
        └── Features/
            ├── train-00000.parquet
            └── ...

The manifest records the repository id, the requested and resolved
revision, the data-file pattern and the size and SHA-256 of every file. A
dataset is considered complete only when the manifest exists, is valid,
matches the requested revision and every listed file exists with the
recorded size. Complete datasets are reused without any network access.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import logging
import os
import posixpath
import tempfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from huggingface_hub import HfApi, hf_hub_download
from huggingface_hub import errors as hf_errors

from MaltaRock.data.schema import BRONZE_CONFIG, REPO_ID, REPO_TYPE
from MaltaRock.exceptions import (
    DatasetDownloadError,
    DatasetIntegrityError,
    InvalidPathError,
)

logger = logging.getLogger(__name__)

DATASET_DIRNAME = "MALTA-WLOGS"
MANIFEST_NAME = "manifest.json"
MANIFEST_VERSION = 1

_NETWORK_ERRORS: tuple[type[BaseException], ...] = (
    hf_errors.HfHubHTTPError,
    hf_errors.HTTPError,
    hf_errors.LocalEntryNotFoundError,
    hf_errors.OfflineModeIsEnabled,
    hf_errors.HFValidationError,
    ConnectionError,
    TimeoutError,
)


@dataclass(frozen=True)
class RemoteFile:
    """A file of the bronze-feature configuration as published on the Hub."""

    path: str
    size: int
    sha256: str | None


@dataclass(frozen=True)
class DatasetManifest:
    """Content of ``manifest.json``; describes one complete local download."""

    repo_id: str
    config_name: str
    data_files: str
    revision: str
    requested_revision: str | None
    files: list[RemoteFile]
    created_at: str = ""
    manifest_version: int = MANIFEST_VERSION
    extra: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, sort_keys=True)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> DatasetManifest:
        files = [RemoteFile(**item) for item in raw["files"]]
        return cls(**{**raw, "files": files})

    def matches_revision(self, revision: str | None) -> bool:
        """Whether this manifest satisfies a request for ``revision``.

        ``None`` accepts whatever revision is cached. A commit hash (full or
        an unambiguous prefix of at least 7 characters) is compared with the
        resolved commit; any other string (branch or tag) is compared with
        the revision that was originally requested.
        """
        if revision is None:
            return True
        if revision == self.requested_revision or revision == self.revision:
            return True
        return len(revision) >= 7 and self.revision.startswith(revision)


def dataset_dir(path: str | os.PathLike[str]) -> Path:
    """Return the directory where the dataset lives inside ``path``."""
    return Path(path).expanduser() / DATASET_DIRNAME


def _manifest_path(path: str | os.PathLike[str]) -> Path:
    return dataset_dir(path) / MANIFEST_NAME


def read_manifest(path: str | os.PathLike[str]) -> DatasetManifest | None:
    """Read the manifest under ``path``; return ``None`` if absent or invalid."""
    manifest_file = _manifest_path(path)
    if not manifest_file.is_file():
        return None
    try:
        manifest = DatasetManifest.from_dict(json.loads(manifest_file.read_text("utf-8")))
    except (OSError, ValueError, TypeError, KeyError) as exc:
        logger.warning("Ignoring invalid manifest %s: %s", manifest_file, exc)
        return None
    if manifest.manifest_version != MANIFEST_VERSION or not manifest.files:
        logger.warning("Ignoring unsupported or empty manifest %s.", manifest_file)
        return None
    return manifest


def _sha256(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _local_file_matches(local: Path, remote: RemoteFile, verify_checksum: bool) -> bool:
    if not local.is_file() or local.stat().st_size != remote.size:
        return False
    if verify_checksum and remote.sha256 is not None:
        return _sha256(local) == remote.sha256
    return True


def is_dataset_complete(
    path: str | os.PathLike[str] = "./data",
    revision: str | None = None,
    verify_checksums: bool = False,
) -> bool:
    """Check whether a complete download matching ``revision`` exists locally.

    This function never accesses the network.

    Args:
        path: Root directory passed to :func:`download_dataset`.
        revision: Requested revision; ``None`` accepts any cached revision.
        verify_checksums: Also recompute the SHA-256 of every file (slower).
    """
    manifest = read_manifest(path)
    if manifest is None or not manifest.matches_revision(revision):
        return False
    root = dataset_dir(path)
    return all(
        _local_file_matches(root / item.path, item, verify_checksums) for item in manifest.files
    )


def _bronze_pattern(card_data: Any) -> str:
    """Extract the ``data_files`` pattern of the bronze configuration."""
    card = card_data.to_dict() if hasattr(card_data, "to_dict") else (card_data or {})
    for config in card.get("configs") or []:
        if config.get("config_name") != BRONZE_CONFIG:
            continue
        data_files = config.get("data_files")
        if isinstance(data_files, str):
            return data_files
        if isinstance(data_files, list) and len(data_files) == 1:
            entry = data_files[0]
            if isinstance(entry, str):
                return entry
            if isinstance(entry, dict) and isinstance(entry.get("path"), str):
                return entry["path"]
        raise DatasetDownloadError(
            f"Unsupported data_files declaration for {BRONZE_CONFIG!r}: {data_files!r}"
        )
    raise DatasetDownloadError(
        f"The dataset card of {REPO_ID} does not declare a {BRONZE_CONFIG!r} configuration."
    )


def select_bronze_files(pattern: str, siblings: list[Any]) -> list[RemoteFile]:
    """Select the repository files matching the bronze ``data_files`` pattern.

    Matching is restricted to the pattern's directory (``*`` never crosses a
    ``/``), so e.g. ``LogsFeatures/*.parquet`` is not matched by
    ``Features/*.parquet``.
    """
    directory, name_pattern = posixpath.split(pattern)
    selected = []
    for sibling in siblings:
        name = sibling.rfilename
        if posixpath.dirname(name) != directory:
            continue
        if not fnmatch.fnmatchcase(posixpath.basename(name), name_pattern):
            continue
        lfs = getattr(sibling, "lfs", None)
        sha256 = getattr(lfs, "sha256", None) if lfs is not None else None
        size = getattr(lfs, "size", None) if lfs is not None else None
        if size is None:
            size = sibling.size
        if size is None:
            raise DatasetDownloadError(f"The Hub did not report a size for {name!r}.")
        selected.append(RemoteFile(path=name, size=int(size), sha256=sha256))
    return sorted(selected, key=lambda item: item.path)


def resolve_remote_files(revision: str | None = None) -> tuple[str, str, list[RemoteFile]]:
    """Query the Hub for the bronze-feature files of ``revision``.

    Returns:
        ``(commit_sha, data_files_pattern, files)``.

    Raises:
        DatasetDownloadError: On network errors, unknown revisions or an
            unexpected repository structure.
    """
    try:
        info = HfApi().dataset_info(REPO_ID, revision=revision, files_metadata=True)
    except _NETWORK_ERRORS as exc:
        raise DatasetDownloadError(
            f"Could not query {REPO_ID} at revision {revision or 'main'}: {exc}"
        ) from exc
    pattern = _bronze_pattern(info.card_data)
    files = select_bronze_files(pattern, list(info.siblings or []))
    if not files:
        raise DatasetDownloadError(
            f"No file of {REPO_ID}@{info.sha} matches the {BRONZE_CONFIG!r} pattern {pattern!r}."
        )
    if info.sha is None:
        raise DatasetDownloadError(f"The Hub did not report a commit for {REPO_ID}.")
    return info.sha, pattern, files


def _write_atomic(target: Path, content: str) -> None:
    descriptor, temp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


def _prepare_root(path: str | os.PathLike[str]) -> Path:
    root = Path(path).expanduser()
    if root.exists() and not root.is_dir():
        raise InvalidPathError(f"Dataset path {root} exists and is not a directory.")
    try:
        dataset_dir(root).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise InvalidPathError(f"Cannot create dataset directory under {root}: {exc}") from exc
    return root


def download_dataset(
    path: str | os.PathLike[str] = "./data",
    force: bool = False,
    revision: str | None = None,
    verify_checksums: bool = False,
) -> Path:
    """Download the MALTA-WLOGS bronze-feature table if it is not cached.

    Calling this function repeatedly is cheap: when ``path`` already holds a
    complete download matching ``revision`` it returns immediately without
    any network access.

    Args:
        path: Root directory for the dataset. Files are stored under
            ``<path>/MALTA-WLOGS``.
        force: Re-download every file even if a complete cache exists.
        revision: Branch, tag or commit of the Hub repository. ``None`` reuses
            any complete cache, or downloads ``main`` when nothing is cached.
            Pass a commit hash to pin the dataset for reproducibility.
        verify_checksums: When checking an existing cache, also recompute
            the SHA-256 of every file. Fresh downloads are always verified.

    Returns:
        The dataset directory (``<path>/MALTA-WLOGS``).

    Raises:
        InvalidPathError: ``path`` is not a usable directory.
        DatasetDownloadError: The Hub cannot be reached or the repository
            structure differs from the expected one.
        DatasetIntegrityError: A downloaded file does not match the size or
            checksum published on the Hub.
    """
    root = _prepare_root(path)
    target_dir = dataset_dir(root)

    if not force and is_dataset_complete(root, revision, verify_checksums):
        manifest = read_manifest(root)
        assert manifest is not None
        logger.info("Using cached %s@%s in %s.", REPO_ID, manifest.revision[:12], target_dir)
        return target_dir

    commit, pattern, remote_files = resolve_remote_files(revision)
    logger.info(
        "Downloading %d bronze-feature file(s) of %s@%s into %s.",
        len(remote_files), REPO_ID, commit[:12], target_dir,
    )

    # Invalidate any previous manifest before touching data files so that an
    # interrupted download is never mistaken for a complete one.
    _manifest_path(root).unlink(missing_ok=True)

    for index, remote in enumerate(remote_files, start=1):
        local = target_dir / remote.path
        if not force and _local_file_matches(local, remote, verify_checksum=True):
            logger.debug("[%d/%d] %s already present.", index, len(remote_files), remote.path)
            continue
        # A stale or truncated file must not be trusted by hf_hub_download's
        # own local_dir metadata either.
        local.unlink(missing_ok=True)
        logger.info("[%d/%d] Downloading %s.", index, len(remote_files), remote.path)
        try:
            downloaded = hf_hub_download(
                repo_id=REPO_ID,
                filename=remote.path,
                repo_type=REPO_TYPE,
                revision=commit,
                local_dir=target_dir,
                force_download=force,
            )
        except _NETWORK_ERRORS as exc:
            raise DatasetDownloadError(f"Failed to download {remote.path}: {exc}") from exc
        except OSError as exc:
            raise InvalidPathError(f"Cannot write {local}: {exc}") from exc
        if Path(downloaded).resolve() != local.resolve():
            raise DatasetIntegrityError(f"{remote.path} was saved to {downloaded}, expected {local}.")
        if not _local_file_matches(local, remote, verify_checksum=True):
            local.unlink(missing_ok=True)
            raise DatasetIntegrityError(
                f"{remote.path} does not match the size/SHA-256 published for {REPO_ID}@{commit}."
            )

    _remove_stale_files(target_dir, pattern, remote_files)

    manifest = DatasetManifest(
        repo_id=REPO_ID,
        config_name=BRONZE_CONFIG,
        data_files=pattern,
        revision=commit,
        requested_revision=revision or "main",
        files=remote_files,
        created_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )
    _write_atomic(_manifest_path(root), manifest.to_json())
    logger.info("Dataset ready in %s.", target_dir)
    return target_dir


def _remove_stale_files(target_dir: Path, pattern: str, keep: list[RemoteFile]) -> None:
    """Delete data files left over from another revision."""
    keep_paths = {item.path for item in keep}
    for candidate in target_dir.glob(pattern):
        relative = candidate.relative_to(target_dir).as_posix()
        if relative not in keep_paths and candidate.is_file():
            logger.info("Removing stale file %s.", relative)
            candidate.unlink()
