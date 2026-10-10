"""Dataset download, loading, feature preparation and well-wise splitting."""

from MaltaRock.data.download import (
    DatasetManifest,
    download_dataset,
    is_dataset_complete,
    read_manifest,
)
from MaltaRock.data.features import fetch_data, prepare_features, select_wells
from MaltaRock.data.schema import CORE15_LABELS, LOG_COLUMNS, REPO_ID
from MaltaRock.data.sequences import build_sequences
from MaltaRock.data.split import (
    TrainTestSplit,
    get_groups,
    load_well_assignment,
    save_well_assignment,
    split_train_test,
    split_wells,
)

__all__ = [
    "CORE15_LABELS",
    "LOG_COLUMNS",
    "REPO_ID",
    "DatasetManifest",
    "TrainTestSplit",
    "build_sequences",
    "download_dataset",
    "fetch_data",
    "get_groups",
    "is_dataset_complete",
    "load_well_assignment",
    "prepare_features",
    "read_manifest",
    "save_well_assignment",
    "select_wells",
    "split_train_test",
    "split_wells",
]
