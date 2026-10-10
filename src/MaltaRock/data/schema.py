"""Constants describing the published MALTA-WLOGS bronze-feature table.

Every value in this module was verified against the public Hugging Face
repository ``MALTA-Lab/MALTA-WLOGS`` (commit
``dd7db33c4803780d629ced50ebb61ddd41575a64``) and against the reference
implementation (``GeoAI-ANP-DataExtractor/slide/models``) and the dataset
publisher (``MALTA-WELD/weld/huggingface.py``).
"""

from __future__ import annotations

from typing import Final

REPO_ID: Final = "MALTA-Lab/MALTA-WLOGS"
REPO_TYPE: Final = "dataset"

#: Dataset-card configuration that holds the bronze features. Its
#: ``data_files`` entry is ``Features/*.parquet`` (the SQLite ``Features``
#: table exported in 500k-row chunks by MALTA-WELD).
BRONZE_CONFIG: Final = "BronzeFeatures"

WELL: Final = "well"
DEPTH: Final = "depth"
TARGET: Final = "rock"

#: Conventional well-log curves, in the column order used by the reference
#: implementation.
LOG_COLUMNS: Final[tuple[str, ...]] = (
    "GR", "SP", "DT", "CALI", "ILD", "RHOB", "NPHI", "MSFL", "DRHO", "SFLU",
)

#: Columns read from the published Parquet files. ``top`` and ``bottom``
#: (lithology interval limits) are also published but are not used by the
#: methodology, so they are not loaded.
REQUIRED_COLUMNS: Final[tuple[str, ...]] = (WELL, DEPTH, *LOG_COLUMNS, TARGET)

#: Columns that must never be used as predictive features.
METADATA_COLUMNS: Final[tuple[str, ...]] = (WELL, DEPTH, TARGET)

#: Physically plausible value ranges (inclusive). Values outside the range
#: are replaced by NaN. Copied verbatim from the reference SQL
#: (``FeatureProcessingPolicy.__select_features``).
VALID_RANGES: Final[dict[str, tuple[float, float]]] = {
    "GR": (0, 200),
    "SP": (-100, 100),
    "DT": (40, 140),
    "CALI": (6, 16),
    "ILD": (0.2, 2000),
    "RHOB": (1.95, 2.95),
    "NPHI": (-15, 45),
    "MSFL": (0.2, 2000),
    "DRHO": (-0.3, 0.3),
    "SFLU": (0.2, 2000),
}

#: The 15 lithology classes used by the paper's experiments (the ``labels``
#: list in every ``models/configs`` YAML file).
CORE15_LABELS: Final[tuple[str, ...]] = (
    "ARENITO", "FOLHELHO", "SILTITO", "CALCILUTITO", "CALCARENITO",
    "ARGILITO", "CALCARIO", "CONGLOMERADO", "DIABASIO", "ANIDRITA",
    "AREIA", "DIAMICTITO", "MARGA", "ARGILA", "DOLOMITO",
)
