"""Classification metrics used in the paper's experiments."""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def compute_metrics(
    y_true: Iterable[Any],
    y_pred: Iterable[Any],
    labels: Sequence[Any] | None = None,
) -> dict[str, Any]:
    """Compute the research metrics for one evaluation partition.

    ``accuracy``, ``f1_macro``, ``f1_weighted`` and ``classification_report``
    are computed exactly as in the reference implementation (sklearn
    defaults: macro/weighted averages over the classes present in
    ``y_true`` or ``y_pred``), so the numbers are comparable with the paper.

    Classes absent from the partition are handled explicitly: ``per_class``
    and ``confusion_matrix`` cover every class in ``labels`` (normally the
    classes seen during training) plus any class in ``y_true``/``y_pred``;
    classes with zero support are listed in ``absent_classes`` and
    ``f1_macro_all_classes`` averages over that full class list, scoring
    absent classes as 0.

    Args:
        y_true: True class names.
        y_pred: Predicted class names.
        labels: Known class names (e.g. the training classes).

    Returns:
        A JSON-serialisable dictionary.
    """
    y_true = np.asarray(list(y_true), dtype=object)
    y_pred = np.asarray(list(y_pred), dtype=object)
    if len(y_true) != len(y_pred):
        raise ValueError(f"y_true and y_pred differ in length ({len(y_true)} != {len(y_pred)})")
    if len(y_true) == 0:
        raise ValueError("Cannot evaluate an empty partition.")

    known = list(labels) if labels is not None else []
    all_labels = sorted(set(known) | set(y_true.tolist()) | set(y_pred.tolist()), key=str)

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=all_labels, zero_division=0
    )
    per_class = {
        str(label): {
            "precision": float(p), "recall": float(r), "f1": float(f), "support": int(s)
        }
        for label, p, r, f, s in zip(all_labels, precision, recall, f1, support, strict=True)
    }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_macro_all_classes": float(np.mean(f1)),
        "per_class": per_class,
        "classification_report": classification_report(
            y_true, y_pred, output_dict=True, zero_division=0
        ),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=all_labels).tolist(),
        "labels": [str(label) for label in all_labels],
        "absent_classes": [str(label) for label, s in zip(all_labels, support, strict=True) if s == 0],
        "unseen_classes": sorted(str(label) for label in set(y_true.tolist()) - set(known)) if known else [],
        "n_samples": int(len(y_true)),
    }


def save_results(results: dict[str, Any], path: str | os.PathLike[str]) -> None:
    """Write a results dictionary to ``path`` as JSON."""
    Path(path).write_text(json.dumps(results, indent=2, default=_to_builtin), encoding="utf-8")


def _to_builtin(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)
