"""Sequence-window construction for recurrent models.

Windows are built independently for each well, so a window never contains
rows from two wells. Rows must already be sorted by depth within each well
(as returned by :func:`MaltaRock.split_train_test`).
"""

from __future__ import annotations

from typing import Literal

import numpy as np

SequenceMode = Literal["centered", "trailing"]


def window_targets(
    groups: np.ndarray,
    sequence_length: int,
    mode: SequenceMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(window_starts, target_rows)`` for every valid window.

    Modes (both reproduce the reference implementations, but per well):

    * ``"centered"`` (LSTM): the window ``[t - L//2, t + L - L//2)`` labelled
      with row ``t``, for ``t`` in ``range(L//2, n - (L - L//2))``.
    * ``"trailing"`` (BiLSTM): the window ``[t, t + L)`` labelled with the
      row *following* it, ``t + L``, for ``t`` in ``range(n - L)``.

    Args:
        groups: Well id per row; rows of a well must be contiguous.
        sequence_length: Window length ``L``.
        mode: ``"centered"`` or ``"trailing"``.
    """
    if sequence_length < 1:
        raise ValueError("sequence_length must be positive")
    groups = np.asarray(groups)
    n_rows = len(groups)
    if n_rows == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    boundaries = np.flatnonzero(groups[1:] != groups[:-1]) + 1
    starts_of_wells = np.concatenate([[0], boundaries])
    ends_of_wells = np.concatenate([boundaries, [n_rows]])
    if len(np.unique(groups[starts_of_wells])) != len(starts_of_wells):
        raise ValueError("Rows of each well must be contiguous (sort by well and depth).")

    half = sequence_length // 2
    window_starts, targets = [], []
    for begin, end in zip(starts_of_wells, ends_of_wells, strict=True):
        n = end - begin
        if mode == "centered":
            centers = np.arange(half, n - (sequence_length - half))
            window_starts.append(begin + centers - half)
            targets.append(begin + centers)
        elif mode == "trailing":
            offsets = np.arange(0, n - sequence_length)
            window_starts.append(begin + offsets)
            targets.append(begin + offsets + sequence_length)
        else:
            raise ValueError(f"Unknown sequence mode {mode!r}")
    return (
        np.concatenate(window_starts).astype(np.int64),
        np.concatenate(targets).astype(np.int64),
    )


def build_sequences(
    values: np.ndarray,
    groups: np.ndarray,
    sequence_length: int,
    mode: SequenceMode,
) -> tuple[np.ndarray, np.ndarray]:
    """Stack per-well windows of ``values``.

    Returns:
        ``(sequences, target_rows)`` where ``sequences`` has shape
        ``(n_windows, sequence_length, n_features)`` and ``target_rows`` holds
        the row position labelled by each window.
    """
    values = np.asarray(values)
    window_starts, targets = window_targets(groups, sequence_length, mode)
    if len(window_starts) == 0:
        return np.empty((0, sequence_length, values.shape[1]), dtype=values.dtype), targets
    offsets = np.arange(sequence_length)
    return values[window_starts[:, None] + offsets[None, :]], targets
