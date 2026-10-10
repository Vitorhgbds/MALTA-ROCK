"""Bidirectional LSTM classifier (reference: ``slide/models/bilstm.py``). Requires ``torch``."""

from __future__ import annotations

from MaltaRock.models.sequence import SequenceModel


class BILSTM(SequenceModel):
    """Bidirectional LSTM over trailing per-well windows.

    As in the reference, each window ``[t, t + sequence_length)`` is labelled
    with the row that follows it (``t + sequence_length``). See
    :class:`~MaltaRock.models.sequence.SequenceModel` for the arguments.
    """

    mode = "trailing"
    bidirectional = True
