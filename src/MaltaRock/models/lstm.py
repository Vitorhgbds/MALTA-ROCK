"""LSTM classifier (reference: ``slide/models/lstm.py``). Requires ``torch``."""

from __future__ import annotations

from MaltaRock.models.sequence import SequenceModel


class LSTM(SequenceModel):
    """Unidirectional LSTM over centred per-well windows.

    Each window of ``sequence_length`` rows is labelled with its centre row.
    As in the reference, dropout is disabled entirely when ``num_layers == 1``.
    See :class:`~MaltaRock.models.sequence.SequenceModel` for the arguments.
    """

    mode = "centered"
    bidirectional = False

    def _effective_dropout(self) -> float:
        return self.dropout if self.num_layers > 1 else 0.0
