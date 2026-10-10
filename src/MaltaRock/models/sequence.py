"""Shared training logic of the recurrent models (requires ``torch``).

Ported from ``slide/models/lstm.py`` and ``slide/models/bilstm.py`` with
three corrections (documented in the README):

* windows are built per well, so they never span two wells;
* early stopping monitors a validation subset of the *training* wells
  (``validation_size``) instead of the held-out test set;
* ``predict`` applies the fitted scaler (the reference forgot it) and
  ``learning_rate`` is honoured (the reference hard-coded ``1e-3``; that
  remains the default).
"""

from __future__ import annotations

import gc
import logging
from typing import Any, Self

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from MaltaRock.data.sequences import SequenceMode, build_sequences
from MaltaRock.data.split import get_groups
from MaltaRock.evaluation.metrics import compute_metrics
from MaltaRock.models.base import BaseModel, _drop_missing_targets, _unpack

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class LSTMClassifier(nn.Module):
    """LSTM (optionally bidirectional) followed by a two-layer head."""

    def __init__(
        self,
        num_features: int,
        hidden_size: int,
        num_layers: int,
        num_classes: int,
        dropout: float,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        self.bidirectional = bidirectional
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * (2 if bidirectional else 1), 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (hidden, _) = self.lstm(x)
        if self.bidirectional:
            return self.fc(torch.cat((hidden[-2], hidden[-1]), dim=1))
        return self.fc(hidden[-1])


class SequenceModel(BaseModel):
    """Base class of :class:`~MaltaRock.LSTM` and :class:`~MaltaRock.BILSTM`.

    ``X`` must carry the well of every row, either in the ``(well, depth)``
    index produced by :func:`MaltaRock.split_train_test` or through the
    ``groups`` argument, and rows must be sorted by depth within each well.
    Predictions are made only for rows that have a complete window inside
    their own well.
    """

    mode: SequenceMode = "centered"
    bidirectional: bool = False

    def __init__(
        self,
        sequence_length: int = 11,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.3,
        batch_size: int = 128,
        epochs: int = 256,
        learning_rate: float = 1e-3,
        patience: int = 16,
        num_workers: int = 4,
        validation_size: float = 0.1,
        random_state: int | None = 42,
        device: str | None = None,
    ) -> None:
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.batch_size = batch_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.patience = patience
        self.num_workers = num_workers
        self.validation_size = validation_size
        self.random_state = random_state
        self.device = device
        self.model = None
        self.imputer: SimpleImputer | None = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names_: list[str] | None = None

    # -- helpers -------------------------------------------------------------
    def _device(self) -> torch.device:
        return torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))

    def _effective_dropout(self) -> float:
        return self.dropout

    @staticmethod
    def _groups(X: Any, groups: Any) -> np.ndarray:
        return np.asarray(groups) if groups is not None else get_groups(X)

    def _scaled(self, X: Any) -> np.ndarray:
        if self.feature_names_ is not None and isinstance(X, pd.DataFrame):
            X = X[self.feature_names_]
        assert self.imputer is not None
        return self.scaler.transform(self.imputer.transform(X)).astype(np.float32)

    def _windows(self, values: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return build_sequences(values, groups, self.sequence_length, self.mode)

    def _loader(self, X: np.ndarray, y: np.ndarray, sampler: Any = None) -> DataLoader:
        dataset = TensorDataset(torch.as_tensor(X), torch.as_tensor(y, dtype=torch.long))
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            sampler=sampler,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self._device().type == "cuda",
        )

    def _predict_windows(self, sequences: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not trained. Call train() first or load a trained model.")
        device = self._device()
        self.model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(sequences), self.batch_size):
                batch = torch.as_tensor(sequences[start:start + self.batch_size]).to(device)
                outputs.append(self.model(batch).argmax(dim=1).cpu().numpy())
        return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.int64)

    # -- public API ------------------------------------------------------------
    def train(self, X: Any, y: Any = None, groups: Any = None) -> Self:
        """Fit on training wells; early-stop on a validation subset of them."""
        X, y = _unpack(X, y, "train")
        groups = self._groups(X, groups)
        mask = pd.Series(np.asarray(y, dtype=object)).notna().to_numpy()
        X, y = _drop_missing_targets(X, y)
        groups = groups[mask]

        if self.random_state is not None:
            torch.manual_seed(self.random_state)
        device = self._device()
        self.feature_names_ = list(X.columns) if isinstance(X, pd.DataFrame) else None
        self.imputer = SimpleImputer(strategy="median")
        self.scaler = StandardScaler()
        values = self.scaler.fit_transform(self.imputer.fit_transform(X)).astype(np.float32)
        self.label_encoder = LabelEncoder()
        targets = self.label_encoder.fit_transform(np.asarray(y, dtype=object))

        sequences, rows = self._windows(values, groups)
        if len(sequences) == 0:
            raise ValueError("No training window fits inside any well; reduce sequence_length.")
        window_targets, window_groups = targets[rows], groups[rows]

        fit_idx = np.arange(len(sequences))
        val_idx = np.empty(0, dtype=np.int64)
        if self.validation_size and len(np.unique(window_groups)) >= 2:
            splitter = GroupShuffleSplit(n_splits=1, test_size=self.validation_size,
                                         random_state=self.random_state)
            fit_idx, val_idx = next(splitter.split(sequences, window_targets, window_groups))

        X_fit, y_fit = sequences[fit_idx], window_targets[fit_idx]
        num_classes = len(self.label_encoder.classes_)

        # Effective-number class weights and weighted sampling (reference).
        counts = np.bincount(y_fit, minlength=num_classes)
        beta = 0.9999
        class_weights = (1.0 - beta) / np.maximum(1.0 - np.power(beta, counts), 1e-12)
        class_weights = class_weights / class_weights.mean()
        generator = torch.Generator()
        if self.random_state is not None:
            generator.manual_seed(self.random_state)
        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(class_weights[y_fit], dtype=torch.double),
            num_samples=len(y_fit), replacement=True, generator=generator,
        )
        criterion = nn.CrossEntropyLoss(weight=torch.tensor(class_weights, dtype=torch.float32, device=device))
        train_loader = self._loader(X_fit, y_fit, sampler)

        self.model = LSTMClassifier(
            num_features=values.shape[1], hidden_size=self.hidden_size, num_layers=self.num_layers,
            num_classes=num_classes, dropout=self._effective_dropout(), bidirectional=self.bidirectional,
        ).to(device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        use_amp = device.type == "cuda"
        grad_scaler = torch.amp.GradScaler(enabled=use_amp)

        best_f1, best_state, no_improve = -1.0, None, 0
        try:
            for epoch in range(self.epochs):
                self.model.train()
                total_loss = 0.0
                for batch_x, batch_y in train_loader:
                    batch_x = batch_x.to(device, non_blocking=True)
                    batch_y = batch_y.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                        loss = criterion(self.model(batch_x), batch_y)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    total_loss += loss.item()
                if len(val_idx) == 0:
                    logger.info("Epoch %d/%d | loss %.4f", epoch + 1, self.epochs,
                                total_loss / max(1, len(train_loader)))
                    continue
                val_pred = self._predict_windows(sequences[val_idx])
                val_f1 = f1_score(window_targets[val_idx], val_pred, average="macro", zero_division=0)
                logger.info("Epoch %d/%d | loss %.4f | val F1 macro %.4f", epoch + 1, self.epochs,
                            total_loss / max(1, len(train_loader)), val_f1)
                if val_f1 > best_f1:
                    best_f1, no_improve = val_f1, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                else:
                    no_improve += 1
                if no_improve >= self.patience:
                    logger.info("Early stopping at epoch %d.", epoch + 1)
                    break
            if best_state is not None:
                self.model.load_state_dict(best_state, strict=True)
        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        return self

    def predict_rows(self, X: Any, groups: Any = None) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(predicted_labels, row_positions)`` for every complete window."""
        groups = self._groups(X, groups)
        sequences, rows = self._windows(self._scaled(X), groups)
        return self.label_encoder.inverse_transform(self._predict_windows(sequences)), rows

    def predict(self, X: Any, groups: Any = None) -> pd.Series:
        """Predict class names; the result is indexed by the rows that have a full window."""
        labels, rows = self.predict_rows(X, groups)
        index = X.index[rows] if isinstance(X, pd.DataFrame) else pd.Index(rows)
        return pd.Series(labels, index=index, name="prediction")

    def evaluate(self, X: Any, y: Any = None, groups: Any = None) -> dict[str, Any]:
        """Evaluate on held-out wells (only rows with a complete window are scored)."""
        X, y = _unpack(X, y, "test")
        groups = self._groups(X, groups)
        y_array = np.asarray(y, dtype=object)
        predictions, rows = self.predict_rows(X, groups)
        y_rows = y_array[rows]
        keep = pd.Series(y_rows).notna().to_numpy()
        results = compute_metrics(y_rows[keep], predictions[keep], labels=list(self.label_encoder.classes_))
        results["n_rows_without_window"] = int(len(y_array) - len(rows))
        logger.info("%s — accuracy %.4f | F1 macro %.4f | F1 weighted %.4f", type(self).__name__,
                    results["accuracy"], results["f1_macro"], results["f1_weighted"])
        return results

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        if self.model is not None:
            state["model"] = None
            state["_model_state_dict"] = {k: v.cpu() for k, v in self.model.state_dict().items()}
            state["_model_shape"] = (self.model.lstm.input_size, self.model.fc[-1].out_features)
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        model_state = state.pop("_model_state_dict", None)
        shape = state.pop("_model_shape", None)
        self.__dict__.update(state)
        if model_state is not None and shape is not None:
            self.model = LSTMClassifier(
                num_features=shape[0], hidden_size=self.hidden_size, num_layers=self.num_layers,
                num_classes=shape[1], dropout=self._effective_dropout(), bidirectional=self.bidirectional,
            ).to(self._device())
            self.model.load_state_dict(model_state)
            self.model.eval()
