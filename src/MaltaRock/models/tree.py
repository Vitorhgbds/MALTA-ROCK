"""Decision Tree classifier (reference: ``slide/models/tree.py``)."""

from __future__ import annotations

from sklearn.tree import DecisionTreeClassifier

from MaltaRock.models.base import TabularModel


class DecisionTree(TabularModel):
    """Decision tree with ``class_weight="balanced"`` (as in the reference)."""

    def __init__(
        self,
        min_samples_split: int | None = None,
        min_samples_leaf: int | None = None,
        max_depth: int | None = None,
        random_state: int | None = 42,
    ) -> None:
        super().__init__()
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_depth = max_depth
        self.random_state = random_state

    def _make_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self.max_depth,
            random_state=self.random_state,
            class_weight="balanced",
            **self._without_none(
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
            ),
        )
