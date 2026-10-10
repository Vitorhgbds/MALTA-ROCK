"""k-nearest neighbours classifier (reference: ``slide/models/knn.py``)."""

from __future__ import annotations

from sklearn.neighbors import KNeighborsClassifier

from MaltaRock.models.base import TabularModel


class KNN(TabularModel):
    """k-NN on median-imputed, standardised features."""

    def __init__(
        self,
        n_neighbors: int | None = None,
        weights: str | None = None,
        p: int | None = None,
        algorithm: str | None = None,
        metric: str = "minkowski",
        leaf_size: int = 30,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p
        self.algorithm = algorithm
        self.metric = metric
        self.leaf_size = leaf_size
        self.n_jobs = n_jobs

    def _make_estimator(self) -> KNeighborsClassifier:
        return KNeighborsClassifier(
            metric=self.metric,
            leaf_size=self.leaf_size,
            n_jobs=self.n_jobs,
            **self._without_none(
                n_neighbors=self.n_neighbors, weights=self.weights, p=self.p, algorithm=self.algorithm
            ),
        )
