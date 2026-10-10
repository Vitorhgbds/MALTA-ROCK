"""Random Forest classifier (reference: ``slide/models/forest.py``)."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier

from MaltaRock.models.base import TabularModel


class RandomForest(TabularModel):
    """Random Forest on median-imputed, standardised tabular features.

    ``None`` parameters fall back to scikit-learn's defaults. The class
    weighting present but commented out in the reference implementation is
    likewise not applied.

    Args:
        n_estimators: Number of trees.
        min_samples_split: Minimum samples required to split a node.
        min_samples_leaf: Minimum samples required at a leaf.
        max_features: Features considered per split (``"sqrt"``, ``"log2"`` or float).
        max_samples: Fraction of samples drawn for each tree.
        bootstrap: Whether bootstrap samples are used.
        random_state: Random seed.
        n_jobs: CPU cores (-1 = all).
        verbose: scikit-learn verbosity (the reference used 1).
    """

    def __init__(
        self,
        n_estimators: int | None = None,
        min_samples_split: int | None = None,
        min_samples_leaf: int | None = None,
        max_features: str | float | None = None,
        max_samples: float | None = None,
        bootstrap: bool = True,
        random_state: int | None = 42,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _make_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            bootstrap=self.bootstrap,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            **self._without_none(
                n_estimators=self.n_estimators,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_samples=self.max_samples,
            ),
        )

    def _extra_results(self) -> dict[str, Any]:
        oob = getattr(self.model, "oob_score_", None)
        return {
            "feature_importances": self.model.feature_importances_.tolist(),
            "feature_names": self._input_feature_names(),
            "oob_score": float(oob) if oob is not None else None,
        }
