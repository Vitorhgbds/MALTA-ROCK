"""XGBoost classifier (reference: ``slide/models/xgboost.py``). Requires ``xgboost``."""

from __future__ import annotations

from typing import Any

from MaltaRock.models.base import TabularModel


class XGBoost(TabularModel):
    """Gradient-boosted trees on median-imputed, standardised features.

    Defaults (``tree_method="hist"``, ``eval_metric="mlogloss"``,
    ``learning_rate=0.0001``) are those of the reference implementation.
    """

    def __init__(
        self,
        n_estimators: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        tree_method: str = "hist",
        eval_metric: str = "mlogloss",
        learning_rate: float | None = 0.0001,
        max_depth: int | None = None,
        min_child_weight: int | None = None,
        random_state: int | None = 42,
        n_jobs: int = -1,
    ) -> None:
        super().__init__()
        self.n_estimators = n_estimators
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.tree_method = tree_method
        self.eval_metric = eval_metric
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_child_weight = min_child_weight
        self.random_state = random_state
        self.n_jobs = n_jobs

    def _make_estimator(self) -> Any:
        try:
            import xgboost as xgb
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ImportError("XGBoost requires the optional dependency: pip install 'malta-rock[xgboost]'") from exc
        return xgb.XGBClassifier(
            tree_method=self.tree_method,
            eval_metric=self.eval_metric,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
            **self._without_none(
                n_estimators=self.n_estimators,
                subsample=self.subsample,
                colsample_bytree=self.colsample_bytree,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                min_child_weight=self.min_child_weight,
            ),
        )

    def _extra_results(self) -> dict[str, Any]:
        return {
            "feature_importances": self.model.feature_importances_.tolist(),
            "feature_names": self._input_feature_names(),
        }
