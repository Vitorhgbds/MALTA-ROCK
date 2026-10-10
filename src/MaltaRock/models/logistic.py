"""Logistic Regression classifier (reference: ``slide/models/logistic.py``)."""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression as LogisticRegressionClassifier

from MaltaRock.models.base import TabularModel


class LogisticRegression(TabularModel):
    """Multinomial logistic regression on median-imputed, standardised features.

    ``n_jobs`` is accepted for compatibility with the reference configs but
    not forwarded: scikit-learn >= 1.8 ignores it for this estimator and
    will remove it in 1.10.
    """

    def __init__(
        self,
        max_iter: int | None = None,
        C: float | None = None,
        solver: str = "lbfgs",
        random_state: int | None = 42,
        n_jobs: int = -1,
        verbose: int = 0,
    ) -> None:
        super().__init__()
        self.max_iter = max_iter
        self.C = C
        self.solver = solver
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.verbose = verbose

    def _make_estimator(self) -> LogisticRegressionClassifier:
        return LogisticRegressionClassifier(
            solver=self.solver,
            random_state=self.random_state,
            verbose=self.verbose,
            **self._without_none(max_iter=self.max_iter, C=self.C),
        )
