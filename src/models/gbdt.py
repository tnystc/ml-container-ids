"""
Histogram Gradient Boosting classifier for stage 2.
sklearn-native, no extra dependencies.

Usually beats Random Forest on tabular classification,
especially with imbalanced classes.
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier


class GBDT:
    def __init__(
        self,
        max_iter: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
        class_weight=None,
    ):
        self.clf = HistGradientBoostingClassifier(
            max_iter=max_iter,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            class_weight=class_weight,
            early_stopping=True,
            n_iter_no_change=max_iter,  # never stop — just track scores
            validation_fraction=0.15,
            scoring="loss",
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
