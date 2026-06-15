"""
Random Forest wrapper for stage 2 multi-class attack classification.
Matches report: n_estimators=200, all other sklearn defaults.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForest:
    def __init__(self, n_estimators: int = 200, random_state: int = 42, class_weight=None):
        self.clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight=class_weight,
        )

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
