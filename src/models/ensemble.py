"""
Weighted probability ensemble over arbitrary classifiers.

All member models must output probabilities over the same class ordering.
"""

import numpy as np


class WeightedEnsemble:
    def __init__(self, members):
        """members: list of (model, weight) tuples. Weights should sum to 1."""
        total = sum(w for _, w in members)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Ensemble weights must sum to 1, got {total}")
        self.members = members

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return sum(w * m.predict_proba(X) for m, w in self.members)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)
