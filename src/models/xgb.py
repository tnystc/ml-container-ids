"""
XGBoost classifier wrapper with sklearn-style interface.

Note: on macOS, XGBoost may segfault with n_jobs=-1 due to libomp conflicts.
We default to n_jobs=1 and set OMP_NUM_THREADS=1 at import time.
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
from xgboost import XGBClassifier


class XGB:
    def __init__(
        self,
        n_estimators: int = 300,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        random_state: int = 42,
        eval_metric: str = "mlogloss",
    ):
        self.clf = XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
            eval_metric=eval_metric,
            use_label_encoder=False,
            n_jobs=1,
            verbosity=0,
        )
        self.eval_results_ = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict_proba(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.clf.predict(X)
