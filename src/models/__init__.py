"""
Model registry for swappable classifiers.

Every model wrapper exposes a sklearn-style interface:
    fit(X, y=None)   -> self
    predict(X)       -> (n,) int labels
    predict_proba(X) -> (n, n_classes) float probabilities  (supervised only)

Add a new model by creating a wrapper in this package and registering it in
`_REGISTRY` below. Instantiate with `build_model("name", **kwargs)` from train
scripts, so swapping architectures is a command-line change.

Why this file has code but `src/__init__.py` and `src/data/__init__.py` are empty:
    Python needs a file named `__init__.py` in a directory to treat it as an
    importable package. The ones in `src/` and `src/data/` are empty package
    markers — they exist only so `from src.data.preprocessing import ...` works.
    This file is also a package marker, but since the models package has a
    public API (the registry), we put that API here.
"""

from src.models.random_forest import RandomForest
from src.models.gbdt import GBDT
from src.models.protonet import ProtoNet
from src.models.xgb import XGB
from src.models.reptile import Reptile
from src.models.mlp import MLP

_REGISTRY = {
    "rf":       RandomForest,
    "gbdt":     GBDT,
    "protonet": ProtoNet,
    "xgb":      XGB,
    "reptile":  Reptile,
    "mlp":      MLP,
}


def build_model(name: str, **kwargs):
    if name not in _REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def list_models():
    return sorted(_REGISTRY)
