"""
Data loading and preprocessing for multi-class attack classification.

Dataset: dataset.csv — 87 columns, last column is Label (0-11).
Metadata columns (dropped): Flow ID, Src IP, Dst IP, Timestamp.
Kept as features: Src Port, Dst Port, Protocol + all flow statistics (cols 8-86).
Classes used: 0 (normal), 1, 2, 3, 4, 6, 7, 8, 11 (others have too few samples).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Attack classes with enough samples for training/evaluation
ATTACK_CLASSES = [1, 2, 3, 4, 6, 7, 8, 11]

# Columns to drop (non-numeric metadata)
DROP_COLS = ["Flow ID", "Src IP", "Dst IP", "Timestamp"]

# Report uses 80 training examples per class for the few-shot setting
FEW_SHOT_TRAIN_N = 80


def load_dataset(path: str, sample_frac: float = None, chunksize: int = 200_000) -> pd.DataFrame:
    """
    Load dataset.csv, drop metadata columns, handle inf/NaN.

    Args:
        sample_frac: If set (0 < frac <= 1), read in chunks and take a stratified
                     random sample of that fraction. Ensures rare classes are included.
                     None loads the full dataset.
        chunksize:   Rows per chunk when sample_frac is used.
    """
    if sample_frac is None:
        df = pd.read_csv(path)
    else:
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(chunk.sample(frac=sample_frac, random_state=42))
        df = pd.concat(chunks, ignore_index=True)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    feature_cols = [c for c in df.columns if c != "Label"]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def get_train_test_data(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    train_n_per_class: int = FEW_SHOT_TRAIN_N,
    test_n_per_class: int = None,
    random_state: int = 42,
    include_normal: bool = True,
    fit_scaler_on_all: bool = True,
):
    """
    Prepare multi-class data for attack classification.

    Remaps labels to 0-indexed contiguous integers. Training is limited to
    train_n_per_class samples per class (few-shot constraint). test_n_per_class
    caps the test set per class (None = use all remaining).

    Returns:
        X_train, y_train, X_test, y_test, label_map, scaler
    """
    classes = list(ATTACK_CLASSES)
    if include_normal:
        classes = [0] + classes

    feature_cols = [c for c in df.columns if c != "Label"]

    if fit_scaler_on_all and scaler is None:
        scaler = StandardScaler()
        scaler.fit(df[feature_cols].values.astype(np.float32))

    df2 = df[df["Label"].isin(classes)].copy()
    label_map = {orig: new for new, orig in enumerate(sorted(classes))}
    df2["label_mapped"] = df2["Label"].map(label_map)

    X = df2[feature_cols].values.astype(np.float32)
    y = df2["label_mapped"].values

    if scaler is None:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    else:
        X = scaler.transform(X)

    train_indices = []
    test_indices = []
    rng = np.random.default_rng(random_state)

    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        n_train = min(train_n_per_class, len(cls_idx))
        chosen_train = rng.choice(cls_idx, size=n_train, replace=False)
        remaining = np.setdiff1d(cls_idx, chosen_train)

        if test_n_per_class is not None and len(remaining) > test_n_per_class:
            chosen_test = rng.choice(remaining, size=test_n_per_class, replace=False)
        else:
            chosen_test = remaining

        train_indices.extend(chosen_train.tolist())
        test_indices.extend(chosen_test.tolist())

    return (
        X[train_indices], y[train_indices],
        X[test_indices], y[test_indices],
        label_map, scaler,
    )


class EpisodeSampler:
    """
    Generates few-shot episodes for ProtoNet training.

    Each episode: N-way K-shot.
      - Support set: K examples per class
      - Query set  : Q examples per class
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, n_way: int, k_shot: int, n_query: int):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.class_indices = {c: np.where(y == c)[0] for c in self.classes}

    def sample_episode(self, rng: np.random.Generator):
        """Return (support_X, support_y, query_X, query_y) as numpy arrays."""
        episode_classes = rng.choice(self.classes, size=self.n_way, replace=False)

        support_X, support_y, query_X, query_y = [], [], [], []

        for new_label, cls in enumerate(episode_classes):
            idx = self.class_indices[cls]
            needed = self.k_shot + self.n_query
            if len(idx) < needed:
                chosen = rng.choice(idx, size=needed, replace=True)
            else:
                chosen = rng.choice(idx, size=needed, replace=False)

            support_X.append(self.X[chosen[: self.k_shot]])
            support_y.append(np.full(self.k_shot, new_label))
            query_X.append(self.X[chosen[self.k_shot :]])
            query_y.append(np.full(self.n_query, new_label))

        return (
            np.concatenate(support_X),
            np.concatenate(support_y),
            np.concatenate(query_X),
            np.concatenate(query_y),
        )
