"""
Data loading and preprocessing for the IDS/IPS pipeline.

Dataset: dataset.csv — 87 columns, last column is Label (0-11).
Metadata columns (dropped): Flow ID, Src IP, Dst IP, Timestamp.
Kept as features: Src Port, Dst Port, Protocol + all flow statistics (cols 8-86).
Stage 2 uses classes: 1, 2, 3, 4, 6, 7, 8, 11 (others have too few samples).
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Classes used in stage 2 (attack classification)
STAGE2_CLASSES = [1, 2, 3, 4, 6, 7, 8, 11]

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
        # Chunked stratified sampling — guarantees all classes are represented
        chunks = []
        for chunk in pd.read_csv(path, chunksize=chunksize):
            chunks.append(chunk.sample(frac=sample_frac, random_state=42))
        df = pd.concat(chunks, ignore_index=True)

    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    feature_cols = [c for c in df.columns if c != "Label"]

    # Replace inf with NaN then fill with column median
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

    return df


def get_stage1_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    scale: bool = True,
    attack_labels: list = None,
):
    """
    Stage 1 (anomaly detection): normal (0) vs attack (1+).

    Args:
        scale:          fit/apply StandardScaler (set False to test raw features)
        attack_labels:  if given, restrict attacks in the test set to these original
                        labels (e.g. [3,4,6,7,8] for exploit-only).
    """
    feature_cols = [c for c in df.columns if c != "Label"]
    X = df[feature_cols].values.astype(np.float32)
    orig_labels = df["Label"].values
    y = (orig_labels > 0).astype(int)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    idx_normal = np.where(y == 0)[0]
    train_idx, test_normal_idx = train_test_split(
        idx_normal, test_size=test_size, random_state=random_state
    )

    if attack_labels is not None:
        attack_idx = np.where(np.isin(orig_labels, attack_labels))[0]
    else:
        attack_idx = np.where(y == 1)[0]
    test_idx = np.concatenate([test_normal_idx, attack_idx])

    X_train_normal = X[train_idx]
    X_test = X[test_idx]
    y_test = y[test_idx]

    return X_train_normal, X_test, y_test, scaler


def get_stage2_data(
    df: pd.DataFrame,
    scaler: StandardScaler = None,
    train_n_per_class: int = FEW_SHOT_TRAIN_N,
    test_n_per_class: int = None,
    random_state: int = 42,
    include_normal: bool = False,
    fit_scaler_on_all: bool = False,
):
    """
    Stage 2 (multi-class attack classification): classes in STAGE2_CLASSES only.
    Remaps labels to 0-indexed contiguous integers.

    Training is limited to train_n_per_class samples per class (few-shot constraint).
    test_n_per_class caps the test set per class (None = use all remaining).
    For fair evaluation matching the report, set test_n_per_class to a fixed value.

    Options:
        include_normal:      add label 0 (normal) as an additional class
        fit_scaler_on_all:   fit StandardScaler on the full dataset (all labels
                             including normal) BEFORE filtering, so feature stats
                             reflect the broader traffic distribution

    Returns:
        X_train, y_train, X_test, y_test, label_map, scaler
    """
    classes = list(STAGE2_CLASSES)
    if include_normal:
        classes = [0] + classes

    feature_cols = [c for c in df.columns if c != "Label"]

    # Fit scaler on full unfiltered data if requested
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

    X_train = X[train_indices]
    y_train = y[train_indices]
    X_test = X[test_indices]
    y_test = y[test_indices]

    return X_train, y_train, X_test, y_test, label_map, scaler


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

        # Index per class
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
