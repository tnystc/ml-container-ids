"""
SMOTE-family oversampling for the few-shot training set.

Kept for literature comparison only — all three variants (SMOTE,
Borderline-SMOTE, ADASYN) were shown to hurt the ensemble (see README.md).
Not part of the recommended pipeline. `random` is naive replication.
"""

import numpy as np


def apply_oversampler(X_train, y_train, kind, k, target, seed, target_names):
    """Synthesize minority samples to balance the training set.

    Args:
        kind: "smote" | "borderline" | "adasyn"
        k: requested k_neighbors (clipped to min_class_count - 1)
        target: target samples per class (None = match the largest class)
        seed: random_state
        target_names: per-class display names, for the printout

    Returns: (X_aug, y_aug) as float32 / int arrays.
    """
    from imblearn.over_sampling import (
        SMOTE, BorderlineSMOTE, ADASYN, RandomOverSampler,
    )

    counts = np.bincount(y_train)
    min_count = int(counts.min())
    k = max(1, min(k, min_count - 1))
    target = target or int(counts.max())
    strategy = {c: max(int(counts[c]), target) for c in range(len(counts))}

    if kind == "random":
        # Naive replication: duplicate existing minority samples with replacement.
        # No synthetic interpolation, no neighbor parameter.
        sampler = RandomOverSampler(sampling_strategy=strategy, random_state=seed)
    else:
        sampler_cls = {
            "smote":      SMOTE,
            "borderline": BorderlineSMOTE,
            "adasyn":     ADASYN,
        }[kind]
        # ADASYN names its neighbor parameter `n_neighbors`; SMOTE family uses `k_neighbors`.
        nbr_kw = "n_neighbors" if kind == "adasyn" else "k_neighbors"
        sampler = sampler_cls(sampling_strategy=strategy, random_state=seed,
                              **{nbr_kw: k})

    X_aug, y_aug = sampler.fit_resample(X_train, y_train)
    print(f"\n  Oversampler={kind}  k_neighbors={k}  target/class={target}")
    print(f"  Train rows: {len(X_train)} -> {len(X_aug)}")
    for i in range(len(counts)):
        before, after = int(counts[i]), int((y_aug == i).sum())
        if before != after:
            print(f"    {target_names[i]:20s}: {before} -> {after}")
    return X_aug.astype(np.float32), y_aug
