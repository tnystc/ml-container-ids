"""
Stage 1: OneClassSVM anomaly detection (normal vs. attack).

Trains on normal traffic only (label 0).
Reports accuracy, precision, recall, F1 on the held-out test set.

Usage:
    .venv/bin/python -m src.train_stage1 --dataset dataset.csv
"""

import argparse
import pickle
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocessing import load_dataset, get_stage1_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Fraction of data to use (e.g. 0.1 for quick runs). Stratified across chunks.")
    p.add_argument("--output", default="models/stage1_ocsvm.pkl")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  Loaded {len(df):,} rows")

    print("Preparing stage 1 data ...")
    X_train_normal, X_test, y_test, scaler = get_stage1_data(df)
    print(f"  Train (normal only): {len(X_train_normal):,}")
    print(f"  Test (normal + attack): {len(X_test):,}  (attacks: {y_test.sum():,})")

    # Best hyperparameters from report
    print("Training OneClassSVM ...")
    clf = OneClassSVM(
        coef0=0.5,
        degree=2,
        gamma=1.0,
        kernel="sigmoid",
        nu=0.01,
    )
    clf.fit(X_train_normal)

    # OneClassSVM returns +1 (normal) and -1 (anomaly)
    raw_pred = clf.predict(X_test)
    y_pred = (raw_pred == -1).astype(int)  # 1 = attack, 0 = normal

    print("\n--- Stage 1 Results ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=["Normal", "Attack"]))

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler}, f)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
