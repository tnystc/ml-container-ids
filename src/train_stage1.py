"""
Stage 1: binary anomaly detection (normal vs attack).

Choose an unsupervised detector via --model. Default is IsolationForest since
it gives the highest attack recall on this dataset. See FINDINGS.md for why
none of these match the report's claimed 98% accuracy.

Usage:
    .venv/bin/python -m src.train_stage1 --dataset dataset.csv --model iforest
    .venv/bin/python -m src.train_stage1 --dataset dataset.csv --model iforest --exploit-only
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocessing import load_dataset, get_stage1_data
from src.models import build_model, list_models


ANOMALY_MODELS = ["ocsvm-sigmoid", "ocsvm-rbf", "iforest"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Fraction of data to load (stratified). None = full dataset.")
    p.add_argument("--model", choices=ANOMALY_MODELS, default="iforest")
    p.add_argument("--train-cap", type=int, default=50_000,
                   help="Max normal training samples (OCSVM is O(n^2)).")
    p.add_argument("--nu", type=float, default=None,
                   help="OCSVM nu / IForest contamination. Model-specific default if unset.")
    p.add_argument("--gamma", default=None,
                   help="OCSVM gamma: 'scale', 'auto', or float.")
    p.add_argument("--no-scale", action="store_true",
                   help="Skip StandardScaler.")
    p.add_argument("--exploit-only", action="store_true",
                   help="Test only against rare exploit classes (3,4,6,7,8).")
    p.add_argument("--output", default="models/stage1.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  {len(df):,} rows")

    attack_labels = [3, 4, 6, 7, 8] if args.exploit_only else None
    X_train, X_test, y_test, scaler = get_stage1_data(
        df,
        scale=not args.no_scale,
        attack_labels=attack_labels,
        random_state=args.seed,
    )
    print(f"  Train (normal): {len(X_train):,}  "
          f"Test: {len(X_test):,} ({int(y_test.sum())} attacks)")

    if len(X_train) > args.train_cap:
        rng = np.random.default_rng(args.seed)
        idx = rng.choice(len(X_train), size=args.train_cap, replace=False)
        X_train = X_train[idx]
        print(f"  Subsampled train to {args.train_cap:,}")

    kwargs = {}
    if args.nu is not None:
        kwargs["nu" if args.model.startswith("ocsvm") else "contamination"] = args.nu
    if args.gamma is not None and args.model.startswith("ocsvm"):
        kwargs["gamma"] = args.gamma

    print(f"Training {args.model} {kwargs or '(defaults)'} ...")
    clf = build_model(args.model, **kwargs).fit(X_train)
    y_pred = clf.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(
        y_test, y_pred, target_names=["Normal", "Attack"], zero_division=0,
    ))

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"clf": clf, "scaler": scaler, "model": args.model}, f)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
