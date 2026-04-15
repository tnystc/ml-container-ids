"""
Stage 2: RF + ProtoNet ensemble for multi-class attack classification.

Training is limited to 80 examples per class (few-shot constraint from report).
Ensemble weights: ProtoNet=0.6, RF=0.4 (from report).

Evaluation modes:
  --test-n N   : balanced test set (N samples per class) — matches report conditions
  (default)    : all remaining samples (imbalanced, dominated by classes 1 and 2)

Usage:
    # Full balanced evaluation (matches report)
    .venv/bin/python -m src.train_stage2 --dataset dataset.csv --test-n 500

    # Quick smoke test
    .venv/bin/python -m src.train_stage2 --dataset dataset.csv --sample-frac 0.05 --epochs 1 --epoch-size 100
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocessing import load_dataset, get_stage2_data, STAGE2_CLASSES
from src.models.random_forest import RandomForest
from src.models.protonet import ProtoNet
from src.models.ensemble import RFProtoNetEnsemble


LABEL_NAMES = {
    0: "Grafana SSRF",
    1: "Node-RED Recon",
    2: "Node-RED RCE",
    3: "Node-RED Escape",
    4: "InfluxDB JWT",
    5: "runc race",
    6: "kubelet symlink",
    7: "Nuclei scanner",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--sample-frac", type=float, default=None,
                   help="Fraction of data to load (e.g. 0.05 for quick runs). Stratified per chunk.")
    p.add_argument("--train-n", type=int, default=80, help="Training samples per class")
    p.add_argument("--test-n", type=int, default=None,
                   help="Test samples per class cap (None = all remaining). "
                        "Set to a fixed value (e.g. 500) for balanced evaluation matching report.")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--epoch-size", type=int, default=2000)
    p.add_argument("--pn-weight", type=float, default=0.6)
    p.add_argument("--rf-weight", type=float, default=0.4)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--k-shot", type=int, default=5)
    p.add_argument("--n-query", type=int, default=10)
    p.add_argument("--output", default="models/stage2_ensemble.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def print_report(label, y_true, y_pred, target_names):
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{label} (accuracy={acc:.4f}):")
    print(classification_report(y_true, y_pred, target_names=target_names, zero_division=0))


def main():
    args = parse_args()

    print(f"Loading dataset from {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  Loaded {len(df):,} rows")

    print(f"Preparing stage 2 data (classes {STAGE2_CLASSES}, train={args.train_n}/class, test={args.test_n or 'all'}/class) ...")
    X_train, y_train, X_test, y_test, label_map, scaler = get_stage2_data(
        df,
        train_n_per_class=args.train_n,
        test_n_per_class=args.test_n,
        random_state=args.seed,
    )
    n_classes = len(label_map)
    n_features = X_train.shape[1]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  Features: {n_features}  Classes: {n_classes}")
    for orig, new in sorted(label_map.items()):
        tr = (y_train == new).sum()
        te = (y_test == new).sum()
        print(f"    class {orig} ({LABEL_NAMES.get(new, '?')}): {tr} train / {te} test")

    target_names = [LABEL_NAMES.get(i, f"class_{i}") for i in range(n_classes)]

    # ---- Random Forest ----
    print("\nTraining Random Forest ...")
    rf = RandomForest(n_estimators=200, random_state=args.seed)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # ---- ProtoNet ----
    print("\nTraining ProtoNet ...")
    pn = ProtoNet(
        n_features=n_features,
        embedding_dim=args.embedding_dim,
        n_way=n_classes,
        k_shot=args.k_shot,
        n_query=args.n_query,
        epochs=args.epochs,
        epoch_size=args.epoch_size,
        lr=1e-3,
        random_state=args.seed,
    )
    pn.fit(X_train, y_train)
    pn_pred = pn.predict(X_test)

    # ---- Ensemble ----
    ensemble = RFProtoNetEnsemble(rf, pn, pn_weight=args.pn_weight, rf_weight=args.rf_weight)
    ens_pred = ensemble.predict(X_test)

    # ---- Results ----
    print("\n" + "=" * 60)
    print("STAGE 2 RESULTS")
    print("=" * 60)

    rf_acc = accuracy_score(y_test, rf_pred)
    pn_acc = accuracy_score(y_test, pn_pred)
    ens_acc = accuracy_score(y_test, ens_pred)

    print(f"RF:         {rf_acc:.4f}")
    print(f"ProtoNet:   {pn_acc:.4f}")
    print(f"Ensemble:   {ens_acc:.4f}  (pn={args.pn_weight}, rf={args.rf_weight})")

    print_report("Random Forest", y_test, rf_pred, target_names)
    print_report("ProtoNet", y_test, pn_pred, target_names)
    print_report(f"Ensemble (pn={args.pn_weight}, rf={args.rf_weight})", y_test, ens_pred, target_names)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({"ensemble": ensemble, "scaler": scaler, "label_map": label_map}, f)
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
