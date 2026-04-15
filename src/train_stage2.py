"""
Stage 2: multi-class attack classification with a weighted ensemble.

Default config reflects findings (see FINDINGS.md): RF+GBDT on the 9-class
setup with scaler fit on the full dataset. ProtoNet is available but excluded
by default because the weight sweep shows it's dead weight.

Usage:
    # Best baseline
    .venv/bin/python -m src.train_stage2 --dataset dataset.csv --test-n 500

    # Include ProtoNet (report's original design)
    .venv/bin/python -m src.train_stage2 --dataset dataset.csv --test-n 500 \\
        --models rf gbdt protonet --weights 0.4 0.3 0.3

    # Quick smoke test
    .venv/bin/python -m src.train_stage2 --dataset dataset.csv \\
        --sample-frac 0.05 --epochs 1 --epoch-size 100
"""

import argparse
import os
import pickle
import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocessing import load_dataset, get_stage2_data
from src.models import build_model
from src.models.ensemble import WeightedEnsemble


ORIG_LABEL_NAMES = {
    0: "Normal",
    1: "Grafana SSRF",
    2: "Node-RED Recon",
    3: "Node-RED RCE",
    4: "Node-RED Escape",
    6: "InfluxDB JWT",
    7: "runc race",
    8: "kubelet symlink",
    11: "Nuclei scanner",
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--train-n", type=int, default=80)
    p.add_argument("--test-n", type=int, default=500,
                   help="Balanced test samples per class. None = use all remaining.")

    # Data options (defaults reflect best config from FINDINGS.md)
    p.add_argument("--no-normal", dest="include_normal", action="store_false",
                   help="Exclude the normal class (8-class mode).")
    p.add_argument("--no-scaler-all", dest="scaler_all", action="store_false",
                   help="Fit scaler on filtered data only instead of full dataset.")
    p.set_defaults(include_normal=True, scaler_all=True)

    # Model selection
    p.add_argument("--models", nargs="+", default=["rf", "gbdt"],
                   help="Models to ensemble. Any combination of rf, gbdt, protonet.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Ensemble weights matching --models (must sum to 1). "
                        "Default: equal.")

    # ProtoNet hyperparameters (only used if 'protonet' in --models)
    p.add_argument("--encoder", choices=["cnn", "mlp"], default="mlp")
    p.add_argument("--distance", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=2000)
    p.add_argument("--k-shot", type=int, default=10)
    p.add_argument("--n-query", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=64)

    p.add_argument("--sweep", action="store_true",
                   help="Print the top-10 weight combinations over the trained models.")
    p.add_argument("--output", default="models/stage2.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model(name: str, args, n_features: int, n_classes: int):
    if name == "rf":
        return build_model("rf", n_estimators=200, random_state=args.seed)
    if name == "gbdt":
        return build_model("gbdt", random_state=args.seed)
    if name == "protonet":
        return build_model(
            "protonet",
            n_features=n_features,
            embedding_dim=args.embedding_dim,
            encoder=args.encoder,
            distance=args.distance,
            n_way=n_classes,
            k_shot=args.k_shot,
            n_query=args.n_query,
            epochs=args.epochs,
            epoch_size=args.epoch_size,
            random_state=args.seed,
        )
    raise ValueError(f"Unknown stage 2 model: {name}")


def resolve_weights(names, weights):
    if weights is None:
        return [1.0 / len(names)] * len(names)
    if len(weights) != len(names):
        raise ValueError(f"Got {len(weights)} weights for {len(names)} models")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
    return weights


def print_report(label, y_true, y_pred, target_names):
    print(f"\n{label} (accuracy={accuracy_score(y_true, y_pred):.4f}):")
    labels = list(range(len(target_names)))
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0,
    ))


def sweep_weights(probs_by_model, y_test, step: float = 0.1):
    """Exhaustive grid search over simplex weights for the trained models."""
    names = list(probs_by_model)
    n = len(names)
    n_steps = int(round(1 / step))
    results = []

    def walk(remaining, acc_weights):
        if len(acc_weights) == n - 1:
            acc_weights = acc_weights + [remaining / n_steps]
            w = dict(zip(names, acc_weights))
            combined = sum(w[k] * probs_by_model[k] for k in names)
            pred = np.argmax(combined, axis=1)
            results.append((accuracy_score(y_test, pred), w))
            return
        for i in range(remaining + 1):
            walk(remaining - i, acc_weights + [i / n_steps])

    walk(n_steps, [])
    return sorted(results, key=lambda r: -r[0])


def main():
    args = parse_args()

    print(f"Loading {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  {len(df):,} rows")

    X_train, y_train, X_test, y_test, label_map, scaler = get_stage2_data(
        df,
        train_n_per_class=args.train_n,
        test_n_per_class=args.test_n,
        random_state=args.seed,
        include_normal=args.include_normal,
        fit_scaler_on_all=args.scaler_all,
    )
    n_classes = len(label_map)
    n_features = X_train.shape[1]
    inv_label_map = {new: orig for orig, new in label_map.items()}
    target_names = [ORIG_LABEL_NAMES.get(inv_label_map[i], f"class_{i}")
                    for i in range(n_classes)]

    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  "
          f"Features: {n_features}  Classes: {n_classes}")
    for i in range(n_classes):
        tr = int((y_train == i).sum())
        te = int((y_test == i).sum())
        print(f"    {target_names[i]:20s}: {tr} train / {te} test")

    # Train each selected model
    trained = {}
    probs = {}
    for name in args.models:
        print(f"\nTraining {name} ...")
        m = make_model(name, args, n_features, n_classes)
        m.fit(X_train, y_train)
        trained[name] = m
        probs[name] = m.predict_proba(X_test)

    weights = resolve_weights(args.models, args.weights)
    ensemble = WeightedEnsemble([(trained[n], w) for n, w in zip(args.models, weights)])
    ens_pred = ensemble.predict(X_test)

    # Results
    print("\n" + "=" * 60)
    print("STAGE 2 RESULTS")
    print("=" * 60)
    for name in args.models:
        pred = np.argmax(probs[name], axis=1)
        print(f"{name:12s}: {accuracy_score(y_test, pred):.4f}")
    wstr = " ".join(f"{n}={w}" for n, w in zip(args.models, weights))
    print(f"ensemble    : {accuracy_score(y_test, ens_pred):.4f}  ({wstr})")

    if args.sweep:
        print("\nWeight sweep (top 10, grid step 0.1):")
        header = "  ".join(f"{n:>6}" for n in args.models)
        print(f"{'acc':>8}  {header}")
        for acc, w in sweep_weights(probs, y_test)[:10]:
            row = "  ".join(f"{w[n]:6.1f}" for n in args.models)
            print(f"{acc:8.4f}  {row}")

    for name in args.models:
        print_report(name, y_test, np.argmax(probs[name], axis=1), target_names)
    print_report(f"ensemble ({wstr})", y_test, ens_pred, target_names)

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "ensemble": ensemble,
            "scaler": scaler,
            "label_map": label_map,
            "models": args.models,
            "weights": weights,
        }, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
