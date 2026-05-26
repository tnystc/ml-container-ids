"""
Multi-class attack classification with a weighted ensemble.

Best baseline is RF + XGB + Reptile on the 9-class setup (see PROJECT_REPORT.md
for the full experimental record). ProtoNet and MLP are available as documented
baselines but excluded by default because the weight sweep shows they're dead
weight. Plots are generated automatically after evaluation unless --no-plots.

Usage:
    # Best baseline (~10 min, 0.7503 / 0.60 macro F1)
    .venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \\
        --models rf gbdt xgb reptile --epochs 10 --epoch-size 2000 --sweep

    # Kill-chain stages (4 classes, 0.8501 / 0.84 macro F1)
    .venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \\
        --label-config killchain --models rf gbdt xgb reptile --sweep

    # Quick smoke test
    .venv/bin/python -m src.train --sample-frac 0.05 --epochs 1 --epoch-size 100
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, accuracy_score

from src.data.preprocessing import load_dataset, get_train_test_data
from src.labels import LABEL_CONFIGS, apply_label_config
from src.sampling import apply_oversampler
from src.models import build_model
from src.models.ensemble import WeightedEnsemble
from src import plotting

PLOTS_ROOT = Path(__file__).resolve().parents[1] / "plots"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--train-n", type=int, default=80)
    p.add_argument("--test-n", type=int, default=500,
                   help="Balanced test samples per class. None = use all remaining.")

    p.add_argument("--no-normal", dest="include_normal", action="store_false",
                   help="Exclude the normal class (8-class mode).")
    p.add_argument("--no-scaler-all", dest="scaler_all", action="store_false",
                   help="Fit scaler on filtered data only instead of full dataset.")
    p.set_defaults(include_normal=True, scaler_all=True)

    p.add_argument("--models", nargs="+", default=["rf", "gbdt", "xgb"],
                   help="Models to ensemble: rf gbdt xgb reptile protonet mlp.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Ensemble weights matching --models (must sum to 1).")

    # ProtoNet / Reptile hyperparameters (episodic models)
    p.add_argument("--encoder", choices=["cnn", "mlp"], default="mlp")
    p.add_argument("--distance", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=2000)
    p.add_argument("--k-shot", type=int, default=10)
    p.add_argument("--n-query", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=64)

    # MLP hyperparameters
    p.add_argument("--mlp-epochs", type=int, default=100)

    # SMOTE-family oversampling (literature comparison; hurts — see PROJECT_REPORT.md)
    p.add_argument("--smote", choices=["none", "smote", "borderline", "adasyn"],
                   default="none",
                   help="Apply SMOTE-family oversampling to the few-shot training set.")
    p.add_argument("--smote-k", type=int, default=5,
                   help="k_neighbors for SMOTE/Borderline/ADASYN (clipped to "
                        "min_class_count - 1).")
    p.add_argument("--smote-target", type=int, default=None,
                   help="Target samples per class after oversampling (default: "
                        "match the largest training class).")

    p.add_argument("--label-config", choices=sorted(LABEL_CONFIGS.keys()),
                   default="none",
                   help="Label-grouping preset for ablation. Applied after "
                        "few-shot sampling, so each constituent class still "
                        "gets --train-n samples and supergroups inherit them.")

    p.add_argument("--sweep", action="store_true",
                   help="Run weight sweep and plot the top combos.")
    p.add_argument("--no-plots", action="store_true", help="Skip plot generation.")
    p.add_argument("--plots-subdir", default=None,
                   help="If set, save plots under plots/<subdir>/ instead of plots/.")
    p.add_argument("--output", default="models/ensemble.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def make_model(name: str, args, n_features: int, n_classes: int):
    """Build a registered model with run-specific hyperparameters from args."""
    if name == "rf":
        return build_model("rf", n_estimators=200, random_state=args.seed)
    if name == "gbdt":
        return build_model("gbdt", random_state=args.seed)
    if name == "xgb":
        return build_model("xgb", random_state=args.seed)
    if name == "protonet":
        return build_model(
            "protonet", n_features=n_features, embedding_dim=args.embedding_dim,
            encoder=args.encoder, distance=args.distance, n_way=n_classes,
            k_shot=args.k_shot, n_query=args.n_query, epochs=args.epochs,
            epoch_size=args.epoch_size, random_state=args.seed,
        )
    if name == "reptile":
        return build_model(
            "reptile", n_features=n_features, embedding_dim=args.embedding_dim,
            n_way=min(5, n_classes), k_shot=args.k_shot, n_query=args.n_query,
            epochs=args.epochs, epoch_size=args.epoch_size, random_state=args.seed,
        )
    if name == "mlp":
        return build_model(
            "mlp", n_features=n_features, n_classes=n_classes,
            epochs=args.mlp_epochs, random_state=args.seed,
        )
    raise ValueError(f"Unknown model: {name}")


def resolve_weights(names, weights):
    if weights is None:
        return [1.0 / len(names)] * len(names)
    if len(weights) != len(names):
        raise ValueError(f"Got {len(weights)} weights for {len(names)} models")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
    return weights


def sweep_weights(probs_by_model, y_test, step: float = 0.1):
    """Exhaustive grid sweep over ensemble weights; returns combos sorted by acc."""
    names = list(probs_by_model)
    n = len(names)
    n_steps = int(round(1 / step))
    results = []

    def walk(remaining, acc_weights):
        if len(acc_weights) == n - 1:
            acc_weights = acc_weights + [remaining / n_steps]
            w = dict(zip(names, acc_weights))
            combined = sum(w[k] * probs_by_model[k] for k in names)
            results.append((accuracy_score(y_test, np.argmax(combined, axis=1)), w))
            return
        for i in range(remaining + 1):
            walk(remaining - i, acc_weights + [i / n_steps])

    walk(n_steps, [])
    return sorted(results, key=lambda r: -r[0])


def print_report(label, y_true, y_pred, target_names):
    print(f"\n{label} (accuracy={accuracy_score(y_true, y_pred):.4f}):")
    print(classification_report(
        y_true, y_pred, labels=list(range(len(target_names))),
        target_names=target_names, zero_division=0,
    ))


def main():
    args = parse_args()
    plots_dir = PLOTS_ROOT / args.plots_subdir if args.plots_subdir else PLOTS_ROOT

    print(f"Loading {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  {len(df):,} rows")
    feature_names = [c for c in df.columns if c != "Label"]

    X_train, y_train, X_test, y_test, label_map, scaler = get_train_test_data(
        df, train_n_per_class=args.train_n, test_n_per_class=args.test_n,
        random_state=args.seed, include_normal=args.include_normal,
        fit_scaler_on_all=args.scaler_all,
    )
    X_train, y_train, X_test, y_test, target_names = apply_label_config(
        X_train, y_train, X_test, y_test, label_map, args.label_config,
    )
    n_classes = len(target_names)
    n_features = X_train.shape[1]

    if args.label_config != "none":
        print(f"  Label config: {args.label_config} -> {n_classes} classes")
    print(f"  Train: {len(X_train):,}  Test: {len(X_test):,}  "
          f"Features: {n_features}  Classes: {n_classes}")
    for i in range(n_classes):
        print(f"    {target_names[i]:20s}: "
              f"{int((y_train == i).sum())} train / {int((y_test == i).sum())} test")

    if args.smote != "none":
        X_train, y_train = apply_oversampler(
            X_train, y_train, args.smote, args.smote_k, args.smote_target,
            args.seed, target_names,
        )

    # Train models
    trained, probs = {}, {}
    for name in args.models:
        print(f"\nTraining {name} ...")
        m = make_model(name, args, n_features, n_classes)
        m.fit(X_train, y_train)
        trained[name] = m
        probs[name] = m.predict_proba(X_test)

    weights = resolve_weights(args.models, args.weights)
    ensemble = WeightedEnsemble([(trained[n], w) for n, w in zip(args.models, weights)])
    ens_pred = ensemble.predict(X_test)
    wstr = " ".join(f"{n}={w:.2f}" for n, w in zip(args.models, weights))

    print("\n" + "=" * 60 + "\nRESULTS\n" + "=" * 60)
    model_accs = {}
    for name in args.models:
        acc = accuracy_score(y_test, np.argmax(probs[name], axis=1))
        model_accs[name] = acc
        print(f"{name:12s}: {acc:.4f}")
    ens_acc = accuracy_score(y_test, ens_pred)
    print(f"{'ensemble':12s}: {ens_acc:.4f}  ({wstr})")

    sweep_results = None
    if args.sweep:
        sweep_results = sweep_weights(probs, y_test)
        print("\nWeight sweep (top 10, grid step 0.1):")
        print(f"{'acc':>8}  " + "  ".join(f"{n:>6}" for n in args.models))
        for acc, w in sweep_results[:10]:
            print(f"{acc:8.4f}  " + "  ".join(f"{w[n]:6.1f}" for n in args.models))

    for name in args.models:
        print_report(name, y_test, np.argmax(probs[name], axis=1), target_names)
    print_report(f"ensemble ({wstr})", y_test, ens_pred, target_names)

    if not args.no_plots:
        ens_probs = sum(w * probs[n] for n, w in zip(args.models, weights))
        pngs = plotting.generate_plots(
            plots_dir, model_names=args.models, probs=probs, ens_probs=ens_probs,
            ens_pred=ens_pred, y_test=y_test, target_names=target_names,
            model_accs=model_accs, ens_acc=ens_acc, trained=trained,
            feature_names=feature_names, sweep_results=sweep_results,
        )
        print(f"\nPlots saved to {plots_dir}/")
        for p in pngs:
            print(f"  {p.name}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump({
            "ensemble": ensemble, "scaler": scaler, "label_map": label_map,
            "models": args.models, "weights": weights,
        }, f)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
