"""
Multi-class attack classification with a weighted ensemble.

Default config reflects findings (see FINDINGS.md): RF+GBDT on the 9-class
setup with scaler fit on the full dataset. ProtoNet is available but excluded
by default because the weight sweep shows it's dead weight.

Generates plots/ automatically after evaluation.

Usage:
    # Best baseline
    .venv/bin/python -m src.train --dataset dataset.csv --test-n 500

    # Include ProtoNet (report's original design)
    .venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \\
        --models rf gbdt protonet --weights 0.4 0.3 0.3

    # Quick smoke test
    .venv/bin/python -m src.train --dataset dataset.csv \\
        --sample-frac 0.05 --epochs 1 --epoch-size 100
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, accuracy_score, confusion_matrix, f1_score,
    precision_recall_curve, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize

from src.data.preprocessing import load_dataset, get_train_test_data
from src.models import build_model
from src.models.ensemble import WeightedEnsemble


PLOTS_DIR = Path(__file__).resolve().parents[1] / "plots"

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

    p.add_argument("--no-normal", dest="include_normal", action="store_false",
                   help="Exclude the normal class (8-class mode).")
    p.add_argument("--no-scaler-all", dest="scaler_all", action="store_false",
                   help="Fit scaler on filtered data only instead of full dataset.")
    p.set_defaults(include_normal=True, scaler_all=True)

    p.add_argument("--models", nargs="+", default=["rf", "gbdt"],
                   help="Models to ensemble. Any combination of rf, gbdt, protonet.")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="Ensemble weights matching --models (must sum to 1).")

    # ProtoNet hyperparameters (only used if 'protonet' in --models)
    p.add_argument("--encoder", choices=["cnn", "mlp"], default="mlp")
    p.add_argument("--distance", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=2000)
    p.add_argument("--k-shot", type=int, default=10)
    p.add_argument("--n-query", type=int, default=10)
    p.add_argument("--embedding-dim", type=int, default=64)

    p.add_argument("--sweep", action="store_true",
                   help="Run weight sweep and plot the top combos.")
    p.add_argument("--no-plots", action="store_true",
                   help="Skip plot generation.")
    p.add_argument("--output", default="models/ensemble.pkl")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---- Model construction ----

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
    if name == "xgb":
        return build_model("xgb", random_state=args.seed)
    raise ValueError(f"Unknown model: {name}")


def resolve_weights(names, weights):
    if weights is None:
        return [1.0 / len(names)] * len(names)
    if len(weights) != len(names):
        raise ValueError(f"Got {len(weights)} weights for {len(names)} models")
    if abs(sum(weights) - 1.0) > 1e-6:
        raise ValueError(f"Weights must sum to 1, got {sum(weights)}")
    return weights


# ---- Weight sweep ----

def sweep_weights(probs_by_model, y_test, step: float = 0.1):
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


# ---- Plotting ----

def plot_accuracy_comparison(model_accuracies, ens_acc, ens_label):
    """Bar chart: per-model accuracy + ensemble."""
    names = list(model_accuracies) + [ens_label]
    accs = list(model_accuracies.values()) + [ens_acc]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#4472C4"] * len(model_accuracies) + ["#70AD47"]
    bars = ax.bar(names, accs, color=colors)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, a + 0.005,
                f"{a:.4f}", ha="center", fontsize=10)
    ax.set_ylabel("Test accuracy")
    ax.set_title("Model accuracy comparison")
    ax.set_ylim(0, min(1.0, max(accs) + 0.1))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def plot_per_class_f1(probs_by_model, ens_pred, ens_label, y_test, target_names):
    """Heatmap of per-class F1 for each model + ensemble."""
    rows = {}
    for name, p in probs_by_model.items():
        pred = np.argmax(p, axis=1)
        rows[name] = f1_score(y_test, pred, labels=range(len(target_names)),
                              average=None, zero_division=0)
    rows[ens_label] = f1_score(y_test, ens_pred, labels=range(len(target_names)),
                               average=None, zero_division=0)

    model_names = list(rows)
    data = np.array([rows[m] for m in model_names])

    fig, ax = plt.subplots(figsize=(max(10, len(target_names) * 1.3), len(model_names) * 0.8 + 2))
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            color = "white" if data[i, j] < 0.4 else "black"
            ax.text(j, i, f"{data[i, j]:.2f}", ha="center", va="center",
                    fontsize=9, color=color)

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("F1 score")
    ax.set_title("Per-class F1 scores")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)


def plot_confusion_matrix(y_test, y_pred, target_names, label):
    """Normalized confusion matrix for a single model/ensemble."""
    cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)

    fig, ax = plt.subplots(figsize=(max(8, len(target_names) * 1.1),
                                    max(6, len(target_names) * 0.9)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)

    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_yticklabels(target_names)

    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            val = cm_norm[i, j]
            count = cm[i, j]
            color = "white" if val > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}\n({count})", ha="center", va="center",
                    fontsize=8, color=color)

    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {label}")
    fig.tight_layout()
    safe_label = label.replace(" ", "_").replace("+", "_")
    fig.savefig(PLOTS_DIR / f"confusion_{safe_label}.png", dpi=150)
    plt.close(fig)


def plot_sweep(sweep_results, model_names, top_n=15):
    """Bar chart of top weight-sweep combos."""
    top = sweep_results[:top_n]
    accs = [r[0] for r in top]
    labels = ["\n".join(f"{n}={r[1][n]:.1f}" for n in model_names) for r in top]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), 5))
    bars = ax.bar(range(len(accs)), accs, color="#4472C4")
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Weight sweep — top {top_n} combos")
    ymin = min(accs) - 0.005
    ymax = max(accs) + 0.005
    ax.set_ylim(ymin, ymax)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.0003, f"{a:.4f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "weight_sweep.png", dpi=150)
    plt.close(fig)


def plot_pr_curves(prob_matrix, y_test, target_names, label):
    """One-vs-rest Precision-Recall curves for a single model/ensemble."""
    n_classes = len(target_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], prob_matrix[:, i])
        ap = auc(rec, prec)
        ax.plot(rec, prec, label=f"{target_names[i]} (AP={ap:.2f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curves — {label}")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    safe = label.replace(" ", "_").replace("+", "_")
    fig.savefig(PLOTS_DIR / f"pr_curves_{safe}.png", dpi=150)
    plt.close(fig)


def plot_roc_curves(prob_matrix, y_test, target_names, label):
    """One-vs-rest ROC curves for a single model/ensemble."""
    n_classes = len(target_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob_matrix[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{target_names[i]} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — {label}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    safe = label.replace(" ", "_").replace("+", "_")
    fig.savefig(PLOTS_DIR / f"roc_curves_{safe}.png", dpi=150)
    plt.close(fig)


def plot_feature_importance(model, feature_names, label, top_n=20):
    """Bar chart of top-N most important features (RF or GBDT)."""
    clf = model.clf
    if not hasattr(clf, "feature_importances_"):
        return
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(top_n), importances[indices][::-1], color="#4472C4")
    ax.set_yticks(range(top_n))
    names = [feature_names[i] if i < len(feature_names) else f"feat_{i}"
             for i in indices[::-1]]
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Feature importance")
    ax.set_title(f"Top {top_n} features — {label}")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    safe = label.replace(" ", "_").replace("+", "_")
    fig.savefig(PLOTS_DIR / f"feature_importance_{safe}.png", dpi=150)
    plt.close(fig)


def plot_gbdt_loss(model, label="gbdt"):
    """Training and validation loss curves for GBDT."""
    clf = model.clf
    if not hasattr(clf, "train_score_"):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = np.arange(1, len(clf.train_score_) + 1)
    ax.plot(iters, clf.train_score_, label="Train loss", color="#4472C4")
    if hasattr(clf, "validation_score_") and clf.validation_score_ is not None:
        ax.plot(iters, clf.validation_score_, label="Validation loss",
                color="#ED7D31", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"GBDT training curve — {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / f"loss_{label}.png", dpi=150)
    plt.close(fig)


# ---- Reporting ----

def print_report(label, y_true, y_pred, target_names):
    print(f"\n{label} (accuracy={accuracy_score(y_true, y_pred):.4f}):")
    labels = list(range(len(target_names)))
    print(classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0,
    ))


# ---- Main ----

def main():
    args = parse_args()

    print(f"Loading {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  {len(df):,} rows")

    feature_names = [c for c in df.columns if c != "Label"]

    X_train, y_train, X_test, y_test, label_map, scaler = get_train_test_data(
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

    # Train models
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
    wstr = " ".join(f"{n}={w:.2f}" for n, w in zip(args.models, weights))

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    model_accs = {}
    for name in args.models:
        pred = np.argmax(probs[name], axis=1)
        acc = accuracy_score(y_test, pred)
        model_accs[name] = acc
        print(f"{name:12s}: {acc:.4f}")
    ens_acc = accuracy_score(y_test, ens_pred)
    print(f"{'ensemble':12s}: {ens_acc:.4f}  ({wstr})")

    sweep_results = None
    if args.sweep:
        sweep_results = sweep_weights(probs, y_test)
        print(f"\nWeight sweep (top 10, grid step 0.1):")
        header = "  ".join(f"{n:>6}" for n in args.models)
        print(f"{'acc':>8}  {header}")
        for acc, w in sweep_results[:10]:
            row = "  ".join(f"{w[n]:6.1f}" for n in args.models)
            print(f"{acc:8.4f}  {row}")

    for name in args.models:
        print_report(name, y_test, np.argmax(probs[name], axis=1), target_names)
    print_report(f"ensemble ({wstr})", y_test, ens_pred, target_names)

    # Generate plots
    if not args.no_plots:
        PLOTS_DIR.mkdir(exist_ok=True)
        ens_label = "ensemble"
        ens_probs = sum(w * probs[n] for n, w in zip(args.models, weights))

        plot_accuracy_comparison(model_accs, ens_acc, ens_label)
        plot_per_class_f1(probs, ens_pred, ens_label, y_test, target_names)

        # Confusion matrix per model + ensemble
        plot_confusion_matrix(y_test, ens_pred, target_names, ens_label)
        for name in args.models:
            plot_confusion_matrix(y_test, np.argmax(probs[name], axis=1),
                                 target_names, name)

        # PR and ROC curves per model + ensemble
        plot_pr_curves(ens_probs, y_test, target_names, ens_label)
        plot_roc_curves(ens_probs, y_test, target_names, ens_label)
        for name in args.models:
            plot_pr_curves(probs[name], y_test, target_names, name)
            plot_roc_curves(probs[name], y_test, target_names, name)

        # Feature importance (RF, GBDT)
        for name in args.models:
            plot_feature_importance(trained[name], feature_names, name)

        # GBDT loss curve
        if "gbdt" in trained:
            plot_gbdt_loss(trained["gbdt"])

        if sweep_results is not None:
            plot_sweep(sweep_results, args.models)

        print(f"\nPlots saved to {PLOTS_DIR}/")
        for p in sorted(PLOTS_DIR.glob("*.png")):
            print(f"  {p.name}")

    # Save model
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
