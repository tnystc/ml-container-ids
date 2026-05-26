"""
Plot generation for training runs.

All figures are written into a given output directory. `generate_plots` is
the single entry point called by train.py; the individual `_plot_*` helpers
are kept private. Matplotlib runs headless (Agg backend).
"""

from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    confusion_matrix, f1_score, precision_recall_curve, roc_curve, auc,
)
from sklearn.preprocessing import label_binarize


def _safe(label: str) -> str:
    return label.replace(" ", "_").replace("+", "_")


def _plot_accuracy_comparison(plots_dir, model_accuracies, ens_acc, ens_label):
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
    fig.savefig(plots_dir / "accuracy_comparison.png", dpi=150)
    plt.close(fig)


def _plot_per_class_f1(plots_dir, probs_by_model, ens_pred, ens_label,
                       y_test, target_names):
    rows = {}
    for name, p in probs_by_model.items():
        pred = np.argmax(p, axis=1)
        rows[name] = f1_score(y_test, pred, labels=range(len(target_names)),
                              average=None, zero_division=0)
    rows[ens_label] = f1_score(y_test, ens_pred, labels=range(len(target_names)),
                               average=None, zero_division=0)

    model_names = list(rows)
    data = np.array([rows[m] for m in model_names])

    fig, ax = plt.subplots(figsize=(max(10, len(target_names) * 1.3),
                                    len(model_names) * 0.8 + 2))
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
    fig.savefig(plots_dir / "per_class_f1.png", dpi=150)
    plt.close(fig)


def _plot_confusion_matrix(plots_dir, y_test, y_pred, target_names, label):
    cm = confusion_matrix(y_test, y_pred, labels=range(len(target_names)))
    cm_norm = np.nan_to_num(cm.astype(float) / cm.sum(axis=1, keepdims=True))

    fig, ax = plt.subplots(figsize=(max(8, len(target_names) * 1.1),
                                    max(6, len(target_names) * 0.9)))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(target_names)))
    ax.set_xticklabels(target_names, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(target_names)))
    ax.set_yticklabels(target_names)
    for i in range(cm_norm.shape[0]):
        for j in range(cm_norm.shape[1]):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]:.2f}\n({cm[i, j]})", ha="center",
                    va="center", fontsize=8, color=color)
    fig.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix — {label}")
    fig.tight_layout()
    fig.savefig(plots_dir / f"confusion_{_safe(label)}.png", dpi=150)
    plt.close(fig)


def _plot_pr_curves(plots_dir, prob_matrix, y_test, target_names, label):
    n_classes = len(target_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], prob_matrix[:, i])
        ax.plot(rec, prec, label=f"{target_names[i]} (AP={auc(rec, prec):.2f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall curves — {label}")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"pr_curves_{_safe(label)}.png", dpi=150)
    plt.close(fig)


def _plot_roc_curves(plots_dir, prob_matrix, y_test, target_names, label):
    n_classes = len(target_names)
    y_bin = label_binarize(y_test, classes=list(range(n_classes)))
    fig, ax = plt.subplots(figsize=(10, 7))
    for i in range(n_classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], prob_matrix[:, i])
        ax.plot(fpr, tpr, label=f"{target_names[i]} (AUC={auc(fpr, tpr):.2f})")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC curves — {label}")
    ax.legend(loc="lower right", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"roc_curves_{_safe(label)}.png", dpi=150)
    plt.close(fig)


def _plot_feature_importance(plots_dir, model, feature_names, label, top_n=20):
    clf = getattr(model, "clf", None)
    if clf is None or not hasattr(clf, "feature_importances_"):
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
    fig.savefig(plots_dir / f"feature_importance_{_safe(label)}.png", dpi=150)
    plt.close(fig)


def _plot_gbdt_loss(plots_dir, model, label="gbdt"):
    clf = getattr(model, "clf", None)
    if clf is None or not hasattr(clf, "train_score_"):
        return
    fig, ax = plt.subplots(figsize=(9, 5))
    iters = np.arange(1, len(clf.train_score_) + 1)
    ax.plot(iters, clf.train_score_, label="Train loss", color="#4472C4")
    if getattr(clf, "validation_score_", None) is not None:
        ax.plot(iters, clf.validation_score_, label="Validation loss",
                color="#ED7D31", linestyle="--")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title(f"GBDT training curve — {label}")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / f"loss_{label}.png", dpi=150)
    plt.close(fig)


def _plot_sweep(plots_dir, sweep_results, model_names, top_n=15):
    top = sweep_results[:top_n]
    accs = [r[0] for r in top]
    labels = ["\n".join(f"{n}={r[1][n]:.1f}" for n in model_names) for r in top]
    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.8), 5))
    ax.bar(range(len(accs)), accs, color="#4472C4")
    ax.set_xticks(range(len(accs)))
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Test accuracy")
    ax.set_title(f"Weight sweep — top {top_n} combos")
    ax.set_ylim(min(accs) - 0.005, max(accs) + 0.005)
    for i, a in enumerate(accs):
        ax.text(i, a + 0.0003, f"{a:.4f}", ha="center", fontsize=7)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "weight_sweep.png", dpi=150)
    plt.close(fig)


def generate_plots(plots_dir, *, model_names, probs, ens_probs, ens_pred,
                   y_test, target_names, model_accs, ens_acc, trained,
                   feature_names, sweep_results):
    """Write the full plot set into `plots_dir`. Returns the list of PNGs."""
    plots_dir = Path(plots_dir)
    plots_dir.mkdir(parents=True, exist_ok=True)
    ens_label = "ensemble"

    _plot_accuracy_comparison(plots_dir, model_accs, ens_acc, ens_label)
    _plot_per_class_f1(plots_dir, probs, ens_pred, ens_label, y_test, target_names)

    _plot_confusion_matrix(plots_dir, y_test, ens_pred, target_names, ens_label)
    _plot_pr_curves(plots_dir, ens_probs, y_test, target_names, ens_label)
    _plot_roc_curves(plots_dir, ens_probs, y_test, target_names, ens_label)
    for name in model_names:
        pred = np.argmax(probs[name], axis=1)
        _plot_confusion_matrix(plots_dir, y_test, pred, target_names, name)
        _plot_pr_curves(plots_dir, probs[name], y_test, target_names, name)
        _plot_roc_curves(plots_dir, probs[name], y_test, target_names, name)
        _plot_feature_importance(plots_dir, trained[name], feature_names, name)

    if "gbdt" in trained:
        _plot_gbdt_loss(plots_dir, trained["gbdt"])
    if sweep_results is not None:
        _plot_sweep(plots_dir, sweep_results, model_names)

    return sorted(plots_dir.glob("*.png"))
