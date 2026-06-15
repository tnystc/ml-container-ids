# Few-Shot Intrusion Detection for Software-Defined Container Networks

Machine-learning intrusion detection for Kubernetes-based software-defined
container networks, with a focus on **few-shot learning** for rare,
CVE-based attack classes. Developed for TÜBİTAK 3501 project 120E537 at
Middle East Technical University.

The detection core is a **weighted ensemble of swappable classifiers** over
CICFlowMeter network-flow features, trained under a strict few-shot budget
(80 samples per class). A model registry lets any classifier that exposes a
scikit-learn-style `fit / predict / predict_proba` interface be added with a
one-line change and combined with the others via an exhaustive weight sweep.

## Results

All numbers use the balanced evaluation protocol (500 test samples per class
where available; rare classes use all held-out samples). **Macro F1** is
reported alongside accuracy because the test set is dominated by a few
well-supported classes — accuracy alone is misleading on this dataset.

| Configuration | Classes | Accuracy | Macro F1 |
|---|---|---|---|
| XGBoost (best single model) | 9 (per-CVE) | 0.7453 | 0.60 |
| **RF + XGB + Reptile ensemble** | 9 (per-CVE) | **0.7503** | **0.60** |
| Asymmetric downsampling (`--train-n-large 500`) | 9 (per-CVE) | 0.8541 | **0.68** |
| Kill-chain stage grouping | 4 supergroups | 0.8501 | 0.84 |
| Kill-chain + asymmetric downsampling | 4 supergroups | 0.8947 | 0.88 |

The three rarest exploit classes (Node-RED RCE, Node-RED container escape,
runc race condition) remain hard (F1 < 0.25) across every few-shot
architecture we evaluated — a **feature-space limitation** of aggregate flow
statistics, not a modelling limitation that a better classifier can lift.

## Dataset

Network traffic captured with `tcpdump` on the `ovn0` interface of a
Kubernetes cluster running kube-OVN, with flow features extracted using
[CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter). 3.2M flow records,
82 numerical features, 12 attack/benign classes (9 used; three have fewer
than 50 samples and are excluded).

The dataset (`dataset.csv`, ~1.8 GB) is **not included in this repository**.
It is publicly available:

- Kaggle: `yigitsever/misuse-detection-in-containers-dataset`
- Aperta (TÜBİTAK ULAKBİM): `aperta.ulakbim.gov.tr/record/273835`

Place `dataset.csv` in the repository root before running.

## Installation

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Run from the repository root. All runs are deterministic (`--seed 42`).

```bash
# Best ensemble — 9-class per-CVE (~10 min on Apple MPS / GPU)
.venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \
    --models rf gbdt xgb reptile --epochs 10 --epoch-size 2000 --sweep

# Kill-chain stage grouping (4 supergroups)
.venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \
    --label-config killchain \
    --models rf gbdt xgb reptile --epochs 10 --epoch-size 2000 --sweep

# Asymmetric downsampling: well-supported classes get 500 training samples,
# rare classes stay few-shot
.venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \
    --train-n-large 500 --models rf gbdt xgb reptile \
    --epochs 10 --epoch-size 2000 --sweep

# Tree-only baseline (fast, no neural meta-learning)
.venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \
    --models rf gbdt xgb --sweep

# Quick smoke test on a 5% sample
.venv/bin/python -m src.train --sample-frac 0.05 --epochs 1 --epoch-size 100
```

Each run prints per-model and ensemble accuracy, an optional weight sweep,
a per-class classification report, and (unless `--no-plots`) writes confusion
matrices, ROC/PR curves, per-class F1 heatmaps and feature-importance charts
into `plots/`.

### Key options

| Flag | Purpose |
|---|---|
| `--models` | Which classifiers to ensemble: `rf gbdt xgb reptile protonet mlp` |
| `--weights` | Fixed ensemble weights (must sum to 1); omit to use uniform + `--sweep` |
| `--sweep` | Exhaustive grid search (step 0.1) over ensemble weights |
| `--train-n` | Few-shot training samples per class (default 80) |
| `--train-n-large` | Asymmetric: training budget for classes with > 500 samples |
| `--test-n` | Balanced test samples per class (default 500) |
| `--label-config` | Label grouping preset: `killchain`, `drop3`, `by-service`, `group-failing`, `drop-2rarest` |
| `--smote` | Oversampling: `smote`, `borderline`, `adasyn`, `random` (naive) |

## Repository structure

```
src/
  train.py              Orchestration: arg parsing, model construction,
                        weight sweep, evaluation, plotting.
  labels.py             Label names and grouping presets (--label-config).
  sampling.py           SMOTE-family and naive oversampling (--smote).
  plotting.py           Figure generation (confusion, ROC/PR, F1, importances).
  data/
    preprocessing.py    Dataset loading, scaling, few-shot train/test split.
  models/
    __init__.py         Model registry: build_model(name, **kwargs).
    ensemble.py         WeightedEnsemble over any probability-output models.
    random_forest.py    Random Forest wrapper.
    gbdt.py             Histogram Gradient Boosting wrapper.
    xgb.py              XGBoost wrapper (best single model).
    reptile.py          Reptile (first-order MAML) meta-learner.
    protonet.py         Prototypical Network.
    mlp.py              Vanilla supervised MLP baseline.
```

## Adding a model

1. Write a wrapper exposing `fit(X, y)`, `predict(X)`, `predict_proba(X)`.
2. Register it in `_REGISTRY` in `src/models/__init__.py`.
3. Add a construction branch to `make_model` in `src/train.py`.
4. Run with `--models yourname`.

## Acknowledgement

Supported by TÜBİTAK 3501 project 120E537,
*"High-Performance Intrusion Detection and Prevention Architecture for
Software-Defined Container Networks in the Cloud"*, Middle East Technical
University.
