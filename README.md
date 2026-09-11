# Few-Shot Intrusion Detection for Software-Defined Container Networks

Machine-learning intrusion detection for Kubernetes-based software-defined
container networks, focused on **few-shot learning** for rare, CVE-based attack
classes. Developed at Middle East Technical University.

The detection core is a weighted ensemble of swappable tree classifiers
(Random Forest, Gradient Boosting, XGBoost) over CICFlowMeter network-flow
features, trained under a strict few-shot budget. A model registry lets any
classifier exposing a scikit-learn-style `fit / predict / predict_proba`
interface be added with a one-line change.

## Dataset

Network traffic captured with `tcpdump` on the `ovn0` interface of a Kubernetes
cluster running kube-OVN; flow features extracted with
[CICFlowMeter](https://github.com/ahlashkari/CICFlowMeter). ~3.2M flow records,
82 numerical features. The dataset (`dataset.csv`, ~1.8 GB) is **not included**
and is publicly available:

- Kaggle: `yigitsever/misuse-detection-in-containers-dataset`
- Aperta (TÜBİTAK ULAKBİM): `aperta.ulakbim.gov.tr/record/273835`

Place `dataset.csv` in the repository root before running.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Usage

Run from the repository root; runs are deterministic (`--seed 42`).

```bash
# Tree ensemble, balanced evaluation (500 test samples per class)
.venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \
    --models rf gbdt xgb --sweep

# Kill-chain stage grouping
.venv/bin/python -m src.train --dataset dataset.csv --test-n 500 \
    --label-config killchain --models rf gbdt xgb --sweep

# Quick smoke test on a 5% sample
.venv/bin/python -m src.train --sample-frac 0.05
```

Key flags: `--models` (classifiers to ensemble), `--label-config`
(`killchain`, `drop3`, `killchain6`, `by-service`, …), `--train-n` / `--test-n`
(few-shot / test budget), `--train-n-large` (asymmetric downsampling), `--sweep`
(grid search over ensemble weights).

## Structure

```
src/
  train.py              Orchestration: args, models, weight sweep, evaluation, plots.
  experiments.py        Multi-config / multi-seed evaluation driver.
  labels.py             Label names and grouping presets (--label-config).
  sampling.py           Oversampling (--smote).
  plotting.py           Figure generation.
  data/preprocessing.py Loading, scaling, few-shot train/test split.
  models/               Registry + classifier wrappers (rf, gbdt, xgb, reptile, …).
```
