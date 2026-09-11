"""
Paper-results driver for the IEEE Transactions revision.

Keeps two label spaces (the 9-class per-CVE space is dropped from the paper):
  * drop3      — 6 per-CVE classes (rare failing classes 3/4/7 removed)
  * killchain6 — kill-chain grouping of those same six classes

Protocol (revised per advisor feedback):
  A1. Ensemble weights are selected by 5-fold CV *macro-F1 on the training set*,
      never on the test set. The test set is touched only for the final metric.
      The selected weights and CV score are logged per cell.
  A2. The test set is FIXED, independent of train_n: for each class we reserve a
      fixed test block up front (size = min(500, total - eff_max_train), where
      eff_max_train = min(320, total - 25% reserve)), then draw the training set
      from the remaining pool. So accuracy is comparable across train_n rows.
      InfluxDB JWT: 193 total -> 48 test (fixed), train pool 145 (train_n capped
      at 145). kubelet symlink: 824 total -> 500 test (fixed), train pool 324.
  A3. Every cell reports the same set: acc, macro F1, balanced acc (macro-recall),
      per-class precision/recall/F1, CV score, selected weights.

Runs are repeated over 5 seeds and reported as mean +/- std.

Usage:
    .venv/bin/python -m src.experiments --validate   # 2 cells, seed 42, timing
    .venv/bin/python -m src.experiments               # full A/B/C/D tables
    .venv/bin/python -m src.experiments --no-reptile  # tree-only (fast/clean)
"""

import argparse
import time
from argparse import Namespace
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    classification_report, precision_recall_fscore_support,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import load_dataset
from src.labels import LABEL_CONFIGS
from src.sampling import apply_oversampler
from src.train import make_model

ROOT = Path(__file__).resolve().parents[1]
PLOTS_ROOT = ROOT / "plots" / "paper"

CONFIGS = ["drop3", "killchain6"]
KEPT = [0, 1, 2, 6, 8, 11]           # classes shared by drop3 and killchain6
MAX_TRAIN_N = 320                     # largest train budget in the sweep
TEST_CAP = 500                        # per-class test ceiling
ABUNDANT = 500                        # asymmetric downsampling threshold
TRAIN_SIZES = [20, 40, 80, 120, 160, 240, 320]
REPTILE_ANCHORS = {80, 160, 320}
BASE_MODELS = ["rf", "gbdt", "xgb"]
SEEDS = [42, 7, 123, 256, 999]
SEEDS_REPTILE_B = [42, 7, 123]        # Table B reptile anchors (reduced seeds)

# Resampling matrix (Table C). asym_large_10000 dropped per advisor.
#   label, train_n_large, smote, smote_target, uses_reptile
RESAMPLING = [
    ("baseline",           None, "none",   None, True),
    ("naive_upsample_500", None, "random", 500,  False),
    ("asym_large_500",     500,  "none",   None, True),
    ("asym_large_2000",    2000, "none",   None, False),
]


def model_args(epochs, epoch_size, seed):
    return Namespace(
        seed=seed, embedding_dim=64, encoder="mlp", distance="euclidean",
        k_shot=10, n_query=10, epochs=epochs, epoch_size=epoch_size,
        mlp_epochs=100,
    )


# ---------------------------------------------------------------------------
# Data: preprocess once, fixed splits
# ---------------------------------------------------------------------------

def preprocess_once(df, scaler):
    """Filter to KEPT classes and scale once. Returns X, y_orig, per-class idx."""
    feature_cols = [c for c in df.columns if c != "Label"]
    dfk = df[df["Label"].isin(KEPT)]
    X = scaler.transform(dfk[feature_cols].values.astype(np.float32))
    y_orig = dfk["Label"].values.astype(np.int64)
    per_class_idx = {c: np.where(y_orig == c)[0] for c in KEPT}
    return X, y_orig, per_class_idx


def build_seed_splits(per_class_idx, seed):
    """Fixed per-class (test_block, train_pool) — independent of train_n.

    test size = min(500, total - eff_max_train); eff_max_train = min(320,
    total - max(1, 25% reserve)). The test block depends only on the seed, so it
    is identical across every train_n row for that seed.
    """
    rng = np.random.default_rng(seed)
    splits = {}
    for c, idx in per_class_idx.items():
        total = len(idx)
        eff_max_train = min(MAX_TRAIN_N, total - max(1, int(total * 0.25)))
        test_size = min(TEST_CAP, total - eff_max_train)
        perm = rng.permutation(idx)
        splits[c] = (perm[:test_size], perm[test_size:])  # (test, pool)
    return splits


def fixed_split(splits, train_n, train_n_large):
    """Draw train/test index arrays for one budget from precomputed splits."""
    train_idx, test_idx, eff = [], [], {}
    for c, (test_block, pool) in splits.items():
        total = len(test_block) + len(pool)
        budget = train_n_large if (train_n_large and total > ABUNDANT) else train_n
        n_train = min(budget, len(pool))
        train_idx.append(pool[:n_train])
        test_idx.append(test_block)
        eff[c] = n_train
    return np.concatenate(train_idx), np.concatenate(test_idx), eff


def group_labels(y_orig, cfg):
    """Map original labels to the config's supergroups (3/4/7 already excluded)."""
    groups = LABEL_CONFIGS[cfg]["groups"]
    names = LABEL_CONFIGS[cfg]["names"]
    return np.array([groups[int(y)] for y in y_orig], dtype=np.int64), names


# ---------------------------------------------------------------------------
# Weight selection by CV macro-F1 (never touches the test set)
# ---------------------------------------------------------------------------

def sweep_f1(probs_by_model, y, step=0.1):
    """Grid sweep (step 0.1) over ensemble weights, maximizing macro-F1."""
    names = list(probs_by_model)
    n = len(names)
    ns = int(round(1 / step))
    best = (-1.0, None)

    def walk(remaining, acc):
        nonlocal best
        if len(acc) == n - 1:
            w = acc + [remaining / ns]
            wd = dict(zip(names, w))
            comb = sum(wd[k] * probs_by_model[k] for k in names)
            score = f1_score(y, np.argmax(comb, axis=1), average="macro",
                             zero_division=0)
            if score > best[0]:
                best = (score, wd)
            return
        for i in range(remaining + 1):
            walk(remaining - i, acc + [i / ns])

    walk(ns, [])
    return best


def cv_oof_probs(models, fitted, X, y, n_cls, margs, seed, smote="none",
                smote_target=None, names=None, n_splits=5):
    """Out-of-fold probabilities per model for weight selection (test untouched).

    Trees are refit on each fold's training split. Reptile reuses the already
    fitted encoder (`fitted['reptile']`, trained once on the full training set)
    and only refits its cheap logistic head per fold — this keeps the test set
    fully clean while avoiding 5x encoder meta-training. When oversampling is
    requested it is applied *inside each fold's training split only* (never the
    validation fold), so duplicated rows cannot leak across folds. Returns OOF
    probs and per-model solo CV macro-F1.
    """
    counts = np.bincount(y, minlength=n_cls)
    k = max(2, min(n_splits, int(counts[counts > 0].min())))
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    oof = {m: np.zeros((len(y), n_cls)) for m in models}
    rep_emb = fitted["reptile"]._encode(X) if "reptile" in models else None

    for tr, va in skf.split(X, y):
        Xf, yf = X[tr], y[tr]
        if smote != "none":
            Xf, yf = apply_oversampler(Xf, yf, smote, 5, smote_target, seed, names)
        for m in models:
            if m == "reptile":  # never combined with smote in our matrix
                head = LogisticRegression(
                    max_iter=1000, C=fitted["reptile"].head_C,
                    random_state=seed).fit(rep_emb[tr], y[tr])
                oof[m][np.ix_(va, head.classes_)] = head.predict_proba(rep_emb[va])
            else:
                mdl = make_model(m, margs, X.shape[1], n_cls)
                mdl.fit(Xf, yf)
                oof[m][va] = mdl.predict_proba(X[va])

    per_model = {m: f1_score(y, np.argmax(oof[m], 1), average="macro",
                             zero_division=0) for m in models}
    return oof, per_model


# ---------------------------------------------------------------------------
# One cell
# ---------------------------------------------------------------------------

def run_cell(X, y_orig, splits, cfg, train_n, models, train_n_large,
             smote, smote_target, seed, margs):
    tr_idx, te_idx, eff = fixed_split(splits, train_n, train_n_large)
    ytr, names = group_labels(y_orig[tr_idx], cfg)
    yte, _ = group_labels(y_orig[te_idx], cfg)
    Xtr, Xte = X[tr_idx], X[te_idx]
    n_cls = len(names)

    # The original (un-oversampled) training set drives CV weight selection;
    # the final models fit on the oversampled set. cv_oof_probs oversamples per
    # fold to avoid duplicate leakage across folds.
    Xtr_cv, ytr_cv = Xtr, ytr
    if smote != "none":
        Xtr, ytr = apply_oversampler(Xtr, ytr, smote, 5, smote_target, seed, names)

    # Fit final models once on the (possibly oversampled) full training set.
    fitted, probs = {}, {}
    for m in models:
        mdl = make_model(m, margs, Xtr.shape[1], n_cls)
        mdl.fit(Xtr, ytr)
        fitted[m] = mdl
        probs[m] = mdl.predict_proba(Xte)

    # Select weights by 5-fold CV macro-F1 on the training set (test untouched);
    # reptile reuses the fitted encoder (see cv_oof_probs).
    oof, per_model_f1 = cv_oof_probs(models, fitted, Xtr_cv, ytr_cv, n_cls,
                                     margs, seed, smote, smote_target, names)
    cv_f1, weights = sweep_f1(oof, ytr_cv)

    combined = sum(weights[m] * probs[m] for m in models)
    pred = np.argmax(combined, axis=1)

    prec, rec, f1, sup = precision_recall_fscore_support(
        yte, pred, labels=list(range(n_cls)), zero_division=0)
    # Per-model solo TEST macro-F1 (answers "is the ensemble better than the
    # best single model?" and quantifies each model's standalone test quality).
    solo_test_f1 = {m: f1_score(yte, np.argmax(probs[m], 1), average="macro",
                                zero_division=0) for m in models}
    return {
        "cfg": cfg, "names": names, "train_n": train_n, "models": models,
        "weights": weights, "cv_f1": cv_f1, "per_model_f1": per_model_f1,
        "solo_test_f1": solo_test_f1, "eff_train": eff,
        "acc": accuracy_score(yte, pred),
        "bal_acc": balanced_accuracy_score(yte, pred),
        "macro_f1": f1_score(yte, pred, average="macro", zero_division=0),
        "per_class": {"precision": prec, "recall": rec, "f1": f1, "support": sup},
        "report": classification_report(yte, pred, labels=list(range(n_cls)),
                                        target_names=names, zero_division=0),
        "probs": probs, "pred": pred, "yte": yte, "combined": combined,
    }


def wtuple(res):
    return ", ".join(f"{m}={res['weights'][m]:.1f}"
                     for m in res["models"] if res["weights"][m] > 0)


# ---------------------------------------------------------------------------
# Cached runner (dedupe cells shared across tables)
# ---------------------------------------------------------------------------

class Runner:
    def __init__(self, X, y_orig, per_class_idx, margs, raw_path):
        self.X, self.y_orig, self.per_class_idx = X, y_orig, per_class_idx
        self.margs, self.raw_path = margs, raw_path
        self.cache = {}
        self.split_cache = {}

    def splits(self, seed):
        if seed not in self.split_cache:
            self.split_cache[seed] = build_seed_splits(self.per_class_idx, seed)
        return self.split_cache[seed]

    def get(self, cfg, train_n, models, large, smote, target, seed, label=""):
        key = (cfg, train_n, tuple(models), large, smote, target, seed)
        if key in self.cache:
            return self.cache[key]
        t0 = time.time()
        res = run_cell(self.X, self.y_orig, self.splits(seed), cfg, train_n,
                       list(models), large, smote, target, seed, self.margs)
        res["label"] = label
        dt = time.time() - t0
        print(f"  [{cfg} {label} tn={train_n} {models} seed={seed}] "
              f"acc={res['acc']:.4f} bal={res['bal_acc']:.4f} "
              f"f1={res['macro_f1']:.4f} cv_f1={res['cv_f1']:.4f} "
              f"({wtuple(res)}) {dt:.0f}s")
        self._log(res, seed, dt)
        self.cache[key] = res
        return res

    def _log(self, res, seed, dt):
        pm = "  ".join(f"{m}:{v:.3f}" for m, v in res["per_model_f1"].items())
        st = "  ".join(f"{m}:{v:.3f}" for m, v in res["solo_test_f1"].items())
        with self.raw_path.open("a") as f:
            f.write(f"\n{'='*70}\n{res['cfg']} | {res.get('label','')} | "
                    f"tn={res['train_n']} | seed={seed} | models={res['models']} "
                    f"| {dt:.0f}s\n")
            f.write(f"selected weights: {wtuple(res)}   CV macro-F1={res['cv_f1']:.4f}\n")
            f.write(f"per-model solo CV macro-F1:   {pm}\n")
            f.write(f"per-model solo TEST macro-F1: {st}\n")
            f.write(f"ensemble acc={res['acc']:.4f}  bal_acc={res['bal_acc']:.4f}  "
                    f"macro_f1={res['macro_f1']:.4f}\n")
            f.write(res["report"])


# ---------------------------------------------------------------------------
# Aggregation + rendering (mean +/- std over seeds)
# ---------------------------------------------------------------------------

def ms(vals, dec=4):
    a = np.array(vals)
    return f"{a.mean():.{dec}f}±{a.std():.{dec}f}"


def agg_field(cells, field):
    return ms([c[field] for c in cells])


def agg_f1(cells):
    return ms([c["macro_f1"] for c in cells], dec=2)


def render(store, out_path, sizes, use_reptile):
    """store: dict table -> {(cfg, param): [cells over seeds]}."""
    L = ["# Paper results — drop3 (6-class) and killchain6\n"]
    L.append("Fixed test set per class (independent of `train_n`); weights "
             "selected by 5-fold CV macro-F1 on the training set (test set used "
             "only for the final metric). Values are mean±std over "
             f"{len(SEEDS)} seeds {SEEDS}. `bal_acc` = balanced accuracy "
             "(macro-recall).\n")
    if not use_reptile:
        L.append("> **Tree-only run** (RF+GBDT+XGB); Reptile excluded.\n")

    if use_reptile:
        L.append("Tables A, C, D report the best 4-model ensemble "
                 "(RF+GBDT+XGB+Reptile, CV-selected weights); Table B is the "
                 "tree-only scaling curve; Table E is the Reptile/ensemble "
                 "ablation.\n")
    else:
        L.append("All main tables use the RF+GBDT+XGB ensemble with CV-selected "
                 "weights. Reptile is dropped: Table E (single models vs tree "
                 "ensemble vs ensemble+Reptile at tn=80) shows it is the weakest "
                 "model and adds ~0 on test. Table B is the training-size scaling "
                 "curve.\n")

    # Table A — granularity at train_n=80
    L.append("## Table A — Label granularity (train_n=80)\n")
    L.append("| Config | #cls | Acc | Macro F1 | Bal. acc | Weights (seed 42) |")
    L.append("|---|---|---|---|---|---|")
    for cfg in CONFIGS:
        cells = store["A"][(cfg, 80)]
        s42 = next(c for c in cells if c is cells[0])
        L.append(f"| {cfg} | {len(cells[0]['names'])} | {agg_field(cells,'acc')} "
                 f"| {agg_f1(cells)} | {agg_field(cells,'bal_acc')} "
                 f"| {wtuple(cells[0])} |")
    L.append("")

    # Table B — training-size scaling (TREE-ONLY, uniform model set)
    if "B" in store and store["B"]:
        L.append("## Table B — Training-size scaling (tree-only)\n")
        L.append(f"Tree-only ensemble (RF+GBDT+XGB), CV-selected weights, {len(SEEDS)} "
                 "seeds, on every row — a uniform model set so the scaling curve is "
                 "monotone and not confounded by a changing model set (Reptile's "
                 "value is isolated in Table E). `eff tn (JWT)` = min(train_n, 145), "
                 "the effective InfluxDB JWT budget after the fixed-test reserve. "
                 "Gap = killchain6 − drop3 (macro F1 means).\n")
        L.append("| train_n | eff tn (JWT) | drop3 Acc | drop3 F1 | killchain6 Acc | killchain6 F1 | Gap F1 |")
        L.append("|---|---|---|---|---|---|---|")
        for tn in sizes:
            d = store["B"][("drop3", tn)]
            k = store["B"][("killchain6", tn)]
            eff_jwt = d[0]["eff_train"][6]
            df1 = np.mean([c["macro_f1"] for c in d])
            kf1 = np.mean([c["macro_f1"] for c in k])
            L.append(f"| {tn} | {eff_jwt} | {agg_field(d,'acc')} | "
                     f"{agg_f1(d)} | {agg_field(k,'acc')} | {agg_f1(k)} | "
                     f"{kf1-df1:+.2f} |")
        L.append("")

    # Table C — resampling
    L.append("## Table C — Resampling\n")
    L.append("Naive upsample replicates minority samples to 500/class; "
             "asymmetric large=N raises only classes with >500 samples to N "
             "(kubelet symlink is capped at its 324-sample pool by the fixed "
             "test reserve). All rows use the same ensemble; "
             "oversampling is applied per-fold inside CV (no duplicate leakage).\n")
    L.append("| | drop3 | | killchain6 | |")
    L.append("|---|---|---|---|---|")
    L.append("| Configuration | Acc | F1 | Acc | F1 |")
    for label, *_ in RESAMPLING:
        d = store["C"][("drop3", label)]
        k = store["C"][("killchain6", label)]
        L.append(f"| {label} | {agg_field(d,'acc')} | {agg_f1(d)} | "
                 f"{agg_field(k,'acc')} | {agg_f1(k)} |")
    L.append("")

    # Table D — per-class, baseline vs asym large=500 (mean over seeds)
    L.append("## Table D — Per-class recall / F1 (baseline vs asym large=500)\n")
    for cfg in CONFIGS:
        base = store["C"][(cfg, "baseline")]
        asym = store["C"][(cfg, "asym_large_500")]
        L.append(f"### {cfg}\n")
        L.append("| Class | Recall (base) | F1 (base) | Recall (large=500) | "
                 "F1 (large=500) | Support (base/asym) |")
        L.append("|---|---|---|---|---|---|")
        for i, nm in enumerate(base[0]["names"]):
            rb = ms([c["per_class"]["recall"][i] for c in base], 2)
            fb = ms([c["per_class"]["f1"][i] for c in base], 2)
            ra = ms([c["per_class"]["recall"][i] for c in asym], 2)
            fa = ms([c["per_class"]["f1"][i] for c in asym], 2)
            sb = int(base[0]["per_class"]["support"][i])
            sa = int(asym[0]["per_class"]["support"][i])
            L.append(f"| {nm} | {rb} | {fb} | {ra} | {fa} | {sb}/{sa} |")
        L.append("")

    # Table E — ablation at train_n=80: single models vs tree-only ensemble vs
    # the same ensemble + Reptile (test macro-F1). The +Reptile column is the
    # evidence for dropping the meta-learner from the main tables.
    if store.get("E"):
        L.append("## Table E — Ensemble & Reptile ablation (train_n=80)\n")
        L.append("Test macro-F1 (mean±std, 5 seeds, CV-selected weights on the "
                 "training set). Solo columns are each model alone; 'Tree-only ens' "
                 "is the RF+GBDT+XGB ensemble used in Tables A–D; '+Reptile ens' "
                 "adds Reptile as a fourth member. Reptile is the weakest single "
                 "model and does not lift the ensemble beyond noise (Δ within std), "
                 "which is why it is dropped from the main tables; the ensemble "
                 "itself barely improves on XGB alone.\n")
        L.append("| Config | RF | GBDT | XGB | Reptile | Tree-only ens | +Reptile ens |")
        L.append("|---|---|---|---|---|---|---|")
        for cfg in CONFIGS:
            tree = store["E"][(cfg, "tree")]
            full = store["E"][(cfg, "full")]
            solo = lambda m, c=full: ms([x["solo_test_f1"][m] for x in c], 2)
            L.append(f"| {cfg} | {solo('rf')} | {solo('gbdt')} | {solo('xgb')} | "
                     f"{solo('reptile')} | {agg_f1(tree)} | {agg_f1(full)} |")
        L.append("")

    out_path.write_text("\n".join(L))
    print(f"\nWrote {out_path}")


def make_plots(res, subdir):
    from src import plotting
    d = PLOTS_ROOT / subdir
    d.mkdir(parents=True, exist_ok=True)
    y, names = res["yte"], res["names"]
    plotting._plot_confusion_matrix(d, y, res["pred"], names, "ensemble")
    plotting._plot_per_class_f1(d, res["probs"], res["pred"], "ensemble", y, names)
    plotting._plot_pr_curves(d, res["combined"], y, names, "ensemble")
    plotting._plot_roc_curves(d, res["combined"], y, names, "ensemble")
    print(f"  plots -> {d}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="dataset.csv")
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--epoch-size", type=int, default=2000)
    p.add_argument("--out", default="PAPER_RESULTS.md")
    p.add_argument("--no-plots", action="store_true")
    p.add_argument("--no-reptile", action="store_true",
                   help="Tree-only (RF+GBDT+XGB); skip Reptile entirely.")
    p.add_argument("--skip-table-b", action="store_true",
                   help="Skip the (optional) training-size sweep.")
    p.add_argument("--validate", action="store_true",
                   help="Run 2 cells (both configs, tn=80, seed 42) and exit.")
    return p.parse_args()


def main():
    args = parse_args()
    margs = model_args(args.epochs, args.epoch_size, 42)
    all_models = BASE_MODELS if args.no_reptile else BASE_MODELS + ["reptile"]

    print(f"Loading {args.dataset} ...")
    df = load_dataset(args.dataset, sample_frac=args.sample_frac)
    print(f"  {len(df):,} rows; fitting scaler + preprocessing once ...")
    feature_cols = [c for c in df.columns if c != "Label"]
    scaler = StandardScaler().fit(df[feature_cols].values.astype(np.float32))
    X, y_orig, per_class_idx = preprocess_once(df, scaler)
    del df
    print(f"  kept {len(y_orig):,} rows across classes {KEPT}")

    raw_path = ROOT / ("PAPER_RESULTS_raw_validate.txt" if args.validate
                       else "PAPER_RESULTS_raw.txt")
    raw_path.write_text("")
    runner = Runner(X, y_orig, per_class_idx, margs, raw_path)

    if args.validate:
        print("\n=== VALIDATION (2 cells, seed 42, CV weight selection) ===")
        for cfg in CONFIGS:
            res = runner.get(cfg, 80, all_models, None, "none", None, 42)
            print(f"\n{cfg}: selected weights {wtuple(res)}  "
                  f"CV macro-F1={res['cv_f1']:.4f}")
            print("  per-model solo CV macro-F1: " +
                  "  ".join(f"{m}={v:.3f}" for m, v in res["per_model_f1"].items()))
            print(res["report"])
        print(f"\nRaw log -> {raw_path}")
        return

    store = {"A": {}, "B": {}, "C": {}, "E": {}}

    # Table A — granularity at tn=80, best 4-model ensemble, 5 seeds.
    for cfg in CONFIGS:
        store["A"][(cfg, 80)] = [
            runner.get(cfg, 80, all_models, None, "none", None, s, "baseline")
            for s in SEEDS]

    # Table C — resampling, best 4-model ensemble on every row (same model set
    # for a fair comparison), 5 seeds. baseline reuses Table A's tn=80 cell.
    for cfg in CONFIGS:
        for label, large, smote, target, _ in RESAMPLING:
            models = BASE_MODELS if args.no_reptile else all_models
            store["C"][(cfg, label)] = [
                runner.get(cfg, 80, models, large, smote, target, s, label)
                for s in SEEDS]

    # Table B — training-size scaling, TREE-ONLY across all rows and 5 seeds, so
    # the scaling curve is not confounded by a changing model set. Reptile's
    # marginal value is isolated in the ablation (Table E) instead.
    if not args.skip_table_b:
        for cfg in CONFIGS:
            for tn in TRAIN_SIZES:
                store["B"][(cfg, tn)] = [
                    runner.get(cfg, tn, BASE_MODELS, None, "none", None, s,
                               f"scaling_tn{tn}")
                    for s in SEEDS]

    # Table E — ablation at tn=80: tree-only ensemble vs +Reptile, same 5 seeds
    # and CV weights. Always runs the +Reptile cells (even in a --no-reptile
    # run) so the evidence for dropping Reptile lives in THIS run's raw. The
    # tree-only cells reuse Table A/B tn=80 from the cache.
    for cfg in CONFIGS:
        store["E"][(cfg, "tree")] = [
            runner.get(cfg, 80, BASE_MODELS, None, "none", None, s, "ablation_tree")
            for s in SEEDS]
        store["E"][(cfg, "full")] = [
            runner.get(cfg, 80, BASE_MODELS + ["reptile"], None, "none", None, s,
                       "ablation_+reptile")
            for s in SEEDS]

    # Headline plots (seed 42)
    if not args.no_plots:
        for cfg in CONFIGS:
            make_plots(store["C"][(cfg, "baseline")][0], f"{cfg}/baseline")
            make_plots(store["C"][(cfg, "asym_large_500")][0], f"{cfg}/asym500")

    render(store, ROOT / args.out, TRAIN_SIZES, not args.no_reptile)
    print(f"Raw classification reports -> {raw_path}")


if __name__ == "__main__":
    main()
