#!/usr/bin/env python3
"""
topg_run_repeated_cv.py
=======================
Repeated cross-validation of the ten-dataset benchmark (reviewer request: results must not hinge on a
single random seed).

Each repeat r uses a different seed for
  - the StratifiedKFold split (5 folds),
  - every selector (TOPG / criterion variants: random_state = seed; baselines: module RANDOM_STATE = seed).
The data subsampling (>1,500 samples) and the downstream classifiers keep seed 42, so that only the
selection side of the pipeline is re-randomised. Protocol otherwise identical to the benchmark:
select K_max features once per fold, evaluate the first K for every K target with four classifiers.

The stored single-seed benchmark (seed 42) serves as repeat 0; this script adds repeats with other seeds.

Usage (from the topg folder):
    python topg_run_repeated_cv.py --seeds 1 2 3 4 --methods TOPG ANOVA mRMR ...   # any subset
    python topg_run_repeated_cv.py --seeds 42 --methods ANOVA --datasets synthetic  # sanity check vs stored

Outputs: exp_result/topg_repeated_cv/topg_rep_s<seed>_<dataset>.csv  (resumable per method)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import topg_baselines as tb  # noqa: E402
import topg_common as tc  # noqa: E402
import topg_run_wrapper_baselines as w  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_repeated_cv"
WRAPPER_KEYS = {"TOPG": "M6", "Net probability gain": "M9", "OOB log-loss": "M10", "OOB Brier score": "M11",
                "OOB accuracy": "M7", "OOB balanced accuracy": "M8"}
ALL_METHODS = list(WRAPPER_KEYS) + list(tb.BASELINES)


def select(method: str, X_tr, y_tr, max_k: int, seed: int) -> list:
    if method in WRAPPER_KEYS:
        _, params = w.VARIANTS[WRAPPER_KEYS[method]]
        common = dict(max_features=max_k, surrogate_estimators=w.SURROGATE_TREES, random_state=seed,
                      use_oob=True, n_jobs=-1)
        if "_criterion" in params:
            sel = w.OOBAccuracyWrapper(criterion=params["_criterion"], enable_g4_g3=False, enable_class_weight=False,
                                       enable_smoothing=False, pair_search=False, min_g4=0.0,
                                       objective_mode="g1_only", **common)
        else:
            sel = w.SRAFSAblation(**common, **params)
        return list(sel.fit(X_tr, y_tr).selected_features_)
    tb.RANDOM_STATE = seed
    try:
        return list(tb.BASELINES[method](X_tr, y_tr, max_k))
    except Exception as e:  # same fallback as the original benchmark
        print(f"    {method} failed ({e}); falling back to ANOVA")
        return list(tb.select_anova(X_tr, y_tr, max_k))


def run(seed: int, name: str, did, methods: list[str], cache: dict):
    out_csv = OUT / f"topg_rep_s{seed}_{name}.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    todo = [m for m in methods if done.empty or m not in set(done["method"])]
    if not todo:
        return
    if name not in cache:
        cache[name] = w.load_dataset(name, did)
    X, y = cache[name]
    p, n_classes = X.shape[1], y.nunique()
    k_targets = w.k_targets_for(p)
    max_k = max(k_targets.values())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sel = {m: [] for m in todo}
    times = {m: [] for m in todo}
    evals = {m: {lab: {pr: [] for pr in w.PREDICTORS} for lab in k_targets} for m in todo}
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te, y_tr, y_te = X.iloc[tr], X.iloc[te], y.iloc[tr], y.iloc[te]
        for m in todo:
            t0 = time.time()
            chosen = select(m, X_tr, y_tr, max_k, seed)
            times[m].append(time.time() - t0)
            sel[m].append(chosen)
            for lab, k in k_targets.items():
                sub = chosen[:k]
                res = tc.evaluate_predictors(X_tr[sub].values, y_tr.values, X_te[sub].values, y_te.values, n_classes)
                for pr, met in res.items():
                    evals[m][lab][pr].append(met)
        print(f"  seed {seed} {name} fold {fold}: " + ", ".join(f"{m} {times[m][-1]:.0f}s" for m in todo))
    col_idx = {c: i for i, c in enumerate(X.columns)}
    rows = []
    for m in todo:
        for lab, k in k_targets.items():
            Z = np.zeros((5, p))
            for f in range(5):
                for feat in sel[m][f][:k]:
                    Z[f, col_idx[feat]] = 1.0
            row = {"seed": seed, "dataset": name, "method": m, "k_label": lab, "K": k,
                   "nogueira_stability": tc.nogueira_stability(Z), "mean_selection_time_sec": float(np.mean(times[m]))}
            for pr in w.PREDICTORS:
                fl = evals[m][lab][pr]
                for met in ["roc_auc", "accuracy", "macro_f1"]:
                    row[f"{w.PREFIX[pr]}_mean_{met}"] = float(np.mean([x[met] for x in fl]))
            rows.append(row)
    new = pd.DataFrame(rows)
    (pd.concat([done, new], ignore_index=True) if not done.empty else new).to_csv(out_csv, index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--datasets", nargs="+", default=[d for d, _ in w.DATASETS])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    cache = {}
    for seed in args.seeds:
        t0 = time.time()
        for name, did in w.DATASETS:
            if name in args.datasets:
                run(seed, name, did, args.methods, cache)
        print(f"[seed {seed}] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
