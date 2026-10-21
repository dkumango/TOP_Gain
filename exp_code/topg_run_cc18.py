#!/usr/bin/env python3
"""
topg_run_cc18.py
================
Main benchmark on the OpenML-CC18 datasets selected by topg_dataset_selection_protocol.md
(eligible rows of exp_result/topg_dataset_screening/topg_screening_cc18.csv).

Protocol
--------
- Repeated stratified 5-fold CV; repeat r uses seed r for the folds and for every selector
  (TOPG / criteria: random_state = r; Python baselines: module RANDOM_STATE = r; R baselines: set.seed(r)).
  Downstream classifiers and the >1,500-instance subsampling keep seed 42.
- Numeric missing values are imputed INSIDE each fold with training-fold medians (categorical values are
  mode-imputed before one-hot encoding, which uses no label information).
- Panel sizes K in {5, 10, 15, 20} with K <= P/2 (protocol amendment A2); each selector returns one ranked
  list of length max(K) per fold, and its first K features are evaluated with four classifiers.
- A selector that raises an error is NOT replaced by another method: the error is written to
  topg_cc18_errors.csv and that (seed, dataset, method) cell stays empty.

Usage (from the topg folder):
    python topg_run_cc18.py --seeds 1 2 3 [--methods ...] [--datasets ...]
Outputs: exp_result/topg_cc18/topg_cc18_s<seed>_<dataset>.csv  (resumable per method)
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import topg_baselines as tb  # noqa: E402
import topg_common as tc  # noqa: E402
import topg_run_wrapper_baselines as w  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_cc18"
SCREEN = HERE.parent / "exp_result" / "topg_dataset_screening" / "topg_screening_cc18.csv"
K_GRID = (5, 10, 15, 20)
WRAPPER_KEYS = {"TOPG": "M6", "Net probability gain": "M9", "OOB log-loss": "M10", "OOB Brier score": "M11",
                "OOB accuracy": "M7", "OOB balanced accuracy": "M8"}
PY_BASELINES = {**tb.BASELINES, **{k: v for k, v in tb.RF_BASELINES.items()}}
ALL_METHODS = list(WRAPPER_KEYS) + list(PY_BASELINES)


def k_targets(p: int) -> dict:
    return {f"K_{k}": k for k in K_GRID if k <= p / 2}


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
        return list(sel.fit(X_tr, y_tr).selected_features_)[:max_k]
    tb.RANDOM_STATE = seed
    return list(PY_BASELINES[method](X_tr, y_tr, max_k))[:max_k]


def impute_fold(X_tr: pd.DataFrame, X_te: pd.DataFrame):
    if not X_tr.isna().any().any() and not X_te.isna().any().any():
        return X_tr, X_te
    med = X_tr.median().fillna(0.0)
    return X_tr.fillna(med), X_te.fillna(med)


def run(seed: int, name: str, did: int, methods: list[str], cache: dict):
    out_csv = OUT / f"topg_cc18_s{seed}_{name}.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    todo = [m for m in methods if done.empty or m not in set(done["method"])]
    if not todo:
        return
    if name not in cache:
        cache[name] = tc.load_bio_dataset(name, did, impute_numeric=False)
    X, y = cache[name]
    p, n_classes = X.shape[1], y.nunique()
    kt = k_targets(p)
    max_k = max(kt.values())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    sel = {m: [] for m in todo}
    times = {m: [] for m in todo}
    evals = {m: {lab: {pr: [] for pr in w.PREDICTORS} for lab in kt} for m in todo}
    failed = set()
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = impute_fold(X.iloc[tr], X.iloc[te])
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        for m in todo:
            if m in failed:
                continue
            t0 = time.time()
            try:
                chosen = select(m, X_tr, y_tr, max_k, seed)
            except Exception as e:
                failed.add(m)
                err = pd.DataFrame([dict(seed=seed, dataset=name, method=m, fold=fold,
                                         error=f"{type(e).__name__}: {e}"[:500],
                                         trace=traceback.format_exc()[-1500:])])
                err_csv = OUT / "topg_cc18_errors.csv"
                err.to_csv(err_csv, mode="a", header=not err_csv.exists(), index=False)
                print(f"  !! {m} failed on {name} (seed {seed}, fold {fold}): {e}")
                continue
            times[m].append(time.time() - t0)
            sel[m].append(chosen)
            for lab, k in kt.items():
                sub = chosen[:k]
                res = tc.evaluate_predictors(X_tr[sub].values, y_tr.values, X_te[sub].values, y_te.values, n_classes)
                for pr, met in res.items():
                    evals[m][lab][pr].append(met)
        print(f"  seed {seed} {name} fold {fold}: " +
              ", ".join(f"{m} {times[m][-1]:.0f}s" for m in todo if m not in failed and times[m]))
    col_idx = {c: i for i, c in enumerate(X.columns)}
    rows = []
    for m in todo:
        if m in failed:
            continue
        for lab, k in kt.items():
            Z = np.zeros((5, p))
            for f in range(5):
                for feat in sel[m][f][:k]:
                    Z[f, col_idx[feat]] = 1.0
            row = {"seed": seed, "did": did, "dataset": name, "method": m, "k_label": lab, "K": k,
                   "nogueira_stability": tc.nogueira_stability(Z), "mean_selection_time_sec": float(np.mean(times[m]))}
            for pr in w.PREDICTORS:
                fl = evals[m][lab][pr]
                for met in ["roc_auc", "accuracy", "macro_f1"]:
                    row[f"{w.PREFIX[pr]}_mean_{met}"] = float(np.mean([x[met] for x in fl]))
            rows.append(row)
    if rows:
        new = pd.DataFrame(rows)
        (pd.concat([done, new], ignore_index=True) if not done.empty else new).to_csv(out_csv, index=False)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--methods", nargs="+", default=ALL_METHODS, choices=ALL_METHODS)
    ap.add_argument("--datasets", nargs="+", default=None)
    ap.add_argument("--screen", default=str(SCREEN), help="screening CSV whose eligible rows are run")
    ap.add_argument("--outdir", default=str(OUT))
    args = ap.parse_args()
    OUT = Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)
    scr = pd.read_csv(args.screen)
    eligible = scr[scr["eligible"]].sort_values("P_used")  # small datasets first
    if args.datasets:
        eligible = eligible[eligible["name"].isin(args.datasets)]
    cache = {}
    for seed in args.seeds:
        t0 = time.time()
        for _, r in eligible.iterrows():
            run(seed, r["name"], int(r["did"]), args.methods, cache)
        print(f"[seed {seed}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
