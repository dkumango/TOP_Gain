#!/usr/bin/env python3
"""
topg_run_marlfs.py
==================
Size-matched comparison of MARLFS (baseline B1, protocol amendment A10) with TOP-Gain on the CC18 and AMLB
datasets: repeat 1 only (seed 1), the benchmark's 5 folds, fold-wise imputation, the four classifiers.

Per fold: MARLFS selects a subset of its own size k on the training fold. TOP-Gain is run on the same training
fold with max_features = min(k, 20) (20 = the largest panel of the main benchmark); both panels are evaluated on
the test fold with the four classifiers. When k > 20, TOP-Gain is evaluated with fewer features than MARLFS,
which favours MARLFS.

Usage (from the exp_code folder):
    python topg_run_marlfs.py                      # CC18
    python topg_run_marlfs.py --screen ../exp_result/topg_dataset_screening/topg_screening_amlb.csv --suite amlb
Outputs: exp_result/topg_marlfs/topg_marlfs_<suite>_<dataset>.csv (one row per fold and method), resumable;
         errors in exp_result/topg_marlfs/topg_marlfs_errors.csv
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
import topg_common as tc  # noqa: E402
import topg_run_cc18 as rc  # noqa: E402
import topg_run_wrapper_baselines as w  # noqa: E402
from topg_marlfs import select_marlfs  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_marlfs"
SEED = 1
CAP = 20


def run_dataset(suite, name, did):
    out = OUT / f"topg_marlfs_{suite}_{name}.csv"
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["fold"])
    X, y = tc.load_bio_dataset(name, did, impute_numeric=False)
    n_classes = y.nunique()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        if fold in set(done["fold"]):
            continue
        X_tr, X_te = rc.impute_fold(X.iloc[tr], X.iloc[te])
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        try:
            t0 = time.time(); sub_m = select_marlfs(X_tr, y_tr, SEED); t_m = time.time() - t0
        except Exception as e:
            err = pd.DataFrame([dict(suite=suite, dataset=name, fold=fold, method="MARLFS",
                                     error=f"{type(e).__name__}: {e}"[:500], trace=traceback.format_exc()[-1500:])])
            ef = OUT / "topg_marlfs_errors.csv"
            err.to_csv(ef, mode="a", header=not ef.exists(), index=False)
            print(f"  !! MARLFS failed on {name} fold {fold}: {e}", flush=True)
            continue
        k = len(sub_m); kk = min(k, CAP)
        t0 = time.time(); sub_t = rc.select("TOPG", X_tr, y_tr, kk, SEED)[:kk]; t_t = time.time() - t0
        rows = []
        for method, sub, tt in [("MARLFS", sub_m, t_m), ("TOPG (size-matched)", sub_t, t_t)]:
            res = tc.evaluate_predictors(X_tr[sub].values, y_tr.values, X_te[sub].values, y_te.values, n_classes)
            row = dict(suite=suite, dataset=name, did=did, P=X.shape[1], fold=fold, method=method, n_selected=len(sub),
                       marlfs_k=k, selection_time_sec=tt, selected=";".join(map(str, sub)))
            for pr, met in res.items():
                for m in ["roc_auc", "accuracy", "macro_f1"]:
                    row[f"{w.PREFIX[pr]}_{m}"] = met[m]
            rows.append(row)
        new = pd.DataFrame(rows)
        done = pd.concat([done, new], ignore_index=True) if len(done) else new
        done.to_csv(out, index=False)
        print(f"  {suite} {name} fold {fold}: MARLFS k={k} ({t_m:.0f}s), TOPG k={kk} ({t_t:.0f}s)", flush=True)


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", default=str(rc.SCREEN))
    ap.add_argument("--suite", default="cc18")
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    flag = OUT / f"topg_marlfs_{args.suite}_runs_on_server.txt"
    if flag.exists() and not args.datasets:  # A10 note: this suite is run on the Linux server instead
        print(f"[{args.suite}] skipped on this machine: see {flag.name}", flush=True)
        return
    scr = pd.read_csv(args.screen)
    el = scr[scr["eligible"]].sort_values("P_used")
    if args.datasets:
        el = el[el["name"].isin(args.datasets)]
    t0 = time.time()
    for _, r in el.iterrows():
        run_dataset(args.suite, r["name"], int(r["did"]))
    print(f"[{args.suite}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
