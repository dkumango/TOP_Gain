#!/usr/bin/env python3
"""
topg_runtime_benchmark.py
=========================
Selection-time study under identical, stated parallel settings (self-review item B4).

The CC18 benchmark ran every selector with its own default parallelism (TOPG and Python baselines: all
10 cores via joblib; VSURF: PSOCK cluster with 9 workers; varSelRF: single-threaded by design), so its
timings are not a like-for-like comparison. Here every selector is timed on the same folds, one call at a
time on an otherwise idle machine, under one setting per run:

  --cores 1   every method on one core (joblib n_jobs=1, VSURF parallel=FALSE, BLAS/OpenMP threads = 1)
  --cores 10  every method may use all 10 cores (joblib n_jobs=10, VSURF ncores=10); varSelRF has no
              parallel option and stays single-threaded

Unit: CC18 datasets (topg_screening_cc18.csv, eligible rows), repeat seed 1, fold 1 of the benchmark's
StratifiedKFold, fold-wise median imputation, max_k = max(K) as in topg_run_cc18.py. Selection only; no
classifiers are trained. One warm-up call per method on a small synthetic problem precedes the timings
(joblib worker start-up, Rscript start-up), and is discarded.

Recorded per (dataset, method): wall-clock seconds of the selector call; for varSelRF / VSURF also the
R-internal fitting time (without the CSV bridge); the selected list, to check that the worker count does
not change any selection.

Usage (from the topg folder):  python topg_runtime_benchmark.py --cores 10 ; python topg_runtime_benchmark.py --cores 1
Outputs: exp_result/topg_runtime/topg_runtime_c<cores>.csv (resumable)
"""
from __future__ import annotations

import argparse
import os
import sys

_ap = argparse.ArgumentParser()
_ap.add_argument("--cores", type=int, required=True)
_ap.add_argument("--datasets", nargs="+", default=None)
_ap.add_argument("--methods", nargs="+", default=None)
ARGS = _ap.parse_args()
if ARGS.cores == 1:  # must be set before numpy / sklearn are imported
    for _v in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        os.environ[_v] = "1"

import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_baselines as tb  # noqa: E402
import topg_common as tc  # noqa: E402
import topg_run_cc18 as rc  # noqa: E402
import topg_run_wrapper_baselines as w  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_runtime"
SEED = 1
METHODS = ["TOPG"] + list(rc.PY_BASELINES)


def select(method, X_tr, y_tr, max_k, cores):
    if method == "TOPG":
        _, params = w.VARIANTS["M6"]
        sel = w.SRAFSAblation(max_features=max_k, surrogate_estimators=w.SURROGATE_TREES, random_state=SEED,
                              use_oob=True, n_jobs=cores, **params)
        return list(sel.fit(X_tr, y_tr).selected_features_)[:max_k]
    tb.RANDOM_STATE = SEED
    tb.N_JOBS = cores
    return list(rc.PY_BASELINES[method](X_tr, y_tr, max_k))[:max_k]


def main():
    cores = ARGS.cores
    methods = ARGS.methods or METHODS
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / f"topg_runtime_c{cores}.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame(columns=["dataset", "method"])
    have = set(zip(done["dataset"], done["method"]))
    print(f"cores={cores}; {platform.processor()}; logical CPUs {os.cpu_count()}; Python {platform.python_version()}")

    # warm-up (discarded)
    rng = np.random.default_rng(0)
    Xw = pd.DataFrame(rng.normal(size=(300, 30)), columns=[f"w{i}" for i in range(30)])
    yw = pd.Series((Xw["w0"] + Xw["w1"] + rng.normal(size=300) > 0).astype(int))
    for m in methods:
        select(m, Xw, yw, 10, cores)
    print("warm-up done", flush=True)

    scr = pd.read_csv(rc.SCREEN)
    eligible = scr[scr["eligible"]].sort_values("P_used")
    if ARGS.datasets:
        eligible = eligible[eligible["name"].isin(ARGS.datasets)]
    for _, r in eligible.iterrows():
        name, did = r["name"], int(r["did"])
        todo = [m for m in methods if (name, m) not in have]
        if not todo:
            continue
        X, y = tc.load_bio_dataset(name, did, impute_numeric=False)
        max_k = max(rc.k_targets(X.shape[1]).values())
        tr, te = next(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X, y))
        X_tr, _ = rc.impute_fold(X.iloc[tr], X.iloc[te])
        y_tr = y.iloc[tr]
        rows = []
        for m in todo:
            tb.LAST_R_FIT_SEC = None
            t0 = time.perf_counter()
            chosen = select(m, X_tr, y_tr, max_k, cores)
            wall = time.perf_counter() - t0
            rows.append(dict(dataset=name, P=X.shape[1], N_train=len(tr), method=m, cores=cores, wall_sec=wall,
                             r_fit_sec=tb.LAST_R_FIT_SEC if m in ("varSelRF", "VSURF") else np.nan,
                             selected=";".join(map(str, chosen))))
            print(f"  {name} (P={X.shape[1]}) {m}: {wall:.1f}s", flush=True)
        new = pd.DataFrame(rows)
        done = pd.concat([done, new], ignore_index=True) if len(done) else new
        done.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
