#!/usr/bin/env python3
"""
topg_run_sensitivity.py
=======================
Hyperparameter sensitivity of TOPG (self-review item B2), as registered in protocol amendment A8.
One setting is changed at a time from the default (T = 40 trees, maximum depth 5, pool factor 1.0):

  Part A (CC18, all 26 eligible datasets, repeats 1-3):  T in {20, 80, 160};  max_depth in {3, 8}
  Part B (AMLB datasets with P > 150, i.e. where the candidate pool is active, repeat 1):
         pool_size_factor in {0.5, 2.0}  (pool m = min(P, factor * max(200, 5K)))

Everything else (folds, K grid, imputation, four classifiers, seeds) is exactly as in topg_run_cc18.py; the
default TOPG results are taken from the main benchmark files.

Usage (from the topg folder):  python topg_run_sensitivity.py --part A ; python topg_run_sensitivity.py --part B
Outputs: exp_result/topg_sensitivity/topg_cc18_s<seed>_<dataset>.csv (same format as the benchmark)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_run_cc18 as rc  # noqa: E402
import topg_run_wrapper_baselines as w  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_sensitivity"
SCREEN = {"A": rc.SCREEN, "B": HERE.parent / "exp_result" / "topg_dataset_screening" / "topg_screening_amlb.csv"}
SENS = {
    "A": {"TOPG T=20": dict(surrogate_estimators=20), "TOPG T=80": dict(surrogate_estimators=80),
          "TOPG T=160": dict(surrogate_estimators=160), "TOPG depth=3": dict(max_depth=3),
          "TOPG depth=8": dict(max_depth=8)},
    "B": {"TOPG pool x0.5": dict(pool_size_factor=0.5), "TOPG pool x2": dict(pool_size_factor=2.0)},
}
ALL = {**SENS["A"], **SENS["B"]}
_orig_select = rc.select


def select(method, X_tr, y_tr, max_k, seed):
    if method not in ALL:
        return _orig_select(method, X_tr, y_tr, max_k, seed)
    _, params = w.VARIANTS["M6"]
    kw = dict(max_features=max_k, surrogate_estimators=w.SURROGATE_TREES, random_state=seed, use_oob=True, n_jobs=-1)
    kw.update(params)
    kw.update(ALL[method])
    sel = w.SRAFSAblation(**kw)
    return list(sel.fit(X_tr, y_tr).selected_features_)[:max_k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B"], required=True)
    ap.add_argument("--datasets", nargs="+", default=None)
    args = ap.parse_args()
    rc.select = select
    rc.OUT = OUT
    OUT.mkdir(parents=True, exist_ok=True)
    scr = pd.read_csv(SCREEN[args.part])
    eligible = scr[scr["eligible"]].sort_values("P_used")
    seeds = [1, 2, 3]
    if args.part == "B":
        eligible = eligible[eligible["P_used"] > 150]
        seeds = [1]
    if args.datasets:
        eligible = eligible[eligible["name"].isin(args.datasets)]
    methods = list(SENS[args.part])
    cache = {}
    for seed in seeds:
        t0 = time.time()
        for _, r in eligible.iterrows():
            rc.run(seed, r["name"], int(r["did"]), methods, cache)
        print(f"[part {args.part} seed {seed}] {(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
