#!/usr/bin/env python3
"""
topg_dataset_screening.py
=========================
Applies the eligibility rules R1-R3 of topg_dataset_selection_protocol.md to OpenML-CC18 and computes
the pre-specified dataset moderators. No feature selector is run here.

Usage (from the topg folder):  python topg_dataset_screening.py
Output: exp_result/topg_dataset_screening/topg_screening_cc18.csv  (one row per CC18 task; resumable)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np
import openml
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_common as tc  # noqa: E402

warnings.filterwarnings("ignore")
OUT = HERE.parent / "exp_result" / "topg_dataset_screening"
SEED = 42
CC18_SUITE = 99  # OpenML benchmark suite id of OpenML-CC18
COLUMNS = ["did", "task_id", "name", "N", "P", "cat_share", "R1_P_ge_20", "R2_N_ok", "N_used", "P_used",
           "n_classes", "imbalance", "M_red", "M_pca", "auc_rf", "auc_lr", "M_nl", "M_head", "M_rel", "log10_P",
           "N_over_P", "R3_headroom", "error", "screen_time_s", "eligible"]


def cv_auc(model, X, y) -> float:
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    classes = np.unique(y)
    aucs = []
    for tr, te in skf.split(X, y):
        model.fit(X[tr], y[tr])
        proba = model.predict_proba(X[te])
        if len(classes) == 2:
            aucs.append(roc_auc_score(y[te], proba[:, 1]))
        else:
            aucs.append(roc_auc_score(y[te], proba, multi_class="ovr", average="macro", labels=classes))
    return float(np.mean(aucs))


def bh_fraction(pvals: np.ndarray, q: float = 0.05) -> float:
    p = np.sort(np.nan_to_num(pvals, nan=1.0))
    m = len(p)
    ok = p <= q * np.arange(1, m + 1) / m
    return float((np.max(np.where(ok)[0]) + 1) / m) if ok.any() else 0.0


def moderators(X: pd.DataFrame, y: pd.Series) -> dict:
    Xv = X.values.astype(float)
    yv = y.values
    keep = Xv.std(axis=0) > 0
    Xs = Xv[:, keep]
    # redundancy: mean over features of max |Spearman| with another feature
    rho = spearmanr(Xs).correlation if Xs.shape[1] > 1 else np.array([[1.0]])
    rho = np.nan_to_num(np.abs(np.atleast_2d(rho)), nan=0.0)
    np.fill_diagonal(rho, 0.0)
    red = float(rho.max(axis=1).mean())
    pca = PCA().fit(StandardScaler().fit_transform(Xs))
    n95 = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), 0.95) + 1)
    auc_rf = cv_auc(RandomForestClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), Xv, yv)
    auc_lr = cv_auc(make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)), Xv, yv)
    _, pv = f_classif(Xs, yv)
    counts = np.bincount(pd.factorize(yv)[0])
    return dict(N_used=len(yv), P_used=Xv.shape[1], n_classes=len(counts), imbalance=float(counts.max() / counts.min()),
                M_red=red, M_pca=n95 / Xs.shape[1], auc_rf=auc_rf, auc_lr=auc_lr, M_nl=auc_rf - auc_lr,
                M_head=1 - auc_rf, M_rel=bh_fraction(pv), log10_P=float(np.log10(Xv.shape[1])),
                N_over_P=len(yv) / Xv.shape[1])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", type=int, default=CC18_SUITE, help="OpenML suite id (99 = CC18, 271 = AMLB)")
    ap.add_argument("--exclude-suite", type=int, default=None, help="skip datasets of this suite (by id and name)")
    ap.add_argument("--max-n", type=int, default=50000, help="rule R2 (protocol: 50,000 for CC18; A3: 1,000,000)")
    ap.add_argument("--out", default="topg_screening_cc18.csv")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    out_csv = OUT / args.out
    done = set(pd.read_csv(out_csv)["did"]) if out_csv.exists() else set()
    excl_ids, excl_names = set(), set()
    if args.exclude_suite is not None:
        excl_ids = set(openml.study.get_suite(args.exclude_suite).data)
        excl_names = {openml.datasets.get_dataset(d, download_data=False).name.lower() for d in excl_ids}
    suite = openml.study.get_suite(args.suite)
    for tid in suite.tasks:
        task = openml.tasks.get_task(tid, download_data=False, download_qualities=False)
        did = task.dataset_id
        if did in done or did in excl_ids:
            continue
        ds = openml.datasets.get_dataset(did, download_data=False)
        if ds.name.lower() in excl_names:
            continue
        q = ds.qualities or {}
        N, P = int(q.get("NumberOfInstances", 0)), int(q.get("NumberOfFeatures", 1)) - 1
        n_sym = int(q.get("NumberOfSymbolicFeatures", 0)) - 1
        row = dict(did=did, task_id=tid, name=ds.name, N=N, P=P, cat_share=max(n_sym, 0) / max(P, 1))
        row["R1_P_ge_20"] = P >= 20
        row["R2_N_ok"] = N <= args.max_n
        if row["R1_P_ge_20"] and row["R2_N_ok"]:
            t0 = time.time()
            try:
                X, y = tc.load_bio_dataset(ds.name, did)
                row.update(moderators(X, y))
                row["R3_headroom"] = 0.60 <= row["auc_rf"] < 0.99
            except Exception as e:  # recorded, never silently dropped
                row["error"] = f"{type(e).__name__}: {e}"[:200]
            row["screen_time_s"] = round(time.time() - t0, 1)
        row["eligible"] = bool(row.get("R1_P_ge_20") and row.get("R2_N_ok") and row.get("R3_headroom", False))
        pd.DataFrame([row]).reindex(columns=COLUMNS).to_csv(out_csv, mode="a", header=not out_csv.exists(), index=False)
        print(f"{ds.name:<40} N={N:>6} P={P:>5}  eligible={row['eligible']}  "
              f"auc_rf={row.get('auc_rf', float('nan')):.3f}  nl={row.get('M_nl', float('nan')):+.3f}  "
              f"red={row.get('M_red', float('nan')):.2f}", flush=True)


if __name__ == "__main__":
    main()
