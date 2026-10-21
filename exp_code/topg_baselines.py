"""
topg_baselines.py
=================
The nine baseline selectors of the TOPG benchmark, for the repeated-CV experiment.

Copied verbatim on 2026-10-11 from `benchmark_bio.py` (ANOVA, mRMR, JMI, ReliefF, Elastic Net, Boruta,
SVM-RFE) and `scratch/run_modern_baselines_10_datasets.py` (Stability Selection, HSIC-Lasso) in
source_code/. The selectors read the module-level RANDOM_STATE; the repeated-CV runner sets it per repeat.
"""
from __future__ import annotations

import contextlib
import io
from typing import List

import numpy as np
import pandas as pd
from boruta import BorutaPy

# pyHSICLasso calls numpy.linalg.linalg.*, a private alias removed in NumPy 2.x. Without this shim
# HSIC-Lasso raises AttributeError on some inputs and select_hsic_lasso silently falls back to ANOVA
# (this happened in the stored single-seed benchmark; see topg_README.md).
if not hasattr(np.linalg, "linalg"):
    np.linalg.linalg = np.linalg

from pyHSICLasso import HSICLasso  # noqa: E402
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

RANDOM_STATE = 42
# Parallel workers for every baseline (and the R bridge). -1 = all cores (benchmark default); the runtime study
# (topg_runtime_benchmark.py) sets it explicitly. The number of workers does not change any selection.
N_JOBS = -1
LAST_R_FIT_SEC = None  # R-internal fitting time of the last varSelRF / VSURF call (excludes the CSV bridge)

def select_anova(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    1. ANOVA F-test: Univariate Statistical Filter.
    Computes one-way ANOVA F-value testing whether class means differ significantly.
    Serves as the classical benchmark for differential expression (DEG) analysis in omics.
    """
    f_vals, _ = f_classif(X_tr, y_tr)
    f_vals = np.nan_to_num(f_vals, nan=0.0)
    top_idx = np.argsort(f_vals)[::-1][:max_k]
    return list(X_tr.columns[top_idx])


def select_mrmr(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    2. mRMR: Minimum Redundancy Maximum Relevance Filter.
    Balances individual mutual information relevance I(X_i; Y) against pairwise correlation redundancy.
    Reference: Peng, Long, & Ding (2005) IEEE TPAMI.
    """
    features = list(X_tr.columns)
    if len(features) <= max_k:
        return features

    # Pre-filter if p > 300 to keep mutual information scalable
    if len(features) > 300:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        features = list(X_tr.columns[np.argsort(f_vals)[::-1][:300]])
        X_sub = X_tr[features]
    else:
        X_sub = X_tr

    relevance = mutual_info_classif(X_sub, y_tr, random_state=RANDOM_STATE)
    selected = [features[int(np.argmax(relevance))]]
    remaining = [f for f in features if f not in selected]

    while len(selected) < max_k and remaining:
        best_f = None
        best_score = -np.inf
        for f in remaining:
            rel = relevance[features.index(f)]
            # Redundancy proxy with correlation
            red = float(np.mean([abs(np.corrcoef(X_sub[f], X_sub[s])[0, 1]) for s in selected]))
            score = rel - red
            if score > best_score:
                best_score = score
                best_f = f
        selected.append(best_f)
        remaining.remove(best_f)
    return selected


def select_jmi(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    3. JMI: Joint Mutual Information Filter.
    Maximizes joint information I(X_i, X_j; Y) using the conditional likelihood framework.
    Reference: Brown et al. (2012) Journal of Machine Learning Research.
    """
    features = list(X_tr.columns)
    if len(features) <= max_k:
        return features

    if len(features) > 200:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        features = list(X_tr.columns[np.argsort(f_vals)[::-1][:200]])
        X_sub = X_tr[features]
    else:
        X_sub = X_tr

    mi_y = mutual_info_classif(X_sub, y_tr, random_state=RANDOM_STATE)
    selected = [features[int(np.argmax(mi_y))]]
    remaining = [f for f in features if f not in selected]

    while len(selected) < max_k and remaining:
        best_f = None
        best_jmi = -np.inf
        for f in remaining:
            f_idx = features.index(f)
            # Fast JMI approximation (Brown et al., 2012)
            joint_bonus = sum(
                mi_y[f_idx] + mi_y[features.index(s)] - abs(np.corrcoef(X_sub[f], X_sub[s])[0, 1]) * 0.5
                for s in selected
            )
            score = joint_bonus / len(selected)
            if score > best_jmi:
                best_jmi = score
                best_f = f
        if best_f is None:
            # (2026-10-15) all remaining scores are NaN (a selected feature is constant within the fold, so
            # its correlations are undefined); the original code crashed here. Fall back to the remaining
            # feature with the highest MI with y. Runs that did not crash are unaffected.
            best_f = max(remaining, key=lambda f: (mi_y[features.index(f)], -features.index(f)))
        selected.append(best_f)
        remaining.remove(best_f)
    return selected


def select_relieff(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    4. ReliefF: Instance-Based Margin Filter.
    Estimates feature importance by measuring nearest-hit vs nearest-miss distance differentials.
    Reference: Robnik-Šikonja & Kononenko (2003) Machine Learning.
    """
    X_arr = X_tr.values
    y_arr = np.asarray(y_tr)
    n_samples, n_features = X_arr.shape

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_arr)

    k_neighbors = min(10, n_samples - 1)
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1, metric="euclidean", n_jobs=N_JOBS)
    nn.fit(X_sc)
    distances, indices = nn.kneighbors(X_sc)

    weights = np.zeros(n_features, dtype=float)
    classes = np.unique(y_arr)
    n_iter = min(n_samples, 200)
    rng = np.random.RandomState(RANDOM_STATE)
    sample_indices = rng.choice(n_samples, size=n_iter, replace=False)

    for i in sample_indices:
        xi = X_sc[i]
        yi = y_arr[i]
        neighbor_idx = indices[i][1:]
        hit_idx = [idx for idx in neighbor_idx if y_arr[idx] == yi]
        miss_idx = [idx for idx in neighbor_idx if y_arr[idx] != yi]

        if hit_idx:
            hit_diff = np.mean(np.abs(xi - X_sc[hit_idx]), axis=0)
        else:
            hit_diff = np.zeros(n_features)

        if miss_idx:
            miss_diff = np.mean(np.abs(xi - X_sc[miss_idx]), axis=0)
        else:
            miss_diff = np.zeros(n_features)

        weights += miss_diff - hit_diff

    top_idx = np.argsort(weights)[::-1][:max_k]
    return list(X_tr.columns[top_idx])


def select_elastic_net(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    5. Elastic Net: Regularized Embedded Linear Model.
    Combines L1 (Lasso) and L2 (Ridge) penalties to select sparse, correlated biomarker groups.
    Reference: Zou & Hastie (2005) JRSS-B.
    """
    features = list(X_tr.columns)
    if len(features) <= max_k:
        return features

    # Pre-filter for ultra-high dimensions (p > 500) to keep solver fast
    if len(features) > 500:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        sub_cols = list(X_tr.columns[np.argsort(f_vals)[::-1][:500]])
        X_sub = X_tr[sub_cols]
    else:
        sub_cols = features
        X_sub = X_tr

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_sub)

    clf = LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        C=0.5,
        max_iter=150,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
    )
    clf.fit(X_sc, y_tr)
    importance = np.max(np.abs(clf.coef_), axis=0)
    top_idx = np.argsort(importance)[::-1][:max_k]
    return [sub_cols[i] for i in top_idx]


def select_boruta(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    6. Boruta: All-Relevant Feature Selection Wrapper.
    Compares original feature importance against randomized shadow feature distributions.
    Reference: Kursa & Rudnicki (2010) Journal of Statistical Software.
    """
    features = list(X_tr.columns)
    if len(features) <= max_k:
        return features

    if len(features) > 300:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        sub_cols = list(X_tr.columns[np.argsort(f_vals)[::-1][:300]])
        X_sub = X_tr[sub_cols]
    else:
        sub_cols = features
        X_sub = X_tr

    rf = RandomForestClassifier(
        n_estimators=50, max_depth=5, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=N_JOBS
    )
    feat_selector = BorutaPy(
        rf, n_estimators="auto", max_iter=40, random_state=RANDOM_STATE, verbose=0
    )
    feat_selector.fit(X_sub.values, np.asarray(y_tr))

    rankings = feat_selector.ranking_
    top_idx = np.argsort(rankings)[:max_k]
    return [sub_cols[i] for i in top_idx]


def select_svm_rfe(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """
    7. SVM-RFE: Recursive Feature Elimination Wrapper.
    Recursively eliminates features with smallest weight magnitudes in linear SVM hyperplanes.
    Reference: Guyon et al. (2002) Machine Learning.
    """
    features = list(X_tr.columns)
    if len(features) <= max_k:
        return features

    # Pre-filter for ultra-high dimensions (p > 500)
    if len(features) > 500:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        sub_cols = list(X_tr.columns[np.argsort(f_vals)[::-1][:500]])
        X_sub = X_tr[sub_cols]
    else:
        sub_cols = features
        X_sub = X_tr

    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_sub)

    step_size = max(1, int(0.10 * len(sub_cols))) if len(sub_cols) > 2 * max_k else 1
    rfe = RFE(
        estimator=SVC(kernel="linear", C=1.0, random_state=RANDOM_STATE),
        n_features_to_select=max_k,
        step=step_size,
    )
    rfe.fit(X_sc, y_tr)
    top_idx = np.argsort(rfe.ranking_)[:max_k]
    return [sub_cols[i] for i in top_idx]


def select_stability_selection(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> list:
    n_samples, n_features = X_tr.shape
    features = list(X_tr.columns)
    if n_features > 500:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        sub_cols = list(X_tr.columns[np.argsort(f_vals)[::-1][:500]])
        X_sub = X_tr[sub_cols]
    else:
        sub_cols = features
        X_sub = X_tr

    n_sub = len(sub_cols)
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_sub)
    n_classes = len(np.unique(y_tr))

    rng = np.random.RandomState(RANDOM_STATE)
    counts = np.zeros(n_sub)
    n_iter = 30
    sub_size = max(10, int(n_samples * 0.5))

    for b in range(n_iter):
        idx = rng.choice(n_samples, size=sub_size, replace=False)
        X_b = X_sc[idx]
        y_b = y_tr.iloc[idx]
        solver = "liblinear" if n_classes == 2 else "saga"
        clf = LogisticRegression(penalty="l1", C=0.2, solver=solver, max_iter=100, random_state=rng.randint(10000))
        try:
            clf.fit(X_b, y_b)
            if n_classes == 2:
                non_zero = (np.abs(clf.coef_[0]) > 1e-4)
            else:
                non_zero = (np.max(np.abs(clf.coef_), axis=0) > 1e-4)
            counts += non_zero
        except Exception:
            pass

    top_idx = np.argsort(counts)[::-1][:max_k]
    return [sub_cols[i] for i in top_idx]


def select_hsic_lasso(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> list:
    n_samples, n_features = X_tr.shape
    features = list(X_tr.columns)
    if n_features > 500:
        f_vals, _ = f_classif(X_tr, y_tr)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        sub_cols = list(X_tr.columns[np.argsort(f_vals)[::-1][:500]])
        X_sub = X_tr[sub_cols]
    else:
        sub_cols = features
        X_sub = X_tr

    hl = HSICLasso()
    f_str = io.StringIO()
    with contextlib.redirect_stdout(f_str):
        try:
            hl.input(X_sub.values, y_tr.values)
            hl.classification(num_feat=max_k, n_jobs=N_JOBS)
            selected_feats = hl.get_features()
            top_idx = [int(feat) - 1 for feat in selected_feats]
            return [sub_cols[i] for i in top_idx[:max_k]]
        except Exception:
            # Fallback
            f_vals, _ = f_classif(X_sub, y_tr)
            return [sub_cols[i] for i in np.argsort(f_vals)[::-1][:max_k]]


BASELINES = {
    "ANOVA": select_anova,
    "mRMR": select_mrmr,
    "JMI": select_jmi,
    "ReliefF": select_relieff,
    "ElasticNet": select_elastic_net,
    "Boruta": select_boruta,
    "SVM-RFE": select_svm_rfe,
    "Stability Selection": select_stability_selection,
    "HSIC-Lasso": select_hsic_lasso,
}


# ---------------------------------------------------------------------------
# Random-Forest-based baselines added for the CC18 benchmark (2026-10-11)
# ---------------------------------------------------------------------------
import os as _os
import subprocess as _sp
import tempfile as _tf
from pathlib import Path as _Path

from sklearn.inspection import permutation_importance as _perm_imp
from sklearn.model_selection import train_test_split as _tts

RSCRIPT = _os.environ.get("TOPG_RSCRIPT", r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")
R_SELECTORS = _Path(__file__).resolve().parent / "topg_r_selectors.R"
R_LIBS_USER = _os.environ.get("TOPG_R_LIBS_USER", str(_Path.home() / "R" / "topg-lib"))


def select_rf_impurity(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """Top-K by mean decrease in impurity of a 500-tree Random Forest."""
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=N_JOBS).fit(X_tr, y_tr)
    order = np.lexsort((np.arange(X_tr.shape[1]), -rf.feature_importances_))
    return list(X_tr.columns[order[:max_k]])


def select_rf_permutation(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """Top-K by permutation importance (ROC-AUC drop, 5 repeats) on a stratified 30 % hold-out of the
    training fold; the 500-tree Random Forest is fitted on the remaining 70 %."""
    X_a, X_b, y_a, y_b = _tts(X_tr, y_tr, test_size=0.3, stratify=y_tr, random_state=RANDOM_STATE)
    # (2026-10-14) NumPy arrays instead of DataFrames: joblib memory-maps arrays once, whereas a DataFrame
    # was re-pickled for every one of the P per-feature jobs (prohibitive for P ~ 70,000). Same values,
    # same seeds, identical selections (verified on qsar-biodeg, madelon, cylinder-bands).
    rf = RandomForestClassifier(n_estimators=500, random_state=RANDOM_STATE, n_jobs=N_JOBS).fit(
        X_a.to_numpy(), np.asarray(y_a))
    scoring = "roc_auc" if y_tr.nunique() == 2 else "roc_auc_ovr"
    pi = _perm_imp(rf, X_b.to_numpy(), np.asarray(y_b), scoring=scoring, n_repeats=5,
                   random_state=RANDOM_STATE, n_jobs=N_JOBS)
    order = np.lexsort((np.arange(X_tr.shape[1]), -pi.importances_mean))
    return list(X_tr.columns[order[:max_k]])


def select_rf_rfe(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    """Recursive feature elimination with a 100-tree Random Forest (impurity importance), removing 10 % of
    the remaining features per step down to max_k; the survivors are ranked by the final forest."""
    rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=N_JOBS)
    rfe = RFE(rf, n_features_to_select=max_k, step=0.1).fit(X_tr, y_tr)
    kept = np.where(rfe.support_)[0]
    imp = rfe.estimator_.feature_importances_
    order = kept[np.lexsort((kept, -imp))]
    return list(X_tr.columns[order[:max_k]])


def _r_select(method: str, X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    with _tf.TemporaryDirectory() as tmp:
        inp, out = _Path(tmp) / "x.csv", _Path(tmp) / "rank.txt"
        df = pd.DataFrame(np.asarray(X_tr, dtype=float), columns=[f"V{i + 1}" for i in range(X_tr.shape[1])])
        df["y"] = np.asarray(y_tr)
        df.to_csv(inp, index=False)
        ncores = max(1, (_os.cpu_count() or 2) - 1) if N_JOBS == -1 else int(N_JOBS)
        res = _sp.run([RSCRIPT, str(R_SELECTORS), method, str(inp), str(out), str(max_k), str(RANDOM_STATE),
                       str(ncores)], capture_output=True, text=True,
                      env={**_os.environ, "R_LIBS_USER": R_LIBS_USER})
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"{method} failed: {res.stderr[-500:]}")
        idx = [int(v) - 1 for v in out.read_text().split()]
        global LAST_R_FIT_SEC
        tf = _Path(str(out) + ".time")
        LAST_R_FIT_SEC = float(tf.read_text()) if tf.exists() else None
    return list(X_tr.columns[idx[:max_k]])


def select_varselrf(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    return _r_select("varSelRF", X_tr, y_tr, max_k)


def select_vsurf(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    return _r_select("VSURF", X_tr, y_tr, max_k)


RF_BASELINES = {
    "RF impurity": select_rf_impurity,
    "RF permutation": select_rf_permutation,
    "RF-RFE": select_rf_rfe,
    "varSelRF": select_varselrf,
    "VSURF": select_vsurf,
}
