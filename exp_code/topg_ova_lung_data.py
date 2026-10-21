"""
topg_ova_lung_data.py
=====================
OVA_Lung (OpenML 1130) loader and imbalance-tier subsampling.

Copied verbatim on 2026-10-11 from `run_omics_imbalance_sweep_ova_lung.py` (source_code/) so that the TOPG project
does not depend on the earlier SRA-FS experiment files.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RANDOM_STATE = 42
TARGET_K = 10
N_MIN_FIXED = 70
TIERS = [("natural", None), ("1", 1.0), ("5", 5.0), ("10", 10.0), ("20", 20.0)]


def load_ova_lung() -> tuple[pd.DataFrame, pd.Series]:
    import openml
    ds = openml.datasets.get_dataset(1130, download_data=True)
    X, y, _, _ = ds.get_data(dataset_format="dataframe", target=ds.default_target_attribute)
    X = X.apply(pd.to_numeric, errors="coerce").astype(np.float32)
    if X.isna().any().any():
        X = X.fillna(X.median())
    y = y.astype(str)
    counts = y.value_counts()
    minority = counts.idxmin()
    y_bin = pd.Series((y == minority).astype(int).values, name="target")
    X.columns = [str(c) for c in X.columns]
    print(f"OVA_Lung loaded: N={len(X)}, P={X.shape[1]}, minority class '{minority}' = {int(y_bin.sum())}, "
          f"majority = {int((1 - y_bin).sum())}")
    return X.reset_index(drop=True), y_bin


def subsample(X: pd.DataFrame, y: pd.Series, rho: float | None, seed: int):
    if rho is None:
        return X, y
    rng = np.random.RandomState(seed)
    idx_min = np.where(y.values == 1)[0]
    idx_maj = np.where(y.values == 0)[0]
    n_min = N_MIN_FIXED
    n_maj = int(round(rho * n_min))
    if n_maj > len(idx_maj) or n_min > len(idx_min):
        raise ValueError(f"rho={rho}: need {n_maj}/{n_min}, have {len(idx_maj)}/{len(idx_min)}")
    sel = np.r_[rng.choice(idx_maj, n_maj, replace=False), rng.choice(idx_min, n_min, replace=False)]
    rng.shuffle(sel)
    return X.iloc[sel].reset_index(drop=True), y.iloc[sel].reset_index(drop=True)
