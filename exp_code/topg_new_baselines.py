#!/usr/bin/env python3
"""
topg_new_baselines.py
=====================
Recent baselines added after the KAIS review (protocol amendment A9). Each function has the signature of the
other baselines, select_xxx(X_tr, y_tr, max_k) -> ranked list of column names, and reads the seed from
topg_baselines.RANDOM_STATE (set by topg_run_cc18.select).

  STG                  Yamada et al., "Feature selection using stochastic gates", ICML 2020  (pip package `stg`)
  Sequential Attention Yasuda et al., ICLR 2023  (official Google Research code, vendored in topg_vendor_seqatt/;
                       needs Keras 2, so it runs in topg_venv_py312 through topg_seqatt_bridge.py)
  Powershap            Verhaeghe et al., ECML PKDD 2022  (pip package `powershap`; needs Python <= 3.13, so it
                       runs in the separate environment topg_venv_py312 through topg_powershap_bridge.py,
                       in the same way as the R bridge of varSelRF / VSURF)

Common rules (A9):
  - every method sees only the training fold;
  - P > 500: the method is applied to the 500 features with the largest ANOVA F on the training fold, the same
    pre-filter as HSIC-Lasso and Stability Selection;
  - neural methods use standardised inputs (scaler fitted on the training fold);
  - a ranking of length max_k is returned; ties are broken by the secondary score and then by column order.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.preprocessing import LabelEncoder, StandardScaler

import topg_baselines as tb

HERE = Path(__file__).resolve().parent
PREFILTER = 500
PY312 = os.environ.get("TOPG_PY312", str(HERE.parent / "topg_venv_py312" / "Scripts" / "python.exe"))
POWERSHAP_BRIDGE = HERE / "topg_powershap_bridge.py"


def _prefilter(X: pd.DataFrame, y) -> pd.DataFrame:
    if X.shape[1] <= PREFILTER:
        return X
    f, _ = f_classif(X.values, np.asarray(y))
    f = np.nan_to_num(f, nan=0.0)
    keep = np.argsort(-f, kind="stable")[:PREFILTER]
    return X.iloc[:, np.sort(keep)]


def _rank(cols, primary, secondary=None) -> List[str]:
    primary = np.asarray(primary, dtype=float)
    secondary = np.zeros_like(primary) if secondary is None else np.asarray(secondary, dtype=float)
    order = np.lexsort((np.arange(len(cols)), -secondary, -primary))
    return [cols[i] for i in order]


# ---------------------------------------------------------------- STG (ICML 2020)
STG_CONFIG = dict(hidden_dims=[60, 20], activation="tanh", optimizer="SGD", learning_rate=0.1, sigma=0.5, lam=0.5,
                  epochs=2000)  # classification example of the official README; full-batch training


def select_stg(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    import torch
    import collections
    import collections.abc
    for _n in ("Sequence", "Mapping", "Iterable", "Set", "MutableMapping", "MutableSequence", "Callable"):  # the stg package uses aliases removed in Python 3.10
        if not hasattr(collections, _n):
            setattr(collections, _n, getattr(collections.abc, _n))
    from stg import STG
    Xp = _prefilter(X_tr, y_tr)
    Xs = StandardScaler().fit_transform(Xp.values).astype(np.float32)
    y = LabelEncoder().fit_transform(np.asarray(y_tr))
    if os.environ.get("TOPG_N_JOBS"):  # runtime study only (topg_runtime_benchmark.py); unset in the benchmark
        torch.set_num_threads(int(os.environ["TOPG_N_JOBS"]))
    torch.manual_seed(tb.RANDOM_STATE)
    np.random.seed(tb.RANDOM_STATE)
    model = STG(task_type="classification", input_dim=Xs.shape[1], output_dim=int(y.max()) + 1,
                hidden_dims=STG_CONFIG["hidden_dims"], activation=STG_CONFIG["activation"],
                optimizer=STG_CONFIG["optimizer"], learning_rate=STG_CONFIG["learning_rate"], batch_size=Xs.shape[0],
                feature_selection=True, sigma=STG_CONFIG["sigma"], lam=STG_CONFIG["lam"],
                random_state=tb.RANDOM_STATE, device="cpu")
    model.fit(Xs, y, nr_epochs=STG_CONFIG["epochs"], verbose=False, print_interval=STG_CONFIG["epochs"] + 1)
    prob = model.get_gates(mode="prob")
    raw = model.get_gates(mode="raw")
    return _rank(list(Xp.columns), prob, raw)[:max_k]


# ---------------------------------------------------------------- Sequential Attention (ICLR 2023) through the Python 3.12 bridge
SEQATT_BRIDGE = HERE / "topg_seqatt_bridge.py"


def _bridge(script, Xvals, y_tr, max_k, name):
    with tempfile.TemporaryDirectory() as tmp:
        inp, out = Path(tmp) / "x.csv", Path(tmp) / "rank.txt"
        df = pd.DataFrame(Xvals, columns=[f"V{i}" for i in range(Xvals.shape[1])])
        df["y"] = LabelEncoder().fit_transform(np.asarray(y_tr))
        df.to_csv(inp, index=False)
        res = subprocess.run([PY312, str(script), str(inp), str(out), str(max_k), str(tb.RANDOM_STATE)],
                             capture_output=True, text=True, encoding="utf-8", errors="replace")
        if res.returncode != 0 or not out.exists():
            raise RuntimeError(f"{name} failed (exit code {res.returncode}, output file: {out.exists()}): {res.stderr[-800:]}")
        return [int(v) for v in out.read_text().split()]


def select_seqatt(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    Xp = _prefilter(X_tr, y_tr)
    Xs = StandardScaler().fit_transform(Xp.values)
    idx = _bridge(SEQATT_BRIDGE, Xs, y_tr, max_k, "Sequential Attention")
    cols = list(Xp.columns)
    return [cols[i] for i in idx[:max_k]]


# ---------------------------------------------------------------- Powershap (ECML PKDD 2022) through the Python 3.12 bridge
def select_powershap(X_tr: pd.DataFrame, y_tr: pd.Series, max_k: int) -> List[str]:
    Xp = _prefilter(X_tr, y_tr)
    idx = _bridge(POWERSHAP_BRIDGE, Xp.values, y_tr, max_k, "Powershap")
    cols = list(Xp.columns)
    return [cols[i] for i in idx[:max_k]]


NEW_BASELINES = {"STG": select_stg, "Sequential Attention": select_seqatt, "Powershap": select_powershap}
