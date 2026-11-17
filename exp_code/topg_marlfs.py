#!/usr/bin/env python3
"""
topg_marlfs.py
==============
Adapter for MARLFS (multi-agent reinforcement-learning feature selection; K. Liu, Y. Fu, et al., TKDE 2021,
doi 10.1109/TKDE.2021.3115477) as implemented in the CAPS repository (Liu et al., KDD 2025), baseline B1 of
protocol amendment A10.

The CAPS code is imported from topg_external/caps_code (not redistributed; see the README there). Its
FeatureEvaluator normally reads an HDF file by task name; here the same object is built from the training fold
in memory, with exactly the attributes its __init__ sets (internal 80/20 split with random_state=0).

select_marlfs(X_tr, y_tr, seed) returns the selected columns (MARLFS decides the subset size itself).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

HERE = Path(__file__).resolve().parent
CAPS = HERE.parent / "topg_external" / "caps_code"
EXPLORE_STEPS = 500   # A10: > MEMORY_CAPACITY (200) of the implementation, so that the DQN agents learn
N_STATES = 64         # as in the CAPS records collector
PREFILTER = 500       # A9/A10: P > 500 -> top-500 ANOVA F on the training fold


def _caps_imports():
    if str(CAPS) not in sys.path:
        sys.path.insert(0, str(CAPS))
    import importlib.util
    import feature_env  # noqa: F401  (CAPS module)
    spec = importlib.util.spec_from_file_location("caps_marlfs", CAPS / "baseline" / "model" / "MARLFS.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return feature_env, mod


def _make_env(feature_env, X: pd.DataFrame, y: np.ndarray, task_type: str):
    from record import RecordList
    fe = feature_env.FeatureEvaluator.__new__(feature_env.FeatureEvaluator)
    original = pd.DataFrame(X.values.astype(float))
    original[original.shape[1]] = y.astype(float)
    original.columns = np.arange(original.shape[1])
    fe.original_report = None
    fe.records = RecordList()
    fe.task_name = "topg_fold"
    fe.task_type = task_type
    fe.col_names = list(X.columns) + ["y"]
    fe.original = original.fillna(value=0)
    xx, yy = fe.original.iloc[:, :-1], fe.original.iloc[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(xx, yy, test_size=0.2, random_state=0, shuffle=True)
    fe.train = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)
    fe.test = pd.concat([pd.DataFrame(X_test), pd.DataFrame(y_test)], axis=1)
    fe.ds_size = fe.original.shape[1] - 1
    return fe


def select_marlfs(X_tr: pd.DataFrame, y_tr: pd.Series, seed: int) -> list:
    import torch
    feature_env, mod = _caps_imports()
    Xp = X_tr
    if X_tr.shape[1] > PREFILTER:
        f, _ = f_classif(X_tr.values, np.asarray(y_tr))
        keep = np.sort(np.argsort(-np.nan_to_num(f, nan=0.0), kind="stable")[:PREFILTER])
        Xp = X_tr.iloc[:, keep]
    y = LabelEncoder().fit_transform(np.asarray(y_tr))
    task_type = "cls" if len(np.unique(y)) == 2 else "mcls"
    fe = _make_env(feature_env, Xp, y, task_type)
    torch.manual_seed(seed)   # DQN weights; the implementation seeds NumPy itself (np.random.seed(0))
    cwd = os.getcwd()
    try:
        os.chdir(CAPS)
        _, optimal_set, k = mod.gen_marlfs(fe, N_STATES=N_STATES, N_ACTIONS=2, EXPLORE_STEPS=EXPLORE_STEPS)
    finally:
        os.chdir(cwd)
    mask = np.asarray(optimal_set).astype(bool)
    return [c for c, m in zip(Xp.columns, mask) if m]
