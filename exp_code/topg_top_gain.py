#!/usr/bin/env python3
"""
topg_top_gain.py
================
Minimal scikit-learn-style interface to TOP-Gain (Truncated Out-of-bag Probability Gain) forward selection.

It runs exactly the configuration used in the paper (code key M6 of topg_selector.SRAFSAblation): plain
RF-OOB forward selection scored by mean_i max(0, p_i(S + f) - p_i(S)), with no loss penalty, class weights,
smoothing or pair search.

Example
-------
    from sklearn.datasets import make_classification
    from topg_top_gain import TOPGainSelector

    X, y = make_classification(n_samples=600, n_features=40, n_informative=6, random_state=0)
    sel = TOPGainSelector(n_features=10, random_state=1).fit(X, y)
    print(sel.selected_features_)        # ordered list: the first k entries are the best panel of size k
    X_small = sel.transform(X)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from topg_selector import SRAFSAblation

# the TOP-Gain configuration of the paper (identical to VARIANTS["M6"] in topg_run_wrapper_baselines.py)
_TOP_GAIN = dict(enable_g4_g3=False, enable_class_weight=False, enable_smoothing=False, pair_search=False,
                 min_g4=0.0, objective_mode="g1_only")


class TOPGainSelector(BaseEstimator, TransformerMixin):
    """TOP-Gain forward feature selection with a Random-Forest OOB surrogate.

    Parameters
    ----------
    n_features : int
        Number of features to select (K_max). The result is an ordered list, so its first k entries are the
        selection for every smaller panel size k.
    n_trees : int, default 40
        Trees of each surrogate forest (T). Larger values give more stable selections at higher cost.
    max_depth : int, default 5
        Maximum depth of the surrogate trees.
    pool_factor : float, default 1.0
        Scales the candidate-pool size used when the data have more than 150 features.
    random_state : int, default 0
    n_jobs : int, default -1
        Parallel workers for the candidate evaluations (the selection does not depend on it).
    """

    def __init__(self, n_features: int = 10, n_trees: int = 40, max_depth: int = 5, pool_factor: float = 1.0,
                 random_state: int = 0, n_jobs: int = -1):
        self.n_features = n_features
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.pool_factor = pool_factor
        self.random_state = random_state
        self.n_jobs = n_jobs

    def fit(self, X, y):
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(np.asarray(X),
                                                                    columns=[f"x{i}" for i in range(np.shape(X)[1])])
        self.feature_names_in_ = np.asarray(X_df.columns, dtype=object)
        sel = SRAFSAblation(max_features=self.n_features, surrogate_estimators=self.n_trees, max_depth=self.max_depth,
                            pool_size_factor=self.pool_factor, random_state=self.random_state, use_oob=True,
                            n_jobs=self.n_jobs, **_TOP_GAIN)
        sel.fit(X_df, np.asarray(y))
        self.selected_features_ = list(sel.selected_features_)[: self.n_features]
        pos = {c: i for i, c in enumerate(X_df.columns)}
        self.selected_indices_ = [pos[c] for c in self.selected_features_]
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            return X[self.selected_features_]
        return np.asarray(X)[:, self.selected_indices_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.asarray(self.selected_indices_)
        mask = np.zeros(len(self.feature_names_in_), dtype=bool)
        mask[self.selected_indices_] = True
        return mask
