#!/usr/bin/env python3
"""
topg_powershap_bridge.py
========================
Runs Powershap (Verhaeghe et al., ECML PKDD 2022) with its package defaults in the Python 3.12 environment
topg_venv_py312 and writes a ranking. Called by topg_new_baselines.select_powershap.

Usage: python topg_powershap_bridge.py <in.csv> <out.txt> <max_k> <seed>
  in.csv : features V0..Vp-1 and a last column y (integer class labels)
  out.txt: one 0-based feature index per line, best first (length max_k)

Ranking: Powershap's own output table (all features), sorted by p-value (ascending) and then by mean absolute
SHAP impact (descending); ties by column order.
Default model: the package default (CatBoostClassifier); automatic mode; stratified validation split.
"""
import os
import sys

import numpy as np
import pandas as pd
from powershap import PowerShap


def main():
    inp, out, max_k, seed = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    df = pd.read_csv(inp)
    X, y = df.drop(columns="y"), df["y"].values
    np.random.seed(seed)
    from catboost import CatBoostClassifier
    n_threads = int(os.environ.get("TOPG_N_JOBS", "-1"))  # set only by the runtime study; default -1 = all cores
    sel = PowerShap(model=CatBoostClassifier(verbose=0, n_estimators=250, random_seed=seed, thread_count=n_threads),
                    stratify=True, show_progress=False, verbose=False)
    sel.fit(X, y)
    tab = sel._processed_shaps_df.copy()
    tab = tab.loc[[c for c in tab.index if c in X.columns]]
    pos = {c: i for i, c in enumerate(X.columns)}
    tab["pos"] = [pos[c] for c in tab.index]
    tab = tab.sort_values(["p_value", "impact", "pos"], ascending=[True, False, True])
    order = [pos[c] for c in tab.index]
    order += [i for i in range(X.shape[1]) if i not in set(order)]
    with open(out, "w") as f:
        f.write("\n".join(str(i) for i in order[:max_k]))


if __name__ == "__main__":
    main()
