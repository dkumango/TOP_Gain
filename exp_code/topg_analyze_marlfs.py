#!/usr/bin/env python3
"""
topg_analyze_marlfs.py
======================
Analysis of the A10 comparison (MARLFS vs size-matched TOP-Gain), as registered in protocol amendment A10:
unit = dataset (metrics averaged over the five folds and the four classifiers); mean paired difference with a
bootstrap 95% CI, wins/ties/losses and a two-sided Wilcoxon test per suite; Nogueira's Phi over the five folds;
panel sizes k.

Usage:  python topg_analyze_marlfs.py --suite cc18|amlb
Outputs: exp_result/topg_marlfs_analysis/topg_marlfs_<suite>_{per_dataset,summary}.csv
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_common as tc  # noqa: E402

RES = HERE.parent / "exp_result" / "topg_marlfs"
OUT = HERE.parent / "exp_result" / "topg_marlfs_analysis"
PRED = ["RF", "XGB", "SVM", "LR"]
METRICS = ["roc_auc", "accuracy", "macro_f1"]
M, T = "MARLFS", "TOPG (size-matched)"


def phi(sel_lists, p):
    feats = sorted({f for s in sel_lists for f in s})
    idx = {f: i for i, f in enumerate(feats)}
    Z = np.zeros((len(sel_lists), p))
    for r, s in enumerate(sel_lists):
        for f in s:
            Z[r, idx[f]] = 1.0
    return tc.nogueira_stability(Z)


def boot_ci(d, B=10000, seed=1):
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(B, len(d)), replace=True).mean(1)
    return np.percentile(means, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cc18")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / f"topg_marlfs_{a.suite}_*.csv"))], ignore_index=True)
    rows = []
    for ds, g in df.groupby("dataset"):
        if g[g.method == M].fold.nunique() < 5 or g[g.method == T].fold.nunique() < 5:
            print(f"  incomplete: {ds}")
            continue
        r = dict(dataset=ds, P=int(g.P.iloc[0]))
        for meth, tag in [(M, "marlfs"), (T, "topg")]:
            gm = g[g.method == meth].sort_values("fold")
            for met in METRICS:
                r[f"{tag}_{met}"] = gm[[f"{p}_{met}" for p in PRED]].values.mean()
            r[f"{tag}_k_mean"] = gm.n_selected.mean()
            r[f"{tag}_phi"] = phi([s.split(";") for s in gm.selected.astype(str)], r["P"])
            r[f"{tag}_time_mean"] = gm.selection_time_sec.mean()
        r["marlfs_k_over20_folds"] = int((g[g.method == M].marlfs_k > 20).sum())
        rows.append(r)
    per = pd.DataFrame(rows).sort_values("P")
    per.to_csv(OUT / f"topg_marlfs_{a.suite}_per_dataset.csv", index=False)
    summ = []
    for met in METRICS + ["phi"]:
        d = (per[f"topg_{met}"] - per[f"marlfs_{met}"]).values
        lo, hi = boot_ci(d)
        nz = d[np.abs(d) > 1e-12]
        p = wilcoxon(nz).pvalue if len(nz) >= 1 else np.nan
        summ.append(dict(metric=met, n_datasets=len(d), topg_mean=per[f"topg_{met}"].mean(),
                         marlfs_mean=per[f"marlfs_{met}"].mean(), diff_mean=d.mean(), ci_lo=lo, ci_hi=hi,
                         wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
                         losses=int((d < -1e-12).sum()), wilcoxon_p=p))
    s = pd.DataFrame(summ)
    s.to_csv(OUT / f"topg_marlfs_{a.suite}_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s.round(4).to_string(index=False))
    print(f"\nMARLFS k: median {per.marlfs_k_mean.median():.1f}, mean {per.marlfs_k_mean.mean():.1f}; "
          f"folds with k > 20: {per.marlfs_k_over20_folds.sum()} of {5 * len(per)}")
    print(f"selection time (s/fold): MARLFS {per.marlfs_time_mean.mean():.0f}, TOPG {per.topg_time_mean.mean():.1f}")


if __name__ == "__main__":
    main()
