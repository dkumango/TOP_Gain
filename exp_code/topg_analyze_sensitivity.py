#!/usr/bin/env python3
"""
topg_analyze_sensitivity.py
===========================
Analysis of the hyperparameter sensitivity study (topg_run_sensitivity.py; protocol amendment A8).

For every variant (one setting changed from the default T = 40, depth 5, pool factor 1.0), per dataset
(mean over K, the four classifiers and the repeats run for that part):
  - difference to default TOPG in ROC-AUC, accuracy, Macro-F1 and stability Phi: mean, bootstrap 95 % CI,
    wins/losses and two-sided Wilcoxon p, Holm-adjusted over the variants of the part;
  - Part A only: the variant's Friedman mean rank of ROC-AUC and of Phi when it replaces default TOPG among
    the 15 CC18 methods, and its paired comparison with the net-gain criterion (Phi).

Usage (from the topg folder):  python topg_analyze_sensitivity.py
Outputs: exp_result/topg_sensitivity_analysis/topg_sens_*.csv
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_stats as ts  # noqa: E402
import topg_run_sensitivity as rs  # noqa: E402

RES = HERE.parent / "exp_result"
OUT = RES / "topg_sensitivity_analysis"
PREFIX = ["RF", "XGB", "SVM", "LR"]
METRICS = ["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]
CC18_BASE = ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection", "Boruta",
             "SVM-RFE", "RF impurity", "RF permutation", "RF-RFE", "varSelRF", "VSURF"]


def load(folder, seeds):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / folder / "topg_cc18_s*_*.csv"))], ignore_index=True)
    d = d[d["seed"].isin(seeds)]
    for met in METRICS[:3]:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    return d


def table(d, met):
    full = d.groupby(["dataset", "method"])["seed"].nunique()
    n = d["seed"].nunique()
    ok = full[full == n].index
    d = d.set_index(["dataset", "method"]).loc[lambda x: x.index.isin(ok)].reset_index()
    return d.groupby(["dataset", "method"])[met].mean().unstack()


def paired(x):
    x = x.dropna().values
    lo, hi = ts.bootstrap_ci(x)
    p = wilcoxon(x).pvalue if np.any(x != 0) else 1.0
    return dict(n=len(x), mean_diff=x.mean(), ci_low=lo, ci_high=hi, win=int((x > 0).sum()), loss=int((x < 0).sum()),
                p=p)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    for part, (main_folder, seeds) in {"A": ("topg_cc18", [1, 2, 3]), "B": ("topg_amlb", [1])}.items():
        variants = list(rs.SENS[part])
        sens = load("topg_sensitivity", seeds)
        sens = sens[sens["method"].isin(variants)]
        if sens.empty:
            print(f"part {part}: no results yet"); continue
        base = load(main_folder, seeds)
        base = base[base["dataset"].isin(set(sens["dataset"]))]
        d = pd.concat([base, sens], ignore_index=True)
        rows = []
        for met in METRICS:
            t = table(d, met)
            for v in variants:
                if v in t:
                    rows.append(dict(part=part, variant=v, metric=ts.LABEL[met], **paired(t[v] - t["TOPG"])))
        res = pd.DataFrame(rows)
        for (m,), g in res.groupby(["metric"]):
            res.loc[g.index, "p_holm"] = ts.holm(g["p"].values)
        res.to_csv(OUT / f"topg_sens_part{part}_vs_default.csv", index=False)
        print(f"\n######## part {part}: variant minus default TOPG ########")
        print(res.round(4).to_string(index=False))

        if part == "A":
            rk = []
            for v in variants:
                for met in ["roc_auc", "nogueira_stability"]:
                    t = table(d, met)
                    if v not in t:
                        continue
                    tab = t[[v] + CC18_BASE].dropna().rename(columns={v: "TOPG"})
                    _, ranks, fr = ts.analyse(tab)
                    rk.append(dict(variant=v, metric=ts.LABEL[met], n_datasets=fr["n"], rank_variant=ranks["TOPG"],
                                   position=int(list(ranks.index).index("TOPG")) + 1, best_other=ranks.drop("TOPG").idxmin(),
                                   rank_best_other=ranks.drop("TOPG").min()))
                t = table(d, "nogueira_stability")
                if v in t:
                    rk.append(dict(variant=v, metric="Stability vs net gain",
                                   **{k: val for k, val in paired(t[v] - t["Net probability gain"]).items()}))
            rk = pd.DataFrame(rk)
            rk.to_csv(OUT / "topg_sens_partA_ranks.csv", index=False)
            print("\n######## part A: rank among the 15 CC18 methods, and stability vs net gain ########")
            print(rk.round(4).to_string(index=False))
        tt = d[d["method"].isin(["TOPG"] + variants)].groupby("method")["mean_selection_time_sec"].median()
        print("\nmedian selection time per fold (s):", tt.round(1).to_dict())


if __name__ == "__main__":
    main()
