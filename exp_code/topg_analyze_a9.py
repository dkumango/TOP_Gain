#!/usr/bin/env python3
"""
topg_analyze_a9.py
==================
CC18 / AMLB comparison including the recent baselines of protocol amendment A9 (STG, Sequential Attention,
Powershap), with the same dataset-level statistics as topg_analyze_cc18.py (Friedman + Nemenyi CD, Holm-adjusted
Wilcoxon over all baselines of the suite, bootstrap 95 % CIs). The registered final analyses are not changed.

Usage (from the exp_code folder):  python topg_analyze_a9.py --suite cc18|amlb
Outputs: exp_result/topg_a9_analysis/topg_a9_<suite>_*.csv / .png
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_stats as ts  # noqa: E402

RES = HERE.parent / "exp_result"
OUT = RES / "topg_a9_analysis"
PREFIX = ["RF", "XGB", "SVM", "LR"]
OLD = ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection", "Boruta", "SVM-RFE",
       "RF impurity", "RF permutation", "RF-RFE", "varSelRF", "VSURF"]
NEW = ["STG", "Sequential Attention", "Powershap"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="cc18", choices=["cc18", "amlb"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    folder = {"cc18": "topg_cc18", "amlb": "topg_amlb"}[args.suite]
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / folder / "topg_cc18_s*_*.csv"))], ignore_index=True)
    d = d[d.seed.isin([1, 2, 3])]
    for met in ["roc_auc", "accuracy", "macro_f1"]:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    full = d.groupby(["dataset", "method"]).seed.nunique()
    ok = full[full == 3].index
    d = d.set_index(["dataset", "method"]).loc[lambda x: x.index.isin(ok)].reset_index()
    base = [m for m in OLD + NEW if m in set(d.method)]
    methods = ["TOPG"] + base
    rows, summ = [], []
    pd.set_option("display.width", 250)
    for met in ["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]:
        tab = d[d.method.isin(methods)].groupby(["dataset", "method"])[met].mean().unstack()[methods].dropna()
        res, ranks, fr = ts.analyse(tab)
        res.insert(0, "metric", ts.LABEL[met]); rows.append(res)
        summ.append(dict(metric=ts.LABEL[met], n_datasets=fr["n"], k=fr["k"], friedman_p=fr["friedman_p"], cd=fr["cd"],
                         rank_TOPG=ranks["TOPG"], position=int(list(ranks.index).index("TOPG")) + 1,
                         ranks="; ".join(f"{m} {r:.2f}" for m, r in ranks.items())))
        ts.cd_diagram(ranks, fr["cd"], ts.LABEL[met], OUT / f"topg_a9_{args.suite}_cd_{met}.png")
    res = pd.concat(rows, ignore_index=True); sm = pd.DataFrame(summ)
    res.to_csv(OUT / f"topg_a9_{args.suite}_baselines.csv", index=False)
    sm.to_csv(OUT / f"topg_a9_{args.suite}_friedman.csv", index=False)
    desc = d[d.method.isin(methods)].groupby(["dataset", "method"])[["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]].mean() \
        .groupby("method").mean()
    desc.sort_values("roc_auc", ascending=False).to_csv(OUT / f"topg_a9_{args.suite}_descriptive.csv")
    print(f"{args.suite}: {d.dataset.nunique()} datasets, {len(base)} baselines")
    print(sm[["metric", "n_datasets", "k", "friedman_p", "cd", "rank_TOPG", "position"]].round(4).to_string(index=False))
    for m in sm.metric:
        print(f"[{m}]", sm.loc[sm.metric == m, "ranks"].iloc[0])
    print(res[res.method.isin(NEW + ["VSURF", "RF-RFE"])][["metric", "method", "mean_diff", "ci_low", "ci_high", "win", "tie", "loss", "p_holm"]].round(4).to_string(index=False))
    print(desc.sort_values("roc_auc", ascending=False).round(4).to_string())


if __name__ == "__main__":
    main()
