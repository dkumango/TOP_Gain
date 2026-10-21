#!/usr/bin/env python3
"""
topg_make_revision_tables.py
============================
Descriptive tables for the revised manuscript (CC18 main benchmark, AMLB confirmatory set). The inferential
statistics come from topg_analyze_cc18.py / topg_analyze_amlb.py; this script only adds the descriptive
means, per-classifier ranks, dataset lists and selection times, from the same raw result files and with the
same aggregation (dataset = unit; mean over K, the four classifiers and the three repeats).

Usage (from the topg folder):  python topg_make_revision_tables.py
Outputs: exp_result/topg_tables/topg_rev_*.csv
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "exp_result"
OUT = RES / "topg_tables"
PREFIX = {"RF": "Random Forest", "XGB": "XGBoost", "SVM": "RBF-SVM", "LR": "Logistic regression"}
METRICS = ["roc_auc", "accuracy", "macro_f1"]
SETS = {"cc18": ("topg_cc18", "topg_screening_cc18.csv"), "amlb": ("topg_amlb", "topg_screening_amlb.csv")}


def load(folder):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / folder / "topg_cc18_s*_*.csv"))], ignore_index=True)
    d = d[d["seed"].isin([1, 2, 3])]
    for met in METRICS:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    full = d.groupby(["dataset", "method"])["seed"].nunique()
    ok = full[full == 3].index
    return d.set_index(["dataset", "method"]).loc[lambda x: x.index.isin(ok)].reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    for tag, (folder, screen) in SETS.items():
        d = load(folder)
        methods = sorted(d["method"].unique())
        # dataset-level means, then the mean over datasets (descriptive)
        cols = METRICS + ["nogueira_stability"]
        per = d.groupby(["dataset", "method"])[cols + ["mean_selection_time_sec"]].mean()
        desc = per.groupby("method")[cols].mean()
        desc["median_time_per_fold_s"] = per.groupby("method")["mean_selection_time_sec"].median()
        desc["total_time_per_fold_s"] = per.groupby("method")["mean_selection_time_sec"].sum()
        desc["total_time_ratio_vs_TOPG"] = desc["total_time_per_fold_s"] / desc.loc["TOPG", "total_time_per_fold_s"]
        desc.sort_values("roc_auc", ascending=False).to_csv(OUT / f"topg_rev_{tag}_descriptive.csv")
        print(f"\n===== {tag}: descriptive =====")
        print(desc.sort_values("roc_auc", ascending=False).round(4).to_string())

        # per-classifier Friedman mean ranks of ROC-AUC (TOPG vs the baselines, criteria excluded)
        base = [m for m in methods if m not in ("Net probability gain", "OOB Brier score", "OOB log-loss",
                                                "OOB accuracy", "OOB balanced accuracy")]
        rows = []
        for p, lab in PREFIX.items():
            for met in METRICS:
                t = d[d["method"].isin(base)].groupby(["dataset", "method"])[f"{p}_mean_{met}"].mean().unstack().dropna()
                r = t.rank(axis=1, ascending=False).mean()
                rows.append(dict(classifier=lab, metric=met, n_datasets=len(t), k=t.shape[1],
                                 rank_TOPG=r["TOPG"], position_TOPG=int((r < r["TOPG"]).sum()) + 1,
                                 best_other=r.drop("TOPG").idxmin(), rank_best_other=r.drop("TOPG").min()))
        pc = pd.DataFrame(rows)
        pc.to_csv(OUT / f"topg_rev_{tag}_per_classifier.csv", index=False)
        print(f"\n===== {tag}: per classifier (baselines) =====")
        print(pc.round(2).to_string(index=False))

        # dataset list
        scr = pd.read_csv(RES / "topg_dataset_screening" / screen)
        scr = scr[scr["eligible"]]
        keep = ["name", "did", "N", "N_used", "P_used", "n_classes", "imbalance", "M_rel", "M_nl", "M_red", "M_head",
                "auc_rf"]
        ds = scr[keep].sort_values("P_used")
        ds.to_csv(OUT / f"topg_rev_{tag}_datasets.csv", index=False)
        print(f"\n===== {tag}: datasets ({len(ds)}) =====")
        print(ds.round(3).to_string(index=False))
        print("ranges: N_used", ds.N_used.min(), ds.N_used.max(), "P", ds.P_used.min(), ds.P_used.max(),
              "classes", ds.n_classes.min(), ds.n_classes.max(), "M_rel", round(ds.M_rel.min(), 3), round(ds.M_rel.max(), 3))


if __name__ == "__main__":
    main()
