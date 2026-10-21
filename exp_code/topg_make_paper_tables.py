#!/usr/bin/env python3
"""
topg_make_paper_tables.py
=========================
Builds the tables of the original ten-dataset benchmark (Online Resource 1, Section S4) from the stored CSVs
(no selector is re-run).

Inputs
------
  exp_result/topg_reference/topg_ref_baselines_benchmark_summary.csv
                                                          nine baselines (+ SRA-FS v8) on the 10-dataset benchmark
                                                          (copy of git_open/exp_results/all_bio_datasets_summary.csv)
  exp_result/topg_benchmark/topg_<dataset>_summary.csv  TOPG (M6) and the criterion / extension variants
  exp_result/topg_core_validation/*.csv                 synthetic epistasis / imbalance and OVA_Lung sweeps

Outputs (exp_result/topg_tables/)
-----------------------------------
  topg_table_main_benchmark.csv      TOPG vs nine baselines: means, stability, runtime, Grand Rank, Wilcoxon
  topg_table_per_dataset.csv         per-dataset mean ROC-AUC / Macro-F1 / stability for TOPG and baselines
  topg_table_criteria.csv            TOPG vs alternative criteria inside the same wrapper (M7-M11)
  topg_table_extensions.csv          TOPG vs TOPG + SRA-FS extensions (M4, SRA-FS v8)
  topg_table_robustness_seed.csv     M12 (p_base refitted with the candidates' seed) vs TOPG, criteria, baselines

Usage:  python topg_make_paper_tables.py   (from the topg folder)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

HERE = Path(__file__).resolve().parent
BENCH = HERE.parent / "exp_result" / "topg_benchmark"
OUT = HERE.parent / "exp_result" / "topg_tables"

PREFIX = ["RF", "XGB", "SVM", "LR"]
METRICS = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}

RENAME = {
    "RF-OOB Wrapper (M6)": "TOPG",
    "RF-OOB Wrapper, OOB accuracy (M7)": "OOB accuracy",
    "RF-OOB Wrapper, OOB balanced accuracy (M8)": "OOB balanced accuracy",
    "RF-OOB Wrapper, OOB mean true-class prob. (M9)": "Net probability gain",
    "RF-OOB Wrapper, OOB log-loss (M10)": "OOB log-loss",
    "RF-OOB Wrapper, OOB Brier score (M11)": "OOB Brier score",
    "SRA-FS w/o Pair Search (M4)": "TOPG + loss penalty, class weights, smoothing",
    "SRA-FS v8": "TOPG + loss penalty, class weights, smoothing, pair synergy search",
    "TOPG, common-seed reference (M12)": "TOPG (common seed)",
}
BASELINES = ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso",
             "Stability Selection", "Boruta", "SVM-RFE"]
CRITERIA = ["TOPG", "Net probability gain", "OOB Brier score", "OOB log-loss",
            "OOB accuracy", "OOB balanced accuracy"]
EXTENSIONS = ["TOPG", "TOPG + loss penalty, class weights, smoothing",
              "TOPG + loss penalty, class weights, smoothing, pair synergy search"]


def load_all() -> pd.DataFrame:
    ref = pd.read_csv(HERE.parent / "exp_result" / "topg_reference" / "topg_ref_baselines_benchmark_summary.csv")
    # The stored HSIC-Lasso rows silently fell back to ANOVA on many folds (pyHSICLasso / NumPy 2 bug);
    # replace them with the corrected seed-42 rerun of topg_run_repeated_cv.py.
    hsic = list((HERE.parent / "exp_result" / "topg_repeated_cv").glob("topg_rep_s42_*.csv"))
    if hsic:
        fixed = pd.concat([pd.read_csv(f) for f in hsic], ignore_index=True)
        fixed = fixed[fixed["method"] == "HSIC-Lasso"].drop(columns="seed")
        if fixed["dataset"].nunique() == ref["dataset"].nunique():
            ref = pd.concat([ref[ref["method"] != "HSIC-Lasso"], fixed], ignore_index=True)
        else:
            print("WARNING: corrected HSIC-Lasso rerun incomplete; using the stored HSIC-Lasso rows")
    files = [f for f in BENCH.glob("topg_*_summary.csv") if f.name != "topg_wrapper_all_summary.csv"]
    new = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = pd.concat([ref, new], ignore_index=True)
    d = d[d["dataset"].isin(new["dataset"].unique())].copy()
    d["method"] = d["method"].replace(RENAME)
    for met in METRICS:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    return d


def grand_rank(d: pd.DataFrame, methods: list[str]) -> pd.Series:
    sub = d[d["method"].isin(methods)]
    long = []
    for met in METRICS:
        for p in PREFIX:
            c = f"{p}_mean_{met}"
            t = sub[["dataset", "k_label", "method", c]].rename(columns={c: "val"})
            t["slice"] = t["dataset"] + "|" + t["k_label"].astype(str) + "|" + p + "|" + met
            long.append(t)
    long = pd.concat(long, ignore_index=True)
    long["rank"] = long.groupby("slice")["val"].rank(ascending=False, method="average")
    return long.groupby("method")["rank"].mean()


def paired(d: pd.DataFrame, a: str, b: str, col: str, level: str):
    """Paired comparison a vs b. level='dataset' (n=10) or 'slice' (dataset x K, n=40)."""
    keys = ["dataset"] if level == "dataset" else ["dataset", "k_label"]
    piv = d[d["method"].isin([a, b])].groupby(keys + ["method"])[col].mean().unstack()[[a, b]].dropna()
    diff = piv[a] - piv[b]
    w, t, l = int((diff > 1e-9).sum()), int((diff.abs() <= 1e-9).sum()), int((diff < -1e-9).sum())
    p = float(wilcoxon(piv[a], piv[b]).pvalue) if (diff.abs() > 1e-12).any() else 1.0
    return dict(n=len(diff), mean_diff=diff.mean(), win=w, tie=t, loss=l, p=p)


def summary_rows(d, methods, ref="TOPG"):
    gr = grand_rank(d, methods)
    rows = []
    for m in methods:
        s = d[d["method"] == m]
        row = {"method": m, "grand_rank": gr[m],
               **{f"mean_{k}": s[k].mean() for k in METRICS},
               "mean_stability": s["nogueira_stability"].mean(),
               "mean_time_s": s.groupby("dataset")["mean_selection_time_sec"].mean().mean()}
        if m != ref:
            for k in list(METRICS) + ["nogueira_stability"]:
                for lvl in ["dataset", "slice"]:
                    r = paired(d, ref, m, k, lvl)
                    tag = "stab" if k == "nogueira_stability" else k
                    row[f"{tag}_{lvl}_W/T/L"] = f"{r['win']}/{r['tie']}/{r['loss']}"
                    row[f"{tag}_{lvl}_p"] = r["p"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("grand_rank")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_all()
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)

    main_tab = summary_rows(d, ["TOPG"] + BASELINES)
    main_tab.to_csv(OUT / "topg_table_main_benchmark.csv", index=False)
    print("=== Main benchmark (TOPG vs nine baselines) ===")
    print(main_tab.round(4).to_string(index=False))

    per_ds = d[d["method"].isin(["TOPG"] + BASELINES)].groupby(["dataset", "method"]).agg(
        roc_auc=("roc_auc", "mean"), macro_f1=("macro_f1", "mean"),
        stability=("nogueira_stability", "mean")).reset_index()
    per_ds.to_csv(OUT / "topg_table_per_dataset.csv", index=False)
    print("\n=== Per-dataset ROC-AUC ===")
    print(per_ds.pivot(index="dataset", columns="method", values="roc_auc").round(4).to_string())
    print("\n=== Per-dataset stability ===")
    print(per_ds.pivot(index="dataset", columns="method", values="stability").round(3).to_string())

    crit = summary_rows(d, CRITERIA)
    crit.to_csv(OUT / "topg_table_criteria.csv", index=False)
    print("\n=== Criteria (same wrapper) ===")
    print(crit.round(4).to_string(index=False))

    ext = summary_rows(d, EXTENSIONS)
    ext.to_csv(OUT / "topg_table_extensions.csv", index=False)
    print("\n=== Extensions ===")
    print(ext.round(4).to_string(index=False))

    if "TOPG (common seed)" in set(d["method"]):
        ref = "TOPG (common seed)"
        rob = pd.concat([summary_rows(d, [ref, "TOPG"] + CRITERIA[1:], ref=ref).assign(comparison="criteria"),
                         summary_rows(d, [ref] + BASELINES, ref=ref).assign(comparison="baselines")],
                        ignore_index=True)
        rob.to_csv(OUT / "topg_table_robustness_seed.csv", index=False)
        print("\n=== Robustness: common-seed TOPG ===")
        print(rob[["comparison", "method", "grand_rank", "mean_roc_auc", "mean_macro_f1", "mean_stability",
                   "roc_auc_dataset_W/T/L", "roc_auc_dataset_p", "roc_auc_slice_p",
                   "stab_dataset_W/T/L", "stab_dataset_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
