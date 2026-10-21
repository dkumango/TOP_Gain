#!/usr/bin/env python3
"""
topg_analyze_amlb.py
====================
Confirmatory analysis of the AMLB benchmark (OpenML suite 271 minus CC18), exactly as registered in
topg_dataset_selection_protocol.md, amendment A3.

1. H4a (primary):   one-sided Spearman permutation test (10,000 permutations, alternative rho < 0) of M_rel
                    against dAUC = ROC-AUC(TOPG) - median ROC-AUC of the twelve Python baselines, alpha = 0.05.
   H4b (secondary): the same test of M_rel against dPhi = Phi(TOPG) - Phi(net gain).
2. Replication of the overall comparison: Friedman + Nemenyi CD, bootstrap 95 % CIs and Holm-adjusted
   Wilcoxon tests (topg_stats.analyse) for TOPG vs the twelve baselines, and TOPG vs net gain.
3. Exploratory (unadjusted, two-sided): the other moderators against both outcomes.

Unit = dataset; per dataset the mean over K, the four classifiers and the three repeats.

Usage (from the topg folder):  python topg_analyze_amlb.py
Outputs: exp_result/topg_amlb_analysis/topg_amlb_*.csv / .png
"""
from __future__ import annotations

import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, wilcoxon

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_stats as ts  # noqa: E402

RES = HERE.parent / "exp_result"
OUT = RES / "topg_amlb_analysis"
SCREEN = RES / "topg_dataset_screening" / "topg_screening_amlb.csv"
PREFIX = ["RF", "XGB", "SVM", "LR"]
SEEDS = [1, 2, 3]
BASELINES = ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection", "Boruta",
             "SVM-RFE", "RF impurity", "RF permutation", "RF-RFE"]
EXPLORATORY = ["M_nl", "M_red", "M_head", "M_pca", "log10_P", "N_over_P", "imbalance", "n_classes", "cat_share",
               "auc_rf"]


def load():
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(RES / "topg_amlb" / "topg_cc18_s*_*.csv"))],
                  ignore_index=True)
    d = d[d["seed"].isin(SEEDS)]
    for met in ["roc_auc", "accuracy", "macro_f1"]:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    full = d.groupby(["dataset", "method"])["seed"].nunique()
    ok = full[full == len(SEEDS)].index
    return d.set_index(["dataset", "method"]).loc[lambda x: x.index.isin(ok)].reset_index()


def table(d, met, methods):
    t = d[d["method"].isin(methods)].groupby(["dataset", "method"])[met].mean().unstack()
    return t[methods].dropna()


def perm_spearman(x, y, alternative, n_perm=10000, seed=0):
    rho = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n_perm)])
    hits = np.sum(null <= rho) if alternative == "less" else np.sum(np.abs(null) >= abs(rho))
    return rho, float((hits + 1) / (n_perm + 1))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    d = load()
    n_ds = d["dataset"].nunique()
    print(f"seeds {SEEDS}; datasets {n_ds}; methods {d['method'].nunique()}")

    # ---------------- 1. confirmatory tests H4a / H4b ----------------
    scr = pd.read_csv(SCREEN).set_index("name")
    auc = table(d, "roc_auc", ["TOPG"] + BASELINES)
    stab = table(d, "nogueira_stability", ["TOPG", "Net probability gain"])
    per = pd.DataFrame({"dAUC": auc["TOPG"] - auc[BASELINES].median(axis=1),
                        "dPhi": stab["TOPG"] - stab["Net probability gain"]})
    per = per.join(scr[["M_rel"] + EXPLORATORY])
    per.to_csv(OUT / "topg_amlb_per_dataset.csv")
    rows = []
    for h, outc, role in [("H4a", "dAUC", "primary"), ("H4b", "dPhi", "secondary")]:
        q = per[["M_rel", outc]].dropna()
        rho, p = perm_spearman(q["M_rel"].values, q[outc].values, alternative="less")
        rows.append(dict(hypothesis=h, role=role, moderator="M_rel", outcome=outc, n_datasets=len(q),
                         spearman_rho=rho, perm_p_one_sided=p, alpha=0.05, rejected=p < 0.05))
    hyp = pd.DataFrame(rows)
    hyp.to_csv(OUT / "topg_amlb_hypotheses.csv", index=False)
    print("\n######## confirmatory hypotheses (A3) ########")
    print(hyp.round(4).to_string(index=False))

    # ---------------- 2. replication of the overall comparison ----------------
    rows, summ = [], []
    for met in ["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]:
        tab = table(d, met, ["TOPG"] + BASELINES)
        res, ranks, fr = ts.analyse(tab)
        res.insert(0, "metric", ts.LABEL[met])
        rows.append(res)
        summ.append(dict(metric=ts.LABEL[met], n_datasets=fr["n"], k=fr["k"], friedman_p=fr["friedman_p"],
                         cd=fr["cd"], rank_TOPG=ranks["TOPG"],
                         rank_position_TOPG=int(list(ranks.index).index("TOPG")) + 1,
                         ranks="; ".join(f"{m} {r:.2f}" for m, r in ranks.items())))
        ts.cd_diagram(ranks, fr["cd"], ts.LABEL[met], OUT / f"topg_amlb_cd_baselines_{met}.png")
    res = pd.concat(rows, ignore_index=True)
    res.to_csv(OUT / "topg_amlb_baselines.csv", index=False)
    sm = pd.DataFrame(summ)
    sm.to_csv(OUT / "topg_amlb_baselines_friedman.csv", index=False)
    print("\n######## TOPG vs twelve baselines ########")
    print(sm[["metric", "n_datasets", "k", "friedman_p", "cd", "rank_TOPG", "rank_position_TOPG"]].round(4)
          .to_string(index=False))
    for met in sm["metric"]:
        print(f"  [{met}] " + sm.loc[sm.metric == met, "ranks"].iloc[0])
    print(res[["metric", "method", "mean_diff", "ci_low", "ci_high", "win", "tie", "loss", "p_holm"]]
          .round(4).to_string(index=False))

    ng = []
    for met in ["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]:
        t = table(d, met, ["TOPG", "Net probability gain"])
        x = (t["TOPG"] - t["Net probability gain"]).values
        lo, hi = ts.bootstrap_ci(x)
        ng.append(dict(metric=ts.LABEL[met], n_datasets=len(x), mean_diff=x.mean(), ci_low=lo, ci_high=hi,
                       win=int((x > 0).sum()), tie=int((x == 0).sum()), loss=int((x < 0).sum()),
                       wilcoxon_p=wilcoxon(x).pvalue))
    ng = pd.DataFrame(ng)
    ng["p_holm"] = ts.holm(ng["wilcoxon_p"].values)
    ng.to_csv(OUT / "topg_amlb_vs_netgain.csv", index=False)
    print("\n######## TOPG vs net gain ########")
    print(ng.round(4).to_string(index=False))

    tt = d[d["method"].isin(["TOPG", "Net probability gain"] + BASELINES)].groupby("method")[
        "mean_selection_time_sec"].agg(["median", "sum"])
    tt["sum_ratio_vs_TOPG"] = tt["sum"] / tt.loc["TOPG", "sum"]
    tt.sort_values("sum").to_csv(OUT / "topg_amlb_selection_time.csv")

    # ---------------- 3. exploratory ----------------
    ex = []
    for mod in EXPLORATORY:
        for outc in ["dAUC", "dPhi"]:
            q = per[[mod, outc]].dropna()
            rho, p = perm_spearman(q[mod].values, q[outc].values, alternative="two-sided", n_perm=5000)
            ex.append(dict(moderator=mod, outcome=outc, spearman_rho=rho, perm_p_unadjusted=p))
    ex = pd.DataFrame(ex)
    ex.to_csv(OUT / "topg_amlb_exploratory.csv", index=False)
    print("\n######## exploratory (two-sided, unadjusted) ########")
    print(ex.round(3).to_string(index=False))
    print("\nper-dataset outcomes:")
    print(per[["dAUC", "dPhi", "M_rel"]].sort_values("M_rel").round(4).to_string())


if __name__ == "__main__":
    main()
