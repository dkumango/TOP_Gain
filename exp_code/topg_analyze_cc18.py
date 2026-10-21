#!/usr/bin/env python3
"""
topg_analyze_cc18.py
====================
Analysis of the CC18 benchmark (topg_run_cc18.py) following topg_dataset_selection_protocol.md.

1. Overall comparison (dataset = unit; per dataset: mean over K, classifiers and the available repeats):
   Friedman + Nemenyi CD, TOPG vs each method: bootstrap 95 % CI of the mean difference and
   Holm-adjusted Wilcoxon p (topg_stats.analyse), for
     (a) TOPG vs all baselines available, (b) TOPG vs the alternative wrapper criteria.
2. Pre-specified hypotheses (protocol Section 5/6), per dataset:
     dAUC = ROC-AUC(TOPG) - median ROC-AUC of the baselines ; dPhi = Phi(TOPG) - Phi(net gain)
     H1: M_nl vs dAUC   H2: M_red vs dPhi   H3: M_head vs dAUC   (primary, Holm across the three)
   Spearman rho with a two-sided permutation p (10,000 permutations) and a median split (Mann-Whitney).
   Exploratory: every other moderator against both outcomes.

Usage (from the topg folder):  python topg_analyze_cc18.py [--seeds 1 2 3] [--tag interim]
Outputs: exp_result/topg_cc18_analysis/topg_cc18_<tag>_*.csv / .png
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_stats as ts  # noqa: E402

RES = HERE.parent / "exp_result"
OUT = RES / "topg_cc18_analysis"
PREFIX = ["RF", "XGB", "SVM", "LR"]
CRITERIA = ["Net probability gain", "OOB Brier score", "OOB log-loss", "OOB accuracy", "OOB balanced accuracy"]
BASELINES = ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection", "Boruta",
             "SVM-RFE", "RF impurity", "RF permutation", "RF-RFE", "varSelRF", "VSURF"]
PRIMARY = [("H1", "M_nl", "dAUC"), ("H2", "M_red", "dPhi"), ("H3", "M_head", "dAUC")]
EXPLORATORY = ["M_pca", "M_rel", "log10_P", "N_over_P", "imbalance", "n_classes", "cat_share", "auc_rf"]


def load(seeds):
    files = glob.glob(str(RES / "topg_cc18" / "topg_cc18_s*_*.csv"))
    d = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    d = d[d["seed"].isin(seeds)]
    for met in ["roc_auc", "accuracy", "macro_f1"]:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    # keep only (dataset, method) cells complete over the requested seeds
    full = d.groupby(["dataset", "method"])["seed"].nunique()
    ok = full[full == len(seeds)].index
    d = d.set_index(["dataset", "method"]).loc[lambda x: x.index.isin(ok)].reset_index()
    return d


def tables(d, methods):
    out = {}
    for met in ["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]:
        t = d[d["method"].isin(methods)].groupby(["dataset", "method"])[met].mean().unstack()
        out[met] = t[[m for m in methods if m in t.columns]].dropna()
    return out


def perm_spearman(x, y, n_perm=10000, seed=0):
    rho = spearmanr(x, y).correlation
    rng = np.random.default_rng(seed)
    null = np.array([spearmanr(x, rng.permutation(y)).correlation for _ in range(n_perm)])
    return rho, float((np.sum(np.abs(null) >= abs(rho)) + 1) / (n_perm + 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    ap.add_argument("--tag", default="final")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 230)
    d = load(args.seeds)
    avail_base = [b for b in BASELINES if b in set(d["method"])]
    print(f"seeds {args.seeds}; datasets {d['dataset'].nunique()}; baselines available: {avail_base}")

    # ---------------- 1. overall comparisons ----------------
    for comp, methods in {"baselines": ["TOPG"] + avail_base, "criteria": ["TOPG"] + CRITERIA}.items():
        tabs = tables(d, methods)
        rows, summ = [], []
        for met, tab in tabs.items():
            res, ranks, fr = ts.analyse(tab)
            res.insert(0, "metric", ts.LABEL[met])
            rows.append(res)
            summ.append(dict(metric=ts.LABEL[met], n_datasets=fr["n"], k=fr["k"], friedman_p=fr["friedman_p"],
                             cd=fr["cd"], rank_TOPG=ranks["TOPG"],
                             rank_position_TOPG=int(list(ranks.index).index("TOPG")) + 1,
                             ranks="; ".join(f"{m} {r:.2f}" for m, r in ranks.items())))
            if True:
                ts.cd_diagram(ranks, fr["cd"], ts.LABEL[met], OUT / f"topg_cc18_{args.tag}_cd_{comp}_{met}.png")
        res = pd.concat(rows, ignore_index=True)
        res.to_csv(OUT / f"topg_cc18_{args.tag}_{comp}.csv", index=False)
        sm = pd.DataFrame(summ)
        sm.to_csv(OUT / f"topg_cc18_{args.tag}_{comp}_friedman.csv", index=False)
        print(f"\n######## {comp} ########")
        print(sm[["metric", "n_datasets", "k", "friedman_p", "rank_TOPG", "rank_position_TOPG"]].round(4).to_string(index=False))
        for met in sm["metric"]:
            print(f"  [{met}] " + sm.loc[sm.metric == met, "ranks"].iloc[0])
        print(res[["metric", "method", "mean_diff", "ci_low", "ci_high", "win", "tie", "loss", "p_holm"]]
              .round(4).to_string(index=False))

    # ---------------- 2. pre-specified hypotheses ----------------
    scr = pd.read_csv(RES / "topg_dataset_screening" / "topg_screening_cc18.csv").set_index("name")
    auc = tables(d, ["TOPG"] + avail_base)["roc_auc"]
    stab = tables(d, ["TOPG", "Net probability gain"])["nogueira_stability"]
    per = pd.DataFrame({"dAUC": auc["TOPG"] - auc[avail_base].median(axis=1),
                        "dPhi": stab["TOPG"] - stab["Net probability gain"]})
    per = per.join(scr[["M_nl", "M_red", "M_head"] + EXPLORATORY])
    per.to_csv(OUT / f"topg_cc18_{args.tag}_per_dataset.csv")
    rows = []
    for h, mod, outc in PRIMARY:
        rho, p = perm_spearman(per[mod].values, per[outc].values)
        hi = per[per[mod] > per[mod].median()][outc]
        lo = per[per[mod] <= per[mod].median()][outc]
        mw = mannwhitneyu(hi, lo, alternative="two-sided").pvalue
        rows.append(dict(hypothesis=h, moderator=mod, outcome=outc, spearman_rho=rho, perm_p=p,
                         median_low=lo.median(), median_high=hi.median(), mannwhitney_p=mw))
    hyp = pd.DataFrame(rows)
    hyp["perm_p_holm"] = ts.holm(hyp["perm_p"].values)
    hyp.to_csv(OUT / f"topg_cc18_{args.tag}_hypotheses.csv", index=False)
    print("\n######## pre-specified hypotheses ########")
    print(hyp.round(4).to_string(index=False))
    ex = []
    for mod in ["M_nl", "M_red", "M_head"] + EXPLORATORY:
        for outc in ["dAUC", "dPhi"]:
            if (mod, outc) in [(m, o) for _, m, o in PRIMARY]:
                continue
            rho, p = perm_spearman(per[mod].values, per[outc].values, n_perm=5000)
            ex.append(dict(moderator=mod, outcome=outc, spearman_rho=rho, perm_p_unadjusted=p))
    ex = pd.DataFrame(ex)
    ex.to_csv(OUT / f"topg_cc18_{args.tag}_exploratory.csv", index=False)
    print("\n######## exploratory (unadjusted) ########")
    print(ex.round(3).to_string(index=False))
    print("\nper-dataset outcomes:")
    print(per[["dAUC", "dPhi", "M_nl", "M_red", "M_head"]].sort_values("M_nl", ascending=False).round(3).to_string())


if __name__ == "__main__":
    main()
