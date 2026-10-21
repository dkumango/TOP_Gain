#!/usr/bin/env python3
"""
topg_stats.py
=============
Statistical comparison of TOPG with the baselines and the alternative criteria, following
Demšar (2006): the unit of analysis is the DATASET (n = 10). Dataset x K pairs are not used as
independent units (nested panels from one ordered list -> pseudo-replication).

For every metric (ROC-AUC, accuracy, Macro-F1, stability), averaged per dataset over K and classifiers:
  1. Friedman test over all compared methods (+ Iman-Davenport F) and the Nemenyi critical difference
     (alpha = 0.05) -> CD diagram (PNG).
  2. TOPG versus each other method: mean difference with a bootstrap 95 % CI over datasets
     (10,000 resamples), wins/ties/losses, two-sided Wilcoxon signed-rank p, Holm-adjusted p.

Sources
-------
  single   : stored seed-42 benchmark (HSIC-Lasso rows replaced by the corrected seed-42 rerun)
  repeated : seed 42 + repeats in exp_result/topg_repeated_cv (per dataset: mean over available seeds;
             only seeds for which every compared method is available are used)

Usage (from the topg folder):  python topg_stats.py --source single|repeated
Outputs: exp_result/topg_stats/topg_stats_<source>_<comparison>.csv, topg_cd_<source>_<comparison>_<metric>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "exp_result"
OUT = RES / "topg_stats"
PREFIX = ["RF", "XGB", "SVM", "LR"]
METRICS = ["roc_auc", "accuracy", "macro_f1", "nogueira_stability"]
LABEL = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "macro_f1": "Macro-F1", "nogueira_stability": "Stability"}

RENAME = {
    "RF-OOB Wrapper (M6)": "TOPG",
    "RF-OOB Wrapper, OOB accuracy (M7)": "OOB accuracy",
    "RF-OOB Wrapper, OOB balanced accuracy (M8)": "OOB balanced accuracy",
    "RF-OOB Wrapper, OOB mean true-class prob. (M9)": "Net probability gain",
    "RF-OOB Wrapper, OOB log-loss (M10)": "OOB log-loss",
    "RF-OOB Wrapper, OOB Brier score (M11)": "OOB Brier score",
}
BASELINES = ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection", "Boruta", "SVM-RFE"]
CRITERIA = ["Net probability gain", "OOB Brier score", "OOB log-loss", "OOB accuracy", "OOB balanced accuracy"]
COMPARISONS = {"baselines": ["TOPG"] + BASELINES, "criteria": ["TOPG"] + CRITERIA}

# Nemenyi critical values q_0.05 (Studentized range / sqrt(2)), Demšar (2006), Table 5(a)
Q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728, 6: 2.850, 7: 2.949, 8: 3.031, 9: 3.102, 10: 3.164}


def q05(k: int) -> float:
    """Nemenyi critical value; tabulated for k <= 10, otherwise studentized range q(0.95, k, inf) / sqrt(2)."""
    if k in Q05:
        return Q05[k]
    from scipy.stats import studentized_range
    return float(studentized_range.ppf(0.95, k, np.inf) / np.sqrt(2))


def load_single() -> pd.DataFrame:
    ref = pd.read_csv(RES / "topg_reference" / "topg_ref_baselines_benchmark_summary.csv")
    hsic = sorted((RES / "topg_repeated_cv").glob("topg_rep_s42_*.csv"))
    if hsic:
        fixed = pd.concat([pd.read_csv(f) for f in hsic], ignore_index=True)
        fixed = fixed[fixed["method"] == "HSIC-Lasso"]
        if fixed["dataset"].nunique() == ref["dataset"].nunique():
            ref = pd.concat([ref[ref["method"] != "HSIC-Lasso"], fixed.drop(columns="seed")], ignore_index=True)
        else:
            print("WARNING: corrected HSIC-Lasso rerun incomplete; using stored HSIC-Lasso rows")
    files = [f for f in (RES / "topg_benchmark").glob("topg_*_summary.csv") if f.name != "topg_wrapper_all_summary.csv"]
    own = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    own["method"] = own["method"].replace(RENAME)
    d = pd.concat([ref, own], ignore_index=True)
    d["seed"] = 42
    return d


def load_repeated() -> pd.DataFrame:
    d = [load_single()]
    for f in sorted((RES / "topg_repeated_cv").glob("topg_rep_s*_*.csv")):
        r = pd.read_csv(f)
        if (r["seed"] != 42).all():
            d.append(r)
    return pd.concat(d, ignore_index=True)


def per_dataset(d: pd.DataFrame, methods: list[str]) -> dict[str, pd.DataFrame]:
    d = d[d["method"].isin(methods)].copy()
    # keep only seeds with every method on every dataset
    full = d.groupby("seed").apply(lambda g: g.groupby("method")["dataset"].nunique().reindex(methods).min())
    n_ds = d["dataset"].nunique()
    seeds = sorted(full[full == n_ds].index)
    d = d[d["seed"].isin(seeds)]
    for met in ["roc_auc", "accuracy", "macro_f1"]:
        d[met] = d[[f"{p}_mean_{met}" for p in PREFIX]].mean(axis=1)
    out = {}
    for met in METRICS:
        out[met] = d.groupby(["dataset", "method"])[met].mean().unstack()[methods]
    out["_seeds"] = seeds
    return out


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx])
        adj[idx] = min(1.0, run)
    return adj


def bootstrap_ci(x: np.ndarray, B: int = 10000, seed: int = 0):
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def analyse(tab: pd.DataFrame, ref: str = "TOPG"):
    k, n = tab.shape[1], tab.shape[0]
    ranks = tab.apply(lambda r: pd.Series(rankdata(-r.values), index=r.index), axis=1)
    avg_rank = ranks.mean()
    chi2, p_f = friedmanchisquare(*[tab[c].values for c in tab.columns])
    ff = (n - 1) * chi2 / (n * (k - 1) - chi2) if n * (k - 1) != chi2 else np.inf
    cd = q05(k) * np.sqrt(k * (k + 1) / (6.0 * n))
    rows = []
    for c in tab.columns:
        if c == ref:
            continue
        diff = (tab[ref] - tab[c]).values
        lo, hi = bootstrap_ci(diff)
        w, t, l = int((diff > 1e-12).sum()), int((np.abs(diff) <= 1e-12).sum()), int((diff < -1e-12).sum())
        p = float(wilcoxon(tab[ref], tab[c]).pvalue) if np.any(np.abs(diff) > 1e-12) else 1.0
        rows.append(dict(method=c, mean_diff=diff.mean(), ci_low=lo, ci_high=hi, win=w, tie=t, loss=l, p_wilcoxon=p,
                         rank_gap=avg_rank[c] - avg_rank[ref], nemenyi_sig=(avg_rank[c] - avg_rank[ref]) > cd))
    res = pd.DataFrame(rows)
    res["p_holm"] = holm(res["p_wilcoxon"].values)
    return res, avg_rank.sort_values(), dict(friedman_chi2=chi2, friedman_p=p_f, iman_davenport_F=ff, cd=cd, n=n, k=k)


def cd_diagram(avg_rank: pd.Series, cd: float, title: str, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    k = len(avg_rank)
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * k + 1.3))
    ax.set_xlim(1, k)
    ax.set_ylim(0, k + 2)
    ax.invert_xaxis()
    ax.axis("off")
    ax.hlines(k + 1, 1, k, color="black", lw=1)
    for r in range(1, k + 1):
        ax.vlines(r, k + 1, k + 1.2, color="black", lw=1)
        ax.text(r, k + 1.35, str(r), ha="center", va="bottom", fontsize=8)
    best = float(avg_rank.iloc[0])  # CD bar anchored at the best-ranked method
    ax.hlines(k + 1.75, best, min(k, best + cd), color="black", lw=2)
    ax.text(best + cd / 2, k + 1.85, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=8)
    for i, (m, r) in enumerate(avg_rank.items()):
        y = k - i
        ax.vlines(r, y, k + 1, color="0.6", lw=0.6)
        ax.plot(r, k + 1, "o", color="black", ms=3)
        ax.text(r, y, f" {m} ({r:.2f})", ha="right" if i < k / 2 else "left", va="center", fontsize=8,
                fontweight="bold" if m == "TOPG" else "normal")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["single", "repeated"], default="single")
    ap.add_argument("--common-seed", action="store_true",
                    help="analyse the common-seed TOPG variant (M12) in place of TOPG (single source only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    d = load_single() if args.source == "single" else load_repeated()
    if args.common_seed:
        d = d[d["method"] != "TOPG"].copy()
        d["method"] = d["method"].replace({"TOPG, common-seed reference (M12)": "TOPG"})
        args.source = args.source + "_commonseed"
    pd.set_option("display.width", 220)
    for comp, methods in COMPARISONS.items():
        tabs = per_dataset(d, methods)
        seeds = tabs.pop("_seeds")
        allrows, summ = [], []
        for met in METRICS:
            res, ranks, fr = analyse(tabs[met])
            res.insert(0, "metric", LABEL[met])
            allrows.append(res)
            summ.append(dict(metric=LABEL[met], **fr, avg_rank_TOPG=ranks["TOPG"], best=ranks.index[0],
                             avg_ranks="; ".join(f"{m} {r:.2f}" for m, r in ranks.items())))
            cd_diagram(ranks, fr["cd"], LABEL[met],
                       OUT / f"topg_cd_{args.source}_{comp}_{met}.png")
        res = pd.concat(allrows, ignore_index=True)
        res.to_csv(OUT / f"topg_stats_{args.source}_{comp}.csv", index=False)
        sm = pd.DataFrame(summ)
        sm.to_csv(OUT / f"topg_stats_{args.source}_{comp}_friedman.csv", index=False)
        print(f"\n######## {comp} | source={args.source} | seeds={seeds} ########")
        print(sm[["metric", "friedman_p", "iman_davenport_F", "cd", "avg_rank_TOPG", "best"]].round(4).to_string(index=False))
        print(res.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
