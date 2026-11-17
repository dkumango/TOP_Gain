#!/usr/bin/env python3
"""
topg_analyze_a9_ranks.py
========================
Rank analyses that were computed among the 15 CC18 methods before amendment A9, recomputed among all methods
after A9 (18 on CC18, 16 on AMLB). Stored results only; no selector is run.

  1. Friedman mean rank of TOP-Gain per downstream classifier (ROC-AUC, accuracy, Macro-F1), as in
     topg_make_revision_tables.py (Table S5), now among TOP-Gain and all baselines.
  2. Sensitivity study, part A: rank of ROC-AUC and of Phi of each surrogate variant when it replaces default
     TOP-Gain among the 18 CC18 methods, as in topg_analyze_sensitivity.py (Table 9).

Outputs: exp_result/topg_a9_analysis/topg_a9_{cc18,amlb}_per_classifier.csv, topg_a9_cc18_sensitivity_ranks.csv
Usage (from the exp_code folder):  python topg_analyze_a9_ranks.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_analyze_sensitivity as sa  # noqa: E402
import topg_make_revision_tables as rt  # noqa: E402
import topg_run_sensitivity as rs  # noqa: E402
import topg_stats as ts  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_a9_analysis"
CRITERIA = {"Net probability gain", "OOB Brier score", "OOB log-loss", "OOB accuracy", "OOB balanced accuracy"}


def per_classifier(tag, folder):
    d = rt.load(folder)
    base = [m for m in sorted(d["method"].unique()) if m not in CRITERIA]
    rows = []
    for p, lab in rt.PREFIX.items():
        for met in rt.METRICS:
            t = d[d["method"].isin(base)].groupby(["dataset", "method"])[f"{p}_mean_{met}"].mean().unstack().dropna()
            r = t.rank(axis=1, ascending=False).mean()
            rows.append(dict(classifier=lab, metric=met, n_datasets=len(t), k=t.shape[1], rank_TOPG=r["TOPG"],
                             position_TOPG=int((r < r["TOPG"]).sum()) + 1, best_other=r.drop("TOPG").idxmin(),
                             rank_best_other=r.drop("TOPG").min()))
    pc = pd.DataFrame(rows)
    pc.to_csv(OUT / f"topg_a9_{tag}_per_classifier.csv", index=False)
    print(f"\n===== {tag}: per classifier, among {pc['k'].iloc[0]} methods =====")
    print(pc.round(2).to_string(index=False))


def sensitivity_ranks():
    variants = list(rs.SENS["A"])
    sens = sa.load("topg_sensitivity", [1, 2, 3])
    sens = sens[sens["method"].isin(variants)]
    base = sa.load("topg_cc18", [1, 2, 3])
    base = base[base["dataset"].isin(set(sens["dataset"]))]
    d = pd.concat([base, sens], ignore_index=True)
    others = sorted(m for m in base["method"].unique() if m not in CRITERIA and m != "TOPG")
    rk = []
    for v in ["TOPG"] + variants:
        for met in ["roc_auc", "nogueira_stability"]:
            t = sa.table(d, met)
            tab = t[[v] + others].dropna().rename(columns={v: "TOPG"})
            _, ranks, fr = ts.analyse(tab)
            rk.append(dict(variant="default" if v == "TOPG" else v, metric=ts.LABEL[met], n_datasets=fr["n"],
                           n_methods=tab.shape[1], rank_variant=ranks["TOPG"],
                           position=int(list(ranks.index).index("TOPG")) + 1,
                           best_other=ranks.drop("TOPG").idxmin(), rank_best_other=ranks.drop("TOPG").min(),
                           ahead=", ".join(ranks.index[:list(ranks.index).index("TOPG")])))
    rk = pd.DataFrame(rk)
    rk.to_csv(OUT / "topg_a9_cc18_sensitivity_ranks.csv", index=False)
    print("\n===== sensitivity part A: rank among the CC18 methods after A9 =====")
    print(rk.round(2).to_string(index=False))
    # selection time aggregated as in Table 2: median over datasets of the dataset-level mean time per fold
    rows = []
    for part, (folder, seeds) in {"A": ("topg_cc18", [1, 2, 3]), "B": ("topg_amlb", [1])}.items():
        v = list(rs.SENS[part])
        s = sa.load("topg_sensitivity", seeds)
        s = s[s["method"].isin(v)]
        b = sa.load(folder, seeds)
        b = b[b["dataset"].isin(set(s["dataset"])) & (b["method"] == "TOPG")]
        t = pd.concat([b, s]).groupby(["dataset", "method"])["mean_selection_time_sec"].mean().unstack().median()
        rows += [dict(part=part, variant=m, median_time_per_fold_s=x) for m, x in t.items()]
    tt = pd.DataFrame(rows)
    tt.to_csv(OUT / "topg_a9_sensitivity_times.csv", index=False)
    print("\n===== sensitivity: median over datasets of the selection time per fold (s) =====")
    print(tt.round(1).to_string(index=False))


def main():
    pd.set_option("display.width", 230)
    per_classifier("cc18", "topg_cc18")
    per_classifier("amlb", "topg_amlb")
    sensitivity_ranks()


if __name__ == "__main__":
    main()
