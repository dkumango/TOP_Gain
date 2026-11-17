#!/usr/bin/env python3
"""
topg_make_r2_tables.py
======================
Markdown tables for the resubmission (revision R2): the A9 baseline comparisons (main-text subsets and full
Online Resource tables), the A10 MARLFS comparison and the correlation-aware stability analysis. All numbers are
read from the stored analysis outputs; nothing is typed by hand.

Usage:  python topg_make_r2_tables.py
Output: exp_result/topg_tables/topg_r2_tables.md
"""
from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
A9 = HERE.parent / "exp_result" / "topg_a9_analysis"
MA = HERE.parent / "exp_result" / "topg_marlfs_analysis"
SR = HERE.parent / "exp_result" / "topg_stability_redundancy"
OUT = HERE.parent / "exp_result" / "topg_tables" / "topg_r2_tables.md"

NAME = {"TOPG": "TOP-Gain", "ElasticNet": "Elastic Net"}
MAIN = ["TOPG", "Powershap", "VSURF", "STG", "Sequential Attention", "RF-RFE", "HSIC-Lasso", "varSelRF",
        "RF impurity", "RF permutation"]
NEW = {"Powershap", "STG", "Sequential Attention"}


def nm(m):
    s = NAME.get(m, m)
    return s + " *" if m in NEW else s


def ranks(suite, metric):
    f = pd.read_csv(A9 / f"topg_a9_{suite}_friedman.csv")
    row = f[f.metric == metric].iloc[0]
    out = {}
    for part in row["ranks"].split(";"):
        name, r = part.strip().rsplit(" ", 1)
        out[name] = float(r)
    return out, row


def times(suite):
    d = pd.concat([pd.read_csv(f) for f in glob.glob(str(HERE.parent / "exp_result" / f"topg_{suite}" / "topg_*_s*_*.csv"))])
    d = d.drop_duplicates(["seed", "dataset", "method"])
    return d.groupby(["method", "dataset"]).mean_selection_time_sec.mean().groupby("method").median()


def fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}" if p < 0.01 else f"{p:.2f}"


def fmt_t(t):
    return f"{t:.3f}" if t < 0.01 else f"{t:.2f}" if t < 1 else f"{t:.1f}"


def baseline_table(suite, methods):
    desc = pd.read_csv(A9 / f"topg_a9_{suite}_descriptive.csv").set_index("method")
    base = pd.read_csv(A9 / f"topg_a9_{suite}_baselines.csv")
    auc = base[base.metric == "ROC-AUC"].set_index("method")
    stab = base[base.metric == "Stability"].set_index("method")
    r_auc, row = ranks(suite, "ROC-AUC")
    r_phi, _ = ranks(suite, "Stability")
    t = times(suite)
    lines = ["| Method | Rank AUC ↓ | ROC-AUC | Accuracy | Macro-F1 | Rank $\\Phi$ ↓ | $\\Phi$ | ΔAUC [95% CI] | W/L | $p_{\\text{Holm}}$ | Time/fold (s) |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|"]
    order = sorted([m for m in methods if m in r_auc], key=lambda m: r_auc[m])
    for m in order:
        d = desc.loc[m]
        if m == "TOPG":
            cells = ["–", "–", "–"]
            name = "**TOP-Gain**"
        else:
            a = auc.loc[m]
            cells = [f"{a.mean_diff:+.3f} [{a.ci_low:.3f}, {a.ci_high:.3f}]", f"{int(a.win)}/{int(a.loss)}", fmt_p(a.p_holm)]
            name = nm(m)
        lines.append(f"| {name} | {r_auc[m]:.2f} | {d.roc_auc:.4f} | {d.accuracy:.4f} | {d.macro_f1:.4f} | "
                     f"{r_phi[m]:.2f} | {d.nogueira_stability:.3f} | " + " | ".join(cells) + f" | {fmt_t(t.get(m, np.nan))} |")
    return "\n".join(lines), row, len(r_auc)


def sig_counts(suite):
    base = pd.read_csv(A9 / f"topg_a9_{suite}_baselines.csv")
    out = {}
    for met, g in base.groupby("metric"):
        out[met] = (int((g.p_holm < 0.05).sum()), g.method.nunique(),
                    ", ".join(NAME.get(m, m) for m in g[g.p_holm >= 0.05].method))
    return out


def marlfs_tables():
    rows = []
    for suite, lab in [("cc18", "CC18 (26)"), ("amlb", "AMLB (24)")]:
        s = pd.read_csv(MA / f"topg_marlfs_{suite}_summary.csv").set_index("metric")
        per = pd.read_csv(MA / f"topg_marlfs_{suite}_per_dataset.csv")
        for met, mlab in [("roc_auc", "ROC-AUC"), ("accuracy", "Accuracy"), ("macro_f1", "Macro-F1"), ("phi", "$\\Phi$")]:
            r = s.loc[met]
            rows.append(f"| {lab} | {mlab} | {r.topg_mean:.3f} | {r.marlfs_mean:.3f} | {r.diff_mean:+.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}] | "
                        f"{int(r.wins)}/{int(r.losses)} | {fmt_p(r.wilcoxon_p)} |")
        rows.append(f"| {lab} | panel size $k$ (median; folds with $k > 20$) | ≤ 20 | {per.marlfs_k_mean.median():.0f}; "
                    f"{int(per.marlfs_k_over20_folds.sum())}/{5 * len(per)} | | | |")
        rows.append(f"| {lab} | selection time per fold (s, mean) | {per.topg_time_mean.mean():.1f} | {per.marlfs_time_mean.mean():.0f} | | | |")
    head = ["| Suite | Outcome | TOP-Gain (size-matched) | MARLFS | Δ [95% CI] | W/L | $p$ (Wilcoxon) |",
            "|---|---|---:|---:|---|---:|---:|"]
    main = "\n".join(head + rows)
    per_tabs = []
    for suite in ["cc18", "amlb"]:
        per = pd.read_csv(MA / f"topg_marlfs_{suite}_per_dataset.csv").sort_values("P")
        t = ["| Dataset | $P$ | MARLFS $k$ (mean) | ROC-AUC MARLFS | ROC-AUC TOP-Gain | Δ | $\\Phi$ MARLFS | $\\Phi$ TOP-Gain |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for _, r in per.iterrows():
            t.append(f"| {r.dataset} | {int(r.P):,} | {r.marlfs_k_mean:.1f} | {r.marlfs_roc_auc:.3f} | {r.topg_roc_auc:.3f} | "
                     f"{r.topg_roc_auc - r.marlfs_roc_auc:+.3f} | {r.marlfs_phi:.3f} | {r.topg_phi:.3f} |")
        per_tabs.append("\n".join(t))
    return main, per_tabs


def stabred_table():
    r = pd.read_csv(SR / "topg_stabred_measures.csv")
    lab = {"TOPG": "TOP-Gain", "Net probability gain": "Net probability gain", "Powershap": "Powershap", "RF impurity": "RF impurity"}
    out = ["| Method | $K$ | Nogueira $\\Phi$ | Sechidis ($\\theta = 0.9$) | Sechidis ($\\theta = 0.7$) | Gain from correlation adjustment ($\\theta = 0.7$) |",
           "|---|---|---:|---:|---:|---:|"]
    from scipy.stats import wilcoxon
    tests = {}
    for klab, sub in [("10", r[r.K == 10]), ("max (≤ 20)", r.loc[r.groupby(["dataset", "method"]).K.idxmax()])]:
        sub = sub.assign(gain=sub.sechidis70 - sub.nogueira)
        m = sub.groupby("method")[["nogueira", "sechidis90", "sechidis70", "gain"]].mean()
        for meth in ["TOPG", "Powershap", "RF impurity", "Net probability gain"]:
            x = m.loc[meth]
            out.append(f"| {lab[meth]} | {klab} | {x.nogueira:.3f} | {x.sechidis90:.3f} | {x.sechidis70:.3f} | {x.gain:+.3f} |")
        g = sub.pivot(index="dataset", columns="method", values="gain")
        s = sub.pivot(index="dataset", columns="method", values="sechidis70").dropna()
        tests[klab] = dict(gain_vs_powershap=wilcoxon(g.TOPG, g.Powershap).pvalue, gain_vs_rf=wilcoxon(g.TOPG, g["RF impurity"]).pvalue,
                           s70_lower=int((s.TOPG < s.Powershap).sum()), n=len(s), s70_p=wilcoxon(s.TOPG, s.Powershap).pvalue)
    return "\n".join(out), tests


def main():
    parts = ["# Tables for revision R2 (generated by topg_make_r2_tables.py)\n"]
    for suite, lab in [("cc18", "CC18"), ("amlb", "AMLB")]:
        r_auc, _ = ranks(suite, "ROC-AUC")
        main_m = [m for m in MAIN if m in r_auc]
        tab, row, n = baseline_table(suite, main_m)
        rest = [m for m in r_auc if m not in main_m]
        parts += [f"## {lab}: main-text table ({len(main_m)} of {n} methods; CD = {row.cd:.2f})\n", tab, ""]
        tab_all, _, _ = baseline_table(suite, list(r_auc))
        parts += [f"## {lab}: full table, all {n} methods (Online Resource)\n", tab_all, ""]
        parts += [f"Not in main table: {', '.join(NAME.get(m, m) for m in rest)}\n"]
        sc = sig_counts(suite)
        parts += [f"Significant (p_Holm < 0.05) per metric: " + "; ".join(f"{k}: {v[0]}/{v[1]} (n.s.: {v[2]})" for k, v in sc.items()) + "\n"]
        for met in ["Accuracy", "Macro-F1"]:
            rr, _ = ranks(suite, met)
            parts.append(f"{met} ranks: " + "; ".join(f"{NAME.get(k, k)} {v:.2f}" for k, v in sorted(rr.items(), key=lambda x: x[1])[:5]) + "\n")
    main_t, per_t = marlfs_tables()
    parts += ["## A10: MARLFS vs size-matched TOP-Gain\n", main_t, "", "### Per dataset, CC18 (Online Resource)\n", per_t[0], "",
              "### Per dataset, AMLB (Online Resource)\n", per_t[1], ""]
    st, tests = stabred_table()
    parts += ["## Correlation-aware stability (CC18, repeat 1)\n", st, "", f"Tests: {tests}\n"]
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
