#!/usr/bin/env python3
"""
topg_make_revision_figures.py
=============================
Figures 2-4 and Table 6 of the revised manuscript.

  Figure 2  toy simulation of Section 3.3 (net gain vs TOPG, probability of picking the better candidate)
  Figure 3  accuracy-stability-cost trade-off on CC18 (Friedman ranks; marker area ~ single-core time)
  Figure 4  relevance density M-rel vs the stability gain dPhi (CC18 exploratory, AMLB confirmatory H4b)
  Table 6   runtime study (topg_runtime_benchmark.py): one core and ten cores

Usage (from the topg folder):  python topg_make_revision_figures.py
Outputs: exp_result/figures/topg_fig2_toy_simulation.png, topg_fig3_pareto.png, topg_fig4_h4b_mrel_dphi.png,
         exp_result/topg_tables/topg_rev_toy_simulation.csv, topg_rev_runtime.csv
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "exp_result"
TAB = RES / "topg_tables"
MAN = HERE.parent / "exp_result" / "figures"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"   # categorical slots 1-3 (validated all-pairs)
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
NAME = "TOP-Gain"  # display name of the method (final name, 2026-10-20)
plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.dpi": 300, "savefig.bbox": "tight"})


def toy_simulation(n=300, frac=0.2, gain_a=0.10, gain_b=0.07, reps=10000, seed=2026):
    rng = np.random.default_rng(seed)
    n_aff = int(frac * n)
    da = np.zeros(n); da[:n_aff] = gain_a
    db = np.zeros(n); db[n_aff:2 * n_aff] = gain_b
    rows = []
    for sigma in [0.01, 0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15]:
        ea = da + rng.normal(0, sigma, size=(reps, n))
        eb = db + rng.normal(0, sigma, size=(reps, n))
        net = (ea.mean(1) > eb.mean(1)).mean()
        top = (np.maximum(ea, 0).mean(1) > np.maximum(eb, 0).mean(1)).mean()
        rows.append(dict(sigma=sigma, p_correct_net=net, p_correct_topg=top))
    t = pd.DataFrame(rows)
    t.to_csv(TAB / "topg_rev_toy_simulation.csv", index=False)
    print(t.round(4).to_string(index=False))
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
    ax.plot(t.sigma, t.p_correct_net, color=ORANGE, lw=2, marker="s", ms=5, label="Net gain")
    ax.plot(t.sigma, t.p_correct_topg, color=BLUE, lw=2, marker="o", ms=5, label=NAME)
    ax.set_xlabel("OOB noise level σ"); ax.set_ylabel("P(choose the better candidate)")
    ax.set_ylim(0.5, 1.01)
    ax.legend(frameon=False, loc="lower left")
    fig.savefig(MAN / "topg_fig2_toy_simulation.png"); plt.close(fig)


def runtime_table():
    rows = {}
    for c in (1, 10):
        f = RES / "topg_runtime" / f"topg_runtime_c{c}.csv"
        if f.exists():
            rows[c] = pd.read_csv(f)
    if 1 not in rows:
        print("runtime study not available yet"); return None
    out = []
    for m, g in rows[1].groupby("method"):
        r = dict(method=m, n_datasets=len(g), total_c1=g.wall_sec.sum(), median_c1=g.wall_sec.median(),
                 max_c1=g.wall_sec.max(), r_fit_total_c1=g.r_fit_sec.sum() if g.r_fit_sec.notna().any() else np.nan)
        if 10 in rows:
            h = rows[10][rows[10].method == m]
            r.update(total_c10=h.wall_sec.sum(), median_c10=h.wall_sec.median())
            mg = g.merge(h, on="dataset", suffixes=("_1", "_10"))
            r["identical_selections"] = int((mg.selected_1 == mg.selected_10).sum())
            r["n_compared"] = len(mg)
        out.append(r)
    t = pd.DataFrame(out).set_index("method")
    t["ratio_c1_vs_TOPG"] = t.total_c1 / t.loc["TOPG", "total_c1"]
    if "total_c10" in t:
        t["ratio_c10_vs_TOPG"] = t.total_c10 / t.loc["TOPG", "total_c10"]
        t["speedup_10_vs_1"] = t.total_c1 / t.total_c10
    t = t.sort_values("total_c1")
    t.to_csv(TAB / "topg_rev_runtime.csv")
    print(t.round(2).to_string())
    big = rows[1][rows[1].dataset.isin(["Bioresponse", "Internet-Advertisements", "madelon"])]
    print(big.pivot_table(index="dataset", columns="method", values="wall_sec")[
        [m for m in ["TOPG", "VSURF", "varSelRF", "Boruta", "RF permutation"] if m in set(big.method)]].round(1))
    return t


def pareto(rt):
    fr = pd.read_csv(RES / "topg_cc18_analysis" / "topg_cc18_final_baselines_friedman.csv").set_index("metric")
    def ranks(met):
        return dict((x.rsplit(" ", 1)[0], float(x.rsplit(" ", 1)[1])) for x in fr.loc[met, "ranks"].split("; "))
    ra, rs = ranks("ROC-AUC"), ranks("Stability")
    groups = {"Filter / embedded": ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection"],
              "RF-based / SVM-RFE": ["Boruta", "SVM-RFE", "RF impurity", "RF permutation", "RF-RFE", "varSelRF", "VSURF"],
              NAME: ["TOPG"]}
    color = {"Filter / embedded": AQUA, "RF-based / SVM-RFE": ORANGE, NAME: BLUE}
    marker = {"Filter / embedded": "o", "RF-based / SVM-RFE": "s", NAME: "D"}
    if rt is not None:
        tt = rt["total_c1"]; size = lambda m: 25 + 500 * np.sqrt(tt[m] / tt.max())
        size_note = "marker area proportional to √(single-core selection time)"
    else:
        size = lambda m: 60; size_note = "single-core time pending"
    fig, ax = plt.subplots(figsize=(6.85, 4.6))
    ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
    for g, ms in groups.items():
        for m in ms:
            ax.scatter(ra[m], rs[m], s=size(m), color=color[g], marker=marker[g], edgecolor="white", lw=1.5, zorder=3,
                       label=g if m == ms[0] else None, alpha=0.9)
            off = {"varSelRF": (-8, 9), "RF impurity": (8, -12), "ANOVA": (-6, 8), "VSURF": (14, 6)}.get(m, (7, 4))
            ha = "right" if m in ("varSelRF", "ANOVA") else "left"
            ax.annotate(NAME if m == "TOPG" else m.replace("ElasticNet", "Elastic Net"), (ra[m], rs[m]), xytext=off,
                        textcoords="offset points", fontsize=7.5, color=INK, ha=ha)
    ax.set_xlabel("Accuracy: Friedman mean rank of ROC-AUC (lower = better)")
    ax.set_ylabel("Stability: mean rank of Φ (lower = better)")
    ax.set_xlim(2.5, 14.5); ax.set_ylim(1.5, 14)
    leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=7.5, title=size_note,
                    title_fontsize=7.5)
    for h in leg.legend_handles:
        h.set_sizes([40])
    fig.savefig(MAN / "topg_fig3_pareto.png"); plt.close(fig)


def h4b():
    cc = pd.read_csv(RES / "topg_cc18_analysis" / "topg_cc18_final_per_dataset.csv", index_col=0)
    am = pd.read_csv(RES / "topg_amlb_analysis" / "topg_amlb_per_dataset.csv", index_col=0)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, d, title, col, mk in [(axes[0], cc, "CC18 (26 datasets, exploratory)", ORANGE, "s"),
                                  (axes[1], am, "AMLB (24 datasets, confirmatory H4b)", BLUE, "o")]:
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.scatter(d.M_rel, d.dPhi, color=col, marker=mk, s=32, edgecolor="white", lw=1.2, zorder=3)
        rho = spearmanr(d.M_rel, d.dPhi).correlation
        ax.set_title(f"{title}\nSpearman ρ = {rho:.2f}", fontsize=8.5, color=INK)
        ax.set_xlabel("Relevance density M-rel")
    axes[0].set_ylabel(f"ΔΦ = Φ({NAME}) − Φ(net gain)")
    fig.savefig(MAN / "topg_fig4_h4b_mrel_dphi.png"); plt.close(fig)


def main():
    TAB.mkdir(parents=True, exist_ok=True)
    toy_simulation()
    rt = runtime_table()
    pareto(rt)
    h4b()


if __name__ == "__main__":
    main()
