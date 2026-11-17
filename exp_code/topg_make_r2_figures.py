#!/usr/bin/env python3
"""
topg_make_r2_figures.py
=======================
Figures changed in revision R2 (same KAIS format as topg_make_v1_figures.py):

  Fig. 3  critical-difference diagram, ROC-AUC on CC18, now with the 17 baselines of amendment A9
          (the three recent baselines are marked with an asterisk).
  Fig. 5  accuracy-stability-cost trade-off on CC18 with all 18 methods; marker area from the controlled
          single-core runtime study (exp_result/topg_runtime/topg_runtime_c1.csv, A9 baselines added 2026-11-07).

Writes exp_result/figures/topg_fig3_cd_cc18_rocauc_r2.png and the TIFF copy topg_kais_figures/topg_Fig3_r2.tif.
Usage (from the exp_code folder):  python topg_make_r2_figures.py
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402

import topg_make_v1_figures as v1  # noqa: E402  (style, colours, paths)

NEW = {"Powershap", "STG", "Sequential Attention"}


def fig_cd(suite="cc18"):
    fr = pd.read_csv(v1.RES / "topg_a9_analysis" / f"topg_a9_{suite}_friedman.csv").set_index("metric")
    r = pd.Series(dict((x.rsplit(" ", 1)[0], float(x.rsplit(" ", 1)[1])) for x in fr.loc["ROC-AUC", "ranks"].split("; ")))
    r = r.rename(index=lambda m: v1.NAME if m == "TOPG" else "Elastic Net" if m == "ElasticNet" else m + " *" if m in NEW else m)
    r = r.sort_values()
    cd = float(fr.loc["ROC-AUC", "cd"]); k = len(r)
    half = (k + 1) // 2
    fig, ax = plt.subplots(figsize=(174 * v1.MM, (40 + 5.2 * half) * v1.MM))
    ax.set_xlim(0.6, k + 0.4); ax.set_ylim(7.0 - half * 0.95 - 0.3, 9.2); ax.invert_xaxis(); ax.axis("off")
    ax.hlines(8, 1, k, color=v1.INK, lw=0.9)
    for t in range(1, k + 1):
        ax.vlines(t, 8, 8.18, color=v1.INK, lw=0.8); ax.text(t, 8.3, str(t), ha="center", va="bottom", fontsize=7)
    best = float(r.iloc[0])
    ax.hlines(8.95, best, best + cd, color=v1.INK, lw=1.6)
    for x in (best, best + cd):
        ax.vlines(x, 8.85, 9.05, color=v1.INK, lw=1.2)
    ax.text(best + cd / 2, 9.1, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    near = r[r - best < cd]
    ax.hlines(7.55, near.min() - 0.05, near.max() + 0.05, color=v1.BLUE, lw=2.2)
    for j, (m, v) in enumerate(r.items()):
        right = j < half
        y = 7.0 - (j if right else (k - 1 - j)) * 0.95
        xt = 0.6 if right else k + 0.4
        ax.plot([v, v, xt], [8, y, y], color=v1.INK2, lw=0.6)
        ax.plot(v, 8, "o", color=v1.BLUE if m == v1.NAME else v1.INK, ms=3)
        ax.text(xt + (0.08 if not right else -0.08), y, f"{m} ({v:.2f})", ha="right" if not right else "left",
                va="center", fontsize=7.2, fontweight="bold" if m == v1.NAME else "normal", color=v1.INK)
    return fig


def fig_pareto():
    fr = pd.read_csv(v1.RES / "topg_a9_analysis" / "topg_a9_cc18_friedman.csv").set_index("metric")
    def ranks(met):
        return dict((x.rsplit(" ", 1)[0], float(x.rsplit(" ", 1)[1])) for x in fr.loc[met, "ranks"].split("; "))
    ra, rs = ranks("ROC-AUC"), ranks("Stability")
    tt = pd.read_csv(v1.RES / "topg_runtime" / "topg_runtime_c1.csv").groupby("method").wall_sec.sum()
    groups = {"Filter / embedded": ["ANOVA", "mRMR", "JMI", "ReliefF", "ElasticNet", "HSIC-Lasso", "Stability Selection"],
              "RF-based / SVM-RFE": ["Boruta", "SVM-RFE", "RF impurity", "RF permutation", "RF-RFE", "varSelRF", "VSURF"],
              "Recent (neural / Shapley)": ["STG", "Sequential Attention", "Powershap"],
              v1.NAME: ["TOPG"]}
    color = {"Filter / embedded": v1.AQUA, "RF-based / SVM-RFE": v1.ORANGE, "Recent (neural / Shapley)": "#8a5cd1", v1.NAME: v1.BLUE}
    marker = {"Filter / embedded": "o", "RF-based / SVM-RFE": "s", "Recent (neural / Shapley)": "^", v1.NAME: "D"}
    size = lambda m: 25 + 500 * np.sqrt(tt[m] / tt.max())
    fig, ax = plt.subplots(figsize=(174 * v1.MM, 118 * v1.MM))
    ax.grid(color=v1.GRID, lw=0.6); ax.set_axisbelow(True)
    offs = {"varSelRF": (-8, 9), "RF impurity": (8, -12), "ANOVA": (-6, 8), "VSURF": (14, 6), "Powershap": (-12, -12),
            "STG": (-6, 9), "TOPG": (8, 4), "Stability Selection": (7, -10), "HSIC-Lasso": (7, 6)}
    for g, ms in groups.items():
        for m in ms:
            ax.scatter(ra[m], rs[m], s=size(m), color=color[g], marker=marker[g], edgecolor="white", lw=1.5, zorder=3,
                       label=g if m == ms[0] else None, alpha=0.9)
            ha = "right" if m in ("varSelRF", "ANOVA", "Powershap") else "left"
            ax.annotate(v1.NAME if m == "TOPG" else m.replace("ElasticNet", "Elastic Net"), (ra[m], rs[m]),
                        xytext=offs.get(m, (7, 4)), textcoords="offset points", fontsize=7.5, color=v1.INK, ha=ha,
                        fontweight="bold" if m == "TOPG" else "normal")
    ax.set_xlabel("Accuracy: Friedman mean rank of ROC-AUC (lower = better)")
    ax.set_ylabel("Stability: mean rank of Φ (lower = better)")
    ax.set_xlim(4, 17.5); ax.set_ylim(2, 16.5)
    leg = ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4, fontsize=7.5,
                    title="marker area proportional to √(single-core selection time)", title_fontsize=7.5)
    for h in leg.legend_handles:
        h.set_sizes([40])
    return fig


def main():
    png = v1.MAN / "topg_fig3_cd_cc18_rocauc_r2.png"
    fig = fig_cd(); fig.savefig(png); plt.close(fig)
    v1.TIF.mkdir(parents=True, exist_ok=True)
    Image.open(png).convert("RGB").save(v1.TIF / "topg_Fig3_r2.tif", dpi=(300, 300), compression="tiff_lzw")
    print("saved", png.name)
    png5 = v1.MAN / "topg_fig5_pareto_r2.png"
    fig = fig_pareto(); fig.savefig(png5); plt.close(fig)
    Image.open(png5).convert("RGB").save(v1.TIF / "topg_Fig5_r2.tif", dpi=(300, 300), compression="tiff_lzw")
    print("saved", png5.name)


if __name__ == "__main__":
    main()
