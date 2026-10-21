#!/usr/bin/env python3
"""
topg_make_v1_figures.py
=======================
Figures of manuscript v1.0 in the format of Knowledge and Information Systems (Springer): Arial, at most
174 mm wide, 300 dpi RGB, named in order of citation.

  Fig. 1  schematic of the OOB true-class probability and of the truncated gain (new)
  Fig. 2  toy simulation                (from topg_make_revision_figures.py)
  Fig. 3  critical-difference diagram, ROC-AUC on CC18 (redrawn with the final method name)
  Fig. 4  M-rel vs the stability gain   (from topg_make_revision_figures.py)
  Fig. 5  accuracy-stability-cost trade-off (from topg_make_revision_figures.py)

Writes PNG (for the docx) to exp_result/figures/ and TIFF copies to exp_result/figures/topg_kais_figures/.

Usage (from the topg folder):  python topg_make_v1_figures.py   (run topg_make_revision_figures.py first)
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

HERE = Path(__file__).resolve().parent
MAN = HERE.parent / "exp_result" / "figures"
TIF = MAN / "topg_kais_figures"
RES = HERE.parent / "exp_result"
NAME = "TOP-Gain"
BLUE, ORANGE, AQUA, INK, INK2, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"
MM = 1 / 25.4
plt.rcParams.update({"font.family": "Arial", "font.size": 8, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "savefig.dpi": 300, "savefig.bbox": "tight"})


def fig_oob():
    rng = np.random.default_rng(3)
    T, N, i_star = 8, 10, 3
    inbag = rng.random((T, N)) < 0.63
    inbag[[0, 2, 5], i_star] = False          # sample i* is out of bag for trees 1, 3, 6
    inbag[[1, 3, 4, 6, 7], i_star] = True
    fig = plt.figure(figsize=(174 * MM, 72 * MM))
    # (a) bootstrap membership
    ax = fig.add_axes([0.02, 0.12, 0.30, 0.74])
    for t in range(T):
        for n in range(N):
            ob = not inbag[t, n]
            ax.add_patch(Rectangle((n, T - 1 - t), 0.92, 0.85, facecolor="white" if ob else "#cfd9e6",
                                   edgecolor=ORANGE if (ob and n == i_star) else "#9aa3ad", lw=1.6 if (ob and n == i_star) else 0.5))
    ax.add_patch(Rectangle((i_star - 0.08, -0.1), 1.08, T, fill=False, edgecolor=ORANGE, lw=1.0, ls="--"))
    ax.set_xlim(-0.2, N + 0.2); ax.set_ylim(-0.6, T + 0.9); ax.axis("off")
    ax.text(N / 2, T + 0.45, "Bootstrap samples of a forest on subset S", ha="center", fontsize=8, color=INK)
    ax.text(i_star + 0.46, -0.5, "sample i", ha="center", va="top", fontsize=7.5, color=ORANGE)
    for t in range(T):
        ax.text(-0.35, T - 1 - t + 0.42, f"tree {t + 1}", ha="right", va="center", fontsize=6.5, color=INK2)
    ax.text(N / 2, -1.25, "shaded = in bag;  white = out of bag (OOB)", ha="center", fontsize=6.5, color=INK2)
    ax.text(-1.6, T + 0.45, "(a)", fontsize=8.5, color=INK)

    # (b) OOB class-probability vector of sample i
    ax = fig.add_axes([0.43, 0.18, 0.15, 0.58])
    p = np.array([0.20, 0.65, 0.15]); y_true = 1
    ax.bar(range(3), p, color=[INK2, BLUE, INK2], width=0.6)
    ax.set_xticks(range(3), ["class 1", "class 2\n(true)", "class 3"], fontsize=7)
    ax.set_ylim(0, 1); ax.set_yticks([0, 0.5, 1]); ax.spines[["top", "right"]].set_visible(False)
    ax.text(y_true, p[y_true] + 0.04, r"$p_i^{S}$", ha="center", fontsize=8.5, color=BLUE)
    ax.set_title("(b)  OOB class probabilities of\n      sample i (mean over OOB trees)", loc="left", fontsize=7.5, x=-0.32)
    fig.add_artist(FancyArrowPatch((0.335, 0.47), (0.372, 0.47), transform=fig.transFigure, arrowstyle="-|>",
                                   mutation_scale=10, color=ORANGE, lw=1.2))

    # (c) per-sample change and truncation
    ax = fig.add_axes([0.71, 0.18, 0.28, 0.58])
    d = np.array([0.18, -0.03, 0.02, 0.25, -0.05, 0.01, -0.02, 0.12, -0.04, 0.03, -0.01, 0.0])
    cols = [BLUE if v > 0 else ORANGE for v in d]
    ax.bar(range(len(d)), d, color=cols, width=0.7)
    ax.axhline(0, color=INK2, lw=0.8)
    ax.set_xticks([]); ax.set_xlabel("samples i", fontsize=7.5)
    ax.set_ylabel(r"$\Delta p_i = p_i^{S\cup\{f\}} - p_i^{S}$", fontsize=7.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title("(c)  " + NAME + " = mean of max(0, Δp)\n      net gain = mean of Δp", loc="left", fontsize=7.5, x=-0.22)
    ax.text(len(d) - 0.5, 0.22, "counted by both", color=BLUE, ha="right", fontsize=6.8)
    ax.text(len(d) - 0.5, -0.075, "counted only by net gain", color=ORANGE, ha="right", va="top", fontsize=6.8)
    ax.set_ylim(-0.1, 0.28)
    return fig


def fig_cd():
    fr = pd.read_csv(RES / "topg_cc18_analysis" / "topg_cc18_final_baselines_friedman.csv").set_index("metric")
    r = pd.Series(dict((x.rsplit(" ", 1)[0], float(x.rsplit(" ", 1)[1])) for x in fr.loc["ROC-AUC", "ranks"].split("; ")))
    r = r.rename(index={"TOPG": NAME, "ElasticNet": "Elastic Net"}).sort_values()
    cd = float(fr.loc["ROC-AUC", "cd"]); k = len(r)
    fig, ax = plt.subplots(figsize=(174 * MM, 78 * MM))
    ax.set_xlim(0.6, k + 0.4); ax.set_ylim(-0.5, 9.2); ax.invert_xaxis(); ax.axis("off")
    ax.hlines(8, 1, k, color=INK, lw=0.9)
    for t in range(1, k + 1):
        ax.vlines(t, 8, 8.18, color=INK, lw=0.8); ax.text(t, 8.3, str(t), ha="center", va="bottom", fontsize=7)
    best = float(r.iloc[0])
    ax.hlines(8.95, best, best + cd, color=INK, lw=1.6)
    for x in (best, best + cd):
        ax.vlines(x, 8.85, 9.05, color=INK, lw=1.2)
    ax.text(best + cd / 2, 9.1, f"CD = {cd:.2f}", ha="center", va="bottom", fontsize=7)
    # methods within CD of the best (not significantly different from it) get a bar below the axis
    near = r[r - best < cd]
    ax.hlines(7.55, near.min() - 0.05, near.max() + 0.05, color=BLUE, lw=2.2)
    half = (k + 1) // 2
    for j, (m, v) in enumerate(r.items()):
        right = j < half                     # best half labelled on the right (rank 1 side), the rest on the left
        y = 7.0 - (j if right else (k - 1 - j)) * 0.95
        xt = 0.6 if right else k + 0.4
        left = not right
        ax.plot([v, v, xt], [8, y, y], color=INK2, lw=0.6)
        ax.plot(v, 8, "o", color=BLUE if m == NAME else INK, ms=3)
        ax.text(xt + (0.08 if left else -0.08), y, f"{m} ({v:.2f})", ha="right" if left else "left", va="center",
                fontsize=7.2, fontweight="bold" if m == NAME else "normal", color=INK)
    return fig


def save(fig, n, stem):
    png = MAN / f"topg_fig{n}_{stem}.png"
    fig.savefig(png); plt.close(fig)
    to_tif(png, n)


def to_tif(png, n):
    TIF.mkdir(parents=True, exist_ok=True)
    im = Image.open(png).convert("RGB")
    im.save(TIF / f"topg_Fig{n}.tif", dpi=(300, 300), compression="tiff_lzw")


def main():
    save(fig_oob(), 1, "oob_schematic")
    save(fig_cd(), 3, "cd_cc18_rocauc")
    for n, stem in [(2, "toy_simulation"), (4, "h4b_mrel_dphi"), (5, "pareto")]:
        src = {2: "topg_fig2_toy_simulation.png", 4: "topg_fig4_h4b_mrel_dphi.png", 5: "topg_fig3_pareto.png"}[n]
        dst = MAN / f"topg_fig{n}_{stem}.png"
        if src != dst.name:
            Image.open(MAN / src).save(dst, dpi=(300, 300))
        to_tif(dst, n)
    print("saved:", sorted(p.name for p in TIF.glob("*.tif")))


if __name__ == "__main__":
    main()
