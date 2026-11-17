#!/usr/bin/env python3
"""
topg_stability_redundancy.py
============================
Does TOP-Gain's low Nogueira stability come from swapping between correlated (redundant) features?

For the 26 CC18 datasets, repeat 1 (seed 1) and the benchmark's five folds, the ranked lists of four selectors are
stored: TOP-Gain, the net-gain criterion (same wrapper without truncation), Powershap and RF impurity (the most
stable baseline). At K = 10 and at K = max(K) (<= 20) we compute:

  - Nogueira's Phi (exact identity of features);
  - correlation-adjusted stability from the R package stabm 1.2.2: Sechidis et al. (ECML PKDD 2019) and Yu et al.
    (with correction for chance), using the absolute Spearman correlation matrix of the dataset as similarity
    (thresholds 0.9 = package default, and 0.7);
  - the swap analysis: over all pairs of folds, the share of features selected in one fold but not the other that
    have a correlated partner (|rho| >= 0.7) among the other fold's selection.

Usage (from the exp_code folder):  python topg_stability_redundancy.py
Outputs: exp_result/topg_stability_redundancy/topg_stabred_lists.csv, topg_stabred_measures.csv
"""
from __future__ import annotations

import itertools
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import topg_baselines as tb  # noqa: E402
import topg_common as tc  # noqa: E402
import topg_run_cc18 as rc  # noqa: E402
import topg_new_baselines as nb  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_stability_redundancy"
SEED = 1
METHODS = ["TOPG", "Net probability gain", "Powershap", "RF impurity"]
RSCRIPT = os.environ.get("TOPG_RSCRIPT", r"C:\Program Files\R\R-4.6.1\bin\Rscript.exe")
R_LIBS = os.environ.get("TOPG_R_LIBS_USER", str(Path.home() / "R" / "topg-lib"))
R_CODE = r"""
args <- commandArgs(trailingOnly = TRUE)
.libPaths(c(Sys.getenv("R_LIBS_USER"), .libPaths()))
suppressPackageStartupMessages(library(stabm))
feats <- jsonlite::fromJSON(args[1], simplifyVector = FALSE)
sim <- as.matrix(read.csv(args[2], header = FALSE))
p <- ncol(sim)
res <- c()
for (nm in names(feats)) {
  f <- lapply(feats[[nm]], function(v) as.integer(unlist(v)) + 1L)
  set.seed(1)
  res <- rbind(res, data.frame(key = nm,
    nogueira = stabilityNogueira(f, p = p),
    sechidis90 = stabilitySechidis(f, sim.mat = sim, threshold = 0.9),
    sechidis70 = stabilitySechidis(f, sim.mat = sim, threshold = 0.7),
    yu90 = stabilityYu(f, sim.mat = sim, threshold = 0.9, N = 2000),
    yu70 = stabilityYu(f, sim.mat = sim, threshold = 0.7, N = 2000)))
}
write.csv(res, args[3], row.names = FALSE)
"""


def abs_spearman(X: pd.DataFrame) -> np.ndarray:
    R = np.apply_along_axis(rankdata, 0, X.values.astype(float))
    R = (R - R.mean(0)) / np.where(R.std(0) > 0, R.std(0), 1.0)
    C = np.abs(R.T @ R / R.shape[0])
    np.fill_diagonal(C, 1.0)
    return np.clip(np.nan_to_num(C), 0.0, 1.0)


def swap_share(lists, C, thr=0.7):
    hits, total = 0, 0
    for a, b in itertools.permutations(range(len(lists)), 2):
        sb = list(lists[b])
        for f in set(lists[a]) - set(lists[b]):
            total += 1
            if sb and C[f, sb].max() >= thr:
                hits += 1
    return hits / total if total else np.nan, total


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    lists_csv = OUT / "topg_stabred_lists.csv"
    done = pd.read_csv(lists_csv) if lists_csv.exists() else pd.DataFrame(columns=["dataset", "method", "fold"])
    scr = pd.read_csv(rc.SCREEN)
    el = scr[scr["eligible"]].sort_values("P_used")
    tb.RANDOM_STATE = SEED
    for _, r in el.iterrows():
        name, did = r["name"], int(r["did"])
        X, y = tc.load_bio_dataset(name, did, impute_numeric=False)
        max_k = max(rc.k_targets(X.shape[1]).values())
        rows = []
        for fold, (tr, te) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(X, y), start=1):
            X_tr, _ = rc.impute_fold(X.iloc[tr], X.iloc[te])
            for m in METHODS:
                if ((done["dataset"] == name) & (done["method"] == m) & (done["fold"] == fold)).any():
                    continue
                try:
                    if m == "Powershap":
                        sel = nb.select_powershap(X_tr, y.iloc[tr], max_k)
                    else:
                        sel = rc.select(m, X_tr, y.iloc[tr], max_k, SEED)
                except Exception as e:  # logged, not replaced
                    print(f"  !! {m} failed on {name} fold {fold}: {e}", flush=True)
                    continue
                rows.append(dict(dataset=name, method=m, fold=fold, max_k=max_k, P=X.shape[1],
                                 selected=json.dumps([str(c) for c in sel])))
        if rows:
            done = pd.concat([done, pd.DataFrame(rows)], ignore_index=True)
            done.to_csv(lists_csv, index=False)
        print(f"{name}: lists stored", flush=True)

    # ---- measures
    out = []
    for name, g in done.groupby("dataset"):
        did = int(el.loc[el["name"] == name, "did"].iloc[0])
        X, y = tc.load_bio_dataset(name, did, impute_numeric=True)
        cols = [str(c) for c in X.columns]
        pos = {c: i for i, c in enumerate(cols)}
        C = abs_spearman(X)
        max_k = int(g["max_k"].iloc[0])
        feats, meta = {}, {}
        for m, gm in g.groupby("method"):
            if gm["fold"].nunique() < 5:
                continue
            ranked = [[pos[c] for c in json.loads(s)] for s in gm.sort_values("fold")["selected"]]
            for k in sorted({min(10, max_k), max_k}):
                key = f"{m}|{k}"
                feats[key] = [lst[:k] for lst in ranked]
                sh, nsw = swap_share(feats[key], C)
                meta[key] = dict(dataset=name, method=m, K=k, swap_corr_share=sh, n_swaps=nsw)
        with tempfile.TemporaryDirectory() as tmp:
            fj, fs, fo, fr = (Path(tmp) / n for n in ["f.json", "sim.csv", "out.csv", "m.R"])
            fj.write_text(json.dumps(feats)); np.savetxt(fs, C, delimiter=","); fr.write_text(R_CODE)
            res = subprocess.run([RSCRIPT, str(fr), str(fj), str(fs), str(fo)], capture_output=True, text=True,
                                 env={**os.environ, "R_LIBS_USER": R_LIBS}, encoding="utf-8", errors="replace")
            if res.returncode != 0:
                print(f"  !! stabm failed on {name}: {res.stderr[-400:]}", flush=True)
                continue
            st = pd.read_csv(fo)
        for _, s in st.iterrows():
            out.append({**meta[s["key"]], **{c: s[c] for c in st.columns if c != "key"}})
        print(f"{name}: measures done", flush=True)
    res = pd.DataFrame(out)
    res.to_csv(OUT / "topg_stabred_measures.csv", index=False)
    pd.set_option("display.width", 200)
    for k_lab, sub in [("K=10", res[res.K == 10]), ("K=max", res.loc[res.groupby(["dataset", "method"]).K.idxmax()])]:
        print(f"\n== {k_lab}: mean over datasets ==")
        print(sub.groupby("method")[["nogueira", "sechidis90", "sechidis70", "yu90", "yu70", "swap_corr_share"]].mean().round(3).to_string())


if __name__ == "__main__":
    main()
