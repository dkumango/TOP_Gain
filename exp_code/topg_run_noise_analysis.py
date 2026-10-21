#!/usr/bin/env python3
"""
topg_run_noise_analysis.py
==========================

Tests the hypothesis of Section 6 of the manuscript: truncating sample-level decreases makes the
candidate ranking of a forward wrapper less sensitive to estimation noise, which is why TOPG selects
more stable feature sets than net / set-level criteria.

Design
------
For each dataset we take the training part of CV fold 1 (StratifiedKFold(5, seed 42), as in the
benchmark) and the forward order chosen there by TOPG (M6 configuration, 40 trees). For fixed subsets
S = first s features of that order (s in SIZES), every remaining candidate f of the candidate pool is
scored R times under two kinds of perturbation:

  seed : same data, candidate forests use seed r and the reference forest (p_base) seed r + 1000
  data : stratified 80 % subsample r of the training data, fixed seeds (42 / 1042)

From each candidate forest we compute all criteria at once:
  TOPG      mean max(0, p_i(S+f) - p_i(S))          (positive part)
  NEG       mean max(0, p_i(S) - p_i(S+f))          (negative part, net = TOPG - NEG; diagnostic only)
  net       mean p_i(S+f)                           (ranking-equivalent to TOPG - NEG)
  brier     -mean ||p(S+f) - onehot||^2
  logloss   mean log max(p_i(S+f), 1e-3)
  acc       OOB accuracy;  bacc  OOB balanced accuracy

Reliability of a criterion over the R replicates (for each dataset x s x perturbation):
  spearman  mean pairwise Spearman correlation of the candidate score vectors
  top1      probability that two replicates pick the same best candidate
  top5      mean pairwise Jaccard index of the five best candidates
  icc       ICC(1): between-candidate variance of the replicate means divided by the total variance

Usage (from the topg folder):  python topg_run_noise_analysis.py [--replicates 10] [--datasets ...]

Outputs (exp_result/topg_noise_analysis/)
-------------------------------------------
  topg_noise_scores_raw.csv      one row per (dataset, size, perturbation, replicate, candidate)
  topg_noise_reliability.csv     reliability per (dataset, size, perturbation, criterion)
  topg_noise_summary.csv         means over datasets x sizes, and Wilcoxon tests of TOPG vs each criterion
"""
from __future__ import annotations

import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr, wilcoxon
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import topg_common as tc  # noqa: E402
from topg_selector import SRAFSAblation  # noqa: E402

OUT = HERE.parent / "exp_result" / "topg_noise_analysis"
TREES, DEPTH, SEED = 40, 5, 42
SIZES = [1, 3, 6, 10]
EPS = 1e-3
CRITERIA = ["TOPG", "net", "brier", "logloss", "acc", "bacc", "NEG"]
DATASETS = {"synthetic": None, "qsar-biodeg": 1494, "MiceProtein": 40966, "splice": 46,
            "Bioresponse": 4134, "OVA_Breast": 1128}
M6_FLAGS = dict(enable_g4_g3=False, enable_class_weight=False, enable_smoothing=False,
                pair_search=False, min_g4=0.0, objective_mode="g1_only")


def oob_proba(X, y, cols, seed):
    """OOB class probabilities of the TOPG surrogate forest (raw, NaN -> uniform)."""
    classes = np.unique(y)
    if not cols:
        return np.full((len(y), len(classes)), 1.0 / len(classes)), classes
    est = RandomForestClassifier(n_estimators=TREES, max_depth=DEPTH, min_samples_leaf=2, class_weight=None,
                                 oob_score=True, random_state=seed, n_jobs=1)
    est.fit(X[sorted(cols)], y)
    return np.nan_to_num(est.oob_decision_function_, nan=1.0 / len(est.classes_)), est.classes_


def score_candidate(X, y, S, f, seed, p_base):
    proba, classes = oob_proba(X, y, S + [f], seed)
    onehot = (y[:, None] == classes[None, :]).astype(float)
    p_true = (proba * onehot).sum(axis=1)
    delta = p_true - p_base
    pred = classes[np.argmax(proba, axis=1)]
    return dict(candidate=f,
                TOPG=float(np.maximum(0.0, delta).mean()),
                NEG=float(np.maximum(0.0, -delta).mean()),
                net=float(p_true.mean()),
                brier=float(-((proba - onehot) ** 2).sum(axis=1).mean()),
                logloss=float(np.log(np.clip(p_true, EPS, 1.0)).mean()),
                acc=float((pred == y).mean()),
                bacc=float(balanced_accuracy_score(y, pred)))


def p_true_of(X, y, cols, seed):
    proba, classes = oob_proba(X, y, cols, seed)
    onehot = (y[:, None] == classes[None, :]).astype(float)
    return (proba * onehot).sum(axis=1)


def run_dataset(name, did, R):
    X, y = tc.make_synthetic() if did is None else tc.load_bio_dataset(name, did)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=tc.RANDOM_STATE)
    tr, _ = next(skf.split(X, y))
    Xtr, ytr = X.iloc[tr].reset_index(drop=True), np.asarray(y.iloc[tr])
    sizes = [s for s in SIZES if s < X.shape[1]]
    sel = SRAFSAblation(max_features=max(sizes), surrogate_estimators=TREES, random_state=SEED,
                        use_oob=True, n_jobs=-1, **M6_FLAGS)
    sel.fit(Xtr, ytr)
    order = list(sel.selected_features_)
    pool = sel._get_active_candidate_pool(Xtr, ytr, max(sizes))
    print(f"\n=== {name}: N_train={len(ytr)}, P={X.shape[1]}, pool={len(pool)}, TOPG order={order[:max(sizes)]}")
    rows = []
    for s in sizes:
        S = order[:s]
        cands = [f for f in pool if f not in S]
        for pert in ["seed", "data"]:
            t0 = time.time()
            for r in range(R):
                if pert == "seed":
                    Xr, yr, cs, bs = Xtr, ytr, r, r + 1000
                else:
                    idx, _ = train_test_split(np.arange(len(ytr)), train_size=0.8, stratify=ytr, random_state=r)
                    Xr, yr, cs, bs = Xtr.iloc[idx].reset_index(drop=True), ytr[idx], SEED, SEED + 1000
                p_base = p_true_of(Xr, yr, S, bs)
                res = Parallel(n_jobs=-1)(delayed(score_candidate)(Xr, yr, S, f, cs, p_base) for f in cands)
                for d in res:
                    rows.append(dict(dataset=name, size=s, perturbation=pert, replicate=r, **d))
            print(f"  |S|={s:>2} {pert:<4} {len(cands)} candidates x {R} replicates  ({time.time() - t0:5.1f}s)")
    return pd.DataFrame(rows)


def reliability(raw: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (ds, s, pert), g in raw.groupby(["dataset", "size", "perturbation"]):
        for c in CRITERIA:
            M = g.pivot(index="candidate", columns="replicate", values=c)  # candidates x replicates
            names = M.index.to_numpy()
            reps = list(M.columns)
            rho, top1, top5 = [], [], []
            for a, b in combinations(reps, 2):
                va, vb = M[a].to_numpy(), M[b].to_numpy()
                rho.append(spearmanr(va, vb).correlation if np.std(va) > 0 and np.std(vb) > 0 else np.nan)
                # best candidate with the wrapper's tie-break (highest score, then name)
                ba = max(zip(va, names))[1]
                bb = max(zip(vb, names))[1]
                top1.append(float(ba == bb))
                ta = set(names[np.lexsort((names, -va))[:5]])
                tb = set(names[np.lexsort((names, -vb))[:5]])
                top5.append(len(ta & tb) / len(ta | tb))
            means = M.mean(axis=1)
            within = M.var(axis=1, ddof=1).mean()
            between = means.var(ddof=1)
            icc = between / (between + within) if (between + within) > 0 else np.nan
            out.append(dict(dataset=ds, size=s, perturbation=pert, criterion=c, spearman=np.nanmean(rho),
                            top1=np.mean(top1), top5=np.mean(top5), icc=icc))
    return pd.DataFrame(out)


def summarize(rel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for pert, g in rel.groupby("perturbation"):
        for m in ["spearman", "top1", "top5", "icc"]:
            piv = g.pivot_table(index=["dataset", "size"], columns="criterion", values=m)
            for c in CRITERIA:
                row = dict(perturbation=pert, metric=m, criterion=c, mean=piv[c].mean(), n_units=len(piv))
                if c not in ("TOPG", "NEG"):
                    d = piv["TOPG"] - piv[c]
                    row["TOPG_win/tie/loss"] = f"{(d > 1e-9).sum()}/{(d.abs() <= 1e-9).sum()}/{(d < -1e-9).sum()}"
                    row["wilcoxon_p"] = float(wilcoxon(piv["TOPG"], piv[c]).pvalue) if (d.abs() > 1e-12).any() else 1.0
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--replicates", type=int, default=10)
    ap.add_argument("--datasets", nargs="+", default=list(DATASETS), choices=list(DATASETS))
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    raw_csv = OUT / "topg_noise_scores_raw.csv"
    if not args.summarize_only:
        done = set(pd.read_csv(raw_csv, usecols=["dataset"])["dataset"]) if raw_csv.exists() else set()
        for name in args.datasets:
            if name in done:
                print(f"[skip] {name}")
                continue
            t0 = time.time()
            df = run_dataset(name, DATASETS[name], args.replicates)
            df.to_csv(raw_csv, mode="a", header=not raw_csv.exists(), index=False)
            print(f"[{name}] {(time.time() - t0) / 60:.1f} min")
    raw = pd.read_csv(raw_csv)
    rel = reliability(raw)
    rel.to_csv(OUT / "topg_noise_reliability.csv", index=False)
    summ = summarize(rel)
    summ.to_csv(OUT / "topg_noise_summary.csv", index=False)
    pd.set_option("display.width", 200)
    for pert in ["seed", "data"]:
        print(f"\n=== Reliability under {pert} perturbation (mean over datasets x sizes) ===")
        t = summ[summ.perturbation == pert].pivot(index="criterion", columns="metric", values="mean")
        print(t.loc[CRITERIA, ["spearman", "top1", "top5", "icc"]].round(3).to_string())
        print(summ[(summ.perturbation == pert) & summ["wilcoxon_p"].notna()]
              [["metric", "criterion", "TOPG_win/tie/loss", "wilcoxon_p"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
