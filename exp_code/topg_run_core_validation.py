#!/usr/bin/env python3
"""
topg_run_core_validation.py
===========================
(TOPG project; formerly run_g1_core_validation.py. In the TOPG manuscript, M6 = TOPG.)

Purpose
-------
Decides whether G2-G4 (plus class weighting, smoothing and pair search) earn their place
in the regimes they were designed for, and how the truncated continuous gain G1 compares with
other selection criteria inside the same RF-OOB forward wrapper.

Selectors (all share the RF-OOB forward-selection framework and SRAFSAblation defaults:
50 surrogate trees, max_depth 5, the same candidate pool, seed 42 -- i.e. the same settings
used by the original exp_epistasis_imbalance.py and run_omics_imbalance_sweep_ova_lung.py):

  SRA-FS Full (M0) : full objective (G1-G4, class weights, smoothing, pair search)
  M6               : G1 only = mean max(0, Delta p_true)   (uniform weights, raw OOB, no pairs)
  M7 / M8          : OOB accuracy / OOB balanced accuracy of S U {f}   (discrete criteria)
  M9               : OOB mean true-class probability of S U {f}       (net continuous gain)
  M10 / M11        : OOB log-loss / OOB Brier score of S U {f}        (proper scoring rules)
  M12              : M6 with p_base refitted with the candidates' seed (robustness check)

Experiments
-----------
  epistasis : XOR + additive synthetic benchmark (generator in topg_synthetic.py), K in {2, 4, 6}
  imbalance : synthetic imbalance sweep (generator in topg_synthetic.py), rho in {1, 5, 10, 20}, K = 10
  ova_lung  : real omics imbalance sweep (loader in topg_ova_lung_data.py)
              (natural + rho in {1, 5, 10, 20}, 3 subsampling seeds, K = 10). The stored
              SRA-FS / baseline rows of that script are merged into the summary; only the new
              wrapper variants are computed here.

Usage (run from the topg folder; self-contained -- no file outside topg/ is used)
---------------------------------------
    python topg_run_core_validation.py                       # all three experiments
    python topg_run_core_validation.py --parts epistasis imbalance
    python topg_run_core_validation.py --parts ova_lung --seeds 1

Outputs (exp_result/topg_core_validation/)
--------------------------------------------
    topg_epistasis_summary.csv, topg_imbalance_summary.csv   per (setting, method) means + Nogueira stability
    topg_ova_lung_raw.csv                                    one row per (tier, seed, method, fold); resumable
    topg_ova_lung_summary.csv                                new variants + stored rows, mean/SD over folds x seeds
"""
from __future__ import annotations

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent          # .../source_code/topg
sys.path.insert(0, str(HERE))

import topg_ova_lung_data as ol  # noqa: E402
from topg_common import nogueira_stability  # noqa: E402
from topg_selector import SRAFSAblation  # noqa: E402
from topg_synthetic import generate_epistasis_dataset, generate_imbalanced_dataset  # noqa: E402
from topg_run_wrapper_baselines import OOBAccuracyWrapper  # noqa: E402

RANDOM_STATE = 42
OUT_DIR = HERE.parent / "exp_result" / "topg_core_validation"
REF_OVA_LUNG_RAW = HERE.parent / "exp_result" / "topg_reference" / "topg_ref_baselines_ova_lung_raw.csv"

OVA_LUNG_P = 10935  # genes in OVA_Lung (OpenML 1130; 10,936 columns incl. the target)

M6_FLAGS = dict(enable_g4_g3=False, enable_class_weight=False, enable_smoothing=False,
                pair_search=False, min_g4=0.0, objective_mode="g1_only")


def make_selector(key: str, k: int):
    """Returns a fresh selector with SRAFSAblation defaults (50 trees) for the given variant key."""
    if key == "M0":
        return SRAFSAblation(max_features=k, random_state=RANDOM_STATE)
    if key == "M6":
        return SRAFSAblation(max_features=k, random_state=RANDOM_STATE, **M6_FLAGS)
    if key == "M12":
        return SRAFSAblation(max_features=k, random_state=RANDOM_STATE, fixed_base_seed=True, **M6_FLAGS)
    crit = {"M7": "accuracy", "M8": "balanced_accuracy", "M9": "mean_ptrue",
            "M10": "neg_log_loss", "M11": "neg_brier"}[key]
    return OOBAccuracyWrapper(criterion=crit, max_features=k, random_state=RANDOM_STATE, **M6_FLAGS)


LABELS = {
    "M0": "SRA-FS Full (M0)",
    "M6": "RF-OOB G1 (M6)",
    "M7": "RF-OOB accuracy (M7)",
    "M8": "RF-OOB balanced acc. (M8)",
    "M9": "RF-OOB mean p_true (M9)",
    "M10": "RF-OOB log-loss (M10)",
    "M11": "RF-OOB Brier (M11)",
    "M12": "TOPG common-seed (M12)",
}
ALL_KEYS = list(LABELS)


def select(key, X_tr, y_tr, k):
    return list(make_selector(key, k).fit(X_tr, y_tr).selected_features_)[:k]


def stability(selections, columns):
    idx = {c: i for i, c in enumerate(columns)}
    Z = np.zeros((len(selections), len(columns)))
    for r, sel in enumerate(selections):
        for f in sel:
            Z[r, idx[f]] = 1.0
    return nogueira_stability(Z)


def _merge_save(new: pd.DataFrame, path: Path, keys: list) -> pd.DataFrame:
    """Replaces the rows of the methods just run and keeps the others (running a subset never drops rows)."""
    if path.exists():
        old = pd.read_csv(path)
        old = old[~old["method"].isin(set(new["method"]))]
        new = pd.concat([old, new], ignore_index=True).sort_values(keys, kind="stable")
    new.to_csv(path, index=False)
    return new


# ---------------------------------------------------------------------------
def run_epistasis(keys):
    X, y, epi, add = generate_epistasis_dataset()
    info = set(epi + add)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    rows = []
    print("\n=== Epistasis benchmark (XOR pair F_1, F_2; additive F_3-F_6) ===")
    for k in [2, 4, 6]:
        for key in keys:
            sels, pair, tp, aucs, t = [], [], [], [], []
            for tr, te in skf.split(X, y):
                X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
                t0 = time.time()
                sel = select(key, X_tr, y_tr, k)
                t.append(time.time() - t0)
                sels.append(sel)
                pair.append(int(epi[0] in sel and epi[1] in sel))
                tp.append(sum(f in info for f in sel))
                rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
                rf.fit(X_tr[sel], y_tr)
                aucs.append(roc_auc_score(y_te, rf.predict_proba(X_te[sel])[:, 1]))
            row = {"K": k, "method": LABELS[key], "pair_recovery_pct": 100 * np.mean(pair),
                   "informative_tp": np.mean(tp), "rf_roc_auc": np.mean(aucs),
                   "nogueira_stability": stability(sels, X.columns), "time_s": np.mean(t)}
            rows.append(row)
            print(f"  K={k} {LABELS[key]:<26} pair={row['pair_recovery_pct']:5.1f}%  TP={row['informative_tp']:.2f}/6  "
                  f"AUC={row['rf_roc_auc']:.4f}  Phi={row['nogueira_stability']:.3f}")
    return _merge_save(pd.DataFrame(rows), OUT_DIR / "topg_epistasis_summary.csv", ["K", "method"])


# ---------------------------------------------------------------------------
def _imbalance_metrics(X_tr, y_tr, X_te, y_te, sel, n_trees):
    rf = RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_tr[sel], y_tr)
    proba, pred = rf.predict_proba(X_te[sel])[:, 1], rf.predict(X_te[sel])
    return dict(minority_recall=recall_score(y_te, pred, pos_label=1, zero_division=0),
                balanced_accuracy=balanced_accuracy_score(y_te, pred),
                macro_f1=f1_score(y_te, pred, average="macro", zero_division=0),
                roc_auc=roc_auc_score(y_te, proba))


def run_imbalance(keys):
    rows = []
    print("\n=== Synthetic imbalance sweep (K = 10; F_1-F_10 informative) ===")
    for rho in [1.0, 5.0, 10.0, 20.0]:
        X, y = generate_imbalanced_dataset(imbalance_ratio=rho)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
        for key in keys:
            sels, mets, t = [], [], []
            for tr, te in skf.split(X, y):
                X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
                t0 = time.time()
                sel = select(key, X_tr, y_tr, 10)
                t.append(time.time() - t0)
                sels.append(sel)
                m = _imbalance_metrics(X_tr, y_tr, X_te, y_te, sel, 80)
                m["informative_tp"] = sum(int(f.split("_")[1]) <= 10 for f in sel)
                mets.append(m)
            row = {"rho": rho, "method": LABELS[key], **pd.DataFrame(mets).mean().to_dict(),
                   "nogueira_stability": stability(sels, X.columns), "time_s": np.mean(t)}
            rows.append(row)
            print(f"  rho={rho:>4.0f} {LABELS[key]:<26} recall={row['minority_recall']:.3f}  "
                  f"BalAcc={row['balanced_accuracy']:.3f}  F1={row['macro_f1']:.3f}  AUC={row['roc_auc']:.4f}  "
                  f"TP={row['informative_tp']:.1f}/10  Phi={row['nogueira_stability']:.3f}")
    return _merge_save(pd.DataFrame(rows), OUT_DIR / "topg_imbalance_summary.csv", ["rho", "method"])


# ---------------------------------------------------------------------------
def run_ova_lung(keys, n_seeds):
    raw_csv = OUT_DIR / "topg_ova_lung_raw.csv"
    done = set()
    if raw_csv.exists():
        d = pd.read_csv(raw_csv, dtype={"tier": str})
        done = set(zip(d.tier, d.seed, d.method, d.fold))
    X_full, y_full = ol.load_ova_lung()
    t_start = time.time()
    print("\n=== OVA_Lung imbalance sweep (K = 10) ===")
    for tier, rho in ol.TIERS:
        for seed in ([0] if rho is None else range(n_seeds)):
            X, y = ol.subsample(X_full, y_full, rho, RANDOM_STATE + seed)
            n_min = int(y.sum())
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
            for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
                X_tr, y_tr, X_te, y_te = X.iloc[tr], y.iloc[tr], X.iloc[te], y.iloc[te]
                for key in keys:
                    if (tier, seed, LABELS[key], fold) in done:
                        continue
                    t0 = time.time()
                    sel = select(key, X_tr, y_tr, ol.TARGET_K)
                    t_sel = time.time() - t0
                    row = dict(tier=tier, rho=(round(len(y) / n_min - 1, 2) if rho is None else rho), seed=seed,
                               n_total=len(y), n_minority=n_min, method=LABELS[key], fold=fold,
                               **_imbalance_metrics(X_tr, y_tr, X_te, y_te, sel, 100),
                               selection_time_s=round(t_sel, 2), selected=";".join(sel))
                    pd.DataFrame([row]).to_csv(raw_csv, mode="a", header=not raw_csv.exists(), index=False)
                    print(f"  rho={tier:<7} seed={seed} fold {fold} {LABELS[key]:<26} AUC={row['roc_auc']:.4f} "
                          f"BalAcc={row['balanced_accuracy']:.4f} ({t_sel:5.1f}s, elapsed {(time.time() - t_start) / 60:5.1f} min)")
    summarize_ova_lung()


def summarize_ova_lung():
    new = pd.read_csv(OUT_DIR / "topg_ova_lung_raw.csv", dtype={"tier": str})
    frames = [new]
    if REF_OVA_LUNG_RAW.exists():
        frames.insert(0, pd.read_csv(REF_OVA_LUNG_RAW, dtype={"tier": str}))
    d = pd.concat(frames, ignore_index=True)
    metrics = ["minority_recall", "balanced_accuracy", "macro_f1", "roc_auc", "selection_time_s"]
    g = d.groupby(["tier", "method"], sort=False)
    s = g[metrics].agg(["mean", "std"]).round(4)
    s.columns = [f"{a}_{b}" for a, b in s.columns]
    s = s.reset_index()
    # selection stability over the 5 folds of each (tier, seed), averaged over seeds
    stab = []
    for (tier, method, seed), grp in d.groupby(["tier", "method", "seed"]):
        sels = [str(x).split(";") for x in grp["selected"]]
        feats = sorted(set(f for sl in sels for f in sl))
        idx = {f: i for i, f in enumerate(feats)}
        Z = np.zeros((len(sels), max(len(feats), 1)))
        for r, sl in enumerate(sels):
            for f in sl:
                Z[r, idx[f]] = 1.0
        # Nogueira needs the full feature dimension; pad with the unselected genes (all-zero columns)
        Zp = np.hstack([Z, np.zeros((Z.shape[0], OVA_LUNG_P - Z.shape[1]))])
        stab.append(dict(tier=tier, method=method, seed=seed, phi=nogueira_stability(Zp)))
    st = pd.DataFrame(stab).groupby(["tier", "method"])["phi"].mean().round(4).rename("nogueira_stability")
    s = s.merge(st.reset_index(), on=["tier", "method"], how="left")
    order = {t: i for i, (t, _) in enumerate(ol.TIERS)}
    s = s.sort_values(by=["tier", "method"], key=lambda c: c.map(order) if c.name == "tier" else c)
    s.to_csv(OUT_DIR / "topg_ova_lung_summary.csv", index=False)
    pd.set_option("display.width", 200)
    print(s[["tier", "method", "minority_recall_mean", "balanced_accuracy_mean", "macro_f1_mean",
             "roc_auc_mean", "nogueira_stability"]].to_string(index=False))


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["epistasis", "imbalance", "ova_lung"],
                    choices=["epistasis", "imbalance", "ova_lung"])
    ap.add_argument("--variants", nargs="+", default=None, choices=ALL_KEYS,
                    help="default: all for synthetic parts; M6-M11 for ova_lung (SRA-FS rows already stored)")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--outdir", default=None, help="default: exp_result/topg_core_validation")
    args = ap.parse_args()
    global OUT_DIR
    if args.outdir:
        OUT_DIR = Path(args.outdir)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for part in args.parts:
        t0 = time.time()
        if part == "epistasis":
            run_epistasis(args.variants or [k for k in ALL_KEYS if k != "M12"])
        elif part == "imbalance":
            run_imbalance(args.variants or [k for k in ALL_KEYS if k != "M12"])
        else:
            run_ova_lung(args.variants or [k for k in ALL_KEYS if k not in ("M0", "M12")], args.seeds)
        print(f"[{part}] {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
