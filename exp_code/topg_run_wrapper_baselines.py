#!/usr/bin/env python3
"""
topg_run_wrapper_baselines.py
=============================
(TOPG project; formerly run_wrapper_baselines.py. In the TOPG manuscript, M6 = TOPG.)

Purpose
-------
Tests whether the benchmark advantage of SRA-FS comes from its objective (G1-G4,
weighting, smoothing, pair search) or simply from the underlying Random Forest
OOB forward-selection framework.

It runs the following selectors on EXACTLY the same setup as the original benchmark_bio.py
(data loading and evaluation are copied verbatim into topg_common.py)
(same 10 datasets, same preprocessing/subsampling, same StratifiedKFold seed,
same K targets, same four downstream classifiers, same surrogate size of 40 trees):

  M6  RF-OOB Wrapper        : plain RF OOB forward selection
                              (G1 only, uniform weights, raw probabilities, no pairs)
  M4  SRA-FS w/o Pair Search: full single-feature objective (alpha*G1 - beta*G2,
                              weights, smoothing) but no pair beam
  M7  RF-OOB Wrapper, OOB accuracy: same as M6 but each step maximises the OOB accuracy
                              of S U {f} (correct/incorrect criterion, as in CBFI)
  M8  RF-OOB Wrapper, OOB balanced accuracy: same as M7 with balanced accuracy
  M9  RF-OOB Wrapper, OOB mean true-class probability: net (signed) continuous gain,
                              i.e. mean of Delta p without truncating losses at zero
  M10 RF-OOB Wrapper, OOB log-loss: minimises the OOB cross-entropy of S U {f}
  M11 RF-OOB Wrapper, OOB Brier score: minimises the OOB multi-class Brier score of S U {f}
  M12 TOPG with a common-seed reference: identical to M6 except that p_base is refitted with the
                              same seed as the candidate forests (robustness check)
  M0  SRA-FS (ablation impl): optional sanity check that the ablation
                              implementation reproduces the stored SRA-FS v8 results

and then compares them with the stored SRA-FS v8 results (per-dataset win/tie/loss,
Wilcoxon signed-rank test, recomputed Grand Rank).

Usage (run from the topg folder; self-contained -- no file outside topg/ is used)
---------------------------------------
    python topg_run_wrapper_baselines.py                    # M6 and M4 on all 10 datasets
    python topg_run_wrapper_baselines.py --variants M6      # M6 only
    python topg_run_wrapper_baselines.py --variants M7 M8   # discrete-criterion wrappers (M6 vs M7/M8)
    python topg_run_wrapper_baselines.py --variants M9 M10 M11   # other continuous criteria (M6 vs M9-M11)
    python topg_run_wrapper_baselines.py --variants M12      # common-seed robustness check of TOPG
    python topg_run_wrapper_baselines.py --sanity           # also run M0 on 'synthetic' only
    python topg_run_wrapper_baselines.py --compare-only     # rebuild the comparison from saved CSVs

Results are saved per dataset, so an interrupted run resumes where it stopped.

Outputs (exp_result/topg_benchmark/)
--------------------------------------
    topg_<dataset>_summary.csv           same columns as benchmark_bio.py summaries
    topg_wrapper_all_summary.csv         all new rows combined
    topg_comparison_per_dataset.csv      per-dataset mean ROC-AUC / Accuracy / Macro-F1 for every method
    topg_comparison_vs_sra.csv           win/tie/loss, mean difference and Wilcoxon p vs SRA-FS v8
    topg_grand_rank_with_wrappers.csv    Grand Rank over all slices (datasets x K x classifiers x metrics)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.model_selection import StratifiedKFold

HERE = Path(__file__).resolve().parent          # .../source_code/topg
sys.path.insert(0, str(HERE))

import topg_common as bb  # noqa: E402  (data loading, evaluation, stability, seeds)
from topg_selector import SRAFSAblation  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score, balanced_accuracy_score  # noqa: E402


LOGLOSS_EPS = 1e-3  # OOB probabilities from 40-50 trees are often exactly 0; clip before log


def _oob_score_of(features, Xt, yt, n_estimators, max_depth, seed, criterion):
    """
    OOB criterion (higher is better) of a surrogate forest identical to the one used by M6.
    criterion: 'accuracy' | 'balanced_accuracy' | 'mean_ptrue' | 'neg_log_loss' | 'neg_brier'
    """
    est = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=2,
                                 class_weight=None, oob_score=True, random_state=seed, n_jobs=1)
    cols = sorted(features)
    est.fit(Xt[cols], yt)
    proba = np.nan_to_num(est.oob_decision_function_, nan=1.0 / len(est.classes_))
    if criterion in ("accuracy", "balanced_accuracy"):
        preds = est.classes_[np.argmax(proba, axis=1)]
        return float(balanced_accuracy_score(yt, preds) if criterion == "balanced_accuracy"
                     else accuracy_score(yt, preds))
    onehot = (yt[:, None] == est.classes_[None, :]).astype(float)
    p_true = (proba * onehot).sum(axis=1)
    if criterion == "mean_ptrue":
        return float(p_true.mean())
    if criterion == "neg_log_loss":
        return float(np.log(np.clip(p_true, LOGLOSS_EPS, 1.0)).mean())
    if criterion == "neg_brier":
        return float(-((proba - onehot) ** 2).sum(axis=1).mean())
    raise ValueError(f"unknown criterion {criterion}")


class OOBAccuracyWrapper(SRAFSAblation):
    """
    RF-OOB forward wrapper with a set-level criterion: at each step add the candidate that
    maximises the OOB criterion of S U {f} -- discrete (accuracy, balanced accuracy; M7/M8) or
    continuous net criteria (mean true-class probability, log-loss, Brier; M9-M11). Everything else is identical to M6
    (same candidate pool, 40 trees, max_depth 5, min_samples_leaf 2, class_weight=None,
    same seed for all candidates, ties broken by feature name as in SRA-FS).
    M6 vs this class therefore isolates the effect of the continuous true-class probability
    gain (G1, the CBFI relaxation) versus the correct/incorrect criterion.
    """

    def __init__(self, criterion="accuracy", **kwargs):
        super().__init__(**kwargs)
        self.criterion = criterion

    def fit(self, X, y):
        X_df = self._prepare_X(X).reset_index(drop=True)
        y_arr = np.asarray(y)
        pool = self._get_active_candidate_pool(X_df, y_arr, self.max_features)
        selected, remaining = [], list(pool)
        while len(selected) < self.max_features and remaining:
            scores = Parallel(n_jobs=self.n_jobs)(
                delayed(_oob_score_of)(tuple(selected + [f]), X_df, y_arr, self.surrogate_estimators,
                                       self.max_depth, self.random_state, self.criterion)
                for f in remaining)
            best = max(zip(scores, [str(f) for f in remaining]))
            selected.append(best[1])
            remaining.remove(best[1])
        self.selected_features_ = selected
        return self


SURROGATE_TREES = 40  # same as select_sra_fs8 in the original benchmark_bio.py

VARIANTS = {
    "M6": ("RF-OOB Wrapper (M6)", dict(
        enable_g4_g3=False, enable_class_weight=False, enable_smoothing=False,
        pair_search=False, min_g4=0.0, objective_mode="g1_only")),
    "M4": ("SRA-FS w/o Pair Search (M4)", dict(
        enable_g4_g3=True, enable_class_weight=True, enable_smoothing=True,
        pair_search=False, min_g4=0.005, objective_mode="full")),
    "M7": ("RF-OOB Wrapper, OOB accuracy (M7)", dict(_criterion="accuracy")),
    "M8": ("RF-OOB Wrapper, OOB balanced accuracy (M8)", dict(_criterion="balanced_accuracy")),
    "M9": ("RF-OOB Wrapper, OOB mean true-class prob. (M9)", dict(_criterion="mean_ptrue")),
    "M10": ("RF-OOB Wrapper, OOB log-loss (M10)", dict(_criterion="neg_log_loss")),
    "M11": ("RF-OOB Wrapper, OOB Brier score (M11)", dict(_criterion="neg_brier")),
    "M12": ("TOPG, common-seed reference (M12)", dict(
        enable_g4_g3=False, enable_class_weight=False, enable_smoothing=False,
        pair_search=False, min_g4=0.0, objective_mode="g1_only", fixed_base_seed=True)),
    "M0": ("SRA-FS Full (ablation impl., M0)", dict(
        enable_g4_g3=True, enable_class_weight=True, enable_smoothing=True,
        pair_search=True, min_g4=0.005, objective_mode="full")),
}

DATASETS = [
    ("synthetic", None),
    ("MiceProtein", 40966),
    ("qsar-biodeg", 1494),
    ("dna", 40670),
    ("splice", 46),
    ("Bioresponse", 4134),
    ("AP_Colon_Lung", 1126),
    ("OVA_Lung", 1130),
    ("OVA_Breast", 1128),
    ("Ovarian", 45098),
]

PREDICTORS = ["RandomForest", "XGBoost", "SVM", "LogisticRegression"]
PREFIX = {"RandomForest": "RF", "XGBoost": "XGB", "SVM": "SVM", "LogisticRegression": "LR"}
REFERENCE_METHOD = "SRA-FS v8"


def k_targets_for(p: int) -> dict:
    """Identical to run_benchmark_dataset in the original benchmark_bio.py."""
    if p > 500:
        return {"K_10": 10, "K_20": 20, "K_30": 30, "K_50": 50}
    return {f"{int(r * 100)}%": max(2, int(p * r)) for r in [0.10, 0.20, 0.30, 0.40]}


def load_dataset(name, did):
    if name == "synthetic":
        return bb.make_synthetic()
    return bb.load_bio_dataset(name, did)


def run_dataset(name, did, variant_keys, outdir: Path) -> pd.DataFrame:
    out_csv = outdir / f"topg_{name}_summary.csv"
    done = pd.read_csv(out_csv) if out_csv.exists() else pd.DataFrame()
    todo = [v for v in variant_keys
            if done.empty or VARIANTS[v][0] not in set(done["method"])]
    if not todo:
        print(f"[skip] {name}: all requested variants already saved.")
        return done

    X, y = load_dataset(name, did)
    p, n_classes = X.shape[1], y.nunique()
    k_targets = k_targets_for(p)
    max_k = max(k_targets.values())
    print(f"\n=== {name}: N={len(X)}, P={p}, C={n_classes}, K={k_targets} | variants={todo}")

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=bb.RANDOM_STATE)
    sel = {v: [] for v in todo}
    times = {v: [] for v in todo}
    evals = {v: {lab: {pr: [] for pr in PREDICTORS} for lab in k_targets} for v in todo}

    for fold, (tr, te) in enumerate(skf.split(X, y), start=1):
        X_tr, X_te = X.iloc[tr], X.iloc[te]
        y_tr, y_te = y.iloc[tr], y.iloc[te]
        for v in todo:
            label, params = VARIANTS[v]
            t0 = time.time()
            common = dict(max_features=max_k, surrogate_estimators=SURROGATE_TREES,
                          random_state=bb.RANDOM_STATE, use_oob=True, n_jobs=-1)
            if "_criterion" in params:
                selector = OOBAccuracyWrapper(
                    criterion=params["_criterion"], enable_g4_g3=False, enable_class_weight=False,
                    enable_smoothing=False, pair_search=False, min_g4=0.0, objective_mode="g1_only", **common)
            else:
                selector = SRAFSAblation(**common, **params)
            selector.fit(X_tr, y_tr)
            chosen = list(selector.selected_features_)
            times[v].append(time.time() - t0)
            sel[v].append(chosen)
            print(f"  fold {fold} {label:<34} {times[v][-1]:7.1f}s ({len(chosen)} features)")
            for lab, k in k_targets.items():
                sub = chosen[:k]
                res = bb.evaluate_predictors(X_tr[sub].values, y_tr.values,
                                             X_te[sub].values, y_te.values, n_classes)
                for pr, m in res.items():
                    evals[v][lab][pr].append(m)

    rows = []
    col_idx = {c: i for i, c in enumerate(X.columns)}
    for v in todo:
        label = VARIANTS[v][0]
        for lab, k in k_targets.items():
            Z = np.zeros((5, p))
            for f in range(5):
                for feat in sel[v][f][:k]:
                    Z[f, col_idx[feat]] = 1.0
            row = {"dataset": name, "method": label, "k_label": lab, "K": k,
                   "nogueira_stability": bb.nogueira_stability(Z),
                   "mean_selection_time_sec": float(np.mean(times[v]))}
            for pr in PREDICTORS:
                fl = evals[v][lab][pr]
                row[f"{PREFIX[pr]}_mean_roc_auc"] = float(np.mean([m["roc_auc"] for m in fl]))
                row[f"{PREFIX[pr]}_mean_accuracy"] = float(np.mean([m["accuracy"] for m in fl]))
                row[f"{PREFIX[pr]}_mean_macro_f1"] = float(np.mean([m["macro_f1"] for m in fl]))
            rows.append(row)

    new = pd.DataFrame(rows)
    combined = pd.concat([done, new], ignore_index=True) if not done.empty else new
    combined.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")
    return combined


def find_reference() -> Path:
    for cand in [HERE.parent / "exp_result" / "topg_reference" / "topg_ref_baselines_benchmark_summary.csv"]:
        if cand.exists():
            df = pd.read_csv(cand)
            if REFERENCE_METHOD in set(df["method"]) and "Stability Selection" in set(df["method"]):
                return cand
    raise FileNotFoundError("Reference summary with all ten methods not found.")


def compare(outdir: Path):
    new_files = sorted(outdir.glob("topg_*_summary.csv"))
    new_files = [f for f in new_files if f.name != "topg_wrapper_all_summary.csv"]
    if not new_files:
        print("No wrapper results yet.")
        return
    new = pd.concat([pd.read_csv(f) for f in new_files], ignore_index=True)
    new.to_csv(outdir / "topg_wrapper_all_summary.csv", index=False)

    ref_path = find_reference()
    ref = pd.read_csv(ref_path)
    print(f"\nReference results: {ref_path}")
    allres = pd.concat([ref, new], ignore_index=True)
    allres = allres[allres["dataset"].isin(set(new["dataset"]))]
    # keep only methods evaluated on every dataset in this comparison (e.g., drop the M0 sanity run)
    n_ds = allres["dataset"].nunique()
    cover = allres.groupby("method")["dataset"].nunique()
    partial = sorted(cover[cover < n_ds].index)
    if partial:
        print(f"Excluded from comparison (not run on all {n_ds} datasets): {partial}")
    allres = allres[allres["method"].isin(cover[cover == n_ds].index)]
    new = new[new["method"].isin(cover[cover == n_ds].index)]

    metrics = {"roc_auc": "ROC-AUC", "accuracy": "Accuracy", "macro_f1": "Macro-F1"}
    per_ds = []
    for met in metrics:
        cols = [f"{PREFIX[p]}_mean_{met}" for p in PREDICTORS]
        g = allres.assign(val=allres[cols].mean(axis=1)).groupby(["dataset", "method"])["val"].mean()
        per_ds.append(g.rename(met))
    per_ds = pd.concat(per_ds, axis=1).reset_index()
    extra = allres.groupby(["dataset", "method"]).agg(
        stability=("nogueira_stability", "mean"),
        time_sec=("mean_selection_time_sec", "mean")).reset_index()
    per_ds = per_ds.merge(extra, on=["dataset", "method"])
    per_ds.to_csv(outdir / "topg_comparison_per_dataset.csv", index=False)

    rows = []
    for label in sorted(set(new["method"])):
        for met, mname in metrics.items():
            piv = per_ds.pivot(index="dataset", columns="method", values=met)
            piv = piv[[REFERENCE_METHOD, label]].dropna()
            d = piv[REFERENCE_METHOD] - piv[label]
            win, tie, loss = int((d > 1e-9).sum()), int((d.abs() <= 1e-9).sum()), int((d < -1e-9).sum())
            try:
                pval = float(wilcoxon(piv[REFERENCE_METHOD], piv[label]).pvalue) if (d.abs() > 1e-12).any() else 1.0
            except ValueError:
                pval = float("nan")
            rows.append({"variant": label, "metric": mname, "n_datasets": len(d),
                         "SRA_mean": piv[REFERENCE_METHOD].mean(), "variant_mean": piv[label].mean(),
                         "mean_diff_SRA_minus_variant": d.mean(),
                         "SRA_win": win, "tie": tie, "SRA_loss": loss, "wilcoxon_p": pval})
    cmp_df = pd.DataFrame(rows)
    cmp_df.to_csv(outdir / "topg_comparison_vs_sra.csv", index=False)

    # Grand Rank over all slices (dataset x K x classifier x metric), lower is better
    long = []
    for met in metrics:
        for p in PREDICTORS:
            c = f"{PREFIX[p]}_mean_{met}"
            tmp = allres[["dataset", "k_label", "method", c]].rename(columns={c: "val"})
            tmp["slice"] = tmp["dataset"] + "|" + tmp["k_label"].astype(str) + "|" + p + "|" + met
            long.append(tmp)
    long = pd.concat(long, ignore_index=True)
    long["rank"] = long.groupby("slice")["val"].rank(ascending=False, method="average")
    gr = long.groupby("method")["rank"].mean().sort_values().rename("grand_rank").reset_index()
    gr["n_slices"] = long["slice"].nunique()
    gr.to_csv(outdir / "topg_grand_rank_with_wrappers.csv", index=False)

    pd.set_option("display.width", 160)
    print("\n=== Per-dataset mean ROC-AUC ===")
    print(per_ds.pivot(index="dataset", columns="method", values="roc_auc").round(4).to_string())
    print("\n=== SRA-FS v8 vs wrapper variants ===")
    print(cmp_df.round(4).to_string(index=False))
    print("\n=== Grand Rank including wrapper variants ===")
    print(gr.round(3).to_string(index=False))
    print(f"\n[saved] comparison files in {outdir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["M6", "M4"], choices=list(VARIANTS))
    ap.add_argument("--datasets", nargs="+", default=None, help="subset of dataset names")
    ap.add_argument("--sanity", action="store_true", help="also run M0 on 'synthetic'")
    ap.add_argument("--compare-only", action="store_true")
    ap.add_argument("--outdir", default=str(HERE.parent / "exp_result" / "topg_benchmark"))
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not args.compare_only:
        names = args.datasets or [d for d, _ in DATASETS]
        for name, did in DATASETS:
            if name not in names:
                continue
            t0 = time.time()
            try:
                run_dataset(name, did, args.variants, outdir)
            except Exception as e:  # keep going with the other datasets
                print(f"[FAILED] {name}: {e}")
            print(f"[{name}] {time.time() - t0:.1f}s")
        if args.sanity:
            run_dataset("synthetic", None, ["M0"], outdir)
    compare(outdir)


if __name__ == "__main__":
    main()
