"""
topg_selector.py
================
RF-OOB forward selector. TOPG = SRAFSAblation(objective_mode='g1_only', enable_g4_g3=False,
enable_class_weight=False, enable_smoothing=False, pair_search=False, min_g4=0.0).

Copied verbatim on 2026-10-11 from `G1G4_Ablation.py` (source_code/) so that the TOPG project
does not depend on the earlier SRA-FS experiment files.
Only change: the optional `fixed_base_seed` flag of SRAFSAblation (default False = original
behaviour); when True, the reference probabilities p_base are refitted with the same seed as the
candidate forests instead of `random_state + step` (robustness check M12).
"""
from __future__ import annotations

from itertools import combinations
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.metrics import balanced_accuracy_score


def _worker_eval_subset(
    features: Tuple[str, ...],
    Xt: pd.DataFrame,
    yt: np.ndarray,
    Xv: pd.DataFrame,
    yv: np.ndarray,
    use_oob: bool,
    n_estimators: int,
    max_depth: Optional[int],
    class_weight: Optional[str],
    seed: int,
    enable_smoothing: bool,
) -> Tuple[np.ndarray, float]:
    classes = np.unique(yt)
    n_classes = len(classes)
    if len(features) == 0:
        p_prior = 1.0 / n_classes
        return np.full(len(yv), p_prior, dtype=float), p_prior

    features = tuple(sorted(features))

    est = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=seed,
        n_jobs=1,
    )

    if use_oob:
        est.oob_score = True
        est.fit(Xt[list(features)], yt)
        if hasattr(est, "oob_decision_function_"):
            proba = est.oob_decision_function_
            if np.isnan(proba).any():
                proba = np.nan_to_num(proba, nan=1.0 / n_classes)

            # Bayesian Laplacian smoothing ablation check
            if enable_smoothing:
                proba = (n_estimators * proba + (1.0 / n_classes)) / (n_estimators + 1.0)

            preds = est.classes_[np.argmax(proba, axis=1)]
            bacc = float(balanced_accuracy_score(yt, preds))
            class_to_idx = {cls: idx for idx, cls in enumerate(est.classes_)}
            true_indices = [class_to_idx.get(target, 0) for target in yt]
            p_true = proba[np.arange(len(yt)), true_indices]
            return p_true, bacc

    est.fit(Xt[list(features)], yt)
    proba = est.predict_proba(Xv[list(features)])
    if enable_smoothing:
        proba = (n_estimators * proba + (1.0 / n_classes)) / (n_estimators + 1.0)
    preds = est.predict(Xv[list(features)])
    bacc = float(balanced_accuracy_score(yv, preds))
    class_to_idx = {cls: idx for idx, cls in enumerate(est.classes_)}
    true_indices = [class_to_idx.get(target, 0) for target in yv]
    p_true = proba[np.arange(len(yv)), true_indices]
    return p_true, bacc


def _worker_eval_single(
    f: str,
    f_idx: int,
    selected: Tuple[str, ...],
    p_base: np.ndarray,
    Xt: pd.DataFrame,
    yt: np.ndarray,
    Xv: pd.DataFrame,
    yv: np.ndarray,
    weights_eval: np.ndarray,
    alpha: float,
    beta: float,
    use_oob: bool,
    n_estimators: int,
    max_depth: Optional[int],
    class_weight: Optional[str],
    base_seed: int,
    enable_smoothing: bool,
    objective_mode: str,
) -> Tuple[str, float, float, float, np.ndarray, float]:
    cand_features = selected + (f,)
    seed = base_seed
    p_cand, acc_cand = _worker_eval_subset(
        cand_features, Xt, yt, Xv, yv, use_oob, n_estimators, max_depth, class_weight, seed, enable_smoothing
    )

    delta = p_cand - p_base
    g1 = float(np.average(np.maximum(0.0, delta), weights=weights_eval))
    g2 = float(np.average(np.maximum(0.0, -delta), weights=weights_eval))

    if objective_mode == "g1_only":
        gain = g1
    else:
        gain = alpha * g1 - beta * g2

    return (f, gain, g1, g2, p_cand, acc_cand)


def _worker_eval_pair(
    fa: str,
    fb: str,
    pair_idx: int,
    selected: Tuple[str, ...],
    p_base: np.ndarray,
    p_i: np.ndarray,
    p_j: np.ndarray,
    Xt: pd.DataFrame,
    yt: np.ndarray,
    Xv: pd.DataFrame,
    yv: np.ndarray,
    weights_eval: np.ndarray,
    alpha: float,
    beta: float,
    lambda_g4: float,
    lambda_g3: float,
    min_g4: float,
    use_oob: bool,
    n_estimators: int,
    max_depth: Optional[int],
    class_weight: Optional[str],
    base_seed: int,
    enable_smoothing: bool,
    enable_g4_g3: bool,
    objective_mode: str,
) -> Tuple[str, str, float, float, float, float, float, float]:
    pair_features = selected + (fa, fb)
    seed = base_seed
    p_ij, acc_ij = _worker_eval_subset(
        pair_features, Xt, yt, Xv, yv, use_oob, n_estimators, max_depth, class_weight, seed, enable_smoothing
    )

    delta_i = p_i - p_base
    delta_j = p_j - p_base
    delta_ij = p_ij - p_base

    g1_ij = float(np.average(np.maximum(0.0, delta_ij), weights=weights_eval))
    g2_ij = float(np.average(np.maximum(0.0, -delta_ij), weights=weights_eval))
    g3_ij = float(np.average(p_i * p_j, weights=weights_eval))

    synergy = np.maximum(0.0, delta_ij - (delta_i + delta_j))
    g4_ij = float(np.average(synergy, weights=weights_eval))
    effective_g4 = g4_ij if g4_ij >= min_g4 else 0.0

    if objective_mode == "g1_only":
        pair_gain = g1_ij
    else:
        pair_gain = alpha * g1_ij - beta * g2_ij

    if enable_g4_g3:
        pair_score = pair_gain + lambda_g4 * effective_g4 - lambda_g3 * g3_ij
    else:
        pair_score = pair_gain

    return (fa, fb, pair_score, pair_gain, effective_g4, g1_ij, g2_ij, g3_ij)


class SRAFSAblation(BaseEstimator):
    """
    Parametrically Ablated SRA-FS Estimator for Rigorous Component Isolation.

    Provides a modular ablation framework to independently disable each theoretical
    component of the SRA-FS objective, proving that downstream gains stem from the
    mathematical formulation rather than the baseline capacity of the Random Forest surrogate.

    Parameters
    ----------
    max_features : int, default=10
        Target feature panel size K to select.
    alpha : float, default=0.5
        Weight for positive true-class posterior margin gain (G1).
    beta : float, default=0.5
        Penalty weight for negative true-class posterior distortion loss (G2).
    lambda_g4 : float, default=1.0
        Reward weight for super-additive epistatic synergy (G4).
    lambda_g3 : float, default=0.2
        Penalty weight for co-activation redundancy (G3).
    pair_search : bool, default=True
        Whether combinatorial pairwise synergy beam search is active (M4 ablation toggles this).
    pair_beam : int, default=10
        Candidate beam search width.
    min_g4 : float, default=0.005
        Interaction noise shrinkage floor epsilon_min (M5 ablation sets this to 0.0).
    pair_margin : float, default=0.01
        Required improvement margin over best single candidate to commit a pair.
    surrogate_estimators : int, default=50
        Number of decision trees in surrogate Random Forest.
    random_state : int, default=42
        Random seed for reproducibility.
    use_oob : bool, default=True
        Whether to utilize Out-Of-Bag probability estimates.
    max_depth : Optional[int], default=5
        Maximum tree depth constraint for HDLSS regularization.
    prune_threshold : int, default=150
        Dimensionality threshold for multi-perspective candidate pooling.
    n_jobs : int, default=-1
        Number of parallel CPU worker processes.
    enable_g4_g3 : bool, default=True
        Ablation Flag (M1): If False, completely deactivates G4 synergy and G3 redundancy.
    enable_class_weight : bool, default=True
        Ablation Flag (M2): If False, adopts uniform sample weights (1/N) and unweighted trees.
    enable_smoothing : bool, default=True
        Ablation Flag (M3): If False, evaluates raw OOB probabilities without Bayesian smoothing.
    objective_mode : str, default='full'
        Ablation Flag (M6): If 'g1_only', disables G2 penalty, optimizing purely positive G1 gain.
    """

    def __init__(
        self,
        max_features: int = 10,
        alpha: float = 0.5,
        beta: float = 0.5,
        lambda_g4: float = 1.0,
        lambda_g3: float = 0.2,
        pair_search: bool = True,
        pair_beam: int = 10,
        min_g4: float = 0.005,
        pair_margin: float = 0.01,
        surrogate_estimators: int = 50,
        random_state: int = 42,
        use_oob: bool = True,
        max_depth: Optional[int] = 5,
        prune_threshold: int = 150,
        n_jobs: int = -1,
        # Ablation Flags
        enable_g4_g3: bool = True,
        enable_class_weight: bool = True,
        enable_smoothing: bool = True,
        objective_mode: str = "full",  # "full" or "g1_only"
        fixed_base_seed: bool = False,  # TOPG robustness check (M12): refit p_base with random_state
        pool_size_factor: float = 1.0,  # sensitivity check (B2): scales the candidate-pool size; 1.0 = original
    ):
        self.max_features = int(max_features)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_g4 = float(lambda_g4)
        self.lambda_g3 = float(lambda_g3)
        self.pair_search = bool(pair_search)
        self.pair_beam = int(pair_beam)
        self.min_g4 = float(min_g4)
        self.pair_margin = float(pair_margin)
        self.surrogate_estimators = int(surrogate_estimators)
        self.random_state = int(random_state)
        self.use_oob = bool(use_oob)
        self.max_depth = max_depth
        self.prune_threshold = int(prune_threshold)
        self.n_jobs = int(n_jobs)
        self.enable_g4_g3 = bool(enable_g4_g3)
        self.enable_class_weight = bool(enable_class_weight)
        self.enable_smoothing = bool(enable_smoothing)
        self.objective_mode = str(objective_mode)
        self.fixed_base_seed = bool(fixed_base_seed)
        self.pool_size_factor = float(pool_size_factor)

    def _compute_sample_weights(self, y: np.ndarray) -> np.ndarray:
        if not self.enable_class_weight:
            return np.full(len(y), 1.0 / len(y), dtype=float)
        classes, counts = np.unique(y, return_counts=True)
        n_samples = len(y)
        n_classes = len(classes)
        class_weight_dict = {
            cls: n_samples / (n_classes * count)
            for cls, count in zip(classes, counts)
        }
        return np.array([class_weight_dict[val] for val in y], dtype=float)

    def _prepare_X(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = np.asarray(X.columns, dtype=object)
            return X.copy()
        X_arr = np.asarray(X)
        self.feature_names_in_ = np.asarray(
            [f"F{i + 1}" for i in range(X_arr.shape[1])], dtype=object
        )
        return pd.DataFrame(X_arr, columns=self.feature_names_in_)

    def _get_active_candidate_pool(self, X: pd.DataFrame, y: np.ndarray, max_k: int) -> List[str]:
        p = X.shape[1]
        if p <= self.prune_threshold:
            return list(X.columns)

        pool_size = min(p, int(round(self.pool_size_factor * max(200, 5 * max_k))))
        sub_size = pool_size // 2

        surr_cw = "balanced_subsample" if self.enable_class_weight else None
        rf_fast = RandomForestClassifier(
            n_estimators=40,
            max_depth=self.max_depth,
            class_weight=surr_cw,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        rf_fast.fit(X, y)
        top_rf_idx = np.argsort(rf_fast.feature_importances_)[::-1][:sub_size]
        cols_rf = set(X.columns[top_rf_idx])

        f_vals, _ = f_classif(X, y)
        f_vals = np.nan_to_num(f_vals, nan=0.0)
        top_f_idx = np.argsort(f_vals)[::-1][:sub_size]
        cols_f = set(X.columns[top_f_idx])

        variances = np.var(X.values, axis=0)
        top_var_idx = np.argsort(variances)[::-1][:sub_size]
        cols_var = set(X.columns[top_var_idx])

        return list(cols_rf.union(cols_f).union(cols_var))

    def fit(self, X, y):
        X_df = self._prepare_X(X)
        y_arr = np.asarray(y)
        Xt, yt, Xv, yv = X_df.reset_index(drop=True), y_arr, X_df.reset_index(drop=True), y_arr
        weights_eval = self._compute_sample_weights(yv)

        self.feature_to_idx_ = {col: i for i, col in enumerate(self.feature_names_in_)}
        candidate_pool = self._get_active_candidate_pool(Xt, yt, self.max_features)

        n_trees = self.surrogate_estimators
        surr_cw = "balanced_subsample" if self.enable_class_weight else None
        selected: List[str] = []
        remaining: List[str] = [col for col in candidate_pool]
        step = 1

        p_base, _ = _worker_eval_subset(
            (), Xt, yt, Xv, yv, self.use_oob, n_trees, self.max_depth, surr_cw, self.random_state, self.enable_smoothing
        )

        while len(selected) < self.max_features and remaining:
            selected_tuple = tuple(selected)

            # 1. Parallel Single Candidate Screening
            single_results = Parallel(n_jobs=self.n_jobs, max_nbytes="1M")(
                delayed(_worker_eval_single)(
                    f=f,
                    f_idx=self.feature_to_idx_[f],
                    selected=selected_tuple,
                    p_base=p_base,
                    Xt=Xt,
                    yt=yt,
                    Xv=Xv,
                    yv=yv,
                    weights_eval=weights_eval,
                    alpha=self.alpha,
                    beta=self.beta,
                    use_oob=self.use_oob,
                    n_estimators=n_trees,
                    max_depth=self.max_depth,
                    class_weight=surr_cw,
                    base_seed=self.random_state,
                    enable_smoothing=self.enable_smoothing,
                    objective_mode=self.objective_mode,
                )
                for f in remaining
            )

            single_results.sort(key=lambda item: (item[1], str(item[0])), reverse=True)
            best_single = single_results[0]
            best_cand_features = (best_single[0],)
            best_score = best_single[1]

            # 2. Pairwise Synergy Search (if beam search enabled)
            if self.pair_search and len(remaining) >= 2 and (len(selected) + 2 <= self.max_features):
                beam_features = [item[0] for item in single_results[: self.pair_beam]]
                pairs_list = list(combinations(beam_features, 2))
                p_cand_dict = {item[0]: item[4] for item in single_results[: self.pair_beam]}

                pair_eval_results = Parallel(n_jobs=self.n_jobs, max_nbytes="1M")(
                    delayed(_worker_eval_pair)(
                        fa=fa,
                        fb=fb,
                        pair_idx=p_idx,
                        selected=selected_tuple,
                        p_base=p_base,
                        p_i=p_cand_dict[fa],
                        p_j=p_cand_dict[fb],
                        Xt=Xt,
                        yt=yt,
                        Xv=Xv,
                        yv=yv,
                        weights_eval=weights_eval,
                        alpha=self.alpha,
                        beta=self.beta,
                        lambda_g4=self.lambda_g4,
                        lambda_g3=self.lambda_g3,
                        min_g4=self.min_g4,
                        use_oob=self.use_oob,
                        n_estimators=n_trees,
                        max_depth=self.max_depth,
                        class_weight=surr_cw,
                        base_seed=self.random_state,
                        enable_smoothing=self.enable_smoothing,
                        enable_g4_g3=self.enable_g4_g3,
                        objective_mode=self.objective_mode,
                    )
                    for p_idx, (fa, fb) in enumerate(pairs_list)
                )

                if self.enable_g4_g3:
                    valid_pairs = [r for r in pair_eval_results if r[4] > 0.0]
                else:
                    valid_pairs = pair_eval_results

                if valid_pairs:
                    valid_pairs.sort(key=lambda r: (r[2], str(r[0]), str(r[1])), reverse=True)
                    best_pair = valid_pairs[0]
                    if best_pair[2] >= best_single[1] + self.pair_margin:
                        best_cand_features = (best_pair[0], best_pair[1])
                        best_score = best_pair[2]

            # 3. Commit features
            for f in best_cand_features:
                selected.append(f)
                remaining.remove(f)

            p_base, _ = _worker_eval_subset(
                tuple(selected), Xt, yt, Xv, yv, self.use_oob, n_trees, self.max_depth, surr_cw,
                self.random_state if self.fixed_base_seed else self.random_state + step, self.enable_smoothing
            )
            step += 1

        self.selected_features_ = selected
        return self
