"""
topg_common.py
==============
Shared utilities: dataset loading, downstream evaluation, Nogueira stability.

Copied verbatim on 2026-10-11 from `benchmark_bio.py` (source_code/) so that the TOPG project
does not depend on the earlier SRA-FS experiment files.
"""
from __future__ import annotations

import re
import sys
import warnings
from typing import Dict, Tuple

import numpy as np
import openml
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
sys.stdout.reconfigure(line_buffering=True)

RANDOM_STATE = 42
MAX_SAMPLES_FOR_FS = 1500  # Subsample massive datasets for efficient 5-fold CV


def nogueira_stability(Z: np.ndarray) -> float:
    """
    Computes Nogueira's stability measure Phi for M feature subsets across d candidate features.

    Reference:
        Nogueira, S., Sechidis, K., & Brown, G. (2017). On the stability of feature selection algorithms.
        Journal of Machine Learning Research, 18(1), 6345-6398.

    Parameters
    ----------
    Z : np.ndarray of shape (M, d)
        Binary selection matrix where Z[m, j] = 1 if feature j was selected in fold m, 0 otherwise.

    Returns
    -------
    phi : float
        Nogueira stability index in range [-1, 1], where 1 denotes perfect selection consensus
        and <= 0 denotes stability expected by chance.
    """
    Z = np.asarray(Z, dtype=float)
    M, d = Z.shape
    if M <= 1:
        return 1.0
    p = np.mean(Z, axis=0)
    k_bar = float(np.sum(p))
    if k_bar == 0.0 or k_bar == float(d):
        return 1.0
    s2 = (M / (M - 1.0)) * p * (1.0 - p)
    denom = (k_bar / d) * (1.0 - (k_bar / d))
    if denom <= 0.0:
        return 1.0
    phi = 1.0 - (float(np.mean(s2)) / denom)
    return float(phi)


def make_synthetic() -> Tuple[pd.DataFrame, pd.Series]:
    """Generates synthetic benchmark dataset (1,500 samples, 30 features)."""
    X, y = make_classification(
        n_samples=1500,
        n_features=30,
        n_informative=8,
        n_redundant=6,
        n_repeated=2,
        n_clusters_per_class=2,
        class_sep=1.0,
        random_state=RANDOM_STATE,
    )
    feature_names = [f"F_{i + 1}" for i in range(X.shape[1])]
    return pd.DataFrame(X, columns=feature_names), pd.Series(y, name="target")


def load_bio_dataset(name: str, dataset_id: int, impute_numeric: bool = True) -> Tuple[pd.DataFrame, pd.Series]:
    """Downloads, imputes, and preprocesses an OpenML biological dataset."""
    print(f"Loading OpenML dataset '{name}' (ID: {dataset_id}) ...")
    ds = openml.datasets.get_dataset(dataset_id, download_data=True)
    X_df, y_raw, cat_indicator, _ = ds.get_data(
        dataset_format="dataframe", target=ds.default_target_attribute
    )

    # 1. Clean non-informative metadata ID columns
    for col in list(X_df.columns):
        if str(col).lower() in ["molecule_name", "id", "sample_id", "gene_id", "probe_id"]:
            X_df = X_df.drop(columns=[col])

    # 1b. (TOPG amendment 2026-10-11) drop columns that are entirely missing; SimpleImputer would drop
    #     them silently and break the column assignment below (e.g. 'TBG' in OpenML 'sick').
    X_df = X_df.dropna(axis=1, how="all")

    # 2. Identify categorical columns
    cat_cols = []
    num_cols = []
    for col in X_df.columns:
        if (
            pd.api.types.is_categorical_dtype(X_df[col])
            or pd.api.types.is_object_dtype(X_df[col])
            or pd.api.types.is_string_dtype(X_df[col])
        ):
            cat_cols.append(col)
        else:
            num_cols.append(col)

    # 3. Impute numeric columns with median, categorical with most_frequent
    # (TOPG amendment 2026-10-11) impute_numeric=False keeps numeric NaNs so that the caller can impute
    # inside each CV fold with training-fold medians (no information from the test fold).
    if num_cols and impute_numeric:
        num_imp = SimpleImputer(strategy="median")
        X_df[num_cols] = num_imp.fit_transform(X_df[num_cols])

    if cat_cols:
        cat_imp = SimpleImputer(strategy="most_frequent")
        X_df[cat_cols] = cat_imp.fit_transform(X_df[cat_cols].astype(str))
        # (TOPG amendment 2026-10-13) sparse float32 dummies: high-cardinality categoricals (e.g. KDDCup09,
        # >70,000 dummy columns) would otherwise need tens of GB as a dense float64 frame. Values are identical.
        X_df = pd.get_dummies(X_df, columns=cat_cols, drop_first=True, dtype=np.float32, sparse=True)

    # 4. Impute any remaining NaNs
    if impute_numeric and X_df.isna().any().any():
        final_imp = SimpleImputer(strategy="median")
        X_df = pd.DataFrame(final_imp.fit_transform(X_df), columns=X_df.columns)

    # 5. Target encoding
    le = LabelEncoder()
    y_encoded = pd.Series(le.fit_transform(y_raw.astype(str)), name="target")

    # Clean column names (XGBoost compatibility)
    X_df.columns = [re.sub(r"[\[\]<>, \s+]", "_", str(col)) for col in X_df.columns]

    # 6. Stratified subsampling if dataset is excessively large
    #    (TOPG amendment 2026-10-13: subsample BEFORE densifying / casting to float32; the selected rows
    #    depend only on y and the seed, so the result is identical to casting first.)
    if len(X_df) > MAX_SAMPLES_FOR_FS:
        print(f"Subsampling {len(X_df)} samples to {MAX_SAMPLES_FOR_FS} for 5-fold CV scalability...")
        _, X_sub, _, y_sub = train_test_split(
            X_df,
            y_encoded,
            test_size=MAX_SAMPLES_FOR_FS,
            random_state=RANDOM_STATE,
            stratify=y_encoded,
        )
        X_df = X_sub.reset_index(drop=True)
        y_encoded = y_sub.reset_index(drop=True)

    sparse_cols = [c for c, dt in X_df.dtypes.items() if isinstance(dt, pd.SparseDtype)]
    if sparse_cols:
        X_df[sparse_cols] = X_df[sparse_cols].sparse.to_dense()
    X_df = X_df.astype(np.float32)

    print(f"'{name}' ready: {len(X_df)} samples, {X_df.shape[1]} features, {y_encoded.nunique()} classes.")
    return X_df, y_encoded


def evaluate_predictors(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    n_classes: int,
) -> Dict[str, Dict[str, float]]:
    """
    Trains and evaluates four structurally distinct downstream classification architectures
    on the selected feature subset to verify cross-model generalizability:
      1. Random Forest (RF): Non-linear bagging ensemble of decision trees.
      2. XGBoost (XGB): Non-linear gradient boosted tree machine.
      3. Support Vector Machine (SVM): Kernel machine with Radial Basis Function (RBF).
      4. Logistic Regression (LR): Linear regularized generalized linear model.

    Parameters
    ----------
    X_tr, y_tr : np.ndarray
        Training feature matrix and class labels.
    X_te, y_te : np.ndarray
        Independent test partition feature matrix and class labels.
    n_classes : int
        Number of distinct class labels.

    Returns
    -------
    results : Dict[str, Dict[str, float]]
        Nested dictionary of downstream validation metrics (Accuracy, Macro-F1, ROC-AUC) per model.
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    results = {}

    # 1. Random Forest
    rf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    rf.fit(X_tr, y_tr)
    rf_pred = rf.predict(X_te)
    rf_proba = rf.predict_proba(X_te)
    results["RandomForest"] = _calc_metrics(y_te, rf_pred, rf_proba, n_classes)

    # 2. XGBoost
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        eval_metric="mlogloss" if n_classes > 2 else "logloss",
    )
    xgb.fit(X_tr, y_tr)
    xgb_pred = xgb.predict(X_te)
    xgb_proba = xgb.predict_proba(X_te)
    results["XGBoost"] = _calc_metrics(y_te, xgb_pred, xgb_proba, n_classes)

    # 3. SVM (RBF)
    svm = SVC(kernel="rbf", C=1.0, probability=True, random_state=RANDOM_STATE)
    svm.fit(X_tr_sc, y_tr)
    svm_pred = svm.predict(X_te_sc)
    svm_proba = svm.predict_proba(X_te_sc)
    results["SVM"] = _calc_metrics(y_te, svm_pred, svm_proba, n_classes)

    # 4. Logistic Regression
    lr = LogisticRegression(
        max_iter=300, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
    )
    lr.fit(X_tr_sc, y_tr)
    lr_pred = lr.predict(X_te_sc)
    lr_proba = lr.predict_proba(X_te_sc)
    results["LogisticRegression"] = _calc_metrics(y_te, lr_pred, lr_proba, n_classes)

    return results


def _calc_metrics(y_true, y_pred, y_proba, n_classes: int) -> Dict[str, float]:
    acc = float(accuracy_score(y_true, y_pred))
    f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    try:
        if n_classes == 2:
            if y_proba.shape[1] == 2:
                auc = float(roc_auc_score(y_true, y_proba[:, 1]))
            else:
                auc = float(roc_auc_score(y_true, y_proba))
        else:
            auc = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro"))
    except Exception:
        auc = 0.5
    return {"accuracy": acc, "macro_f1": f1, "roc_auc": auc}
