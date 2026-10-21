"""
topg_synthetic.py
=================
Synthetic generators: XOR epistasis benchmark and class-imbalance sweep.

Copied verbatim on 2026-10-11 from `exp_epistasis_imbalance.py` (source_code/) so that the TOPG project
does not depend on the earlier SRA-FS experiment files.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

RANDOM_STATE = 42


def generate_epistasis_dataset(
    n_samples: int = 1500,
    n_features: int = 30,
    noise_dim: int = 20,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series, List[str], List[str]]:
    """
    Constructs a controlled synthetic benchmark with authentic non-linear epistasis.

    Designed specifically to evaluate whether feature selection algorithms can discover
    combinatorial biomarker interactions lacking linear marginal main effects:
      1. Pure Epistatic Pair (X1, X2): Governed by an exclusive-OR (XOR) interaction rule:
             Y_XOR = (X1 > 0) XOR (X2 > 0)
         Marginally, P(Y=1 | X1) = 0.5 and P(Y=1 | X2) = 0.5 (Main effect ~ 0).
      2. Additive Driver Genes (X3, X4, X5, X6): Linear disease drivers with positive main effects.
      3. Redundant Collinear Features (X7, X8, X9, X10): Highly correlated copies of drivers.
      4. Stochastic Noise Features (X11 ~ X30): Pure Gaussian noise with zero phenotypic signal.

    Parameters
    ----------
    n_samples : int, default=1500
        Number of synthetic cohort samples.
    n_features : int, default=30
        Total feature dimensionality.
    noise_dim : int, default=20
        Number of pure stochastic noise features.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    df_X : pd.DataFrame of shape (n_samples, n_features)
        Constructed feature matrix.
    ser_y : pd.Series of shape (n_samples,)
        Binary phenotype target labels.
    epistatic_features : List[str]
        Identifiers of true epistatic feature pair ['F_1', 'F_2'].
    additive_features : List[str]
        Identifiers of true additive linear drivers ['F_3', 'F_4', 'F_5', 'F_6'].
    """
    rng = np.random.RandomState(random_state)
    
    # 1. Epistatic Pair (X1, X2)
    x1 = rng.normal(0, 1, n_samples)
    x2 = rng.normal(0, 1, n_samples)
    # Pure XOR interaction
    y_xor = ((x1 > 0) ^ (x2 > 0)).astype(int)
    
    # 2. Additive Drivers (X3 ~ X6)
    x_add = rng.normal(0, 1, (n_samples, 4))
    score_add = 0.8 * x_add[:, 0] + 0.8 * x_add[:, 1] + 0.6 * x_add[:, 2] + 0.6 * x_add[:, 3]
    y_add = (score_add > 0).astype(int)
    
    # Combined target: Epistatic rule modulated by additive pathways
    logit = 1.8 * (2 * y_xor - 1) + 1.2 * score_add + rng.normal(0, 0.5, n_samples)
    prob = 1.0 / (1.0 + np.exp(-logit))
    y = (prob > 0.5).astype(int)
    
    # 3. Redundant features (collinear combinations of X3~X6)
    r1 = 0.9 * x_add[:, 0] + rng.normal(0, 0.2, n_samples)
    r2 = 0.9 * x_add[:, 1] + rng.normal(0, 0.2, n_samples)
    r3 = 0.85 * x_add[:, 2] + rng.normal(0, 0.2, n_samples)
    r4 = 0.85 * x_add[:, 3] + rng.normal(0, 0.2, n_samples)
    
    # 4. Pure Noise features
    x_noise = rng.normal(0, 1, (n_samples, noise_dim))
    
    X_all = np.column_stack([x1, x2, x_add, r1, r2, r3, r4, x_noise])
    feature_names = [f"F_{i + 1}" for i in range(X_all.shape[1])]
    
    df_X = pd.DataFrame(X_all, columns=feature_names)
    ser_y = pd.Series(y, name="target")
    
    epistatic_features = ["F_1", "F_2"]
    additive_features = ["F_3", "F_4", "F_5", "F_6"]
    
    return df_X, ser_y, epistatic_features, additive_features


def generate_imbalanced_dataset(
    n_total: int = 2000,
    n_features: int = 50,
    imbalance_ratio: float = 1.0,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Constructs a controlled synthetic benchmark with specified class imbalance ratio:
        rho = N_majority / N_minority

    Parameters
    ----------
    n_total : int, default=2000
        Total cohort sample size.
    n_features : int, default=50
        Feature dimensionality (10 latent biomarkers + 40 noise variables).
    imbalance_ratio : float, default=1.0
        Imbalance skew ratio rho in {1.0, 5.0, 10.0, 20.0}.
    random_state : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    df_X : pd.DataFrame of shape (n_total, n_features)
        Generated feature matrix.
    ser_y : pd.Series of shape (n_total,)
        Binary class target labels with specified rarity.
    """
    rng = np.random.RandomState(random_state)
    n_min = int(n_total / (1.0 + imbalance_ratio))
    n_maj = n_total - n_min
    
    # Latent true biomarker patterns for minority class
    x_min = rng.normal(0.8, 1.0, (n_min, 10))
    x_maj = rng.normal(-0.2, 1.0, (n_maj, 10))
    x_bio = np.vstack([x_maj, x_min])
    
    # Noise features
    x_noise = rng.normal(0, 1.0, (n_total, n_features - 10))
    X_mat = np.hstack([x_bio, x_noise])
    
    y = np.array([0] * n_maj + [1] * n_min)
    
    # Shuffle samples
    idx = rng.permutation(n_total)
    X_mat = X_mat[idx]
    y = y[idx]
    
    fnames = [f"F_{i + 1}" for i in range(n_features)]
    return pd.DataFrame(X_mat, columns=fnames), pd.Series(y, name="target")
