"""
Core MCDM models for football scouting.

Pure numerical implementations — no Streamlit, no I/O, no side effects.
All functions accept and return NumPy arrays / pandas Series with strict typing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# TYPE ALIASES
# ============================================================
Array = np.ndarray
Weights = Array  # shape (n_criteria,), sum == 1
Benefits = Array  # shape (n_criteria,), bool
Scores = Array  # shape (n_alternatives,)


# ============================================================
# DATA PREP — Imputation
# ============================================================

def impute_matrix(df: pd.DataFrame, columns: list[str]) -> Array:
    """
    Convert selected columns to numeric and fill NaN with column median.

    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    columns : list[str]
        Column names to extract and impute.

    Returns
    -------
    Array
        Numeric matrix (m, n) with no NaN values.
    """
    X = df[columns].apply(pd.to_numeric, errors="coerce").astype(float)

    for col in columns:
        median = X[col].median()
        if pd.isna(median):
            median = 0.0
        X[col] = X[col].fillna(median)

    # Guarantee no NaN remain
    assert not X.isna().any().any(), "impute_matrix: NaN remain after fillna"
    return X.to_numpy(dtype=float)


# ============================================================
# AHP — Analytic Hierarchy Process
# ============================================================

def ahp_weights(matrix: Array) -> Weights:
    """
    Compute priority vector from a Saaty pairwise comparison matrix.

    Parameters
    ----------
    matrix : Array, shape (n, n)
        Reciprocal matrix with a_ij = 1/a_ji, diagonal = 1.

    Returns
    -------
    Weights
        Normalized principal eigenvector (approx via column-normalized mean).

    Raises
    ------
    ValueError
        If matrix is not square or consistency ratio > 0.1.
    """
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Saaty matrix must be square")

    n = matrix.shape[0]

    # Principal eigenvector approximation (column-normalized mean)
    col_sums = matrix.sum(axis=0)
    norm = matrix / col_sums
    weights = norm.mean(axis=1)
    weights = weights / weights.sum()

    # Consistency check
    lam = (matrix @ weights / weights).mean()
    ci = (lam - n) / (n - 1) if n > 1 else 0.0

    # Random Index (Saaty)
    RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}
    ri = RI.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0

    if cr > 0.1:
        raise ValueError(f"AHP consistency ratio {cr:.3f} exceeds 0.1 threshold")

    return weights


def default_ahp_weights() -> Weights:
    """
    Default 4-criterion Saaty matrix for scouting:
    [NPxG, SoT, xA, MarketValue]
    """
    matrix = np.array([
        [1,   3,   3,   2],   # NPxG
        [1/3, 1,   1,   1/2], # SoT
        [1/3, 1,   1,   1/2], # xA
        [1/2, 2,   2,   1],   # MarketValue (cost)
    ], dtype=float)
    return ahp_weights(matrix)


# ============================================================
# TOPSIS — Technique for Order Preference by Similarity to Ideal Solution
# ============================================================

def topsis(
    X: Array,
    weights: Weights,
    benefits: Benefits,
    normalize: str = "vector",
) -> Scores:
    """
    TOPSIS scoring.

    Parameters
    ----------
    X : Array, shape (m, n)
        Decision matrix (m alternatives, n criteria). Must be numeric, no NaN.
    weights : Weights, shape (n,)
        Non-negative weights summing to 1.
    benefits : Benefits, shape (n,)
        True for benefit criteria (maximize), False for cost (minimize).
    normalize : {"vector", "minmax"}
        Normalization method. "vector" = vector normalization (classic TOPSIS).
        "minmax" = min-max to [0,1] per criterion.

    Returns
    -------
    Scores
        Closeness coefficient in [0, 1]. Higher = better.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    benefits = np.asarray(benefits, dtype=bool)

    m, n = X.shape
    if weights.shape != (n,) or benefits.shape != (n,):
        raise ValueError("weights/benefits must match n_criteria")

    # Normalization
    if normalize == "vector":
        denom = np.sqrt((X ** 2).sum(axis=0))
        denom[denom == 0] = 1.0
        R = X / denom
    elif normalize == "minmax":
        lo = X.min(axis=0)
        hi = X.max(axis=0)
        rng = hi - lo
        rng[rng == 0] = 1.0
        R = (X - lo) / rng
    else:
        raise ValueError(f"Unknown normalize method: {normalize}")

    # Weighted normalized matrix
    V = R * weights

    # Ideal best / worst
    ideal_best = np.where(benefits, V.max(axis=0), V.min(axis=0))
    ideal_worst = np.where(benefits, V.min(axis=0), V.max(axis=0))

    # Euclidean distances
    d_best = np.sqrt(((V - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((V - ideal_worst) ** 2).sum(axis=1))

    denom = d_best + d_worst
    scores = np.full(m, 0.5, dtype=float)
    mask = denom > 0
    scores[mask] = d_worst[mask] / denom[mask]

    return scores


# ============================================================
# MABAC — Multi-Attributive Border Approximation Area
# ============================================================

def mabac(
    X: Array,
    weights: Weights,
    benefits: Benefits,
) -> Scores:
    """
    MABAC scoring.

    Parameters
    ----------
    X : Array, shape (m, n)
        Decision matrix. Must be numeric, no NaN.
    weights : Weights, shape (n,)
        Non-negative weights summing to 1.
    benefits : Benefits, shape (n,)
        True for benefit, False for cost.

    Returns
    -------
    Scores
        Distance to border approximation area. Higher = better.
        Range is unbounded but typically centered near 0.
    """
    X = np.asarray(X, dtype=float)
    weights = np.asarray(weights, dtype=float)
    benefits = np.asarray(benefits, dtype=bool)

    m, n = X.shape
    if weights.shape != (n,) or benefits.shape != (n,):
        raise ValueError("weights/benefits must match n_criteria")

    # Min-max normalization to [0,1]
    N = np.zeros_like(X)
    for j, benefit in enumerate(benefits):
        lo = np.nanmin(X[:, j])
        hi = np.nanmax(X[:, j])
        if np.isclose(lo, hi):
            N[:, j] = 0.5
        elif benefit:
            N[:, j] = (X[:, j] - lo) / (hi - lo)
        else:
            N[:, j] = (hi - X[:, j]) / (hi - lo)

    # Weighted matrix + weights (MABAC formula)
    V = N * weights + weights
    V = np.clip(V, 1e-12, None)  # avoid log(0)

    # Border approximation area (geometric mean per criterion)
    log_V = np.log(V)
    frontier = np.exp(log_V.mean(axis=0))

    # Distance to frontier
    Q = V - frontier
    scores = Q.sum(axis=1)

    return scores


# ============================================================
# MEREC — Method based on the Removal Effects of Criteria
# ============================================================

def merec(
    X: Array,
    benefits: Benefits | None = None,
    eps: float = 1e-12,
) -> tuple[Scores, Weights]:
    """
    MEREC objective weights + scores.

    Parameters
    ----------
    X : Array, shape (m, n)
        Decision matrix. Values must be > 0 (strictly positive).
    benefits : Benefits, shape (n,), optional
        True for benefit, False for cost. Defaults to all True.
    eps : float
        Small constant added before log to avoid log(1)=0 issues.

    Returns
    -------
    scores : Scores
        Final MEREC scores in [0, 1] (min-max normalized weighted sum).
    weights : Weights
        Objective weights derived from criterion removal effects.
    """
    X = np.asarray(X, dtype=float)

    if benefits is None:
        benefits = np.ones(X.shape[1], dtype=bool)
    else:
        benefits = np.asarray(benefits, dtype=bool)

    m, n = X.shape
    if benefits.shape != (n,):
        raise ValueError("benefits must match n_criteria")

    # Ensure strictly positive
    X_safe = np.maximum(X, eps)

    # Step 1: Normalization specific to MEREC
    Z = np.zeros_like(X_safe)
    for j, benefit in enumerate(benefits):
        col = X_safe[:, j]
        if benefit:
            Z[:, j] = np.min(col) / col
        else:
            Z[:, j] = col / np.max(col)

    Z = np.clip(Z, eps, 1.0)

    # Step 2: Overall performance S_i
    log_Z = np.log(Z + eps)
    S = np.log(1.0 + np.abs(log_Z).sum(axis=1) / n)

    # Step 3: Removal effect per criterion
    effects = np.zeros(n)
    for j in range(n):
        Z_without = np.delete(Z, j, axis=1)
        log_Zw = np.log(Z_without + eps)
        S_without = np.log(1.0 + np.abs(log_Zw).sum(axis=1) / (n - 1))
        effects[j] = np.sum(np.abs(S_without - S))

    # Step 4: Objective weights
    if effects.sum() <= eps:
        weights = np.full(n, 1.0 / n)
    else:
        weights = effects / effects.sum()

    # Step 5: Final score via min-max normalization + weights
    N = np.zeros_like(X_safe)
    for j, benefit in enumerate(benefits):
        lo = np.min(X_safe[:, j])
        hi = np.max(X_safe[:, j])
        if np.isclose(lo, hi):
            N[:, j] = 0.5
        elif benefit:
            N[:, j] = (X_safe[:, j] - lo) / (hi - lo)
        else:
            N[:, j] = (hi - X_safe[:, j]) / (hi - lo)

    scores = N @ weights
    return scores, weights


# ============================================================
# HIGH-LEVEL PIPELINE HELPER
# ============================================================

def rank_frame_from_matrix(
    X: Array,
    weights: Weights | None = None,
    benefits: Benefits | None = None,
    criteria_names: list[str] | None = None,
) -> pd.DataFrame:
    """
    Convenience: run all three methods on a decision matrix.

    Returns DataFrame with columns: TOPSIS, MABAC, MEREC, Rank_TOPSIS, ...
    """
    if benefits is None:
        benefits = np.ones(X.shape[1], dtype=bool)
    if weights is None:
        weights = default_ahp_weights()

    scores_topsis = topsis(X, weights, benefits)
    scores_mabac = mabac(X, weights, benefits)
    scores_merec, w_merec = merec(X, benefits)

    df = pd.DataFrame({
        "TOPSIS": scores_topsis,
        "MABAC": scores_mabac,
        "MEREC": scores_merec,
    })

    if criteria_names:
        df.index = criteria_names

    for method in ["TOPSIS", "MABAC", "MEREC"]:
        col = method
        ordered = df.sort_values(col, ascending=False).index
        ranks = pd.Series(np.arange(1, len(ordered) + 1), index=ordered)
        df[f"Rank_{method}"] = ranks.reindex(df.index).astype(int)

    return df, w_merec