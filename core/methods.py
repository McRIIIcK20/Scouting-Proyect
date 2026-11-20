"""
Métodos MCDM adicionales para Scouting Lab.

Implementaciones numéricas puras (sin Streamlit, sin I/O), en el mismo estilo
que ``core/models.py``. Cada método sigue los pasos del ramo
"Toma de Decisiones (MCDM)" — UCN — y, cuando el formulario no lo define,
la formulación original publicada (ver referencias en cada docstring).

Convenciones
------------
X        : matriz de decisión (m alternativas × n criterios), numérica, sin NaN.
weights  : vector (n,) no negativo que suma 1.
benefits : vector (n,) bool. True = beneficio (maximizar), False = costo.

Métodos de PESOS  : entropy, swara, itara, bwm, anp
Métodos de RANKING: saw, copras, vikor, moora, multimoora, waspas, cocoso,
                    mairca, dnmarcos, rafsi, grey, fuzzy_topsis, three_way
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from core.models import mabac, merec, topsis

Array = np.ndarray
EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def _check(X: Array, weights: Array | None, benefits: Array) -> tuple[Array, Array | None, Array]:
    X = np.asarray(X, dtype=float)
    benefits = np.asarray(benefits, dtype=bool)
    if X.ndim != 2:
        raise ValueError("X debe ser una matriz 2D (m, n)")
    n = X.shape[1]
    if benefits.shape != (n,):
        raise ValueError("benefits debe tener largo n_criterios")
    if weights is not None:
        weights = np.asarray(weights, dtype=float)
        if weights.shape != (n,):
            raise ValueError("weights debe tener largo n_criterios")
    return X, weights, benefits


def minmax_normalize(X: Array, benefits: Array) -> Array:
    """(+) (x−min)/(max−min) · (−) (max−x)/(max−min). Columna constante → 0.5."""
    lo, hi = X.min(axis=0), X.max(axis=0)
    rng = hi - lo
    const = np.isclose(rng, 0)
    rng = np.where(const, 1.0, rng)
    N = np.where(benefits, (X - lo) / rng, (hi - X) / rng)
    N[:, const] = 0.5
    return N


def linear_max_normalize(X: Array, benefits: Array) -> Array:
    """(+) x/max(x) · (−) min(x)/x  (normalización lineal del ramo)."""
    Xs = np.maximum(X, EPS)
    return np.where(benefits, Xs / Xs.max(axis=0), Xs.min(axis=0) / Xs)


def vector_normalize(X: Array) -> Array:
    """x / sqrt(Σ x²)."""
    denom = np.sqrt((X ** 2).sum(axis=0))
    denom[denom == 0] = 1.0
    return X / denom


def positive_shift(X: Array) -> Array:
    """Desplaza columnas con valores ≤ 0 para que sean estrictamente positivas."""
    X = X.copy()
    col_min = X.min(axis=0)
    shift = np.where(col_min <= 0, -col_min + 1e-6 * (np.abs(X).max(axis=0) + 1), 0.0)
    return X + shift


def ranks_from_scores(scores: Array, higher_is_better: bool = True, ties: str = "first") -> Array:
    """
    Ranking 1..m (1 = mejor).

    ties="first" → empates en orden estable (permutación 1..m).
    ties="min"   → empates comparten el menor ranking (igual que RANK de Excel).
    """
    scores = np.asarray(scores, dtype=float)
    if ties == "min":
        s = scores if higher_is_better else -scores
        return 1 + (s[None, :] > s[:, None]).sum(axis=1)
    order = np.argsort(-scores if higher_is_better else scores, kind="stable")
    ranks = np.empty(len(scores), dtype=int)
    ranks[order] = np.arange(1, len(scores) + 1)
    return ranks


# ============================================================
# MÉTODOS DE PESOS
# ============================================================

SAATY_RI = {1: 0.0, 2: 0.0, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}


def ahp_consistency(matrix: Array) -> dict[str, float]:
    """λmax, CI y CR de una matriz de Saaty (mismo cálculo que el formulario)."""
    A = np.asarray(matrix, dtype=float)
    n = A.shape[0]
    w = (A / A.sum(axis=0)).mean(axis=1)
    lam = float((A @ w / w).mean())
    ci = (lam - n) / (n - 1) if n > 1 else 0.0
    ri = SAATY_RI.get(n, 1.49)
    return {"lambda_max": lam, "CI": ci, "CR": ci / ri if ri > 0 else 0.0}


def entropy_weights(X: Array) -> Array:
    """
    Entropía de Shannon (pesos objetivos).

    pᵢⱼ = xᵢⱼ/Σᵢxᵢⱼ ;  Eⱼ = −(1/ln m) Σᵢ pᵢⱼ ln pᵢⱼ ;  dⱼ = 1−Eⱼ ;  wⱼ = dⱼ/Σdⱼ
    """
    X = np.asarray(X, dtype=float)
    m, n = X.shape
    if m < 2:
        return np.full(n, 1.0 / n)
    Xp = positive_shift(X)
    P = Xp / Xp.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        plogp = np.where(P > 0, P * np.log(P), 0.0)
    E = -plogp.sum(axis=0) / np.log(m)
    d = 1.0 - E
    if d.sum() <= EPS:
        return np.full(n, 1.0 / n)
    return d / d.sum()


def swara_weights(order: list[int], s: list[float]) -> Array:
    """
    SWARA (Step-wise Weight Assessment Ratio Analysis).

    Parameters
    ----------
    order : índices de criterios ordenados del más al menos importante.
    s     : importancia comparativa sⱼ de cada criterio respecto al anterior
            (mismo orden que ``order``; el primero es 0).

    kⱼ = sⱼ + 1 ;  q₁ = 1, qⱼ = qⱼ₋₁/kⱼ ;  wⱼ = qⱼ/Σq
    Devuelve los pesos en el orden ORIGINAL de los criterios.
    """
    s = np.asarray(s, dtype=float)
    if len(order) != len(s) or sorted(order) != list(range(len(order))):
        raise ValueError("order debe ser una permutación de 0..n-1 del mismo largo que s")
    if np.any(s < 0):
        raise ValueError("sⱼ no puede ser negativo")
    k = s + 1.0
    k[0] = 1.0
    q = np.cumprod(1.0 / k)
    w_sorted = q / q.sum()
    w = np.empty_like(w_sorted)
    w[np.asarray(order)] = w_sorted
    return w


def itara_weights(
    X: Array,
    benefits: Array,
    thresholds: Array | float = 0.01,
    normalization: str = "max",
    aggregation: str = "euclidean",
) -> Array:
    """
    ITARA (Indifference Threshold-based Attribute Ratio Analysis, Hatefi 2019).

    1) Normalizar (formulario: (+) x/max, (−) min/x ; Excel del ramo: x/Σx).
    2) Ordenar cada criterio y tomar diferencias entre valores adyacentes.
    3) δ = max(0, diferencia − umbral de indiferencia normalizado).
    4) vⱼ = sqrt(Σ δ²)  (o Σ δ con aggregation="sum") ;  wⱼ = vⱼ/Σv

    ``thresholds`` está en unidades NORMALIZADAS (escalar o vector por criterio).
    Si ningún criterio supera el umbral, devuelve pesos iguales.
    """
    X, _, benefits = _check(X, None, benefits)
    m, n = X.shape
    if normalization == "max":
        N = linear_max_normalize(X, benefits)
    elif normalization == "sum":
        Xp = positive_shift(X)
        N = Xp / Xp.sum(axis=0)
    else:
        raise ValueError("normalization debe ser 'max' o 'sum'")
    thr = np.broadcast_to(np.asarray(thresholds, dtype=float), (n,))
    S = np.sort(N, axis=0)
    diffs = np.diff(S, axis=0)
    delta = np.maximum(diffs - thr, 0.0)
    if aggregation == "euclidean":
        v = np.sqrt((delta ** 2).sum(axis=0))
    elif aggregation == "sum":
        v = delta.sum(axis=0)
    else:
        raise ValueError("aggregation debe ser 'euclidean' o 'sum'")
    if v.sum() <= EPS:
        return np.full(n, 1.0 / n)
    return v / v.sum()


# Índice de consistencia de BWM (Rezaei, 2015) según a_BW
BWM_CI = {1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73, 8: 4.47, 9: 5.23}


@dataclass
class BWMResult:
    weights: Array
    xi: float | None
    consistency_ratio: float | None


def bwm_weights(best_to_others: list[float], others_to_worst: list[float] | None = None) -> BWMResult:
    """
    BWM (Best-Worst Method), versión del formulario del ramo.

    ASⱼ : preferencia del MEJOR criterio sobre cada criterio j (1–9, el mejor = 1).
    AWⱼ = max(AS)/ASⱼ ;  wⱼ = AWⱼ/Σ AW

    Si se entrega ``others_to_worst`` (preferencia de cada criterio sobre el PEOR),
    se reporta además ξ = max |w_B/w_j − a_Bj|, |w_j/w_W − a_jW| y CR = ξ/CI(a_BW)
    (Rezaei, 2015).
    """
    a_b = np.asarray(best_to_others, dtype=float)
    if np.any(a_b < 1) or np.any(a_b > 9):
        raise ValueError("Las comparaciones BWM deben estar en la escala 1–9")
    aw = a_b.max() / a_b
    w = aw / aw.sum()

    xi = cr = None
    if others_to_worst is not None:
        a_w = np.asarray(others_to_worst, dtype=float)
        if a_w.shape != a_b.shape:
            raise ValueError("best_to_others y others_to_worst deben tener el mismo largo")
        b = int(np.argmin(a_b))
        wi = int(np.argmin(a_w))
        xi = float(max(np.max(np.abs(w[b] / w - a_b)), np.max(np.abs(w / w[wi] - a_w))))
        a_bw = int(round(a_b[wi]))
        ci = BWM_CI.get(a_bw, 5.23)
        cr = xi / ci if ci > 0 else 0.0
    return BWMResult(weights=w, xi=xi, consistency_ratio=cr)


@dataclass
class ANPResult:
    weights: Array
    unweighted: Array
    weighted: Array
    limit: Array
    iterations: int


def anp_weights(influence: Array, tol: float = 1e-10, max_iter: int = 200) -> ANPResult:
    """
    ANP (Analytic Network Process) — red de un solo clúster de criterios.

    ``influence[i, j]`` = cuánto influye/importa el criterio i desde el punto de
    vista del criterio j (columna = matriz de influencia de ese criterio).

    SW  : supermatriz no ponderada = columnas con los vectores de prioridad
    SW* : supermatriz ponderada     = SW / Suma j(SW)  (columna-estocástica)
    SW& : supermatriz límite        = SW* elevada hasta convergencia
    Pesos ANP = primera columna de SW&.
    """
    A = np.asarray(influence, dtype=float)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("La matriz de influencia debe ser cuadrada")
    if np.any(A < 0):
        raise ValueError("La matriz de influencia no admite valores negativos")
    col = A.sum(axis=0)
    if np.any(col <= 0):
        raise ValueError("Cada criterio debe tener al menos una influencia positiva")
    SW = A / col
    SWw = SW / SW.sum(axis=0)
    L = SWw.copy()
    it = 0
    for it in range(1, max_iter + 1):
        L_next = L @ L
        if np.max(np.abs(L_next - L)) < tol:
            L = L_next
            break
        L = L_next
    w = L[:, 0] / L[:, 0].sum()
    return ANPResult(weights=w, unweighted=SW, weighted=SWw, limit=L, iterations=it)


# ============================================================
# MÉTODOS DE RANKING (mayor score = mejor, salvo que se indique)
# ============================================================

def saw(X: Array, weights: Array, benefits: Array) -> Array:
    """SAW: rᵢⱼ = x/max (+) ó min/x (−) ;  Sᵢ = Σ wⱼ rᵢⱼ."""
    X, weights, benefits = _check(X, weights, benefits)
    return linear_max_normalize(X, benefits) @ weights


def copras(X: Array, weights: Array, benefits: Array) -> Array:
    """
    COPRAS: pᵢⱼ = x/Σx ; vᵢⱼ = w·p ; S⁺, S⁻ ;
    Qᵢ = S⁺ᵢ + (S⁻min · ΣS⁻) / (S⁻ᵢ · Σ(S⁻min/S⁻)).
    """
    X, weights, benefits = _check(X, weights, benefits)
    Xp = positive_shift(X)
    V = Xp / Xp.sum(axis=0) * weights
    s_plus = V[:, benefits].sum(axis=1)
    if (~benefits).sum() == 0:
        return s_plus
    s_minus = np.maximum(V[:, ~benefits].sum(axis=1), EPS)
    s_min = s_minus.min()
    return s_plus + (s_min * s_minus.sum()) / (s_minus * (s_min / s_minus).sum())


@dataclass
class VIKORResult:
    Q: Array
    S: Array
    R: Array


def vikor(X: Array, weights: Array, benefits: Array, v: float = 0.5) -> VIKORResult:
    """VIKOR: S (utilidad grupal), R (arrepentimiento), Q (MENOR = mejor)."""
    X, weights, benefits = _check(X, weights, benefits)
    f_best = np.where(benefits, X.max(axis=0), X.min(axis=0))
    f_worst = np.where(benefits, X.min(axis=0), X.max(axis=0))
    rng = f_best - f_worst
    rng = np.where(np.isclose(rng, 0), 1.0, rng)
    D = weights * (f_best - X) / rng
    S, R = D.sum(axis=1), D.max(axis=1)
    s_rng = S.max() - S.min() or 1.0
    r_rng = R.max() - R.min() or 1.0
    Q = v * (S - S.min()) / s_rng + (1 - v) * (R - R.min()) / r_rng
    return VIKORResult(Q=Q, S=S, R=R)


def moora(X: Array, weights: Array, benefits: Array) -> Array:
    """MOORA (sistema de razones): yᵢ = Σ_(+) w·r − Σ_(−) w·r con r vectorial."""
    X, weights, benefits = _check(X, weights, benefits)
    V = vector_normalize(X) * weights
    return V[:, benefits].sum(axis=1) - V[:, ~benefits].sum(axis=1)


@dataclass
class MultiMOORAResult:
    ratio_system: Array        # mayor = mejor
    reference_point: Array     # menor = mejor (desviación máxima Tchebycheff)
    full_multiplicative: Array # mayor = mejor
    rank_sum: Array            # menor = mejor (suma de rankings)
    score: Array               # mayor = mejor (dominancia normalizada 0–1)


def multimoora(X: Array, weights: Array, benefits: Array, weighted_fmf: bool = True) -> MultiMOORAResult:
    """
    MULTIMOORA = MOORA (RS) + Punto de Referencia (RP) + Forma Multiplicativa Completa (FMF).

    FMF: Uᵢ = Π_(+) x^w / Π_(−) x^w  (``weighted_fmf=False`` → productos simples,
    como en el Excel del ramo). Ranking final por suma de rankings.
    """
    X, weights, benefits = _check(X, weights, benefits)
    V = vector_normalize(X) * weights
    rs = V[:, benefits].sum(axis=1) - V[:, ~benefits].sum(axis=1)
    ref = np.where(benefits, V.max(axis=0), V.min(axis=0))
    rp = np.abs(ref - V).max(axis=1)
    Xs = np.maximum(X, EPS)
    exps = weights if weighted_fmf else np.ones_like(weights)
    logs = np.log(Xs) * exps
    fmf = np.exp(logs[:, benefits].sum(axis=1) - logs[:, ~benefits].sum(axis=1))
    rank_sum = (
        ranks_from_scores(rs, True, ties="min")
        + ranks_from_scores(rp, False, ties="min")
        + ranks_from_scores(fmf, True, ties="min")
    ).astype(float)
    m = len(rank_sum)
    score = 1.0 - (rank_sum - 3) / max(3 * (m - 1), 1)
    return MultiMOORAResult(rs, rp, fmf, rank_sum, score)


def waspas(X: Array, weights: Array, benefits: Array, lam: float = 0.5) -> Array:
    """WASPAS: Q = λ·Σ w·n + (1−λ)·Π n^w  con n = x/max (+), min/x (−)."""
    X, weights, benefits = _check(X, weights, benefits)
    N = linear_max_normalize(X, benefits)
    q1 = N @ weights
    q2 = np.exp(np.log(np.maximum(N, EPS)) @ weights)
    return lam * q1 + (1 - lam) * q2


def cocoso(X: Array, weights: Array, benefits: Array, lam: float = 0.5) -> Array:
    """
    CoCoSo (Combined Compromise Solution, Yazdani et al. 2019).

    N min-max ; Sᵢ = Σ w·N ; Pᵢ = Σ N^w ;
    KA = (S+P)/Σ(S+P) ; KB = S/minS + P/minP ;
    KC = (λS+(1−λ)P)/(λ maxS+(1−λ) maxP) ;
    K = (KA·KB·KC)^(1/3) + (KA+KB+KC)/3
    """
    X, weights, benefits = _check(X, weights, benefits)
    N = minmax_normalize(X, benefits)
    S = N @ weights
    P = (np.power(N, weights)).sum(axis=1)

    def _pos_min(a: Array) -> float:
        pos = a[a > EPS]
        return float(pos.min()) if pos.size else 1.0

    ka = (S + P) / (S + P).sum()
    kb = S / _pos_min(S) + P / _pos_min(P)
    kc = (lam * S + (1 - lam) * P) / (lam * S.max() + (1 - lam) * P.max())
    return np.cbrt(ka * kb * kc) + (ka + kb + kc) / 3.0


def mairca(X: Array, weights: Array, benefits: Array) -> Array:
    """
    MAIRCA: PAⱼ = 1/m ; TPⱼ = wⱼ·PA ; TRᵢⱼ = TPⱼ·N (min-max) ;
    Gᵢⱼ = TPⱼ − TRᵢⱼ ; Qᵢ = Σ G  →  MENOR brecha = mejor.
    """
    X, weights, benefits = _check(X, weights, benefits)
    m = X.shape[0]
    tp = weights / m
    tr = tp * minmax_normalize(X, benefits)
    return (tp - tr).sum(axis=1)


def dnmarcos(X: Array, weights: Array, benefits: Array, vector_cost: str = "invert") -> Array:
    """
    DN-MARCOS (MARCOS con doble normalización), formulario del ramo.

    N1 lineal: x/max (+), min/x (−) ; N2 vectorial: x/sqrt(Σx²) ;
    N3 = (N1+N2)/2 ; R = N3·w ; S⁺ = max columna ;  Qᵢ = Σⱼ Rᵢⱼ / Σⱼ S⁺ⱼ

    ``vector_cost="invert"`` usa 1 − x/‖x‖ en criterios de costo (DNMA, Liao & Wu 2020)
    para que la parte vectorial también penalice el costo. ``"raw"`` replica el Excel
    del ramo (x/‖x‖ para todos los criterios).
    """
    X, weights, benefits = _check(X, weights, benefits)
    n1 = linear_max_normalize(X, benefits)
    n2 = vector_normalize(X)
    if vector_cost == "invert":
        n2 = np.where(benefits, n2, 1.0 - n2)
    elif vector_cost != "raw":
        raise ValueError("vector_cost debe ser 'invert' o 'raw'")
    R = (n1 + n2) / 2.0 * weights
    return R.sum(axis=1) / R.max(axis=0).sum()


def rafsi(
    X: Array,
    weights: Array,
    benefits: Array,
    ideal: Array | None = None,
    anti_ideal: Array | None = None,
    margin: float = 0.0,
    n2k: float | None = None,
) -> Array:
    """
    RAFSI (Žižović et al., 2020).

    Mapea cada criterio desde [anti-ideal, ideal] a [1, n2k] (n2k = N° alternativas
    por defecto, como en el formulario), luego:
      A = (1+n2k)/2 (media aritmética) ; H = 2/(1+1/n2k) (media armónica)
      (+) ŝ = f/(2A) ;  (−) ŝ = H/(2f) ;  Vᵢ = Σ w·ŝ
    Si no se entregan ideal/anti-ideal se usan los extremos de los datos ± ``margin``·rango.
    """
    X, weights, benefits = _check(X, weights, benefits)
    m = X.shape[0]
    n2k = float(n2k if n2k is not None else max(m, 2))
    lo, hi = X.min(axis=0), X.max(axis=0)
    pad = margin * (hi - lo)
    if ideal is None:
        ideal = np.where(benefits, hi + pad, lo - pad)
    if anti_ideal is None:
        anti_ideal = np.where(benefits, lo - pad, hi + pad)
    ideal = np.asarray(ideal, dtype=float)
    anti_ideal = np.asarray(anti_ideal, dtype=float)
    # Intervalo creciente [a, b] mapeado a [1, n2k]
    a = np.minimum(ideal, anti_ideal)
    b = np.maximum(ideal, anti_ideal)
    span = np.where(np.isclose(b - a, 0), 1.0, b - a)
    f = 1.0 + (n2k - 1.0) * (np.clip(X, a, b) - a) / span
    A = (1.0 + n2k) / 2.0
    H = 2.0 / (1.0 + 1.0 / n2k)
    s = np.where(benefits, f / (2 * A), H / (2 * f))
    return s @ weights


def grey(X: Array, weights: Array, benefits: Array, zeta: float = 0.5) -> Array:
    """
    Grey (Análisis Relacional Gris, 'Grey TOPSIS' en el formulario).

    N min-max ; Sᵢⱼ = 1 − N ;  Eᵢⱼ = (minS + ζ·maxS)/(Sᵢⱼ + ζ·maxS) ;  GRG = Σ w·E
    """
    X, weights, benefits = _check(X, weights, benefits)
    S = 1.0 - minmax_normalize(X, benefits)
    s_min, s_max = S.min(), S.max()
    if s_max <= EPS:
        return np.full(X.shape[0], float(weights.sum()))
    E = (s_min + zeta * s_max) / (S + zeta * s_max)
    return E @ weights


def fuzzy_topsis(X: Array, weights: Array, benefits: Array, spread: float = 0.10) -> Array:
    """
    Fuzzy TOPSIS (Chen, 2000) con números triangulares (l, m, u).

    Como los datos son nítidos, cada valor se fuzzifica como
    (x·(1−δ), x, x·(1+δ)), con δ = ``spread`` (incertidumbre de la métrica).
    (+) r = (l/u*, m/u*, u/u*) ;  (−) r = (l⁻/u, l⁻/m, l⁻/l)
    v = r·w ; A* = max, A⁻ = min por criterio ;
    d(ã,b̃) = sqrt(⅓ Σ (aₖ−bₖ)²) (método del vértice) ;  CC = D⁻/(D*+D⁻)
    """
    X, weights, benefits = _check(X, weights, benefits)
    Xp = positive_shift(X)
    L, M, U = Xp * (1 - spread), Xp, Xp * (1 + spread)
    u_star = U.max(axis=0)
    l_minus = np.maximum(L.min(axis=0), EPS)
    rl = np.where(benefits, L / u_star, l_minus / np.maximum(U, EPS))
    rm = np.where(benefits, M / u_star, l_minus / np.maximum(M, EPS))
    ru = np.where(benefits, U / u_star, l_minus / np.maximum(L, EPS))
    V = np.stack([rl, rm, ru], axis=-1) * weights[None, :, None]  # (m, n, 3)
    a_star = V[..., 2].max(axis=0)
    a_minus = V[..., 0].min(axis=0)
    d_star = np.sqrt(((V - a_star[None, :, None]) ** 2).mean(axis=-1)).sum(axis=1)
    d_minus = np.sqrt(((V - a_minus[None, :, None]) ** 2).mean(axis=-1)).sum(axis=1)
    denom = d_star + d_minus
    return np.where(denom > 0, d_minus / np.where(denom > 0, denom, 1.0), 0.5)


@dataclass
class ThreeWayResult:
    pc: Array          # probabilidad compuesta (mayor = mejor)
    decision: Array    # "Aceptar" / "Frontera" / "Rechazar"


def three_way(
    X: Array,
    weights: Array,
    benefits: Array,
    alpha: float = 0.6,
    beta: float = 0.4,
) -> ThreeWayResult:
    """
    Three-Way Decision (Yao): PCᵢ = Σ w·N (min-max).
    PC ≥ α → Aceptar ; PC ≤ β → Rechazar ; β < PC < α → Frontera (diferir).
    """
    X, weights, benefits = _check(X, weights, benefits)
    if not 0 <= beta < alpha <= 1:
        raise ValueError("Se requiere 0 ≤ β < α ≤ 1")
    pc = minmax_normalize(X, benefits) @ weights
    decision = np.where(pc >= alpha, "Aceptar", np.where(pc <= beta, "Rechazar", "Frontera"))
    return ThreeWayResult(pc=pc, decision=decision)


# ============================================================
# REGISTRO DE MÉTODOS DE RANKING
# ============================================================

@dataclass(frozen=True)
class RankingMethod:
    key: str
    label: str
    higher_is_better: bool
    description: str
    run: Callable[[Array, Array, Array], Array]
    source: str = "Ramo MCDM"


RANKING_METHODS: dict[str, RankingMethod] = {
    m.key: m
    for m in [
        RankingMethod("TOPSIS", "TOPSIS", True,
                      "Cercanía relativa al jugador ideal y lejanía del anti-ideal.",
                      lambda X, w, b: topsis(X, w, b, normalize="vector")),
        RankingMethod("MABAC", "MABAC", True,
                      "Distancia al área de aproximación fronteriza (media geométrica).",
                      mabac),
        RankingMethod("MEREC", "MEREC", True,
                      "Pesos objetivos por efecto de remover cada criterio; no usa los pesos elegidos.",
                      lambda X, w, b: merec(X, b)[0]),
        RankingMethod("SAW", "SAW", True,
                      "Suma ponderada simple con normalización lineal.", saw),
        RankingMethod("COPRAS", "COPRAS", True,
                      "Separa la contribución de beneficios (S⁺) y costos (S⁻).", copras),
        RankingMethod("VIKOR", "VIKOR", False,
                      "Solución de compromiso entre utilidad grupal (S) y arrepentimiento (R). Q menor = mejor.",
                      lambda X, w, b: vikor(X, w, b).Q),
        RankingMethod("MOORA", "MOORA", True,
                      "Sistema de razones: beneficios menos costos normalizados.", moora),
        RankingMethod("MULTIMOORA", "MULTIMOORA", True,
                      "Consenso de MOORA, punto de referencia y forma multiplicativa (score de dominancia 0–1).",
                      lambda X, w, b: multimoora(X, w, b).score),
        RankingMethod("WASPAS", "WASPAS", True,
                      "Mezcla de suma ponderada y producto ponderado (λ = 0.5).", waspas),
        RankingMethod("CoCoSo", "CoCoSo", True,
                      "Compromiso combinado de tres estrategias de agregación.", cocoso),
        RankingMethod("MAIRCA", "MAIRCA", False,
                      "Brecha entre preferencia teórica y real. Q menor = mejor.", mairca),
        RankingMethod("DNMARCOS", "DN-MARCOS", True,
                      "MARCOS con doble normalización (lineal + vectorial).", dnmarcos),
        RankingMethod("RAFSI", "RAFSI", True,
                      "Mapeo funcional a un intervalo único; robusto a la inversión de ranking.",
                      rafsi, source="Formulario + Žižović et al. (2020)"),
        RankingMethod("GREY", "Grey (GRA)", True,
                      "Grado de relación gris con el jugador de referencia (ζ = 0.5).",
                      grey, source="Formulario + Deng (1989)"),
        RankingMethod("FUZZY_TOPSIS", "Fuzzy TOPSIS", True,
                      "TOPSIS con números difusos triangulares (±10 % de incertidumbre).",
                      fuzzy_topsis, source="Formulario + Chen (2000)"),
        RankingMethod("THREE_WAY", "Three-Way", True,
                      "Probabilidad compuesta y decisión Aceptar / Frontera / Rechazar.",
                      lambda X, w, b: three_way(X, w, b).pc, source="Formulario + Yao (2010)"),
    ]
}


WEIGHT_METHODS = ["Manual (sliders)", "AHP", "Entropía", "MEREC", "SWARA", "ITARA", "BWM", "ANP"]


def run_all_rankings(X: Array, weights: Array, benefits: Array) -> dict[str, tuple[Array, Array]]:
    """Ejecuta todos los métodos registrados → {key: (scores, ranks)}."""
    out: dict[str, tuple[Array, Array]] = {}
    for key, meth in RANKING_METHODS.items():
        scores = np.asarray(meth.run(X, weights, benefits), dtype=float)
        out[key] = (scores, ranks_from_scores(scores, meth.higher_is_better))
    return out
