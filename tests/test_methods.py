"""
Tests de los métodos MCDM agregados (core/methods.py).

Los valores de referencia salen de las planillas del ramo
"Toma de Decisiones (MCDM)" (COCOSO.xlsx, Mairca.xlsx, DNMARCOS.xlsx,
SWARA_ITARA_MOORA.xlsx, Ejercicio_MOORA_y_MULTIMOORA.xlsx y
MCDM_Paso_a_Paso.xlsx), recalculadas con Excel/LibreOffice.

Run with: pytest tests/ -v
"""

import numpy as np
import pytest

from core.methods import (
    RANKING_METHODS,
    ahp_consistency,
    anp_weights,
    bwm_weights,
    cocoso,
    copras,
    dnmarcos,
    entropy_weights,
    fuzzy_topsis,
    grey,
    itara_weights,
    mairca,
    moora,
    multimoora,
    rafsi,
    ranks_from_scores,
    run_all_rankings,
    saw,
    swara_weights,
    three_way,
    vikor,
    waspas,
)
from core.models import mabac, merec, topsis


# ============================================================
# DATOS DE LAS PLANILLAS DEL RAMO
# ============================================================

# Aleaciones (COCOSO.xlsx / Mairca.xlsx) — todos beneficio
ALLOY_X = np.array([
    [191.7, 2.4, 62.1],
    [200.8, 8.3, 68.4],
    [219.7, 6.7, 66.9],
    [222.3, 7.9, 70.2],
])
ALLOY_W = np.array([0.40, 0.35, 0.25])
ALLOY_B = np.array([True, True, True])

# SUVs (MCDM_Paso_a_Paso.xlsx): Precio(−), Consumo(−), Seguridad, Potencia, Maletero
SUV_X = np.array([
    [30, 7.5, 8, 170, 580],
    [32, 8.0, 9, 190, 550],
    [28, 7.0, 7, 180, 600],
    [29, 7.2, 8, 165, 560],
    [27, 6.8, 9, 150, 570],
    [31, 7.8, 7, 175, 590],
], dtype=float)
SUV_W = np.array([0.35, 0.25, 0.20, 0.10, 0.10])
SUV_B = np.array([False, False, True, True, True])

# Celulares (SWARA_ITARA_MOORA.xlsx): Precio(−) + 4 beneficio
PHONE_X = np.array([
    [400, 20, 8, 128, 2.4],
    [350, 18, 9, 64, 2.0],
    [450, 24, 10, 256, 3.0],
    [300, 15, 7, 32, 1.8],
    [500, 30, 9, 128, 2.6],
    [320, 22, 6, 64, 1.6],
], dtype=float)
PHONE_B = np.array([False, True, True, True, True])
PHONE_IT = np.array([20, 2, 0.5, 32, 0.2])

# Evaluación de personal (Ejercicio_MOORA_y_MULTIMOORA.xlsx)
STAFF_X = np.array([
    [6, 1, 4, 4, 8, 10], [4, 6, 3, 5, 8, 7], [9, 9, 2, 7, 8, 9],
    [9, 2, 6, 10, 9, 10], [5, 4, 1, 4, 6, 1], [3, 4, 9, 2, 4, 4],
    [4, 8, 1, 2, 10, 10], [1, 5, 8, 4, 3, 8], [3, 1, 1, 5, 6, 6],
    [7, 9, 5, 2, 5, 10],
], dtype=float)
STAFF_B = np.array([True, True, True, True, False, False])


@pytest.fixture
def scouting_matrix() -> np.ndarray:
    return np.array([
        [0.50, 1.2, 0.15, 20_000_000],
        [0.70, 1.5, 0.20, 50_000_000],
        [0.30, 0.8, 0.10, 5_000_000],
        [0.60, 1.3, 0.18, 10_000_000],
        [0.45, 1.0, 0.00, 2_000_000],   # xA = 0 (caso real del dataset)
    ], dtype=float)


B4 = np.array([True, True, True, False])
W4 = np.array([0.45, 0.14, 0.14, 0.27])


# ============================================================
# REPRODUCCIÓN DE LAS PLANILLAS
# ============================================================

class TestCourseSpreadsheets:

    def test_cocoso(self):
        np.testing.assert_allclose(
            cocoso(ALLOY_X, ALLOY_W, ALLOY_B),
            [0.0, 1.82741506, 2.01537013, 2.32739422], atol=1e-7)

    def test_mairca(self):
        np.testing.assert_allclose(
            mairca(ALLOY_X, ALLOY_W, ALLOY_B),
            [0.25, 0.08415033, 0.05768851, 0.0059322], atol=1e-7)

    def test_dnmarcos_course_version(self):
        X = np.array([[8, 200, 10], [6, 150, 5], [9, 250, 8]], dtype=float)
        q = dnmarcos(X, np.array([.5, .3, .2]), np.array([True, False, False]), vector_cost="raw")
        np.testing.assert_allclose(q, [0.9005836, 0.81872882, 0.95666108], atol=1e-7)

    def test_swara(self):
        np.testing.assert_allclose(
            swara_weights([0, 1, 2, 3, 4], [0, .3, .5, .2, .2]),
            [0.32620818, 0.25092937, 0.16728625, 0.1394052, 0.116171], atol=1e-7)

    def test_swara_reorders_to_original_criteria(self):
        w = swara_weights([2, 0, 1], [0, 0.5, 0.5])
        assert w[2] > w[0] > w[1]
        assert np.isclose(w.sum(), 1)

    def test_itara_excel_version(self):
        w = itara_weights(PHONE_X, PHONE_B, PHONE_IT / PHONE_X.sum(axis=0), normalization="sum")
        np.testing.assert_allclose(
            w, [0.09238924, 0.1294688, 0.0826674, 0.60997367, 0.08550088], atol=1e-7)

    def test_moora(self):
        # En la planilla, MOORA usa los pesos SWARA de la misma hoja
        w = swara_weights([0, 1, 2, 3, 4], [0, .3, .5, .2, .2])
        np.testing.assert_allclose(
            moora(PHONE_X, w, PHONE_B),
            [0.12780472, 0.10817595, 0.21285394, 0.07689605, 0.15285864, 0.10388965], atol=1e-7)

    def test_multimoora(self):
        r = multimoora(STAFF_X, np.full(6, 1 / 6), STAFF_B, weighted_fmf=False)
        np.testing.assert_allclose(
            r.full_multiplicative,
            [1.2, 6.428571, 15.75, 12.0, 13.333333, 13.5, 0.64, 6.666667, 0.416667, 12.6],
            atol=1e-5)
        np.testing.assert_array_equal(r.rank_sum, [19, 16, 7, 8, 17, 11, 27, 15, 27, 14])

    def test_entropy(self):
        np.testing.assert_allclose(
            entropy_weights(SUV_X),
            [0.14376428, 0.14243567, 0.44756143, 0.22842494, 0.03781368], atol=1e-7)

    def test_merec_weights(self):
        np.testing.assert_allclose(
            merec(SUV_X, SUV_B)[1],
            [0.17545485, 0.17298786, 0.27342242, 0.2845108, 0.09362408], atol=1e-7)

    @pytest.mark.parametrize("func, expected", [
        (saw, [4, 5, 2, 3, 1, 6]),
        (topsis, [4, 5, 3, 2, 1, 6]),
        (copras, [4, 5, 2, 3, 1, 6]),
        (mabac, [4, 5, 2, 3, 1, 6]),
    ])
    def test_first_test_rankings(self, func, expected):
        np.testing.assert_array_equal(ranks_from_scores(func(SUV_X, SUV_W, SUV_B)), expected)

    def test_vikor_ranking(self):
        q = vikor(SUV_X, SUV_W, SUV_B).Q
        np.testing.assert_array_equal(ranks_from_scores(q, higher_is_better=False), [4, 6, 3, 2, 1, 5])


# ============================================================
# MÉTODOS DE PESOS
# ============================================================

class TestWeightMethods:

    def test_bwm_formulario(self):
        r = bwm_weights([1, 4, 4, 2], [4, 1, 1, 2])
        np.testing.assert_allclose(r.weights, [0.5, 0.125, 0.125, 0.25])
        assert r.consistency_ratio == pytest.approx(0.0)

    def test_bwm_inconsistent_has_positive_cr(self):
        r = bwm_weights([1, 3, 3, 2], [3, 1, 1, 2])
        assert r.consistency_ratio > 0

    def test_bwm_rejects_out_of_scale(self):
        with pytest.raises(ValueError):
            bwm_weights([1, 12, 3])

    def test_anp_limit_converges_and_columns_equal(self):
        A = np.array([[5, 4, 1, 5], [3, 3, 1, 2], [1, 1, 3, 2], [1, 1, 1, 3]], float)
        r = anp_weights(A)
        assert np.isclose(r.weights.sum(), 1)
        # todas las columnas de la supermatriz límite son iguales
        np.testing.assert_allclose(r.limit, np.tile(r.limit[:, :1], (1, 4)), atol=1e-8)

    def test_anp_identity_like_gives_stationary(self):
        A = np.ones((3, 3))
        np.testing.assert_allclose(anp_weights(A).weights, [1 / 3] * 3)

    def test_ahp_consistency_cr(self):
        M = np.array([[1, 3, 3, 2], [1 / 3, 1, 1, .5], [1 / 3, 1, 1, .5], [.5, 2, 2, 1]])
        assert ahp_consistency(M)["CR"] < 0.1

    def test_objective_weights_handle_zeros(self, scouting_matrix):
        for w in (entropy_weights(scouting_matrix),
                  itara_weights(scouting_matrix, B4),
                  merec(scouting_matrix, B4)[1]):
            assert np.all(np.isfinite(w))
            assert np.isclose(w.sum(), 1)
            assert np.all(w >= 0)

    def test_merec_zero_does_not_dominate(self, scouting_matrix):
        _, w = merec(scouting_matrix, B4)
        assert w[2] < 0.6


# ============================================================
# PROPIEDADES DE TODOS LOS MÉTODOS DE RANKING
# ============================================================

class TestRankingMethods:

    def test_registry_has_16_methods(self):
        assert len(RANKING_METHODS) == 16

    def test_all_finite_and_permutation(self, scouting_matrix):
        res = run_all_rankings(scouting_matrix, W4, B4)
        for key, (scores, ranks) in res.items():
            assert np.all(np.isfinite(scores)), key
            assert sorted(ranks) == list(range(1, len(scouting_matrix) + 1)), key

    @pytest.mark.parametrize("key", [k for k in RANKING_METHODS if k != "MEREC"])
    def test_dominant_alternative_wins(self, key):
        # A domina en todo: más producción y menor costo
        X = np.array([
            [0.8, 2.0, 0.30, 1_000_000],
            [0.5, 1.2, 0.15, 20_000_000],
            [0.3, 0.8, 0.10, 40_000_000],
            [0.4, 1.0, 0.12, 30_000_000],
        ])
        _, ranks = run_all_rankings(X, W4, B4)[key]
        assert ranks[0] == 1, key

    @pytest.mark.parametrize("key", [k for k in RANKING_METHODS if k != "MEREC"])
    def test_cheaper_is_not_worse(self, key):
        X = np.array([
            [0.5, 1.2, 0.15, 20_000_000],
            [0.5, 1.2, 0.15, 10_000_000],   # igual pero más barato
            [0.3, 0.8, 0.10, 40_000_000],
            [0.7, 1.6, 0.25, 60_000_000],
        ])
        meth = RANKING_METHODS[key]
        s = meth.run(X, W4, B4)
        if meth.higher_is_better:
            assert s[1] >= s[0] - 1e-12, key
        else:
            assert s[1] <= s[0] + 1e-12, key

    def test_three_way_regions(self, scouting_matrix):
        r = three_way(scouting_matrix, W4, B4, alpha=0.6, beta=0.4)
        assert set(r.decision) <= {"Aceptar", "Frontera", "Rechazar"}
        assert np.all((r.pc >= 0) & (r.pc <= 1))
        with pytest.raises(ValueError):
            three_way(scouting_matrix, W4, B4, alpha=0.3, beta=0.5)

    def test_grey_in_unit_interval(self, scouting_matrix):
        g = grey(scouting_matrix, W4, B4)
        assert np.all((g > 0) & (g <= 1 + 1e-12))

    def test_fuzzy_topsis_in_unit_interval(self, scouting_matrix):
        cc = fuzzy_topsis(scouting_matrix, W4, B4)
        assert np.all((cc >= 0) & (cc <= 1))

    def test_rafsi_custom_bounds(self, scouting_matrix):
        v = rafsi(scouting_matrix, W4, B4,
                  ideal=[1.0, 2.5, 0.4, 0.0], anti_ideal=[0.0, 0.0, 0.0, 200_000_000])
        assert np.all(np.isfinite(v))

    def test_waspas_lambda_extremes(self, scouting_matrix):
        assert np.allclose(waspas(scouting_matrix, W4, B4, lam=1.0), saw(scouting_matrix, W4, B4))

    def test_ranks_ties_min(self):
        np.testing.assert_array_equal(ranks_from_scores(np.array([3, 1, 3, 2]), ties="min"), [1, 4, 1, 3])
