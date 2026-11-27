"""
Testes da interpolação espaço-tempo por kernel gaussiano
(thwaites.interp.spacetime).
"""

import numpy as np
import pytest

from thwaites.interp.spacetime import build_spacetime_cube, gaussian_spacetime_predict


def _regular_obs(f, nx=21, nt=13, step=1000.0, dt=0.25):
    """Observações numa grade regular (x, y, t) com valores dados por f(x, y, t)."""
    g = np.arange(nx) * step
    tt = 2019.0 + np.arange(nt) * dt
    X, Y, T = np.meshgrid(g, g, tt, indexing="ij")
    return X.ravel(), Y.ravel(), T.ravel(), f(X.ravel(), Y.ravel(), T.ravel())


def test_constant_field_is_exact():
    px, py, pt, pv = _regular_obs(lambda x, y, t: np.full_like(x, 3.7))
    r = gaussian_spacetime_predict(px, py, pt, pv, [7300.0, 12100.0], [5500.0, 9900.0],
                                   [2020.1, 2021.3], alpha_d_m=2000.0, alpha_t_yr=0.5)
    assert np.allclose(r["pred"], 3.7)
    assert np.allclose(r["rmse"], 0.0, atol=1e-12)


def test_linear_field_exact_at_symmetric_point():
    # alvo no centro de uma grade simétrica: os pesos são simétricos e a
    # média ponderada de um campo linear devolve o valor exato
    f = lambda x, y, t: 1.0 + 2e-4 * x - 1e-4 * y - 0.8 * (t - 2019.0)
    px, py, pt, pv = _regular_obs(f)
    x0, y0, t0 = 10_000.0, 10_000.0, 2020.5
    r = gaussian_spacetime_predict(px, py, pt, pv, x0, y0, t0,
                                   alpha_d_m=1500.0, alpha_t_yr=0.3, k=500)
    assert r["pred"][0] == pytest.approx(f(x0, y0, t0), abs=1e-9)


def test_time_distance_reduces_weight():
    # duas observações no mesmo lugar: uma na data do alvo (valor 1),
    # outra a 2 α_t de distância (valor 0)
    px, py = np.zeros(2), np.zeros(2)
    pt = np.array([2020.0, 2021.0])
    pv = np.array([1.0, 0.0])
    r = gaussian_spacetime_predict(px, py, pt, pv, 0.0, 0.0, 2020.0,
                                   alpha_d_m=1000.0, alpha_t_yr=0.5, min_points=1)
    w_far = np.exp(-0.5 * 2.0 ** 2)
    assert r["pred"][0] == pytest.approx(1.0 / (1.0 + w_far))
    # com α_t muito grande o tempo deixa de importar: média simples
    r2 = gaussian_spacetime_predict(px, py, pt, pv, 0.0, 0.0, 2020.0,
                                    alpha_d_m=1000.0, alpha_t_yr=100.0, min_points=1)
    assert r2["pred"][0] == pytest.approx(0.5, abs=1e-3)


def test_prior_error_downweights_noisy_obs():
    px, py, pt = np.zeros(2), np.zeros(2), np.full(2, 2020.0)
    pv = np.array([0.0, 10.0])
    psig = np.array([0.1, 1.0])          # a segunda é 10× mais ruidosa (100× menos peso)
    r = gaussian_spacetime_predict(px, py, pt, pv, 0.0, 0.0, 2020.0, alpha_d_m=1000.0,
                                   alpha_t_yr=0.5, psig=psig, min_points=1)
    assert r["pred"][0] == pytest.approx(10.0 * 1.0 / 101.0)


def test_no_neighbours_gives_nan():
    px, py, pt, pv = np.zeros(10), np.zeros(10), np.full(10, 2020.0), np.ones(10)
    # alvo a 10 α_d de distância, fora do raio de busca (3)
    r = gaussian_spacetime_predict(px, py, pt, pv, 10_000.0, 0.0, 2020.0,
                                   alpha_d_m=1000.0, alpha_t_yr=0.5)
    assert np.isnan(r["pred"][0]) and r["n"][0] == 0
    # alvo no mesmo lugar, mas 10 α_t depois: também sem suporte temporal
    r = gaussian_spacetime_predict(px, py, pt, pv, 0.0, 0.0, 2025.0,
                                   alpha_d_m=1000.0, alpha_t_yr=0.5)
    assert np.isnan(r["pred"][0])


def test_min_points_rule():
    px, py, pt, pv = np.zeros(3), np.zeros(3), np.full(3, 2020.0), np.ones(3)
    r = gaussian_spacetime_predict(px, py, pt, pv, 0.0, 0.0, 2020.0, alpha_d_m=1000.0,
                                   alpha_t_yr=0.5, min_points=5)
    assert np.isnan(r["pred"][0]) and r["n"][0] == 3


def test_nonfinite_inputs_are_ignored():
    px = np.array([0.0, 0.0, np.nan])
    py = np.zeros(3)
    pt = np.full(3, 2020.0)
    pv = np.array([1.0, 1.0, 1e6])
    r = gaussian_spacetime_predict(px, py, pt, pv, 0.0, 0.0, 2020.0, alpha_d_m=1000.0,
                                   alpha_t_yr=0.5, min_points=1)
    assert r["pred"][0] == pytest.approx(1.0) and r["n"][0] == 2


def test_cube_shape_and_chunking_consistency():
    rng = np.random.default_rng(1)
    n = 3000
    px, py = rng.uniform(0, 20_000, n), rng.uniform(0, 20_000, n)
    pt = rng.uniform(2019, 2025, n)
    pv = 1e-4 * px - 0.5 * (pt - 2019) + rng.normal(0, 0.05, n)
    xg, yg, tg = np.arange(0, 20_001, 4000.0), np.arange(0, 20_001, 5000.0), np.arange(2019.5, 2025, 1.0)
    a = build_spacetime_cube(px, py, pt, pv, xg, yg, tg, alpha_d_m=3000.0, alpha_t_yr=0.4, chunk=17)
    b = build_spacetime_cube(px, py, pt, pv, xg, yg, tg, alpha_d_m=3000.0, alpha_t_yr=0.4)
    assert a["pred"].shape == (tg.size, yg.size, xg.size)
    np.testing.assert_allclose(a["pred"], b["pred"], equal_nan=True)
    # tendência temporal recuperada no centro do cubo (≈ −0,5 m/ano)
    serie = a["pred"][:, 2, 2]
    assert np.polyfit(tg, serie, 1)[0] == pytest.approx(-0.5, abs=0.06)


def test_step_change_is_smoothed():
    # documenta a limitação: um degrau de −1 m em 2022,0 sai espalhado no tempo
    px, py, pt, _ = _regular_obs(lambda x, y, t: x, nx=9, nt=49, dt=0.125)
    pv = np.where(pt < 2022.0, 0.0, -1.0)
    r = gaussian_spacetime_predict(px, py, pt, pv, np.full(3, 4000.0), np.full(3, 4000.0),
                                   np.array([2020.0, 2022.0, 2024.0]),
                                   alpha_d_m=1000.0, alpha_t_yr=0.5, k=2000)
    antes, meio, depois = r["pred"]
    assert antes == pytest.approx(0.0, abs=1e-3)
    assert depois == pytest.approx(-1.0, abs=1e-3)
    assert -0.75 < meio < -0.25          # no instante do degrau: valor intermediário


def test_kernel_not_truncated_by_default_k():
    # dados densos: com o k padrão o kernel chega perto do raio de busca;
    # com k pequeno ele é truncado (r_max ≪ search_radius)
    rng = np.random.default_rng(3)
    n = 20_000
    px, py, pt = rng.uniform(0, 20e3, n), rng.uniform(0, 20e3, n), rng.uniform(2019, 2025, n)
    pv = rng.normal(0, 1, n)
    full = gaussian_spacetime_predict(px, py, pt, pv, 10e3, 10e3, 2022.0, alpha_d_m=2e3, alpha_t_yr=0.25)
    trunc = gaussian_spacetime_predict(px, py, pt, pv, 10e3, 10e3, 2022.0, alpha_d_m=2e3,
                                       alpha_t_yr=0.25, k=32)
    assert full["r_max"][0] > 2.5
    assert trunc["r_max"][0] < 1.5


def test_invalid_alpha_raises():
    with pytest.raises(ValueError):
        gaussian_spacetime_predict([0.0], [0.0], [2020.0], [1.0], 0.0, 0.0, 2020.0,
                                   alpha_d_m=0.0, alpha_t_yr=0.5)
