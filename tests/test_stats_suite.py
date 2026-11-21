"""Testes unitários e de identidade analítica para o subpacote ``thwaites.stats``."""

from __future__ import annotations

import numpy as np

from thwaites.stats import (
    benjamini_hochberg_fdr,
    block_bootstrap_normal_eq,
    bretherton_effective_dof,
    cochran_interannual_chi2,
    cr1_clustered_se,
    delete_group_jackknife,
    even_odd_divergence_cv,
    fixed_effects_panel_series,
    fourier_phase_surrogate,
    leave_one_year_out_jackknife,
    mann_kendall_sen,
    normalized_gaussian_derivative,
    pre_post_window_difference,
    pyper_peterman_effective_dof,
    rolstad_effective_samples,
    schur_complement_beta,
    seasonal_mann_kendall_sen,
    spatial_block_assignments,
    standardized_error_calibration,
    step_change_bic_test,
    surrogate_correlation_test,
)


def test_mann_kendall_sen_and_fdr():
    years = np.arange(2019, 2026, dtype=float)
    values = -1.5 * (years - 2019.0) + 10.0
    res = mann_kendall_sen(years, values)
    assert np.isclose(res["sens_slope"], -1.5, atol=1e-6)
    assert res["tau"] < -0.95
    assert res["trend"] == "decrescente"

    reject, q = benjamini_hochberg_fdr([0.001, 0.005, 0.40, 0.85], alpha=0.05)
    assert list(reject) == [True, True, False, False]
    assert np.all(np.diff(q) >= -1e-12)


def test_seasonal_mann_kendall_removes_annual_cycle():
    years = np.repeat(np.arange(2019, 2026), 4)
    months = np.tile([1, 4, 7, 10], 7)
    seasonal_offset = np.tile([3.0, -1.0, -4.0, 2.0], 7)
    values = -0.8 * (years - 2019) + seasonal_offset
    res = seasonal_mann_kendall_sen(years, months, values)
    assert np.isclose(res["sens_slope"], -0.8, atol=1e-6)
    assert res["trend"] == "decrescente"


def test_jackknife_and_block_bootstrap_normal_equations():
    rng = np.random.default_rng(42)
    years = np.repeat(np.arange(2019, 2026), 40)
    dt = years - 2022.0
    y = 2.0 - 0.5 * dt + 0.5 * 0.2 * (dt ** 2) + rng.normal(0, 0.05, size=years.size)
    A = np.c_[np.ones_like(dt), dt, 0.5 * (dt ** 2)]

    theta, se_jk, k = leave_one_year_out_jackknife(
        years, lambda m: float(np.linalg.lstsq(A[m], y[m], rcond=None)[0][1])
    )
    assert k == 7
    assert np.isclose(theta, -0.5, atol=0.05)
    assert se_jk > 0

    boot = block_bootstrap_normal_eq(A, y, years, param_index=2, n_boot=100, seed=7)
    assert len(boot) >= 50
    assert np.isclose(np.median(boot), 0.2, atol=0.05)

    _, se_dag = delete_group_jackknife(
        years, lambda m: np.linalg.lstsq(A[m], y[m], rcond=None)[0], n_groups=5, seed=1
    )
    assert se_dag.shape == (3,)
    assert np.all(se_dag >= 0)


def test_schur_complement_equals_full_ols_and_cr1():
    rng = np.random.default_rng(123)
    n_cells = 15
    n_per_cell = 20
    n = n_cells * n_per_cell
    g = np.repeat(np.arange(n_cells), n_per_cell)
    dx = rng.uniform(-1, 1, size=n)
    dy = rng.uniform(-1, 1, size=n)
    X_loc = np.c_[np.ones(n), dx, dy]
    alpha_true = rng.normal(0, 5.0, size=(n_cells, 3))
    tide = rng.normal(0, 1.0, size=n)
    beta_true = 0.85
    clusters = rng.integers(0, 25, size=n)
    noise = rng.normal(0, 0.1, size=25)[clusters]
    y = np.einsum("ij,ij->i", X_loc, alpha_true[g]) + beta_true * tide + noise

    beta_schur, h_schur, z_proj, resid = schur_complement_beta(X_loc, g, tide, y)

    # Solução conjunta explícita por matriz densa (Frisch-Waugh-Lovell)
    D = np.zeros((n, n_cells * 3 + 1))
    for c in range(n_cells):
        m = g == c
        D[m, 3 * c:3 * c + 3] = X_loc[m]
    D[:, -1] = tide
    beta_full = float(np.linalg.lstsq(D, y, rcond=None)[0][-1])

    assert np.isclose(beta_schur, beta_full, atol=1e-10)
    se_cr1, n_cl = cr1_clustered_se(z_proj, resid, clusters, h_schur)
    assert n_cl == 25
    assert 0.0 < se_cr1 < 0.2


def test_fixed_effects_panel_series_recovers_epoch_signal():
    rng = np.random.default_rng(99)
    n = 600
    px = rng.integers(0, 6, size=n) * 500.0 + rng.uniform(50, 450, size=n)
    py = rng.integers(0, 4, size=n) * 500.0 + rng.uniform(50, 450, size=n)
    k = rng.integers(0, 8, size=n)
    true_s = np.linspace(4.0, -4.0, 8)
    true_s -= true_s.mean()
    cell_offset = 100.0 * np.floor(px / 500.0) - 50.0 * np.floor(py / 500.0)
    H = cell_offset + 0.02 * (px % 500.0) + true_s[k]
    s_hat, counts = fixed_effects_panel_series(px, py, H, k, n_epochs=8, cell_size_m=500.0)
    assert counts.sum() == n
    assert np.allclose(s_hat, true_s, atol=1e-8)


def test_time_series_tests_and_surrogates():
    # Cochran chi2
    const_res = cochran_interannual_chi2([5.0, 5.1, 4.9, 5.0], [0.5, 0.5, 0.5, 0.5])
    assert const_res["p_value"] > 0.90
    var_res = cochran_interannual_chi2([2.0, 9.0, 1.0, 10.0], [0.5, 0.5, 0.5, 0.5])
    assert var_res["p_value"] < 1e-6

    # Pre/post window
    t = np.arange(12, dtype=float) * 0.25 + 2019.0
    y = -2.0 * (t - 2019.0)
    sig = np.full_like(t, 0.2)
    pp = pre_post_window_difference(t, y, sig, pre_idx=[0, 1, 2, 3, 4], post_idx=[6, 7, 8, 9, 10, 11])
    assert np.isclose(pp["delta"], 0.0, atol=1e-8)

    # Surrogates & effective DoF
    rng = np.random.default_rng(2026)
    x = np.cumsum(rng.normal(0, 1, size=32))
    surr = fourier_phase_surrogate(x, rng)
    assert np.isclose(np.mean(surr), np.mean(x), atol=1e-8)
    assert np.isclose(np.var(surr), np.var(x), rtol=1e-6)
    assert bretherton_effective_dof(x, x) < len(x)
    assert pyper_peterman_effective_dof(x, x) < len(x)

    sc = surrogate_correlation_test(rng.normal(size=24), rng.normal(size=24), n_surrogates=200, seed=1)
    assert 0.0 <= sc["p_surrogate"] <= 1.0

    # Step change BIC
    t_step = np.linspace(2019.0, 2025.75, 28)
    y_step = np.where(t_step < 2024.1, 180.0, 140.0) + rng.normal(0, 1.0, size=28)
    bic = step_change_bic_test(t_step, y_step)
    assert bic["delta_bic"] > 15.0
    assert np.isclose(bic["step_size"], -40.0, atol=3.0)


def test_spatial_cv_and_normalized_derivative():
    x = np.repeat(np.arange(0, 20000, 1000), 10)
    y = np.tile(np.arange(0, 10000, 1000), 20)
    folds = spatial_block_assignments(x, y, block_size_m=5000.0, n_folds=4, seed=0)
    assert set(np.unique(folds)) == {0, 1, 2, 3}

    cal = standardized_error_calibration(np.zeros(100), np.linspace(-1, 1, 100), np.ones(100))
    assert np.isclose(cal["bias"], 0.0, atol=1e-8)
    assert cal["z_std"] > 0

    grid_x = np.broadcast_to(np.arange(30, dtype=float)[None, :], (20, 30)).copy()
    grid_x[8:12, 12:16] = np.nan
    d_dx = normalized_gaussian_derivative(grid_x, sigma_px=2.0, axis=1)
    interior = d_dx[:, 4:-4]
    valid = np.isfinite(interior)
    assert valid.sum() > 300
    assert np.isclose(np.median(interior[valid]), 1.0, atol=5e-3)
    assert np.allclose(d_dx[0, 6:-6], 1.0, atol=2e-3)

    cv = even_odd_divergence_cv(np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10)), np.ones((10, 10), bool))
    assert np.isclose(cv["cv_loss"], 0.0)

    rol = rolstad_effective_samples(area_total_m2=200_000e6, correlation_length_m=34_100.0, cell_area_m2=25e6, nugget_fraction=0.2)
    assert rol["n_eff"] > 50.0
