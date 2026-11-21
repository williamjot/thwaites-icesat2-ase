"""Testes das peças numéricas do derretimento basal reconstruído em 2026-11-15.

Cada teste trava uma propriedade que já falhou ou foi verificada à mão durante o trabalho:
- modelo de viga contra a solução analítica;
- convenção de sinal da fase do ApRES (uma versão inverteu o m_b);
- estimador de efeitos fixos com cobertura variável (a média simples erra metros);
- interpolação do firn exata nos nós e linear entre épocas;
- solução exata de β (complemento de Schur) contra mínimos quadrados densos.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))


def test_beam_matches_analytic():
    from thwaites.flexure.beam import beta_analytic_uniform, beta_profile, rigidity
    D = rigidity(3e9, 500.0)
    lam = (4 * D / (1027 * 9.81)) ** 0.25
    L = 12 * lam
    x = np.linspace(0, L, 300)
    b = beta_profile(x, L, lambda xx: np.full_like(np.atleast_1d(xx), D, dtype=float))
    assert np.max(np.abs(b - beta_analytic_uniform(x, D))) < 1e-4


def test_apres_phase_sign_and_magnitude():
    from thwaites.validation.apres import Config, range_profile
    cfg = Config()
    t = np.arange(cfg.n) / 40000.0

    def chirp(R):
        tau = 2 * R / cfg.ci
        return np.cos(2 * np.pi * (cfg.fc - cfg.B / 2) * tau + 2 * np.pi * cfg.K * tau * t
                      - np.pi * cfg.K * tau ** 2)

    _, sa = range_profile(chirp(300.0), cfg)
    _, sb = range_profile(chirp(300.05), cfg)
    k = int(np.argmax(abs(sa)))
    c = np.sum(sa[k - 8:k + 8] * np.conj(sb[k - 8:k + 8]))
    dr = -np.angle(c) * cfg.lam_c / (4 * np.pi)          # convenção usada em apres.py
    assert dr == pytest.approx(0.05, abs=0.003)


def test_fixed_effects_series_with_changing_coverage():
    import run_thickness_ase as r
    rng = np.random.default_rng(0)
    n = 200_000
    x, y = rng.uniform(0, 40e3, n), rng.uniform(0, 40e3, n)
    k = rng.integers(0, 28, n)
    keep = ~((k % 2 == 0) & (x < 20e3))
    x, y, k = x[keep], y[keep], k[keep]
    topo = 300 + 0.002 * x + 50 * np.sin(x / 3e3) * np.cos(y / 4e3)
    s_true = np.cumsum(rng.normal(-0.5, 1, 28))
    s_true -= s_true.mean()
    H = topo + s_true[k] + rng.normal(0, 2, len(x))
    s, _ = r.fe_series(x, y, H, k, 28)
    assert np.nanmax(np.abs(s - s_true)) < 0.2


def test_firn_interpolation_exact_at_nodes(tmp_path):
    import xarray as xr
    import run_thickness_ase as r
    gx, gy = np.meshgrid(np.linspace(-1600e3, -1500e3, 6), np.linspace(-500e3, -400e3, 6))
    x_pts, y_pts = gx.ravel(), gy.ravel()
    times = np.linspace(2019.0, 2025.0, 15)
    fac_vals = (0.5 * np.sin(times)[:, None] + 1e-5 * (x_pts + y_pts)[None, :] + 10.0)
    nc_file = tmp_path / "synthetic_fac.nc"
    xr.Dataset(
        {"fac": (("time", "point"), fac_vals), "x": (("point",), x_pts), "y": (("point",), y_pts)},
        coords={"time": times},
    ).to_netcdf(nc_file)
    f = r.FirnField(nc_file, max_dist=30_000.0)
    i = np.arange(len(f.xy))
    ep = np.full(len(i), 5)
    v, _ = f(f.xy[i, 0], f.xy[i, 1], f.time[ep])
    assert np.nanmax(np.abs(v - f.fac[ep, i])) < 1e-9
    tm = (f.time[ep] + f.time[ep + 1]) / 2
    v2, _ = f(f.xy[i, 0], f.xy[i, 1], tm)
    assert np.nanmax(np.abs(v2 - (f.fac[ep, i] + f.fac[ep + 1, i]) / 2)) < 1e-9


def test_batched_ls_matches_dense():
    from run_tide_response_lagrangian import batched_ls
    rng = np.random.default_rng(3)
    n, ng, p = 3000, 7, 3
    X = rng.normal(size=(n, p))
    g = rng.integers(0, ng, n)
    beta = rng.normal(size=(ng, p))
    y = np.einsum("ij,ij->i", X, beta[g]) + rng.normal(0, 0.01, n)
    est = batched_ls(X, y, g, ng)
    for j in range(ng):
        ref = np.linalg.lstsq(X[g == j], y[g == j], rcond=None)[0]
        assert np.allclose(est[j], ref, atol=1e-8)
