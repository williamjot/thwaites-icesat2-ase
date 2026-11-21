"""Mapa contínuo de derretimento basal: interpolação espacial do DH/Dt com método escolhido por CV.

Só o DH/Dt é limitado às trilhas (células materiais com ≥ 8 passagens e ≥ 3 anos: 21–40% da
área). O SMB e o H·∇·u já são campos contínuos. Por isso interpola-se o DH/Dt e recalcula-se
m_b = SMB − DH/Dt − H·∇·u.

Candidatos (nenhum assumido a priori):
- ``kernel_L``: média gaussiana ponderada por 1/σ² com L = 1, 2, 3, 5, 8 km;
- ``krig``: krigagem ordinária local (48 vizinhos), covariância exponencial ajustada ao
  variograma empírico da unidade, pepita heterocedástica = σ_i² + pepita extra.

Escolha: validação cruzada espacial em 10 dobras de blocos de 5 km, por unidade. Blocos, porque
células vizinhas na mesma trilha são correlacionadas, e deixá-las em dobras diferentes daria uma
CV otimista. Métricas: RMSE; na krigagem, também DP do erro padronizado (≈ 1 se o σ previsto é
honesto).

Saída: ``data/interim/basal_melt/mb_continuo_500m.nc`` e ``outputs/basal_melt_ase/cv_interpolacao.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.optimize import curve_fit
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data" / "interim" / "basal_melt" / "mb_500m.nc"
OUT_NC = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
OUT = ROOT / "outputs" / "basal_melt_ase"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Thwaites_TWGT", "Crosson", "Dotson", "Cosgrove"]
KERNEL_L = (1e3, 2e3, 3e3, 5e3, 8e3)
NFOLD, BLOCK, K_NEIGH = 10, 5e3, 48
RNG = np.random.default_rng(20261115)


# ---------------------------------------------------------------- kernel
def kernel_predict(xo, yo, vo, so, xt, yt, L):
    tree = cKDTree(np.c_[xo, yo])
    pred = np.full(len(xt), np.nan)
    sig = np.full(len(xt), np.nan)
    nb = tree.query_ball_point(np.c_[xt, yt], 3 * L)
    w0 = 1 / np.maximum(so, 1e-2) ** 2
    for i, idx in enumerate(nb):
        if not idx:
            continue
        idx = np.asarray(idx)
        d2 = (xo[idx] - xt[i]) ** 2 + (yo[idx] - yt[i]) ** 2
        w = w0[idx] * np.exp(-0.5 * d2 / L ** 2)
        if w.sum() <= 0:
            continue
        pred[i] = np.sum(w * vo[idx]) / w.sum()
        sig[i] = np.sqrt(np.sum(w ** 2 * so[idx] ** 2)) / w.sum()   # só ruído
    return pred, sig


# ---------------------------------------------------------------- krigagem
def fit_variogram(x, y, v, s, npairs=200_000):
    i = RNG.integers(0, len(x), npairs)
    j = RNG.integers(0, len(x), npairs)
    h = np.hypot(x[i] - x[j], y[i] - y[j])
    g = 0.5 * (v[i] - v[j]) ** 2
    ok = (h > 0) & (h < 25e3)
    bins = np.arange(0, 25e3 + 1, 1000)
    hb = pd.Series(h[ok]).groupby(np.digitize(h[ok], bins)).mean()
    gb = pd.Series(g[ok]).groupby(np.digitize(h[ok], bins)).mean()
    model = lambda hh, n, c, r: n + c * (1 - np.exp(-hh / r))  # noqa: E731
    p0 = [float(np.median(s) ** 2), float(np.var(v)), 5e3]
    try:
        (n, c, r), _ = curve_fit(model, hb.values, gb.values, p0=p0,
                                 bounds=([0, 1e-6, 500], [np.inf, np.inf, 1e5]))
    except RuntimeError:
        n, c, r = p0
    extra = max(n - float(np.median(s ** 2)), 0.0)
    return {"pepita": n, "patamar_parcial": c, "alcance_m": r, "pepita_extra": extra}


def krig_predict(xo, yo, vo, so, xt, yt, vg, chunk=2000):
    tree = cKDTree(np.c_[xo, yo])
    k = min(K_NEIGH, len(xo))
    c, r, ne = vg["patamar_parcial"], vg["alcance_m"], vg["pepita_extra"]
    pred = np.full(len(xt), np.nan)
    sig = np.full(len(xt), np.nan)
    for a in range(0, len(xt), chunk):
        b = slice(a, a + chunk)
        _, idx = tree.query(np.c_[xt[b], yt[b]], k=k)
        idx = np.atleast_2d(idx)
        px, py = xo[idx], yo[idx]                                   # (m, k)
        D = np.hypot(px[:, :, None] - px[:, None, :], py[:, :, None] - py[:, None, :])
        C = c * np.exp(-D / r)
        C[:, np.arange(k), np.arange(k)] += so[idx] ** 2 + ne
        d0 = np.hypot(px - xt[b][:, None], py - yt[b][:, None])
        c0 = c * np.exp(-d0 / r)
        m = len(idx)
        A = np.zeros((m, k + 1, k + 1))
        A[:, :k, :k] = C
        A[:, :k, k] = A[:, k, :k] = 1.0
        rhs = np.concatenate([c0, np.ones((m, 1))], 1)
        sol = np.linalg.solve(A, rhs[..., None])[..., 0]
        lam, mu = sol[:, :k], sol[:, k]
        pred[b] = np.sum(lam * vo[idx], 1)
        sig[b] = np.sqrt(np.maximum(c - np.sum(lam * c0, 1) - mu, 0))
    return pred, sig


# ---------------------------------------------------------------- CV
def block_folds(x, y):
    key = np.floor(x / BLOCK).astype(np.int64) * 1_000_003 + np.floor(y / BLOCK).astype(np.int64)
    u, inv = np.unique(key, return_inverse=True)
    f = RNG.integers(0, NFOLD, len(u))
    return f[inv.ravel()]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    d = xr.open_dataset(SRC)
    xs, ys = d.x.values, d.y.values
    XX, YY = np.meshgrid(xs, ys)
    unit = d.unidade.values
    dh, se = d.DHDt.values, d.se_DHDt.values
    cont = np.full(dh.shape, np.nan)
    cont_se = np.full(dh.shape, np.nan)
    dist_obs = np.full(dh.shape, np.nan)
    rows, chosen = [], {}
    for u in UNITS:
        su = unit == u
        ob = su & np.isfinite(dh) & np.isfinite(se)
        xo, yo, vo, so = XX[ob], YY[ob], dh[ob], se[ob]
        vg = fit_variogram(xo, yo, vo, so)
        folds = block_folds(xo, yo)
        res = {}
        for name in [f"kernel_{int(L / 1e3)}km" for L in KERNEL_L] + ["krig"]:
            pred = np.full(len(xo), np.nan)
            sig = np.full(len(xo), np.nan)
            for f in range(NFOLD):
                tr, te = folds != f, folds == f
                if name == "krig":
                    p, s_ = krig_predict(xo[tr], yo[tr], vo[tr], so[tr], xo[te], yo[te], vg)
                    sig[te] = np.sqrt(s_ ** 2 + so[te] ** 2 + vg["pepita_extra"])
                else:
                    L = float(name.split("_")[1][:-2]) * 1e3
                    p, s_ = kernel_predict(xo[tr], yo[tr], vo[tr], so[tr], xo[te], yo[te], L)
                pred[te] = p
            ok = np.isfinite(pred)
            err = pred[ok] - vo[ok]
            row = {"unidade": u, "metodo": name, "n_obs": len(vo), "frac_prevista": ok.mean(),
                   "rmse_mpa": float(np.sqrt(np.mean(err ** 2))),
                   "mae_mpa": float(np.median(np.abs(err)))}
            if name == "krig":
                z = err / sig[ok]
                row["dp_erro_padronizado"] = float(np.std(z))
                row.update({k: float(v) for k, v in vg.items()})
            rows.append(row)
            res[name] = row["rmse_mpa"] if ok.mean() > 0.95 else np.inf
        best = min(res, key=res.get)
        chosen[u] = best
        xt, yt = XX[su], YY[su]
        if best == "krig":
            p, s_ = krig_predict(xo, yo, vo, so, xt, yt, vg)
        else:
            L = float(best.split("_")[1][:-2]) * 1e3
            p, s_ = kernel_predict(xo, yo, vo, so, xt, yt, L)
        cont[su], cont_se[su] = p, s_
        dist_obs[su] = cKDTree(np.c_[xo, yo]).query(np.c_[xt, yt])[0]
        print(f"{u:14s} melhor: {best}  ({', '.join(f'{k} {v:.2f}' for k, v in res.items())})",
              flush=True)
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "cv_interpolacao.csv", index=False)

    mb = d.SMB.values - cont - d.H_div.values
    xr.Dataset({"mb": (("y", "x"), mb.astype(np.float32)),
                "DHDt_continuo": (("y", "x"), cont.astype(np.float32)),
                "se_DHDt_continuo": (("y", "x"), cont_se.astype(np.float32)),
                "dist_obs_m": (("y", "x"), dist_obs.astype(np.float32)),
                "SMB": d.SMB, "H_div": d.H_div, "unidade": d.unidade},
               coords={"x": xs, "y": ys},
               attrs={"metodo_por_unidade": str(chosen),
                      "unidade_fisica": "m gelo eq./a; m_b > 0 = derretimento"}).to_netcdf(OUT_NC)
    summ = []
    for u in UNITS:
        su = (unit == u) & np.isfinite(mb)
        summ.append({"unidade": u, "metodo": chosen[u],
                     "cobertura": su.sum() / (unit == u).sum(),
                     "mb_media_area_mpa": float(np.mean(mb[su])),
                     "mb_mediana_mpa": float(np.median(mb[su])),
                     "dist_obs_p95_km": float(np.percentile(dist_obs[su], 95) / 1e3)})
    s = pd.DataFrame(summ)
    s.to_csv(OUT / "mb_continuo_por_unidade.csv", index=False)
    pd.set_option("display.width", 220)
    print(t.round(3).to_string(index=False))
    print(s.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
