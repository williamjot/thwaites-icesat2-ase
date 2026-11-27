"""
thwaites.interp.spacetime
=========================
Interpolação ESPAÇO-TEMPO por kernel gaussiano: estima o valor num ponto
(x, y) e numa data t a partir das observações vizinhas no espaço E no tempo,
e monta cubos (tempo × y × x) de anomalia de altura.

Inspirada no `interpgaus3d.py` do captoolkit (Nilsson, Paolo & Gardner, NASA
JPL, licença Apache 2.0); reimplementada aqui de forma vetorizada.

Peso de cada observação i em relação ao alvo (x0, y0, t0):

    w_i = exp(-d_i² / (2 α_d²)) · exp(-Δt_i² / (2 α_t²))  [ / σ_i² ]

onde d_i é a distância horizontal (m), Δt_i a diferença de tempo (anos),
α_d e α_t os comprimentos de correlação no espaço e no tempo e σ_i o erro a
priori (opcional, ponderação por variância inversa).

Truque de implementação: com coordenadas normalizadas
(X, Y, T) = (x/α_d, y/α_d, t/α_t), o produto dos dois kernels é um único
kernel gaussiano isotrópico, exp(-r²/2), com r a distância euclidiana
normalizada. Basta UMA KDTree 3D e uma consulta de vizinhos por raio.

Incerteza (mesma convenção do captoolkit): soma quadrática da dispersão
ponderada dos vizinhos e do erro a priori propagado,

    rmse = sqrt( Σw(v − v̂)² / Σw  +  Σw²σ² / (Σw)² ).

É uma estimativa heurística, não uma variância de predição calibrada: antes
de propagá-la para massa, verifique a calibração por validação cruzada em
blocos (ver `thwaites.stats.spatial_cv.standardized_error_calibration`).

CUIDADOS DE USO
- Interpole RESÍDUOS (p. ex. `h_res = h_corr − REMA`) ou anomalias, nunca a
  altura bruta: o relevo vazaria para o cubo.
- A janela temporal SUAVIZA eventos bruscos: um degrau aparece espalhado por
  ~±2 α_t. Para testar mudanças abruptas use séries por temporada e
  `thwaites.stats.step_change_bic_test`.
- Células sem observação a menos de `search_radius` comprimentos de
  correlação ficam NaN, em vez de emprestar dados distantes.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree

__all__ = ["gaussian_spacetime_predict", "build_spacetime_cube"]


def _normalized(x, y, t, alpha_d_m, alpha_t_yr):
    if alpha_d_m <= 0 or alpha_t_yr <= 0:
        raise ValueError("alpha_d_m e alpha_t_yr precisam ser positivos")
    return np.c_[np.asarray(x, float) / alpha_d_m,
                 np.asarray(y, float) / alpha_d_m,
                 np.asarray(t, float) / alpha_t_yr]


class _SpaceTimeIndex:
    """Índice 3D das observações em coordenadas normalizadas (reutilizável)."""

    def __init__(self, px, py, pt, pv, alpha_d_m, alpha_t_yr, psig=None):
        px, py, pt, pv = (np.asarray(a, float) for a in (px, py, pt, pv))
        if not (px.shape == py.shape == pt.shape == pv.shape):
            raise ValueError("px, py, pt e pv precisam ter o mesmo tamanho")
        ok = np.isfinite(px) & np.isfinite(py) & np.isfinite(pt) & np.isfinite(pv)
        if psig is not None:
            psig = np.asarray(psig, float)
            ok &= np.isfinite(psig) & (psig > 0)
            self.sig = psig[ok]
        else:
            self.sig = None
        self.v = pv[ok]
        self.alpha_d, self.alpha_t = float(alpha_d_m), float(alpha_t_yr)
        self.n_obs = int(ok.sum())
        self.tree = (cKDTree(_normalized(px[ok], py[ok], pt[ok], alpha_d_m, alpha_t_yr))
                     if self.n_obs else None)

    def predict(self, tx, ty, tt, k, search_radius, min_points):
        tx, ty, tt = np.broadcast_arrays(*(np.atleast_1d(np.asarray(a, float)) for a in (tx, ty, tt)))
        shape = tx.shape
        m = tx.size
        out = {key: np.full(m, np.nan) for key in ("pred", "rmse", "n_eff", "r_min", "r_max")}
        out["n"] = np.zeros(m, dtype=np.int64)
        if self.n_obs == 0 or m == 0:
            return {key: val.reshape(shape) for key, val in out.items()}

        q = _normalized(tx.ravel(), ty.ravel(), tt.ravel(), self.alpha_d, self.alpha_t)
        kk = int(min(k, self.n_obs))
        d, idx = self.tree.query(q, k=kk, distance_upper_bound=search_radius)
        if kk == 1:
            d, idx = d[:, None], idx[:, None]
        valid = np.isfinite(d)
        idx_safe = np.where(valid, idx, 0)

        w = np.where(valid, np.exp(-0.5 * np.where(valid, d, 0.0) ** 2), 0.0)
        if self.sig is not None:
            s2 = self.sig[idx_safe] ** 2
            w = w / np.where(valid, s2, 1.0)
        v = self.v[idx_safe]

        n = valid.sum(axis=1)
        sw = w.sum(axis=1)
        good = (n >= min_points) & (sw > 0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pred = (w * v).sum(axis=1) / sw
            spread2 = (w * (v - pred[:, None]) ** 2).sum(axis=1) / sw
            if self.sig is not None:
                prior2 = (w ** 2 * s2).sum(axis=1) / sw ** 2
            else:
                prior2 = np.zeros(m)
            n_eff = sw ** 2 / (w ** 2).sum(axis=1)
            r_min = np.where(valid, d, np.inf).min(axis=1)
            r_max = np.where(valid, d, -np.inf).max(axis=1)

        out["n"] = n
        out["pred"] = np.where(good, pred, np.nan)
        out["rmse"] = np.where(good, np.sqrt(spread2 + prior2), np.nan)
        out["n_eff"] = np.where(good, n_eff, np.nan)
        out["r_min"] = np.where(good, r_min, np.nan)
        out["r_max"] = np.where(good, r_max, np.nan)
        return {key: val.reshape(shape) for key, val in out.items()}


def gaussian_spacetime_predict(px, py, pt, pv, tx, ty, tt, alpha_d_m: float, alpha_t_yr: float,
                               psig=None, k: int = 1024, search_radius: float = 3.0,
                               min_points: int = 5) -> dict:
    """
    Predição espaço-tempo por kernel gaussiano nos alvos (tx, ty, tt).

    Parâmetros
    ----------
    px, py : coordenadas projetadas das observações (m), p. ex. EPSG:3031
    pt     : tempo das observações (ano decimal)
    pv     : valor a interpolar (resíduo ou anomalia, m)
    tx, ty, tt : alvos (arrays de mesmo formato, ou escalares com broadcast)
    alpha_d_m  : comprimento de correlação horizontal α_d (m)
    alpha_t_yr : comprimento de correlação temporal α_t (anos)
    psig       : erro a priori de cada observação (m); se dado, pondera por 1/σ²
    k          : máximo de vizinhos usados por alvo. ATENÇÃO: com dados densos,
                 um k pequeno trunca o kernel antes de `search_radius` e a
                 suavização deixa de depender de α. Confira `r_max` na saída:
                 se ficar bem abaixo de `search_radius`, aumente k.
    search_radius : raio de busca em unidades normalizadas (3 ⇒ peso ≥ 1,1%)
    min_points : mínimo de vizinhos dentro do raio; abaixo disso o alvo fica NaN

    Retorna
    -------
    dict com arrays no formato dos alvos:
      pred  : valor interpolado
      rmse  : incerteza heurística (dispersão ponderada ⊕ erro a priori)
      n     : nº de observações dentro do raio
      n_eff : nº efetivo de observações, (Σw)² / Σw²
      r_min : distância normalizada da observação mais próxima
      r_max : distância normalizada do vizinho mais distante usado
              (≈ search_radius quando k não trunca o kernel)
    """
    idx = _SpaceTimeIndex(px, py, pt, pv, alpha_d_m, alpha_t_yr, psig)
    return idx.predict(tx, ty, tt, k, search_radius, min_points)


def build_spacetime_cube(px, py, pt, pv, x_grid, y_grid, t_grid, alpha_d_m: float,
                         alpha_t_yr: float, psig=None, k: int = 1024,
                         search_radius: float = 3.0, min_points: int = 5,
                         chunk: int = 200_000) -> dict:
    """
    Monta um cubo (tempo × y × x) por interpolação espaço-tempo.

    `x_grid`, `y_grid` e `t_grid` são vetores 1D. O cálculo é feito em blocos
    de `chunk` alvos para limitar a memória. Retorna dict com `pred`, `rmse`,
    `n`, `n_eff`, `r_min`, `r_max` (arrays nt × ny × nx) e os eixos `x`, `y`, `t`.
    """
    x_grid, y_grid, t_grid = (np.asarray(a, float).ravel() for a in (x_grid, y_grid, t_grid))
    T, Y, X = np.meshgrid(t_grid, y_grid, x_grid, indexing="ij")
    idx = _SpaceTimeIndex(px, py, pt, pv, alpha_d_m, alpha_t_yr, psig)

    flat = {key: [] for key in ("pred", "rmse", "n", "n_eff", "r_min", "r_max")}
    xs, ys, ts = X.ravel(), Y.ravel(), T.ravel()
    for i0 in range(0, xs.size, max(int(chunk), 1)):
        sl = slice(i0, i0 + chunk)
        part = idx.predict(xs[sl], ys[sl], ts[sl], k, search_radius, min_points)
        for key in flat:
            flat[key].append(part[key])

    shape = T.shape
    cube = {key: np.concatenate(val).reshape(shape) for key, val in flat.items()}
    cube.update({"x": x_grid, "y": y_grid, "t": t_grid,
                 "alpha_d_m": float(alpha_d_m), "alpha_t_yr": float(alpha_t_yr)})
    return cube
