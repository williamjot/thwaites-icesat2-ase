# -*- coding: utf-8 -*-
"""Ponto euleriano fixo em GELO ATERRADO: dh/dt e massa por temporada.

Analogo direto de outputs/seasonal_H_timeseries (TEIS, flutuante), com as
diferencas que a fisica exige:

  * a relacao hidrostatica NAO vale em gelo aterrado -> a grandeza e a
    elevacao de superficie corrigida h_corr, sem amplificacao 9,34;
  * mare/DAC nao sao subtraidas (contrato vertical: so sobre flutuante);
  * o geoide e irrelevante: so diferencas temporais entram;
  * a massa exige separar ar de gelo -> Delta_h_ice = Delta_h - Delta_FAC
    (GSFC-FDM v1.2.1), e so entao rho_i;
  * gelo aterrado contribui para o nivel do mar -> SLE e calculavel;
  * GIA (Caron 2018) desloca a superficie sem mudar massa -> reportado.

Estimador de janela identico ao da serie TEIS: um intercepto por janela +
plano local comum (dx, dy), Huber IRLS, pesos balanceados por bloco, sigma =
max(SE cluster-robusta sobre blocos, RMSE/sqrt(N_blocos)). Bloco = track_id
(passagem along-track por feixe), a unidade independente destes tiles.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pyproj import Transformer
from scipy.stats import theilslopes

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
sys.path.insert(0, str(ROOT))

from thwaites.atl06_granule_id import (  # noqa: E402
    t_year_to_calendar_decimal_year, t_year_to_datetime)
from thwaites.legacy_fit import block_balanced_weights  # noqa: E402
from thwaites.corrections.gia import GIAField  # noqa: E402

PROC = ROOT / "data" / "processed"
OUT = ROOT / "outputs" / "grounded_H_timeseries"
OUT.mkdir(parents=True, exist_ok=True)

TARGET = ("JJA 2019", "DJF 2019/2020", "JJA 2020", "DJF 2020/2021",
          "JJA 2021", "DJF 2021/2022", "JJA 2022")
ORDER = {w: i for i, w in enumerate(TARGET)}
CELL = 5000.0
AREA_M2 = CELL ** 2
RHO_I = 917.0                 # config mass_balance.ice_density
OCEAN_AREA_M2 = 3.618e14      # config mass_balance.ocean_area_m2
GT_PER_M = RHO_I * AREA_M2 / 1e12
GT_PER_MM_SLE = OCEAN_AREA_M2 * 1e-3 * 1000.0 / 1e12
MIN_OBS = 30
BOOT = 20_000
BOX = (-1560000.0, -1420000.0, -700000.0, -400000.0)
COLS = ["x", "y", "t_year", "mask_class", "track_id", "h_corr", "s_elv",
        "h_elv", "rema", "h_res"]


def window_labels(t_year):
    out = []
    for v in np.asarray(t_year, dtype=float):
        d = t_year_to_datetime(float(v))
        if d.month in (6, 7, 8):
            out.append("JJA %d" % d.year)
        elif d.month == 12:
            out.append("DJF %d/%d" % (d.year, d.year + 1))
        elif d.month in (1, 2):
            out.append("DJF %d/%d" % (d.year - 1, d.year))
        else:
            out.append("OUT")
    return np.asarray(out, dtype=object)


def cell_center(x, y):
    return (np.floor(x / CELL) * CELL + CELL / 2.0,
            np.floor(y / CELL) * CELL + CELL / 2.0)


# ----------------------------------------------------------------- selecao
def select_cells():
    counts = pd.read_csv(PROC / "grounded_seasonal_cell_counts.csv")
    counts = counts[counts.window.isin(TARGET)]
    agg = (counts.groupby(["cx", "cy"], observed=True)
           .agg(n_windows=("window", "nunique"),
                min_obs_window=("n", "min"),
                min_tracks_window=("n_tracks", "min"),
                n_obs_total=("n", "sum")).reset_index())
    pool = agg[(agg.n_windows == len(TARGET))
               & (agg.min_obs_window >= MIN_OBS)
               & (agg.min_tracks_window >= 2)].copy()

    nodes = pd.read_parquet(ROOT / "data/jja/dhdt/dhdt_nodes_qc.parquet")
    nodes = nodes[["x", "y", "mask_class", "dist_gl", "dist_coast",
                   "grounded_frac", "dhdt", "dhdt_err_jack", "reliability"]]
    pool = pool.merge(nodes, left_on=["cx", "cy"], right_on=["x", "y"],
                      how="inner")
    # criterio de gelo aterrado do config: classe 2, buffers de 5 km (GL) e
    # 2 km (costa); grounded_frac exige a celula inteira aterrada.
    pool = pool[(pool.mask_class == 2) & (pool.grounded_frac >= 0.99)
                & (pool.dist_gl >= 5000.0) & (pool.dist_coast >= 2000.0)]
    if pool.empty:
        raise ValueError("nenhuma celula aterrada com as 7 janelas")

    # ranking POR COBERTURA, nunca por dh/dt (evita selecao pelo sinal)
    ranked = pool.sort_values(
        ["min_tracks_window", "min_obs_window", "n_obs_total"],
        ascending=[False, False, False])
    primary = ranked.iloc[0]
    used = {(float(primary.cx), float(primary.cy))}
    q33, q67 = pool.dist_gl.quantile([0.33, 0.67])
    strata = {
        "perto_GL": pool.dist_gl <= q33,
        "intermediario": (pool.dist_gl > q33) & (pool.dist_gl <= q67),
        "interior": pool.dist_gl > q67,
    }
    chosen = {"primary": primary}
    for label, mask in strata.items():
        keys = [(float(a), float(b)) for a, b in zip(pool.cx, pool.cy)]
        free = np.array([k not in used for k in keys])
        sub = pool[mask.to_numpy() & free]
        if sub.empty:
            raise ValueError(f"sem candidato para {label}")
        row = sub.sort_values(["min_tracks_window", "min_obs_window",
                               "n_obs_total"],
                              ascending=[False, False, False]).iloc[0]
        chosen[label] = row
        used.add((float(row.cx), float(row.cy)))
    return chosen, pool


# ------------------------------------------------------------- observacoes
def load_observations(cells):
    keys = {(float(c.cx), float(c.cy)) for c in cells}
    xs = [k[0] for k in keys]
    ys = [k[1] for k in keys]
    win = (min(xs) - 2600.0, max(xs) + 2600.0,
           min(ys) - 2600.0, max(ys) + 2600.0)
    parts = []
    for prof in ("jja", "djf"):
        man = json.load(open(ROOT / f"data/{prof}/tiles/manifest.json"))
        sel = [e for e in man if not (e["x_max"] < win[0] or e["x_min"] > win[1]
                                      or e["y_max"] < win[2]
                                      or e["y_min"] > win[3])]
        for e in sel:
            t = pq.read_table(ROOT / f"data/{prof}/tiles" / e["file"],
                              columns=COLS).to_pandas()
            t = t[(t.x >= e["x_min"]) & (t.x < e["x_max"])
                  & (t.y >= e["y_min"]) & (t.y < e["y_max"])]
            t = t[(t.mask_class == 2) & t.x.between(win[0], win[1])
                  & t.y.between(win[2], win[3])]
            if t.empty:
                continue
            cx, cy = cell_center(t.x.to_numpy(), t.y.to_numpy())
            t["cx"], t["cy"] = cx, cy
            t = t[[k in keys for k in zip(t.cx, t.cy)]]
            if t.empty:
                continue
            t["window"] = window_labels(t.t_year.to_numpy())
            t = t[t.window.isin(TARGET)]
            if len(t):
                t["profile"] = prof
                parts.append(t)
    obs = pd.concat(parts, ignore_index=True)
    obs = obs[np.isfinite(obs.h_corr) & np.isfinite(obs.s_elv)
              & (obs.s_elv > 0)].copy()
    # um track_id se repete entre perfis; separa por perfil
    obs["block"] = obs.profile.map({"jja": 0, "djf": 1}) * 1_000_000 \
        + obs.track_id.astype("int64")
    obs["dx"] = obs.x - obs.cx
    obs["dy"] = obs.y - obs.cy
    return obs


# ------------------------------------------------------------- estimador
def joint_window_fit(obs, cx, cy, value_col="h_res"):
    """Ajusta um intercepto por janela.

    `value_col='h_res'` (h_corr - REMA) e o padrao: REMA e invariante no tempo,
    entao NAO altera nenhuma diferenca temporal, mas remove a topografia
    estatica que um plano de 5 km nao consegue descrever em gelo aterrado. O
    offset constante e devolvido para reconstruir a elevacao absoluta.
    """
    obs = obs[np.isfinite(obs[value_col])].copy()
    support = (obs.groupby("window", observed=True)
               .agg(N_obs=(value_col, "size"), N_blocks=("block", "nunique"))
               .reset_index())
    keep = sorted(support.loc[support.N_obs >= MIN_OBS, "window"].tolist(),
                  key=ORDER.get)
    work = obs[obs.window.isin(keep)].copy()
    if len(keep) < 2:
        raise ValueError("janelas insuficientes")
    cat = pd.Categorical(work.window, categories=keep, ordered=True)
    dummy = np.zeros((len(work), len(keep)))
    dummy[np.arange(len(work)), cat.codes] = 1.0
    design = np.column_stack([dummy,
                              work.dx.to_numpy(float) / 1000.0,
                              work.dy.to_numpy(float) / 1000.0])
    values = work[value_col].to_numpy(float)
    offset = (float(np.nanmean(work.rema.to_numpy(float)))
              if value_col == "h_res" else 0.0)
    base_w = block_balanced_weights(work.s_elv.to_numpy(float),
                                    work.block.to_numpy("int64"))
    aw = design * np.sqrt(base_w)[:, None]
    norms = np.linalg.norm(aw, axis=0)
    norms = np.where(norms > 0, norms, 1.0)
    condition = float(np.linalg.cond(aw / norms))
    if not np.isfinite(condition) or condition > 100.0:
        raise ValueError(f"ajuste mal condicionado ({condition:.1f})")

    weight = base_w.copy()
    beta = np.zeros(design.shape[1])
    for _ in range(12):
        root = np.sqrt(weight)
        new_beta, *_ = np.linalg.lstsq(design * root[:, None], values * root,
                                       rcond=None)
        resid = values - design @ new_beta
        scale = max(1.4826 * np.median(np.abs(resid - np.median(resid))), 0.02)
        huber = np.minimum(1.0, 1.345 * scale / np.maximum(np.abs(resid), 1e-12))
        new_w = base_w * huber
        if np.allclose(new_beta, beta, rtol=1e-8, atol=1e-9):
            beta, weight = new_beta, new_w
            break
        beta, weight = new_beta, new_w
    resid = values - design @ beta

    xtwx = design.T @ (weight[:, None] * design)
    bread = np.linalg.pinv(xtwx)
    meat = np.zeros_like(xtwx)
    blocks = work.block.to_numpy("int64")
    ub = np.unique(blocks)
    for b in ub:
        m = blocks == b
        s = design[m].T @ (weight[m] * resid[m])
        meat += np.outer(s, s)
    cov = bread @ meat @ bread
    n, p, G = len(work), design.shape[1], len(ub)
    if G > 1 and n > p:
        cov *= (G / (G - 1)) * ((n - 1) / (n - p))
    cluster_se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    lon, lat = Transformer.from_crs(3031, 4326, always_xy=True).transform(
        float(cx), float(cy))
    rows = []
    for k, w in enumerate(keep):
        m = work.window.eq(w).to_numpy()
        part = work.loc[m]
        lw, lb = weight[m], base_w[m]
        if lw.sum() <= 0:
            continue
        t_center = float(np.average(part.t_year.to_numpy(float), weights=lb))
        nb = int(part.block.nunique())
        rmse = float(np.sqrt(np.average(resid[m] ** 2, weights=lw)))
        sigma = max(float(cluster_se[k]), rmse / np.sqrt(max(nb, 1)))
        rows.append({
            "cell_x": float(cx), "cell_y": float(cy),
            "lon": float(lon), "lat": float(lat), "window": w,
            "season": w[:3], "t_center_year": t_center,
            "t_center": t_year_to_datetime(t_center).isoformat(),
            "h": float(beta[k]) + offset, "sigma_h": float(sigma),
            "h_anomaly_vs_REMA": float(beta[k]),
            "rema_offset_m": offset, "value_col": value_col,
            "N_obs": int(len(part)), "N_blocks": nb,
            "fit_rmse_window_m": rmse,
            "fit_condition_number": condition,
            "temporal_span_days": float(
                (part.t_year.max() - part.t_year.min()) * 365.25),
        })
    res = pd.DataFrame(rows).sort_values("t_center_year").reset_index(drop=True)
    return res


# ------------------------------------------------------------------- FAC
class FDM:
    """GSFC-FDM v1.2.1 recortado. Respeita a mascara de fill (-9999)."""

    def __init__(self, path):
        import netCDF4 as nc
        d = nc.Dataset(path)
        self.t = np.asarray(d["time"][:], dtype=float)
        self.x = np.asarray(d["x"][:], dtype=float)
        self.y = np.asarray(d["y"][:], dtype=float)
        fac = d["FAC"][:]
        smb = d["SMB_a"][:]
        # np.asarray descartaria a mascara e devolveria -9999 cru
        self.fac = np.ma.filled(fac, np.nan).astype(float)
        self.smb = np.ma.filled(smb, np.nan).astype(float)
        d.close()

    def _series_at(self, cube, cx, cy):
        ix = int(np.argmin(np.abs(self.x - cx)))
        iy = int(np.argmin(np.abs(self.y - cy)))
        return cube[:, iy, ix], float(self.x[ix]), float(self.y[iy])

    def sample(self, cube_name, cx, cy, years):
        cube = self.fac if cube_name == "fac" else self.smb
        series, gx, gy = self._series_at(cube, cx, cy)
        ok = np.isfinite(series)
        out = np.full(len(years), np.nan)
        if ok.sum() >= 2:
            inside = (np.asarray(years) >= self.t[ok].min()) & \
                     (np.asarray(years) <= self.t[ok].max())
            out[inside] = np.interp(np.asarray(years)[inside],
                                    self.t[ok], series[ok])
        dist = float(np.hypot(gx - cx, gy - cy))
        return out, dist, float(self.t[ok].min()), float(self.t[ok].max())


# --------------------------------------------------------------- tendencia
def trend(series, value_col, sigma_col, seed):
    t = series.t_center_year.to_numpy(float)
    y = series[value_col].to_numpy(float)
    s = series[sigma_col].to_numpy(float)
    ok = np.isfinite(y) & np.isfinite(s)
    t, y, s = t[ok], y[ok], s[ok]
    if len(t) < 5 or np.ptp(t) < 2.0:
        return {"statistically_valid": False, "reason": "n<5 ou span<2 anos",
                "n_windows": int(len(t))}
    slope = float(theilslopes(y, t, alpha=0.95).slope)
    tref = float(np.mean(t))
    intercept = float(np.median(y - slope * (t - tref)))
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOT)
    for i in range(BOOT):
        draws[i] = theilslopes(y + rng.normal(0.0, s), t, alpha=0.95).slope
    lo, hi = np.percentile(draws, [2.5, 97.5])
    p = float(2.0 * min((draws <= 0).mean(), (draws >= 0).mean()))
    return {"statistically_valid": True,
            "method": "Theil-Sen; Monte Carlo parametrico condicional a sigma",
            "n_windows": int(len(t)), "time_span_years": float(np.ptp(t)),
            "slope_m_per_year": slope,
            "slope_sigma_m_per_year": float(draws.std(ddof=1)),
            "slope_ci95_low": float(lo), "slope_ci95_high": float(hi),
            "slope_p_two_sided": min(p, 1.0),
            "intercept_at_mean_time": intercept, "t_reference": tref,
            "thinning_supported": bool(hi < 0.0),
            "thickening_supported": bool(lo > 0.0)}


# ============================================================ derivadas
def enrich(series, fdm, cx, cy):
    years = t_year_to_calendar_decimal_year(
        series.t_center_year.to_numpy(float))
    fac, dist_fac, t0, t1 = fdm.sample("fac", cx, cy, years)
    smb, _, _, _ = fdm.sample("smb", cx, cy, years)
    series = series.copy()
    series["calendar_year"] = years
    series["FAC"] = fac
    series["SMB_a"] = smb
    series["h_ice"] = series.h - series.FAC
    # a sigma do FDM nao e fornecida pelo recorte; nao se inventa valor
    series["sigma_h_ice"] = series.sigma_h
    h0 = float(series.h.iloc[0])
    series["dh_cum_m"] = series.h - h0
    series["sigma_dh_cum_m"] = np.hypot(series.sigma_h,
                                        float(series.sigma_h.iloc[0]))
    series.loc[series.index[0], "sigma_dh_cum_m"] = 0.0
    base_ice = float(series.h_ice.iloc[0])
    series["dh_ice_cum_m"] = series.h_ice - base_ice
    series["dM_cum_Gt"] = series.dh_ice_cum_m * GT_PER_M
    series["sigma_dM_cum_Gt"] = series.sigma_dh_cum_m * GT_PER_M
    series["SLE_cum_mm"] = -series.dM_cum_Gt / GT_PER_MM_SLE
    series["fac_window_valid"] = np.isfinite(series.FAC)
    return series, {"fdm_cell_distance_m": dist_fac,
                    "fdm_time_min": t0, "fdm_time_max": t1}


def transitions(series):
    tr = pd.DataFrame({
        "de": series.window.values[:-1], "para": series.window.values[1:],
        "dt_anos": np.diff(series.t_center_year.values),
    })
    tr["dh_m"] = np.diff(series.h.values)
    tr["sigma_dh_m"] = np.hypot(series.sigma_h.values[1:],
                                series.sigma_h.values[:-1])
    tr["dFAC_m"] = np.diff(series.FAC.values)
    tr["dh_ice_m"] = np.diff(series.h_ice.values)
    tr["dhdt_m_ano"] = tr.dh_m / tr.dt_anos
    tr["sigma_dhdt_m_ano"] = tr.sigma_dh_m / tr.dt_anos
    tr["dh_icedt_m_ano"] = tr.dh_ice_m / tr.dt_anos
    tr["dM_Gt"] = tr.dh_ice_m * GT_PER_M
    tr["sigma_dM_Gt"] = tr.sigma_dh_m * GT_PER_M
    tr["dMdt_Gt_ano"] = tr.dh_icedt_m_ano * GT_PER_M
    tr["SLE_mm"] = -tr.dM_Gt / GT_PER_MM_SLE
    tr["z"] = tr.dh_m.abs() / tr.sigma_dh_m
    tr["tipo"] = np.where(tr.de.str.startswith("JJA"), "JJA->DJF", "DJF->JJA")
    return tr


def _short(w):
    return ("JJA " + w[-2:]) if w.startswith("JJA") \
        else ("DJF " + w[6:8] + "/" + w[-2:])
