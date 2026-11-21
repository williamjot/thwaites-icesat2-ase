"""InSAR + ICESat-2, passo C: recuo previsto pela flutuação × recuo observado por InSAR.

Para cada linha de fluxo (passo B):
  HAF(s) = H(s) + (ρ_w/ρ_i)·b(s)   (altura acima da flutuação; BedMachine v4.1; ρ_w = 1027, ρ_i = 917)
  gradiente G = dHAF/ds por ajuste linear em s ∈ [s_GL − 3 km, s_GL] (a montante da linha média
  InSAR de 2019–2025, s_GL);
  dh/dt = média ponderada (1/σ²) das células do passo A a ≤ 750 m da linha de fluxo nesse trecho;
  recuo previsto v_prev = −(dh/dt)/G (m/ano; negativo = para o interior). Supõe leito fixo,
  dH/dt = dh/dt (gelo apoiado; firn e GIA desprezados junto à linha) e que a linha acompanha a
  flutuação (sem flexão, sem intrusão de água).
Incerteza: Monte Carlo (2.000): leito perturbado por ruído gaussiano com σ = errbed e correlação
exponencial de 1 km ao longo da linha; dh/dt perturbado pelo seu σ (inflado por √χ² das células).
Comparação: razão R = v_obs/v_prev e diferença v_obs − v_prev, com v_obs = inclinação InSAR
2019–2025 (passo B). R > 1: recuo mais rápido que a flutuação explica.
Controle de consistência: HAF em s_GL (deveria ser ~0 se BedMachine e InSAR concordam).
Saída: outputs/gl_migration/previsao_vs_observado.parquet
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs" / "gl_migration"
RNG = np.random.default_rng(20261117)
RW, RI = 1027.0, 917.0
JAN = 3000.0
NMC = 2000


def amostra_bm(x, y):
    bm = xr.open_dataset(ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc")
    sub = bm[["thickness", "bed", "errbed", "mask"]].sel(x=slice(x.min() - 2000, x.max() + 2000),
                                                         y=slice(y.max() + 2000, y.min() - 2000)).load()
    X, Y = xr.DataArray(x, dims="p"), xr.DataArray(y, dims="p")
    H = sub.thickness.interp(x=X, y=Y).values
    b = sub.bed.interp(x=X, y=Y).values
    e = sub.errbed.interp(x=X, y=Y).values
    return H, b, e


def ruido_correlacionado(n, ds, L=1000.0):
    rho = np.exp(-ds / L)
    z = np.empty(n)
    z[0] = RNG.standard_normal()
    for i in range(1, n):
        z[i] = rho * z[i - 1] + np.sqrt(1 - rho ** 2) * RNG.standard_normal()
    return z


def main():
    vert = pd.read_parquet(OUT / "fluxo_linhas.parquet")
    rec = pd.read_parquet(OUT / "recuo_insar_por_linha.parquet")
    rec = rec[rec.inicio == 2019]
    dh = pd.read_parquet(OUT / "dhdt_junto_linha.parquet")
    dh = dh[np.isfinite(dh.se) & (dh.se > 0)]
    tdh = cKDTree(np.c_[dh.x, dh.y])
    H, b, e = amostra_bm(vert.x.to_numpy(), vert.y.to_numpy())
    vert = vert.assign(H=H, bed=b, errbed=e, haf=H + RW / RI * b)
    linhas = []
    for _, r in rec.iterrows():
        v = vert[vert.fid == r.fid].sort_values("s")
        tr = v[(v.s >= r.s_medio - JAN) & (v.s <= r.s_medio)]
        if len(tr) < 8 or tr.haf.isna().any():
            continue
        G0 = np.polyfit(tr.s, tr.haf, 1)[0]
        haf_gl = np.interp(r.s_medio, v.s, v.haf)
        idx = sorted(set(j for lst in tdh.query_ball_point(np.c_[tr.x, tr.y], 750) for j in lst))
        if len(idx) < 3:
            continue
        c = dh.iloc[idx]
        w = 1 / c.se ** 2
        m = np.sum(w * c.dhdt) / np.sum(w)
        chi = np.sum(w * (c.dhdt - m) ** 2) / max(len(c) - 1, 1)
        sm = np.sqrt(1 / np.sum(w)) * np.sqrt(max(chi, 1))
        # Monte Carlo
        vs = []
        for _ in range(NMC):
            hafp = tr.haf.to_numpy() + RW / RI * tr.errbed.to_numpy() * ruido_correlacionado(len(tr), 250.0)
            G = np.polyfit(tr.s, hafp, 1)[0]
            dd = m + sm * RNG.standard_normal()
            vs.append(-dd / G if G < 0 else np.nan)
        vs = np.array(vs)
        ok = np.isfinite(vs)
        linhas.append(dict(fid=r.fid, geleira=r.geleira, s_gl=r.s_medio, haf_gl=haf_gl, G=G0,
                           dhdt=m, dhdt_sd=sm, n_cel=len(c), frac_G_neg=ok.mean(),
                           v_prev=-m / G0 if G0 < 0 else np.nan,
                           v_prev_lo=np.nanpercentile(vs, 2.5) if ok.mean() > 0.5 else np.nan,
                           v_prev_hi=np.nanpercentile(vs, 97.5) if ok.mean() > 0.5 else np.nan,
                           v_obs=r.recuo_m_ano, v_obs_lo=r.lo, v_obs_hi=r.hi, t0=r.t0, t1=r.t1,
                           n_meses=r.n_meses, x=np.interp(r.s_medio, v.s, v.x), y=np.interp(r.s_medio, v.s, v.y)))
    d = pd.DataFrame(linhas)
    d["razao"] = d.v_obs / d.v_prev
    d.to_parquet(OUT / "previsao_vs_observado.parquet", index=False)
    with pd.option_context("display.width", 200):
        print(d.groupby("geleira").agg(n=("fid", "size"), haf_gl=("haf_gl", "median"), G=("G", "median"),
                                       dhdt=("dhdt", "median"), v_prev=("v_prev", "median"),
                                       v_obs=("v_obs", "median"), razao=("razao", "median")).round(3))


if __name__ == "__main__":
    main()
