"""Desprendimentos por plataforma: posição da frente (IceLines) e água/mélange no setor da frente (ICESat-2).

1. **Posição da frente (IceLines, Sentinel-1, mensal ou trimestral, 2015–2024,7):** projeta os pontos
   de cada frente na direção média do fluxo da unidade (ITS_LIVE 2019–2025), dentro de um corredor
   central de ±8 km em torno do trecho mais rápido da frente (10% de maior velocidade ITS_LIVE), e usa a mediana da metade
   mais avançada (a face de desprendimento; as margens laterais ficam fora do corredor). Posição relativa à
   primeira época de 2019. **Evento de desprendimento:** recuo > LIMIAR km entre épocas
   consecutivas (avanço contínuo é fluxo; recuo brusco é perda de gelo).
   Associação: PineIsland → PIG; Thwaites1 → TEIS; Thwaites2 → TWGT (pelo centro geométrico);
   Crosson, Dotson, Cosgrove.
2. **Água/mélange no setor da frente (ICESat-2, trimestral, 2019–2025):** fração dos segmentos
   com altura sobre o mar < 5 m, entre os segmentos da máscara de plataforma de 2021 até 10 km da
   frente de 2021. Cobre 2025, onde o IceLines não chega. Independente do IceLines.

Saída: ``outputs/series_v2/frente_posicao.csv``, ``desprendimentos.csv``, ``agua_setor_frente.csv``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
FRONTS = ROOT / "data" / "calving_fronts_densified_500m.parquet"
V2 = ROOT / "data" / "interim" / "basal_melt" / "mb_final_v2_500m.nc"
VEL = ROOT / "data" / "velocity_itslive_annual.nc"
THK = ROOT / "data" / "interim" / "thickness"
OUT = ROOT / "outputs" / "series_v2"
MAP = {"Pine_Island": "PineIsland", "Thwaites_TEIS": "Thwaites1", "Thwaites_TWGT": "Thwaites2",
       "Crosson": "Crosson", "Dotson": "Dotson", "Cosgrove": "Cosgrove"}
SHELF_FILE = {"Pine_Island": "Pine_Island", "Thwaites_TEIS": "Thwaites", "Thwaites_TWGT": "Thwaites",
              "Crosson": "Crosson", "Dotson": "Dotson", "Cosgrove": "Cosgrove"}
LIMIAR_KM = 1.0
CORR_KM = 8.0
SEASON_OF_MONTH = {12: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 2, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    fr = pd.read_parquet(FRONTS)
    v2 = xr.open_dataset(V2)
    XX, YY = np.meshgrid(v2.x.values, v2.y.values)
    unit = v2.unidade.values
    vel = xr.open_dataset(VEL)
    pos_rows, ev_rows, water_rows = [], [], []
    for u, sh in MAP.items():
        su = unit == u
        cx, cy = XX[su].mean(), YY[su].mean()
        vx = float(vel.vx.sel(x=cx, y=cy, method="nearest").mean())
        vy = float(vel.vy.sel(x=cx, y=cy, method="nearest").mean())
        # direção média do fluxo na unidade (amostra de pixels)
        idx = np.random.default_rng(0).choice(su.sum(), min(2000, su.sum()), replace=False)
        px, py = XX[su][idx], YY[su][idx]
        sx = vel.vx.mean("time").interp(x=xr.DataArray(px), y=xr.DataArray(py)).values
        sy = vel.vy.mean("time").interp(x=xr.DataArray(px), y=xr.DataArray(py)).values
        ux, uy = np.nanmean(sx), np.nanmean(sy)
        nrm = np.hypot(ux, uy) or np.hypot(vx, vy)
        ux, uy = ux / nrm, uy / nrm
        f = fr[fr.shelf == sh].copy()
        f["s"] = (f.x * ux + f.y * uy) / 1e3
        f["n"] = (-f.x * uy + f.y * ux) / 1e3
        # corredor central: ±CORR_KM em torno da linha de fluxo pelo centro da unidade. A 1ª versão
        # (30% mais avançados da linha inteira) era dominada pelas margens laterais e não via o
        # grande desprendimento da PIG de 2020.
        # centro do corredor: trecho da frente de maior velocidade (face do tronco rápido). Centrar no
        # centro geométrico da unidade falhava na PIG, cuja unidade junta o lobo principal e o oeste.
        f0 = f[(f.epoch_year >= 2019.0) & (f.epoch_year <= 2020.0)]
        f0 = f0 if len(f0) else f
        spd = np.hypot(vel.vx.mean("time").interp(x=xr.DataArray(f0.x.values), y=xr.DataArray(f0.y.values)).values,
                       vel.vy.mean("time").interp(x=xr.DataArray(f0.x.values), y=xr.DataArray(f0.y.values)).values)
        fast = np.nan_to_num(spd) >= np.nanquantile(spd, 0.9)
        n0 = float(np.median(f0.n.values[fast]))
        f = f[np.abs(f.n - n0) <= CORR_KM]
        g = f.groupby("epoch_year").s.apply(lambda v: float(np.median(v[v >= np.quantile(v, 0.5)]))
                                              if len(v) >= 5 else np.nan).dropna()
        g = g.sort_index()
        ref = g[g.index >= 2019.0]
        ref = float(ref.iloc[0]) if len(ref) else float(g.iloc[0])
        pos = g - ref
        for t, p in pos.items():
            pos_rows.append({"unidade": u, "epoca": t, "posicao_frente_km": p})
        dp = pos.diff()
        for t, step in dp.items():
            if step < -LIMIAR_KM:
                prev = pos.index[pos.index.get_loc(t) - 1]
                ev_rows.append({"unidade": u, "entre": prev, "e": t, "recuo_km": -step})
        # água/mélange no setor da frente (ICESat-2)
        d = pd.read_parquet(THK / f"espessura_{SHELF_FILE[u]}.parquet",
                            columns=["x", "y", "h_sea", "granule_month", "granule_year"])
        f21 = f[(f.epoch_year >= 2020.9) & (f.epoch_year <= 2021.3)]
        if f21.empty:
            f21 = f[f.epoch_year >= 2019]
        dist = cKDTree(f21[["x", "y"]].to_numpy()).query(np.c_[d.x, d.y])[0] / 1e3
        jj = np.clip(np.round((d.x.to_numpy() - v2.x.values[0]) / 500).astype(int), 0, len(v2.x) - 1)
        ii = np.clip(np.round((d.y.to_numpy() - v2.y.values[0]) / -500).astype(int), 0, len(v2.y) - 1)
        sel = (dist <= 10.0) & (unit[ii, jj] == u)
        d = d[sel]
        s_idx = np.vectorize(SEASON_OF_MONTH.get)(d.granule_month.to_numpy())
        yy = d.granule_year.to_numpy() + (d.granule_month.to_numpy() == 12)
        k = (yy - 2019) * 4 + s_idx
        w = pd.DataFrame({"k": k, "agua": d.h_sea.to_numpy() < 5.0}).groupby("k").agg(
            frac_agua=("agua", "mean"), n=("agua", "size"))
        for kk, r in w.iterrows():
            if 0 <= kk < 28:
                water_rows.append({"unidade": u, "t_meio": 2019 + kk // 4 + [0.0, 0.25, 0.5, 0.75][kk % 4] + 0.04,
                                   "frac_agua": r.frac_agua, "n": int(r.n)})
        print(f"{u}: {len(pos)} épocas; {sum(e['unidade'] == u for e in ev_rows)} recuos > {LIMIAR_KM} km",
              flush=True)
    pd.DataFrame(pos_rows).to_csv(OUT / "frente_posicao.csv", index=False)
    ev = pd.DataFrame(ev_rows)
    ev.to_csv(OUT / "desprendimentos.csv", index=False)
    pd.DataFrame(water_rows).to_csv(OUT / "agua_setor_frente.csv", index=False)
    pd.set_option("display.width", 200)
    print(ev.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
