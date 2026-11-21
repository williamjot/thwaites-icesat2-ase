"""Canais basais, passo 2b: separa canais basais de fendas, riftes e ondulações.

As depressões do passo 2 incluem fendas e riftes (estreitos e muitas vezes transversais ao fluxo).
Um canal basal tem duas propriedades verificáveis:
  (1) largura >= LMIN_CANAL: feições basais mais estreitas que ~2–3 espessuras de gelo (H ~ 200–600 m)
      são sustentadas por ponte elástica e quase não aparecem na superfície; 1 km é o limiar adotado
      (testado por sensibilidade no passo 3);
  (2) continuidade ao longo do fluxo: a depressão persiste em seções perpendiculares ao fluxo
      deslocadas ±1 e ±2 km a montante e a jusante (REMA 32 m; direção do fluxo = média ITS_LIVE
      2019–2025). Fendas e riftes transversais não persistem.
Também mede o ângulo θ entre a trilha e o fluxo: com θ < 30° a trilha corre quase ao longo do canal
e a seção não é transversal; descartada.
Saída: data/interim/canais/canais_validados.parquet (todas as candidatas com as métricas e o veredito)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[2]
IN = ROOT / "data" / "interim" / "canais"
LMIN_CANAL = 1000.0
OFFS = (-2000.0, -1000.0, 1000.0, 2000.0)


def main():
    e = pd.read_parquet(IN / "canais_empilhados.parquet")
    s = pd.read_parquet(IN / "secoes.parquet")
    # direção da trilha no centro: dos pontos x,y das seções do mesmo canal não dá (são o mesmo
    # centro); usa todos os segmentos do rgt/feixe próximos do centro
    fl = pd.read_parquet(IN / "flutuante_ase.parquet", columns=["rgt", "beam", "x", "y"])
    v = xr.open_dataset(ROOT / "data" / "velocity_itslive_annual.nc")[["vx", "vy"]]
    rema = rasterio.open(ROOT / "data" / "REMA_ase_32m.tif")
    out = []
    for (rgt, beam), g in e.groupby(["rgt", "beam"]):
        f = fl[(fl.rgt == rgt) & (fl.beam == beam)]
        for _, r in g.iterrows():
            near = f[(np.abs(f.x - r.x) < 3000) & (np.abs(f.y - r.y) < 3000)]
            if len(near) < 20:
                continue
            xy = near[["x", "y"]].to_numpy()
            tdir = np.linalg.svd(xy - xy.mean(0), full_matrices=False)[2][0]
            vv = v.sel(x=r.x, y=r.y, method="nearest")
            vx, vy = float(np.nanmean(vv.vx)), float(np.nanmean(vv.vy))
            sp = np.hypot(vx, vy)
            if not np.isfinite(sp) or sp < 1:
                continue
            fdir = np.array([vx, vy]) / sp
            theta = np.degrees(np.arccos(min(abs(tdir @ fdir), 1.0)))
            pdir = np.array([-fdir[1], fdir[0]])
            L = max(3 * r.largura_m, 3000.0)
            w = rema.read(1, window=from_bounds(r.x - L - 2500, r.y - L - 2500, r.x + L + 2500, r.y + L + 2500,
                                                 rema.transform).round_offsets().round_lengths(), boundless=True,
                          fill_value=rema.nodata).astype(float)
            w[w == rema.nodata] = np.nan
            tr = rasterio.windows.transform(from_bounds(r.x - L - 2500, r.y - L - 2500, r.x + L + 2500, r.y + L + 2500,
                                                       rema.transform).round_offsets().round_lengths(), rema.transform)

            def secao(off):
                c = np.array([r.x, r.y]) + off * fdir
                q = np.arange(-L, L + 1, 32.0)
                px, py = c[0] + q * pdir[0], c[1] + q * pdir[1]
                col = np.floor((px - tr.c) / tr.a).astype(int)
                row = np.floor((py - tr.f) / tr.e).astype(int)
                ok = (row >= 0) & (row < w.shape[0]) & (col >= 0) & (col < w.shape[1])
                z = np.full(len(q), np.nan)
                z[ok] = w[row[ok], col[ok]]
                hw = r.largura_m / 2
                cen = np.abs(q) <= hw
                fl_ = (np.abs(q) >= r.largura_m) & (np.abs(q) <= 2 * r.largura_m)
                if np.isfinite(z[cen]).sum() < 3 or np.isfinite(z[fl_]).sum() < 6:
                    return np.nan
                return np.nanmedian(z[fl_]) - np.nanmin(z[cen])

            d0 = secao(0.0)
            dd = [secao(o) for o in OFFS]
            pers = np.nanmean([x / d0 > 0.5 for x in dd if np.isfinite(x)]) if np.isfinite(d0) and d0 > 0 else np.nan
            # gradiente da profundidade ao longo do fluxo (m de bordo livre por m), para o termo advectivo
            oo = np.array([OFFS[0], OFFS[1], 0.0, OFFS[2], OFFS[3]])
            zz = np.array([dd[0], dd[1], d0, dd[2], dd[3]])
            okg = np.isfinite(zz)
            grad = np.polyfit(oo[okg], zz[okg], 1)[0] if okg.sum() >= 4 else np.nan
            out.append(dict(canal=r.canal, theta=theta, vel=sp, vx=vx, vy=vy, D_rema=d0, persistencia=pers,
                            dDds_rema=grad, **{f"D_rema_{int(o)}": x for o, x in zip(OFFS, dd)}))
    m = e.merge(pd.DataFrame(out), on="canal", how="left")
    m["canal_basal"] = (m.largura_m >= LMIN_CANAL) & (m.theta >= 30) & (m.persistencia >= 0.75)
    m.to_parquet(IN / "canais_validados.parquet", index=False)
    print(m.groupby("plataforma").agg(cand=("canal", "size"), largos=("largura_m", lambda x: (x >= LMIN_CANAL).sum()),
                                       validos=("canal_basal", "sum")))
    print(m[m.canal_basal].groupby("plataforma")[["prom", "largura_m", "theta", "vel"]].median().round(1))


if __name__ == "__main__":
    main()
