"""Exporta para CSV os dados que as figuras em R (manuscrito/figuras_R/*.R) leem.

O R desta máquina não tem leitores de NetCDF nem Parquet (ncdf4, arrow, terra ausentes), então a
leitura binária fica em Python e o R recebe CSV. Nada é recalculado: só recorte, subamostragem da
máscara para o mapa (2 km) e cópia de colunas.

Saída: manuscrito/figuras/dados/R_*.csv
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
I = ROOT / "data" / "interim"
SRC = ROOT / "manuscrito" / "figuras" / "dados"
BM_PATH = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"


def mascara(xmin, xmax, ymin, ymax, step):
    with xr.open_dataset(BM_PATH) as bm:
        sub = bm["mask"].sel(x=slice(xmin, xmax), y=slice(ymax, ymin))[::step, ::step]
        xx, yy = np.meshgrid(sub.x.values / 1e3, sub.y.values / 1e3)
        return pd.DataFrame({"x_km": xx.ravel(), "y_km": yy.ravel(), "mask": sub.values.ravel()})


def main():
    SRC.mkdir(parents=True, exist_ok=True)
    # mapa regional (2 km) e inset continental (30 km); mask BedMachine: 0 oceano, 1 rocha, 2 apoiado,
    # 3 flutuante, 4 lago Vostok
    mascara(-1900e3, -1150e3, -800e3, -50e3, 4).to_csv(SRC / "R_fig1_mascara_ase_2km.csv", index=False)
    mascara(-3.3e6, 3.3e6, -3.3e6, 3.3e6, 60).to_csv(SRC / "R_fig1_mascara_antartica_30km.csv", index=False)

    # produto basal v2 (500 m): pixels com m_b medido, unidade e marca das manchas negativas
    sys.path.insert(0, str(ROOT / "pipelines"))
    from run_blue_spots_ase import negative_flags
    nr, rb = negative_flags()
    with xr.open_dataset(I / "basal_melt" / "mb_final_v2_500m.nc") as d:
        u = d.unidade.values.astype(str)
        m = d.mb.values
        xx, yy = np.meshgrid(d.x.values, d.y.values)
    ok = np.isfinite(m) & (u != "")
    pd.DataFrame({"x_m": xx[ok], "y_m": yy[ok], "unidade": u[ok], "mb_mpa": m[ok],
                  "mancha_ruido": np.asarray(nr)[ok], "mancha_persistente": np.asarray(rb)[ok]}
                 ).to_csv(SRC / "R_mb_pixels_500m.csv", index=False)
    # domínio do produto (com e sem medida) a 2 km, para o mapa da Fig. 1
    with xr.open_dataset(I / "basal_melt" / "mb_final_v2_500m.nc") as d:
        s = d.unidade[::4, ::4]
        xx, yy = np.meshgrid(s.x.values / 1e3, s.y.values / 1e3)
        uu = s.values.astype(str)
    k = uu != ""
    pd.DataFrame({"x_km": xx[k], "y_km": yy[k], "unidade": uu[k]}).to_csv(
        SRC / "R_fig1_dominio_basal_2km.csv", index=False)
    # d* (f = 0,8 / 0,9 / 0,95) e patamar P por plataforma, dos atributos da máscara hidrostática
    import json
    with xr.open_dataset(I / "hydrostatic_mask_500m.nc") as hm:
        par = json.loads(hm.classe.attrs["parametros"])
    pd.DataFrame([{"plataforma": k, "P": v["P"], "d_f080_km": v["d_star_km"]["f080"],
                   "d_f090_km": v["d_star_km"]["f090"], "d_f095_km": v["d_star_km"]["f095"]}
                  for k, v in par.items()]).to_csv(SRC / "R_dstar_plataformas.csv", index=False)
    print("ok")


if __name__ == "__main__":
    main()
