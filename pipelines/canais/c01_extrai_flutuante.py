"""Canais basais (ASE, ICESat-2), passo 1: segmentos ATL06 r007 sobre gelo flutuante hidrostático.

Entrada: data/ase_v2/{djf,mam,jja,son}/ATL06_*.parquet (ATL06 r007 extraído, 40 m; h_elv = h_li;
beam 1..6 = gt1l, gt1r, gt2l, gt2r, gt3l, gt3r). Máscara: data/interim/hydrostatic_mask_500m.nc
(classe 1 = hidrostático além da zona de flexão d*; plataforma 1..6 = PIG, Thwaites, Crosson,
Dotson, Cosgrove, Getz). A máscara é estática (fronteiras de 2021), o que basta para selecionar
segmentos; a frente de cada data é tratada depois.

Filtros: atl06_quality_summary == 0; h finito e |h| < 1e4 (o fill de 3,4e38 é finito);
dentro da classe 1. Nada de correção é aplicado aqui: as alturas, marés, DAC e geoide são
guardados para que cada passo escolha o que subtrair.
Saída: data/interim/canais/flutuante_ase.parquet
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "data" / "interim" / "canais"
COLS = ["rgt", "cycle", "beam", "x", "y", "h_elv", "t_year", "delta_time", "atl06_quality_summary",
        "h_robust_sprd", "sigma_geo_h", "dh_fit_dx", "tide_ocean", "tide_equilibrium", "dac", "geoid",
        "tide_load", "x_atc"]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    m = xr.open_dataset(ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc")
    cl = m.classe.values
    pl = m.plataforma.values
    x0, dx = float(m.x[0]), float(m.x[1] - m.x[0])
    y0, dy = float(m.y[0]), float(m.y[1] - m.y[0])
    arqs = sorted(glob.glob(str(ROOT / "data" / "ase_v2" / "*" / "ATL06_*.parquet")))
    partes, n_lidos = [], 0
    for i, f in enumerate(arqs):
        d = pd.read_parquet(f, columns=COLS)
        n_lidos += len(d)
        d = d[(d.atl06_quality_summary == 0) & np.isfinite(d.h_elv) & (d.h_elv.abs() < 1e4)]
        if d.empty:
            continue
        ix = np.rint((d.x.to_numpy() - x0) / dx).astype(int)
        iy = np.rint((d.y.to_numpy() - y0) / dy).astype(int)
        ok = (ix >= 0) & (ix < cl.shape[1]) & (iy >= 0) & (iy < cl.shape[0])
        c = np.zeros(len(d), np.int8)
        p = np.zeros(len(d), np.int16)
        c[ok] = cl[iy[ok], ix[ok]]
        p[ok] = pl[iy[ok], ix[ok]]
        sel = (c == 1) & (p > 0)
        if sel.any():
            dd = d[sel].drop(columns="atl06_quality_summary").assign(plataforma=p[sel], estacao=Path(f).parent.name)
            partes.append(dd)
        if i % 500 == 0:
            print(i, len(arqs), sum(len(q) for q in partes), flush=True)
    out = pd.concat(partes, ignore_index=True)
    for k in ["tide_ocean", "tide_equilibrium", "dac", "geoid", "tide_load", "sigma_geo_h", "h_robust_sprd"]:
        v = out[k].to_numpy(dtype=float)
        v[np.abs(v) > 1e30] = np.nan
        out[k] = v
    out.to_parquet(OUT / "flutuante_ase.parquet", index=False)
    print("lidos", n_lidos, "flutuante hidrostático", len(out))
    print(out.groupby("plataforma").agg(n=("h_elv", "size"), rgts=("rgt", "nunique"), ciclos=("cycle", "nunique")))


if __name__ == "__main__":
    main()
