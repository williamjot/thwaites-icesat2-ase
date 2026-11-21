"""Segundo produto de velocidade: MEaSUREs anual (NSIDC-0720, SAR, Rignot/Mouginot) × ITS_LIVE.

O ITS_LIVE (óptico/SAR combinados, Gardner) é o único produto de velocidade da cadeia. O
NSIDC-0720 (1 km, anos de julho a junho, 2019/20–2024/25) é independente em sensor e
processamento. Lê-se só o recorte da ASE por acesso remoto (h5netcdf), sem baixar os arquivos
continentais de 1,3 GB.

Comparação: H·∇·u médio por unidade, no domínio final (``dominio_final_500m.nc``), com a mesma
derivada (gaussiana normalizada, ``run_divergence_ase``) e as mesmas escalas (2, 3, 5 km; 1 km não,
porque o NSIDC-0720 tem pixel de 1 km). A diferença entre os produtos entra como mais um fator
estrutural.

Saídas: ``data/velocity_measures_annual_ase.nc`` e ``outputs/velocity_crosscheck/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_divergence_ase import divergence, to_mask_grid  # noqa: E402

OUT_V = ROOT / "data" / "velocity_measures_annual_ase.nc"
OUT = ROOT / "outputs" / "velocity_crosscheck"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
FLUX = ROOT / "data" / "interim" / "flux_term" / "H_div_500m.nc"
DOM = ROOT / "data" / "interim" / "basal_melt" / "dominio_final_500m.nc"
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
UNITS = ["Pine_Island", "Thwaites_TEIS", "Crosson", "Dotson", "Cosgrove"]


def fetch():
    import earthaccess
    if OUT_V.is_file():
        return xr.open_dataset(OUT_V)
    earthaccess.login(strategy="netrc")
    m = xr.open_dataset(MASK)
    ii, jj = np.nonzero(m.classe.values > 0)
    pad = 40e3
    x0, x1 = m.x.values[jj].min() - pad, m.x.values[jj].max() + pad
    y0, y1 = m.y.values[ii].min() - pad, m.y.values[ii].max() + pad
    g = earthaccess.search_data(short_name="NSIDC-0720", temporal=("2019-07-01", "2025-06-30"),
                                count=-1)
    layers = []
    for f, gr in zip(earthaccess.open(g), g):
        name = gr.data_links()[0].split("/")[-1]
        yr = int(name.split("_")[3])                       # ano de início (julho)
        ds = xr.open_dataset(f, engine="h5netcdf")[["VX", "VY", "ERRX", "ERRY", "CNT"]]
        sub = ds.sel(x=slice(x0, x1), y=slice(y1, y0)).load()
        sub = sub.expand_dims(ano_inicio=[yr])
        layers.append(sub)
        print(f"{name}: recorte {sub.sizes}", flush=True)
    v = xr.concat(layers, "ano_inicio").sortby("ano_inicio")
    v.attrs = {"fonte": "NSIDC-0720 MEaSUREs Annual Antarctic Ice Velocity v1 (recorte ASE)"}
    v.to_netcdf(OUT_V)
    return v


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    v = fetch()
    m = xr.open_dataset(MASK)
    mx, my = m.x.values, m.y.values
    fl = xr.open_dataset(FLUX)
    H = fl.H_gemb.values
    dom = xr.open_dataset(DOM).dominio_final.values == 1
    unit = xr.open_dataset(MB).unidade.values
    xs, ys = v.x.values.astype(float), v.y.values.astype(float)
    vx = v.VX.values.astype(np.float32)
    vy = v.VY.values.astype(np.float32)
    for a in (vx, vy):
        a[~np.isfinite(a) | (np.abs(a) > 2e4)] = np.nan
    rows = []
    for sc in (2, 3, 5):
        divs = np.stack([divergence(vx[k], vy[k], xs, ys, sc * 1e3) for k in range(len(v.ano_inicio))])
        dmeas = to_mask_grid(np.nanmean(divs, 0), xs, ys, mx, my)
        hd_its = fl[f"H_div_gemb_{sc}km"].values
        for u in UNITS:
            s = dom & (unit == u) & np.isfinite(dmeas) & np.isfinite(hd_its) & np.isfinite(H)
            a_m, a_i = float(np.mean(H[s] * dmeas[s])), float(np.mean(hd_its[s]))
            rows.append({"unidade": u, "escala_km": sc, "frac_dominio": s.sum() / max((dom & (unit == u)).sum(), 1),
                         "Hdiv_ITS_LIVE_mpa": a_i, "Hdiv_MEaSUREs_mpa": a_m,
                         "diferenca_mpa": a_m - a_i,
                         "r_espacial": float(np.corrcoef(H[s] * dmeas[s], hd_its[s])[0, 1])})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "Hdiv_ITS_LIVE_vs_MEaSUREs.csv", index=False)
    pd.set_option("display.width", 200)
    print(t.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
