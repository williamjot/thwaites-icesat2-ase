"""Domínio final do produto de derretimento basal e médias/massa por unidade.

Exclui um pixel se valer qualquer uma das condições, todas medidas:
1. faixa da frente (≤ 1,5 km de frente IceLines 2019–2024,7) ou perda de gelo > 10%
   (``run_front_zone_ase.py``);
2. dentro de d\\* da linha de aterramento do BedMachine pela resposta à maré (máscara f = 0,9);
3. dentro de d\\*_recente das linhas de aterramento de 2019–2025 (InSAR v2.1 e Sentinel-1 2025),
   pela mesma resposta à maré: PIG 3,0; Thwaites 1,5; Crosson 3,0; Dotson 2,5; Cosgrove 2,0 km;
4. dentro de d\\*_radar: onde a razão MCoRDS radar/hidrostático, normalizada por voo, se afasta de
   1 em mais de 3%: Thwaites 4,0 km (0,94–0,95 entre 2 e 4 km), PIG 2,0, Dotson 0,5,
   Cosgrove 0,5, Crosson 0 km. Na Thwaites, esse erro (~5% de H a montante) gera ~7 m/a de
   afinamento aparente das parcelas, a mesma ordem da diferença entre o produto e o ApRES
   G1–G4 na zona de aterramento da TEIS.

Saídas: ``data/interim/basal_melt/dominio_final_500m.nc`` e
``outputs/basal_melt_ase/mb_final_por_unidade.csv`` (m/a e Gt/a no domínio medido).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
FZ = ROOT / "data" / "interim" / "basal_melt" / "front_zone_500m.nc"
DREC = ROOT / "data" / "interim" / "basal_melt" / "dist_LA_recente_500m.nc"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
OUT_NC = ROOT / "data" / "interim" / "basal_melt" / "dominio_final_500m.nc"
OUT = ROOT / "outputs" / "basal_melt_ase"
D_REC = {"Pine_Island": 3.0, "Thwaites_TEIS": 1.5, "Thwaites_TWGT": 1.5, "Crosson": 3.0,
         "Dotson": 2.5, "Cosgrove": 2.0}
D_RADAR = {"Pine_Island": 2.0, "Thwaites_TEIS": 4.0, "Thwaites_TWGT": 4.0, "Crosson": 0.0,
           "Dotson": 0.5, "Cosgrove": 0.5}
RHO_I = 917.0


def main() -> int:
    mb = xr.open_dataset(MB)
    u, m, se = mb.unidade.values, mb.mb.values, mb.se_mb.values
    front = xr.open_dataset(FZ).exclui_frente.values == 1
    drec = xr.open_dataset(DREC).dist_LA_recente_km.values
    dbm = xr.open_dataset(MASK).dist_gl_km.values
    keep = np.zeros(u.shape, bool)
    rows = []
    for un in D_REC:
        s = (u == un) & np.isfinite(m)
        k = s & ~front & (drec >= D_REC[un]) & (dbm >= D_RADAR[un])
        keep |= k
        rows.append({"unidade": un, "area_km2": k.sum() * 0.25,
                     "frac_do_dominio_hidrostatico": k.sum() / s.sum(),
                     "mb_mpa": float(np.mean(m[k])), "mb_mediana_mpa": float(np.median(m[k])),
                     "se_pixel_mediano_mpa": float(np.median(se[k])),
                     "Gt_dominio_final": float(np.mean(m[k]) * k.sum() * 0.25e6 * RHO_I / 1e12)})
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "mb_final_por_unidade.csv", index=False)
    xr.Dataset({"dominio_final": (("y", "x"), keep.astype(np.int8))},
               coords={"x": mb.x.values, "y": mb.y.values}).to_netcdf(OUT_NC)
    pd.set_option("display.width", 200)
    print(t.round(2).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
