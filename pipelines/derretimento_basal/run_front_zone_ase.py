"""Faixa da frente de gelo: diagnóstico e exclusão (item 3 da lista de pendências do basal).

Duas camadas, ambas medidas:
1. **Perda de gelo:** fração dos segmentos ICESat-2 do pixel removidos pelo filtro de presença de
   gelo (``run_thickness_ase.ice_presence_filter``). Fração > 10% → a frente passou pelo pixel em
   2019–2025 (água ou mélange em parte das passagens). Cobre 2025, que o IceLines não tem.
2. **Distância à frente:** distância de cada pixel à posição mais próxima da frente do IceLines
   (Sentinel-1, DLR, mensal) em qualquer data de 2019,0 a 2024,7. A largura excluída sai do perfil
   do m_b e do σ por pixel em função dessa distância (``perfil_frente.csv``), não do valor fixo
   de 2 km de Meng.

Saída: camada ``frente_dist_km`` e ``perda_gelo_frac`` em ``data/interim/basal_melt/front_zone_500m.nc``
e o perfil em ``outputs/front_zone/``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "pipelines" / "derretimento_basal"))
from run_thickness_ase import ice_presence_filter  # noqa: E402

MB = ROOT / "data" / "interim" / "basal_melt" / "mb_continuo_500m.nc"
THK = ROOT / "data" / "interim" / "thickness"
FRONTS = ROOT / "data" / "calving_fronts_densified_500m.parquet"
OUT_NC = ROOT / "data" / "interim" / "basal_melt" / "front_zone_500m.nc"
OUT = ROOT / "outputs" / "front_zone"
SHELF_FRONTS = {"Pine_Island": ["PineIsland"], "Thwaites_TEIS": ["Thwaites1", "Thwaites2"],
                "Thwaites_TWGT": ["Thwaites1", "Thwaites2"], "Crosson": ["Crosson"],
                "Dotson": ["Dotson"], "Cosgrove": ["Cosgrove"]}
D_FRONT, LOST_MAX = 1.5, 0.10
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    mb = xr.open_dataset(MB)
    xs, ys = mb.x.values, mb.y.values
    XX, YY = np.meshgrid(xs, ys)
    unit = mb.unidade.values
    fr = pd.read_parquet(FRONTS)
    fr = fr[(fr.epoch_year >= 2019.0) & (fr.epoch_year <= 2025.0)]

    dist = np.full(unit.shape, np.nan)
    for u, names in SHELF_FRONTS.items():
        s = unit == u
        f = fr[fr.shelf.isin(names)]
        if f.empty or not s.any():
            continue
        d, _ = cKDTree(f[["x", "y"]].to_numpy()).query(np.c_[XX[s], YY[s]])
        dist[s] = d / 1e3

    # fração de segmentos removidos pelo filtro de presença de gelo, por pixel
    lost = np.full(unit.shape, np.nan)
    for k, name in NAMES.items():
        d = pd.read_parquet(THK / f"espessura_{name}.parquet",
                            columns=["x", "y", "h_sea", "H_gemb", "classe"])
        d = d[(d["classe"] == 1) & np.isfinite(d["H_gemb"])]
        keep = ice_presence_filter(d["x"].to_numpy(), d["y"].to_numpy(), d["h_sea"].to_numpy())
        j = np.floor((d["x"].to_numpy() - xs[0]) / (xs[1] - xs[0]) + 0.5).astype(np.int64)
        i = np.floor((d["y"].to_numpy() - ys[0]) / (ys[1] - ys[0]) + 0.5).astype(np.int64)
        ok = (i >= 0) & (i < len(ys)) & (j >= 0) & (j < len(xs))
        key = i[ok] * len(xs) + j[ok]
        frac = pd.Series((~keep[ok]).astype(float)).groupby(key).mean()
        flat = lost.ravel()
        flat[frac.index.to_numpy()] = frac.to_numpy()
        lost = flat.reshape(unit.shape)


    m, se = mb.mb.values, mb.se_mb.values
    bins = np.r_[0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 20, 60]
    rows = []
    for u in SHELF_FRONTS:
        s = (unit == u) & np.isfinite(m) & np.isfinite(dist)
        cut = pd.cut(dist[s], bins)
        g = pd.DataFrame({"faixa": cut, "mb": m[s], "se": se[s], "perda": lost[s]}).groupby(
            "faixa", observed=True)
        t = g.agg(n=("mb", "size"), mb_mediana=("mb", "median"), mb_media=("mb", "mean"),
                  se_mediano=("se", "median"), perda_media=("perda", "mean")).reset_index()
        t.insert(0, "unidade", u)
        rows.append(t)
    prof = pd.concat(rows)
    prof.to_csv(OUT / "perfil_frente.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    print(prof.round(2).to_string(index=False))
    # exclusão: ≤ D_FRONT km de qualquer frente 2019–2024,7 ou perda de gelo > 10%.
    # D_FRONT = 1,5 km: no perfil, a PIG estabiliza a partir de 1,5 km (mediana 30 → 12 m/a),
    # a TEIS a partir de ~1 km e o Crosson a partir de ~1,5 km.
    excl = (dist <= D_FRONT) | (lost > LOST_MAX)
    summ = []
    for u in SHELF_FRONTS:
        s = (unit == u) & np.isfinite(m)
        keep = s & ~excl
        summ.append({"unidade": u, "frac_perda_gelo": float(np.nanmean(lost[unit == u] > LOST_MAX)),
                     "frac_excluida": float(excl[s].mean()),
                     "mb_antes_mpa": float(np.mean(m[s])), "mb_depois_mpa": float(np.mean(m[keep])),
                     "area_antes_km2": s.sum() * 0.25, "area_depois_km2": keep.sum() * 0.25})
    t = pd.DataFrame(summ)
    t.to_csv(OUT / "efeito_exclusao_frente.csv", index=False)
    print(t.round(2).to_string(index=False))
    ds = xr.Dataset({"frente_dist_km": (("y", "x"), dist.astype(np.float32)),
                     "perda_gelo_frac": (("y", "x"), lost.astype(np.float32)),
                     "exclui_frente": (("y", "x"), excl.astype(np.int8))},
                    coords={"x": xs, "y": ys})
    ds["exclui_frente"].attrs = {"regra": f"dist <= {D_FRONT} km de frente IceLines 2019-2024.7 "
                                          f"ou perda de gelo > {LOST_MAX:.0%}"}
    ds.to_netcdf(OUT_NC)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
