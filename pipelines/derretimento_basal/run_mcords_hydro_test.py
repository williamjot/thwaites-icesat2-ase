"""Teste do erro da hipótese hidrostática com radar MCoRDS (IceBridge L2, 2009–2018).

Cada linha de voo mede a superfície (laser/radar) e a espessura (radar) ao mesmo tempo, então a
comparação não depende da época. Para cada ponto sobre gelo flutuante ou perto da linha:

    H_hidro_real = 9,336·h − 8,336·FAC,   h = SURFACE − geoide(EIGEN-6C4) − MDT(DTU15) − maré(CATS)

é a espessura **total** (com firn) prevista pela flutuação. É a grandeza comparável à espessura
do radar. FAC: firn do BedMachine (RACMO/IMAU-FDM, estático), porque o GEMB e o IMAU locais só
começam em 2018.

Saída: razão e diferença H_radar − H_hidro em função da distância à linha de aterramento do
BedMachine, por plataforma (``outputs/mcords_hydro_test/``), comparadas à máscara hidrostática (d\\*).

Limitações: o radar usa ε do gelo também no firn (erro de alguns metros); a linha de aterramento
do BedMachine não acompanha o recuo de 2009–2018; a qualidade 1 do CReSIS é a única aceita.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thwaites import load_config  # noqa: E402
from thwaites.corrections.cats_tide import CatsTidePredictor  # noqa: E402

SRC = ROOT / "data" / "validation" / "mcords"
MASK = ROOT / "data" / "interim" / "hydrostatic_mask_500m.nc"
BED = ROOT / "data" / "NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
MDT = ROOT / "data" / "vertical_reference" / "dtu15mdt_ase.nc"
OUT = ROOT / "outputs" / "mcords_hydro_test"
A, B = 1027.0 / 110.0, 917.0 / 110.0
NAMES = {1: "Pine_Island", 2: "Thwaites", 3: "Crosson", 4: "Dotson", 5: "Cosgrove"}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    m = xr.open_dataset(MASK)
    xs, ys = m.x.values, m.y.values
    plat, cls = m.plataforma.values, m.classe.values
    bm = xr.open_dataset(BED)
    mdt = xr.open_dataset(MDT)
    # distância assinada à linha do BedMachine na faixa ±15 km (recalculada como no β)
    sub = bm.sel(x=slice(xs.min() - 20e3, xs.max() + 20e3), y=slice(ys.max() + 20e3, ys.min() - 20e3))
    from scipy import ndimage
    mk = sub["mask"].values
    flo, grd = mk == 3, np.isin(mk, [1, 2, 4])
    sd = np.where(flo, ndimage.distance_transform_edt(flo) * 0.5,
                  np.where(grd, -ndimage.distance_transform_edt(grd) * 0.5, np.nan))
    frames = []
    for f in sorted(SRC.glob("*.csv")):
        d = pd.read_csv(f).rename(columns={"UTCTIMESOD": "TIME"})   # 20 dos 84 CSV usam esse nome
        d = d[(d.QUALITY == 1) & (d.THICK > 0) & (d.SURFACE > -9000)]
        if d.empty:
            continue
        day = datetime.strptime(f.name.split("_")[1], "%Y%m%d")
        x, y = tf.transform(d.LON.to_numpy(), d.LAT.to_numpy())
        d = d.assign(x=x, y=y, data=day.strftime("%Y-%m-%d"),
                     t_year=2018.0 + ((day - datetime(2018, 1, 1)).total_seconds()
                                      + (d.TIME.to_numpy() % 86400)) / (365.25 * 86400))
        frames.append(d)
    d = pd.concat(frames, ignore_index=True)
    X, Y = xr.DataArray(d.x.to_numpy()), xr.DataArray(d.y.to_numpy())
    d["dist_gl_km"] = xr.DataArray(sd, coords={"y": sub.y.values, "x": sub.x.values},
                                   dims=("y", "x")).interp(x=X, y=Y, method="nearest").values
    d = d[np.isfinite(d.dist_gl_km) & (d.dist_gl_km.abs() <= 15)].reset_index(drop=True)
    X, Y = xr.DataArray(d.x.to_numpy()), xr.DataArray(d.y.to_numpy())
    d["geoid"] = bm.geoid.interp(x=X, y=Y).values
    d["firn"] = bm.firn.interp(x=X, y=Y).values
    d["mdt"] = mdt.mdt.interp(lat=xr.DataArray(d.LAT.to_numpy()),
                              lon=xr.DataArray(d.LON.to_numpy())).values
    j = np.clip(np.floor((d.x.to_numpy() - xs[0]) / (xs[1] - xs[0]) + 0.5).astype(int), 0, len(xs) - 1)
    i = np.clip(np.floor((d.y.to_numpy() - ys[0]) / (ys[1] - ys[0]) + 0.5).astype(int), 0, len(ys) - 1)
    d["plataforma"] = [NAMES.get(int(p), "") for p in plat[i, j]]
    d["classe"] = cls[i, j]
    with CatsTidePredictor(load_config()) as pred:
        tide = np.concatenate([pred.predict(d.LON.to_numpy()[a:a + 100000], d.LAT.to_numpy()[a:a + 100000],
                                            d.t_year.to_numpy()[a:a + 100000])
                               for a in range(0, len(d), 100000)])
    d["tide"] = tide
    # no L2 do CReSIS, SURFACE é a DISTÂNCIA do avião até a superfície (m) e ELEVATION é a
    # altura do avião sobre o WGS84; a altura da superfície é a diferença. Uma 1ª versão usou
    # SURFACE como altura e deu H_hidro ~4.500 m.
    d["surf_elev"] = d.ELEVATION - d.SURFACE
    h = d.surf_elev - d.geoid - d.mdt.fillna(0) - d.tide
    d["H_hidro"] = A * h - B * d.firn
    d["razao"] = d.THICK / d.H_hidro
    d["dif_m"] = d.THICK - d.H_hidro
    d.to_parquet(OUT / "pontos.parquet", index=False)
    params = json.loads(m.classe.attrs["parametros"])
    bins = np.r_[-6, -4, -2, -1, 0, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 11, 15]
    rows = []
    for p, g in d[d.plataforma != ""].groupby("plataforma"):
        gg = g[np.isfinite(g.razao) & (g.H_hidro > 50)]
        cut = pd.cut(gg.dist_gl_km, bins)
        t = gg.groupby(cut, observed=True).agg(n=("razao", "size"), razao_med=("razao", "median"),
                                               dif_med_m=("dif_m", "median"),
                                               H_radar_med=("THICK", "median"),
                                               n_voos=("data", "nunique")).reset_index()
        t.insert(0, "plataforma", p)
        t["d_star_f090_km"] = params[p]["d_star_km"]["f090"]
        rows.append(t)
    r = pd.concat(rows)
    r.to_csv(OUT / "perfil_razao.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 300)
    print(r.round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
