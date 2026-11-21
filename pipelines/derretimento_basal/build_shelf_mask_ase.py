"""Máscara de plataformas de gelo da ASE para auditoria de cobertura (grade 240 m, EPSG:3031).

Para que serve: dizer a que plataforma pertence cada ponto ATL06 sobre gelo flutuante. Ainda NÃO
é a máscara do produto basal: a zona onde a hipótese hidrostática vale será definida depois, pela
resposta medida à maré, e não por esta máscara estática.

Classe = plataforma (``iceshelf_id`` do NSIDC-0794, Greene et al.)
       ∩ gelo presente na última máscara anual do NSIDC-0794 (mar/2021)
       ∩ flutuante no BedMachine v4.1 (mask == 3).

Limitações declaradas:
- o ``iceshelf_id`` é dilatado e extrudido (README de data/external_geometry); por isso a
  interseção com o BedMachine e com o ``ice_mask`` é obrigatória;
- a linha de aterramento do BedMachine é estática e não acompanha o recuo da Thwaites;
- a frente é a de 2021; o tronco da Thwaites perdeu gelo em 2024 (ver evento_2024/).

Saída: ``data/interim/ase_shelf_mask_240m.nc`` com ``shelf`` (int8) e o atributo de legenda.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from pyproj import Transformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from thwaites import load_config  # noqa: E402

GREENE = ROOT / "data/external_geometry/NSIDC-0794_v1/NSIDC-0794_19971001-20210314_V01.0.nc"
BEDMACHINE = ROOT / "data/NSIDC-0756_BedMachineAntarctica_19700101-20191001_V04.1.nc"
OUT = ROOT / "data/interim/ase_shelf_mask_240m.nc"
SHELVES = ["Pine_Island", "Thwaites", "Crosson", "Dotson", "Cosgrove", "Getz"]


def main() -> int:
    cfg = load_config()
    r = cfg.roi
    tf = Transformer.from_crs(4326, 3031, always_xy=True)
    lon = np.r_[np.linspace(r.lon_min, r.lon_max, 50), np.full(50, r.lon_max),
                np.linspace(r.lon_min, r.lon_max, 50), np.full(50, r.lon_min)]
    lat = np.r_[np.full(50, r.lat_min), np.linspace(r.lat_min, r.lat_max, 50),
                np.full(50, r.lat_max), np.linspace(r.lat_min, r.lat_max, 50)]
    X, Y = tf.transform(lon, lat)
    x0, x1, y0, y1 = X.min() - 20e3, X.max() + 20e3, Y.min() - 20e3, Y.max() + 20e3

    g = xr.open_dataset(GREENE)
    names = g["iceshelf_id"].attrs["flag_meanings"].split()
    values = list(g["iceshelf_id"].attrs["flag_values"])
    ids = {n: int(values[names.index(n)]) for n in SHELVES}
    sub = g.sel(x=slice(x0, x1), y=slice(y1, y0))          # y decrescente no arquivo
    sid = sub["iceshelf_id"].values
    ice = sub["ice_mask"].isel(time=-1).values == 1
    t_last = str(sub["time"].values[-1])[:10]
    xs, ys = sub["x"].values.astype(float), sub["y"].values.astype(float)

    bm = xr.open_dataset(BEDMACHINE)
    bmm = bm["mask"].interp(x=xr.DataArray(xs, dims="x"), y=xr.DataArray(ys, dims="y"),
                            method="nearest").values
    floating = bmm == 3

    shelf = np.zeros(sid.shape, np.int8)
    for k, n in enumerate(SHELVES, 1):
        shelf[(sid == ids[n]) & ice & floating] = k
    area = {n: float((shelf == k).sum()) * 0.24 ** 2 for k, n in enumerate(SHELVES, 1)}
    ds = xr.Dataset({"shelf": (("y", "x"), shelf)}, coords={"x": xs, "y": ys})
    ds["shelf"].attrs = {
        "legenda": "; ".join(f"{k}={n}" for k, n in enumerate(SHELVES, 1)) + "; 0=outro",
        "fontes": f"NSIDC-0794 iceshelf_id e ice_mask ({t_last}); BedMachine v4.1 mask==3",
        "crs": "EPSG:3031"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(OUT)
    print(f"ice_mask usado: {t_last}")
    for n, a in area.items():
        print(f"{n:12s} {a:9.0f} km²")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
