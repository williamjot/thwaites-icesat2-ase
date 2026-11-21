# -*- coding: utf-8 -*-
"""Pass 1: conta observacoes por (celula 5 km, janela sazonal) em gelo aterrado.

Le apenas x, y, t_year, mask_class, track_id dos tiles que intersectam a caixa
do tronco aterrado de Thwaites. Nao decide nada por sinal - so por cobertura.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
sys.path.insert(0, str(ROOT))
from thwaites.atl06_granule_id import t_year_to_datetime  # noqa: E402

OUT = ROOT / "data" / "processed"

BOX = (-1560000.0, -1420000.0, -700000.0, -400000.0)
CELL = 5000.0
TARGET = ("JJA 2019", "DJF 2019/2020", "JJA 2020", "DJF 2020/2021",
          "JJA 2021", "DJF 2021/2022", "JJA 2022")
COLS = ["x", "y", "t_year", "mask_class", "track_id"]


def window_labels(t_year):
    out = []
    for v in np.asarray(t_year, dtype=float):
        d = t_year_to_datetime(float(v))
        if d.month in (6, 7, 8):
            out.append("JJA %d" % d.year)
        elif d.month == 12:
            out.append("DJF %d/%d" % (d.year, d.year + 1))
        elif d.month in (1, 2):
            out.append("DJF %d/%d" % (d.year - 1, d.year))
        else:
            out.append("OUT")
    return np.asarray(out, dtype=object)


def cell_index(x, y):
    """Centros de celula na MESMA convencao da grade legacy de 5 km
    (centros em ...2500, ou seja multiplos de 5000 deslocados de 2500)."""
    cx = np.floor(x / CELL) * CELL + CELL / 2.0
    cy = np.floor(y / CELL) * CELL + CELL / 2.0
    return cx, cy


parts = []
for prof in ("jja", "djf"):
    man = json.load(open(ROOT / f"data/{prof}/tiles/manifest.json"))
    sel = [e for e in man if not (e["x_max"] < BOX[0] or e["x_min"] > BOX[1]
                                  or e["y_max"] < BOX[2] or e["y_min"] > BOX[3])]
    print(f"[{prof}] {len(sel)} tiles", flush=True)
    for i, e in enumerate(sel):
        t = pq.read_table(ROOT / f"data/{prof}/tiles" / e["file"],
                          columns=COLS).to_pandas()
        # nucleo do tile (evita contagem dupla pelo halo)
        t = t[(t.x >= e["x_min"]) & (t.x < e["x_max"])
              & (t.y >= e["y_min"]) & (t.y < e["y_max"])]
        t = t[(t.mask_class == 2)
              & t.x.between(BOX[0], BOX[1]) & t.y.between(BOX[2], BOX[3])]
        if t.empty:
            continue
        t = t[(t.t_year >= 2019.0) & (t.t_year <= 2023.0)]
        if t.empty:
            continue
        t["window"] = window_labels(t.t_year.to_numpy())
        t = t[t.window.isin(TARGET)]
        if t.empty:
            continue
        cx, cy = cell_index(t.x.to_numpy(), t.y.to_numpy())
        t["cx"], t["cy"] = cx, cy
        agg = (t.groupby(["cx", "cy", "window"], observed=True)
               .agg(n=("x", "size"), n_tracks=("track_id", "nunique"))
               .reset_index())
        agg["profile"] = prof
        parts.append(agg)
        print(f"  [{prof}] {i + 1}/{len(sel)} {e['file']} -> {len(agg)} pares",
              flush=True)

counts = pd.concat(parts, ignore_index=True)
counts = (counts.groupby(["cx", "cy", "window"], observed=True)
          .agg(n=("n", "sum"), n_tracks=("n_tracks", "sum")).reset_index())
counts.to_csv(OUT / "grounded_seasonal_cell_counts.csv", index=False)
print("celulas distintas:", counts.groupby(["cx", "cy"]).ngroups)
print("salvo em", OUT / "grounded_seasonal_cell_counts.csv")
