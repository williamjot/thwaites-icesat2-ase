# -*- coding: utf-8 -*-
"""Extrai do legacy_ase (mesma cadeia de dados da serie TEIS) os pontos
aterrados de duas celulas candidatas, 2019-2025, JJA e DJF.

Guarda linhas cruas + source_file/row_idx/rgt/cycle para que o QC da cadeia
legacy seja juntado depois, sem decidir nada aqui.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
SCRATCH = Path(__file__).resolve().parent

# Celulas padrao (comportamento historico do script). Outras entram por
# --cells nome:x:y, e --out manda para outro parquet, para que um recorte novo
# nunca sobrescreva o produto ja usado pela serie publicada.
DEFAULT_CELLS = {"nucleo": (-1462500.0, -642500.0),
                 "celula_52km": (-1472500.0, -437500.0)}
DEFAULT_OUT = ROOT / "data" / "processed" / "grounded_legacy_cells_raw.parquet"

_ap = argparse.ArgumentParser(description="Extrai pontos aterrados por celula.")
_ap.add_argument("--cells", nargs="*", default=None,
                 help="celulas como nome:x:y (metros, EPSG:3031)")
_ap.add_argument("--out", default=None, help="parquet de saida")
_args = _ap.parse_args()

if _args.cells:
    CELLS = {}
    for spec in _args.cells:
        nome, sx, sy = spec.split(":")
        CELLS[nome] = (float(sx), float(sy))
else:
    CELLS = dict(DEFAULT_CELLS)
OUT_PARQUET = Path(_args.out) if _args.out else DEFAULT_OUT
HALF = 2500.0
print("celulas:", CELLS)
print("saida:", OUT_PARQUET)
COLS = ["row_idx", "x", "y", "h_elv", "s_elv", "t_year", "beam", "tide_ocean",
        "tide_equilibrium", "dac", "geoid", "bm_mask"]

man = pd.read_csv(ROOT / "data/processed/legacy_ase_manifest.csv")
man = man[man.fraction_inside_roi > 0].copy()
xmin = min(c[0] for c in CELLS.values()) - HALF
xmax = max(c[0] for c in CELLS.values()) + HALF
ymin = min(c[1] for c in CELLS.values()) - HALF
ymax = max(c[1] for c in CELLS.values()) + HALF

parts = []
for i, r in enumerate(man.itertuples(index=False)):
    f = ROOT / "data/legacy_ase" / r.season.lower() / r.source_file
    if not f.exists():
        continue
    t = pq.read_table(f, columns=COLS,
                      filters=[("x", ">=", xmin), ("x", "<", xmax),
                               ("y", ">=", ymin), ("y", "<", ymax)]).to_pandas()
    if t.empty:
        continue
    keep = np.zeros(len(t), bool)
    lab = np.empty(len(t), dtype=object)
    for name, (cx, cy) in CELLS.items():
        m = ((t.x >= cx - HALF) & (t.x < cx + HALF)
             & (t.y >= cy - HALF) & (t.y < cy + HALF)).to_numpy()
        keep |= m
        lab[m] = name
    if not keep.any():
        continue
    t = t[keep].copy()
    t["cell"] = lab[keep]
    t["source_file"] = r.source_file
    t["season_file"] = r.season
    t["rgt"] = int(r.rgt)
    t["cycle"] = int(r.cycle)
    parts.append(t)
    if i % 100 == 0:
        print(f"{i}/{len(man)} granulos | linhas acumuladas "
              f"{sum(len(p) for p in parts)}", flush=True)

obs = pd.concat(parts, ignore_index=True)
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
obs.to_parquet(OUT_PARQUET, index=False)
print("total", len(obs))
print(obs.groupby(["cell", "season_file", "bm_mask"]).size())
