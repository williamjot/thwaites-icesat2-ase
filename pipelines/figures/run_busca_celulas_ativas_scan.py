# -*- coding: utf-8 -*-
"""Busca de células ativas para a série sazonal (2026-11-19): extrai do legacy_ase (mesma cadeia da série TEIS e
de run_grounded_legacy_cells_scan.py) os pontos aterrados de TODAS as células de 5 km com dh/dt ≤ −1 m/ano no
mapa DJF atual (perfil djf_glrecente) nas caixas da Thwaites e da Pine Island. Uma passada pelos grânulos já em
disco; nada é baixado. Saída nova (não toca em grounded_legacy_cells_raw.parquet):
data/processed/busca_celulas_ativas_raw.parquet, com cell = "x_y" em km do centro.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(r"D:/WILLIAM/ARTIGO PIBIC ANTARTICA")
OUT = ROOT / "data" / "processed" / "busca_celulas_ativas_raw.parquet"
CELLS_CSV = ROOT / "data" / "processed" / "busca_celulas_ativas_candidatas.csv"
HALF = 2500.0
COLS = ["row_idx", "x", "y", "h_elv", "s_elv", "t_year", "beam", "tide_ocean", "tide_equilibrium", "dac", "geoid",
        "bm_mask"]
CAIXAS = {"Thwaites": (-1620e3, -1380e3, -560e3, -370e3), "Pine_Island": (-1700e3, -1520e3, -370e3, -200e3)}


def candidatas():
    exps = sorted((ROOT / "outputs/djf_glrecente/experiments").glob("*_firn"))
    exps = [e for e in exps if (e / "firn_corrected_grid.parquet").exists()]
    g = pd.read_parquet(exps[-1] / "firn_corrected_grid.parquet")[["x", "y", "dhdt_ice"]]
    rows = []
    for nome, (x0, x1, y0, y1) in CAIXAS.items():
        s = g[(g.x > x0) & (g.x < x1) & (g.y > y0) & (g.y < y1) & (g.dhdt_ice <= -1.0)].copy()
        s["geleira"] = nome
        rows.append(s)
    c = pd.concat(rows).reset_index(drop=True)
    c["cell"] = [f"{x / 1e3:.1f}_{y / 1e3:.1f}" for x, y in zip(c.x, c.y)]
    return c


def main():
    c = candidatas()
    c.to_csv(CELLS_CSV, index=False)
    print(len(c), "células candidatas;", c.geleira.value_counts().to_dict(), flush=True)
    cx, cy = c.x.to_numpy(), c.y.to_numpy()
    from scipy.spatial import cKDTree
    tree = cKDTree(np.c_[cx, cy])
    xmin, xmax, ymin, ymax = cx.min() - HALF, cx.max() + HALF, cy.min() - HALF, cy.max() + HALF
    man = pd.read_csv(ROOT / "data/processed/legacy_ase_manifest.csv")
    man = man[man.fraction_inside_roi > 0]
    parts, n = [], 0
    for i, r in enumerate(man.itertuples(index=False)):
        f = ROOT / "data/legacy_ase" / r.season.lower() / r.source_file
        if not f.exists():
            continue
        t = pq.read_table(f, columns=COLS, filters=[("x", ">=", xmin), ("x", "<", xmax), ("y", ">=", ymin),
                                                     ("y", "<", ymax)]).to_pandas()
        if t.empty:
            continue
        d, k = tree.query(np.c_[t.x.to_numpy(), t.y.to_numpy()], p=np.inf)   # dentro do quadrado de 5 km
        m = d < HALF
        if not m.any():
            continue
        t = t[m].copy()
        t["cell"] = c.cell.to_numpy()[k[m]]
        t["source_file"] = r.source_file
        t["season_file"] = r.season
        t["rgt"] = int(r.rgt)
        t["cycle"] = int(r.cycle)
        parts.append(t)
        n += len(t)
        if i % 500 == 0:
            print(f"{i}/{len(man)} grânulos | {n:,} linhas", flush=True)
    out = pd.concat(parts, ignore_index=True)
    out.to_parquet(OUT, index=False)
    print("gravado", OUT, f"{len(out):,} linhas", out.cell.nunique(), "células", flush=True)


if __name__ == "__main__":
    main()
